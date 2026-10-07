"""MCP manager: lazy connection per server, idle stop, restart with backoff,
concurrency and result-size caps per server. One failing server never stops another:
every method that talks to a server raises `McpServerError`, never lets the
underlying SDK/transport exception escape uncaught, so a caller iterating several
servers can catch just that one class.

Transports: `stdio`, restricted to a vetted built-in (`app.mcp.builtin.REGISTRY`,
never an admin-supplied command — see that module's docstring), launched with the
SDK's own safe default environment (only HOME/LOGNAME/PATH/SHELL/TERM/USER; a
server's own declared `env_vars` are added on top, never this application's
secrets, which are never in that default set in the first place); and `http`
(streamable HTTP), an exact URL re-checked by the outbound guard on every
connection, not only when declared, so a name that later repoints does not get a
free pass.
"""

import asyncio
import contextlib
import logging
import sys
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from app.db.models import McpEgress, McpTransport
from app.mcp.builtin import REGISTRY as BUILTIN_REGISTRY
from app.mcp.builtin import builtin_environment
from app.security.outbound import Guard

logger = logging.getLogger("channelagent")

# A session unused this long is closed rather than kept open indefinitely; the next
# call reconnects lazily.
IDLE_TIMEOUT_SECONDS = 300.0

# A server that failed to connect is not retried immediately: exponential backoff,
# capped, so a persistently broken server does not get hammered once per call.
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 60.0


class McpServerError(Exception):
    """A connection or a call failed; the message is safe to show an administrator
    or return to the model as the tool's result."""


@dataclass(frozen=True)
class ServerConfig:
    name: str
    protocol: McpTransport
    builtin_id: str | None = None
    url: str | None = None
    env_vars: dict = field(default_factory=dict)
    egress: McpEgress = McpEgress.LOCAL
    timeout_seconds: int = 20
    concurrency_limit: int = 2
    result_max_bytes: int = 1_000_000
    disabled_tools: tuple = ()
    # Read by app.mcp.catalogue, not by the connection itself.
    tool_policies: dict = field(default_factory=dict)
    approved_hashes: dict = field(default_factory=dict)  # tool name -> sha256
    has_credentials: bool = False
    shared_credentials: bool = False
    confirm_timeout_seconds: int = 120
    tool_classes: dict = field(default_factory=dict)  # Overrides by tool name


class ManagedServer:
    """One server's live connection and its own concurrency limit. Not shared
    across event loops (the anyio streams a connection opens belong to the loop
    that opened them, the same constraint as app.checkpoints' saver).
    """

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._stack: contextlib.AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(config.concurrency_limit)
        self._last_used = 0.0
        self._next_retry_at = 0.0
        self._backoff = BACKOFF_BASE_SECONDS

    async def _connect(self) -> None:
        stack = contextlib.AsyncExitStack()
        try:
            if self.config.protocol == McpTransport.STDIO:
                module = BUILTIN_REGISTRY.get(self.config.builtin_id or "")
                if module is None:
                    raise McpServerError(f"no vetted built-in server {self.config.builtin_id!r}")
                env = builtin_environment(self.config.builtin_id)
                env.update(self.config.env_vars or {})
                params = StdioServerParameters(
                    command=sys.executable,
                    args=["-m", module],
                    env=env or None,
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            elif self.config.protocol == McpTransport.HTTP:
                if not self.config.url:
                    raise McpServerError(f"{self.config.name} has no URL")
                host = urlsplit(self.config.url).hostname or ""
                guard = Guard(
                    allowed_hosts=frozenset({host.lower()}),
                    allow_http=self.config.egress != McpEgress.INTERNET,
                    allow_private=self.config.egress != McpEgress.INTERNET,
                )
                target = guard.prepare(self.config.url)
                read, write, _get_session_id = await stack.enter_async_context(
                    streamable_http_client(target.url)
                )
            else:  # pragma: no cover - the enum has only these two members
                raise McpServerError(f"unknown protocol {self.config.protocol!r}")
            session = await stack.enter_async_context(ClientSession(read, write))
            # Only the handshake itself is time-bounded here, not the context
            # managers above: they open long-lived resources this method keeps
            # alive past its own return (in self._stack), and anyio requires a
            # cancel scope to close within the same task that opened it — wrapping
            # the whole method in one would make it outlive its own scope.
            with anyio.fail_after(self.config.timeout_seconds):
                await session.initialize()
        except BaseException:
            # BaseException, not Exception: a slow handshake cut off by fail_after
            # arrives here as a TimeoutError/Cancelled, and the partially opened
            # stack (a spawned subprocess, an open HTTP connection) must still be
            # closed, not leaked.
            with contextlib.suppress(Exception):
                await stack.aclose()
            raise
        self._stack = stack
        self._session = session
        self._backoff = BACKOFF_BASE_SECONDS

    async def disconnect(self) -> None:
        stack, self._stack, self._session = self._stack, None, None
        if stack is not None:
            try:
                await stack.aclose()
            except BaseException:
                # BaseException, not Exception: closing a stdio/HTTP connection can
                # surface the SDK's own internal anyio task-group cleanup as a
                # CancelledError even on an ordinary, expected disconnect — this is
                # cleanup, so nothing here is worth losing an outer failure over.
                logger.debug(
                    "Closing MCP server %r raised during cleanup, ignoring",
                    self.config.name,
                    exc_info=True,
                )

    async def _get_session(self) -> ClientSession:
        async with self._lock:
            now = time.monotonic()
            if self._session is not None and now - self._last_used > IDLE_TIMEOUT_SECONDS:
                await self.disconnect()
            if self._session is None:
                if now < self._next_retry_at:
                    raise McpServerError(
                        f"{self.config.name} is backing off after a recent failure"
                    )
                try:
                    await self._connect()
                except BaseException as exc:
                    # BaseException, not Exception: a connection that fails fast (a
                    # refused port) can surface from the SDK's own internal anyio
                    # task group as a sibling-task CancelledError rather than the
                    # original error — a real connection failure, not this task
                    # being cancelled from outside, which is caught and tested
                    # separately (a *running* tool call, app.tools.run_tool_loop).
                    self._next_retry_at = time.monotonic() + self._backoff
                    self._backoff = min(self._backoff * 2, BACKOFF_MAX_SECONDS)
                    logger.warning("MCP server %r unreachable: %s", self.config.name, exc)
                    raise McpServerError(f"{self.config.name} could not be reached") from exc
            self._last_used = now
            assert self._session is not None
            return self._session

    async def list_tools(self):
        """Every tool this server offers right now, whether or not an
        administrator turned it off (`disabled_tools`) — the admin API's own
        listing needs to show a disabled tool to let it be turned back on;
        app.mcp.catalogue.build_tools is what excludes a disabled tool from what
        a live turn may call, and call_tool below still refuses to run one.
        """
        try:
            session = await self._get_session()
            with anyio.fail_after(self.config.timeout_seconds):
                result = await session.list_tools()
        except McpServerError:
            raise
        except Exception as exc:
            raise McpServerError(f"{self.config.name} failed to list tools: {exc}") from exc
        return list(result.tools)

    async def call_tool(self, tool_name: str, arguments: dict) -> tuple[str, bool]:
        """Returns `(result_text, is_error)`. `result_text` is capped at
        `result_max_bytes` and is exactly what a large or slow tool produced —
        the caller (app.tools.run_tool_loop) is the one that frames it as
        untrusted data before it reaches the model.
        """
        if tool_name in self.config.disabled_tools:
            raise McpServerError(f"{tool_name!r} is disabled on {self.config.name!r}")
        try:
            session = await self._get_session()
            async with self._semaphore:
                with anyio.fail_after(self.config.timeout_seconds):
                    result = await session.call_tool(tool_name, arguments)
        except McpServerError:
            raise
        except TimeoutError as exc:
            raise McpServerError(
                f"{self.config.name}.{tool_name} did not answer within "
                f"{self.config.timeout_seconds}s"
            ) from exc
        except Exception as exc:
            raise McpServerError(f"{self.config.name}.{tool_name} failed: {exc}") from exc
        text = "".join(
            block.text for block in result.content if getattr(block, "type", None) == "text"
        )
        return text[: self.config.result_max_bytes], bool(result.isError)


    async def list_prompts(self):
        """The prompts this server publishes, [] when it publishes none."""
        try:
            session = await self._get_session()
            with anyio.fail_after(self.config.timeout_seconds):
                result = await session.list_prompts()
        except McpServerError:
            raise
        except Exception as exc:
            raise McpServerError(f"{self.config.name} failed to list prompts: {exc}") from exc
        return list(result.prompts)

    async def get_prompt(self, prompt_name: str, arguments: dict) -> str:
        """The text of a prompt with its arguments filled in: its messages, joined."""
        try:
            session = await self._get_session()
            with anyio.fail_after(self.config.timeout_seconds):
                result = await session.get_prompt(prompt_name, arguments)
        except McpServerError:
            raise
        except Exception as exc:
            raise McpServerError(f"{self.config.name}.{prompt_name} failed: {exc}") from exc
        parts = [
            m.content.text for m in result.messages if getattr(m.content, "type", None) == "text"
        ]
        return "\n\n".join(parts)[: self.config.result_max_bytes]


class Manager:
    """One `ManagedServer` per configured server, created lazily and kept for the
    life of the process (or until `reset()`, tests and a registry change call it).
    """

    def __init__(self) -> None:
        self._servers: dict[str, ManagedServer] = {}

    def configure(self, configs: list[ServerConfig]) -> None:
        """Replace the set of known servers. A server whose configuration changed
        keeps no stale live connection: its ManagedServer is dropped and rebuilt,
        reconnecting lazily on the next call.
        """
        wanted = {c.name: c for c in configs}
        for name in list(self._servers):
            if name not in wanted or self._servers[name].config != wanted[name]:
                stale = self._servers.pop(name)
                asyncio.ensure_future(stale.disconnect())
        for name, config in wanted.items():
            if name not in self._servers:
                self._servers[name] = ManagedServer(config)

    def get(self, name: str) -> ManagedServer | None:
        return self._servers.get(name)

    def servers(self) -> list[ManagedServer]:
        return list(self._servers.values())

    async def reset(self) -> None:
        """Test/dev only: disconnect and forget every server."""
        for server in self._servers.values():
            await server.disconnect()
        self._servers.clear()


manager = Manager()
