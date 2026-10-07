"""Tool catalogue: turns the live tools of enabled, connectable servers
into the OpenAI `tools` list app.tools.run_tool_loop needs, and builds the executor
that dispatches `mcp__<server>__<tool>` back to the right server.

A tool is offered to a turn only when every rule holds (default deny):

1. the agent's own allow-list names it and an administrator did not turn it off;
2. a grant covers this user, this agent and this server or tool;
3. its live definition has the hash an administrator approved (a changed description
   or schema disables it until approved again);
4. a server holding a credential is flagged as shared (M5);
5. its policy is not `deny`.

Read-then-write: once a tool whose results are untrusted (web pages, mail, anything from
outside) has answered in a turn, every later call in that turn to a tool with outbound or write
reach asks the user first, even when its policy is `allow`: text read from outside can carry
instructions meant for the model.

The executor checks the same set again at call time (a model can name a tool it was
not offered), asks the user for a `confirm` tool, and writes exactly one McpCall row
per call, refused ones included: user, agent, server, tool, redacted arguments,
decision, outcome, size, duration.
"""

import json
import logging
import time
from dataclasses import dataclass, field

from app import feed_memory
from app.db.models import McpCall
from app.db.session import session_scope
from app.logging_setup import scrub
from app.mcp import guard, threats
from app.mcp.confirm import current_confirmer, current_standing, current_task_addresses
from app.mcp.manager import Manager, McpServerError
from app.mcp.policy import (
    Decision,
    Policy,
    classes,
    definition_hash,
    effective_policy,
    is_granted,
    redacted_arguments,
    tool_definition,
)

logger = logging.getLogger("channelagent")

NAME_PREFIX = "mcp__"


def catalogue_name(server: str, tool: str) -> str:
    return f"{NAME_PREFIX}{server}__{tool}"


def _split_name(name: str) -> tuple[str, str] | None:
    if not name.startswith(NAME_PREFIX):
        return None
    server, sep, tool = name[len(NAME_PREFIX) :].partition("__")
    return (server, tool) if sep else None


@dataclass
class Offered:
    server: str
    tool: str
    policy: Policy
    classes: dict = field(default_factory=dict)  # Private, untrusted, outbound
    approved_hash: str | None = None  # What a standing approval is bound to
    read_only: bool = False  # the definition's readOnlyHint


@dataclass
class Catalogue:
    """One turn's tools: what the model is shown, and why each other allowed name
    was withheld (so a call to it is refused with the right decision)."""

    tools: list[dict] = field(default_factory=list)
    offered: dict[str, Offered] = field(default_factory=dict)
    withheld: dict[str, Decision] = field(default_factory=dict)


async def build_tools(
    manager: Manager, allowed: set[str], grants: frozenset = frozenset()
) -> Catalogue:
    """The tools of this turn for names in `allowed` (the agent's `Agent.tools`) and
    pairs in `grants` (app.admin.mcp.grants_for). One failing server is skipped, never
    stops the others' tools from being listed.
    """
    catalogue = Catalogue()
    if not allowed:
        return catalogue
    for server in manager.servers():
        config = server.config
        if not any(_split_name(n) and _split_name(n)[0] == config.name for n in allowed):
            continue
        try:
            live_tools = await server.list_tools()
        except McpServerError:
            logger.warning(
                "MCP server %r unavailable, its tools are not offered this turn", config.name
            )
            continue
        for tool in live_tools:
            name = catalogue_name(config.name, tool.name)
            if name not in allowed or tool.name in config.disabled_tools:
                continue
            definition = tool_definition(tool)
            policy = effective_policy(tool.name, definition, config.tool_policies)
            if not is_granted(grants, config.name, tool.name):
                catalogue.withheld[name] = Decision.NOT_GRANTED
            elif config.approved_hashes.get(tool.name) != definition_hash(definition):
                catalogue.withheld[name] = Decision.UNAPPROVED
                logger.warning(
                    "MCP tool %s is not offered: its definition is not the approved one", name
                )
            elif config.has_credentials and not config.shared_credentials:
                catalogue.withheld[name] = Decision.CREDENTIALS
            elif policy == Policy.DENY:
                catalogue.withheld[name] = Decision.DENIED
            else:
                catalogue.offered[name] = Offered(
                    config.name,
                    tool.name,
                    policy,
                    classes(tool.name, definition, config.egress, config.tool_classes),
                    config.approved_hashes.get(tool.name),
                    (definition.get("annotations") or {}).get("readOnlyHint") is True,
                )
                catalogue.tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema,
                        },
                    }
                )
    return catalogue


def narrow(catalogue: Catalogue, shown: list[dict]) -> None:
    """Keep only `shown` of the offered tools (the tool budget); a tool left out is
    refused as not offered if the model names it anyway."""
    keep = {t["function"]["name"] for t in shown}
    for name in [n for n in catalogue.offered if n not in keep]:
        del catalogue.offered[name]
        catalogue.withheld[name] = Decision.NOT_OFFERED
    catalogue.tools = [t for t in catalogue.tools if t["function"]["name"] in keep]


REFUSALS = {
    Decision.NOT_GRANTED: "this user has no grant for this tool",
    Decision.UNAPPROVED: "its definition changed and waits for an administrator's approval",
    Decision.CREDENTIALS: "its server holds a credential and is not flagged as shared",
    Decision.DENIED: "an administrator denied it",
    Decision.NOT_OFFERED: "it is not available to this agent",
    Decision.DECLINED: "the user declined it",
    Decision.TIMEOUT: "the user did not confirm it in time",
    Decision.NO_CHANNEL: "it needs the user's confirmation and nobody can be asked here",
    Decision.BLOCKED: "the guard blocked it",
    Decision.SUSPENDED: "this user's tools are suspended; an administrator can resume them",
    Decision.STALE: (
        "its definition changed since the user approved it for scheduled runs; "
        "the user must approve it again"
    ),
}


def _question(
    server: str, tool: str, arguments_text: str, timeout: int, after: str | None = None
) -> str:
    reason = (
        f"It comes after content read from outside by {after!r}, which may have steered it.\n"
        if after
        else ""
    )
    return (
        f"The assistant wants to run the tool {tool!r} of {server!r} with:\n"
        f"{arguments_text}\n"
        f"{reason}"
        f"Allow it? Answer yes or no within {timeout} seconds."
    )


def make_executor(
    manager: Manager,
    catalogue: Catalogue,
    agent_id: int | None,
    user_id: int | None = None,
):
    """The `executor(name, raw_arguments)` app.graph.call_llm passes to
    app.tools.run_tool_loop. Never raises (a refusal or a failure becomes the tool's
    own text, matching app.tools' contract) and always writes exactly one McpCall row.
    """

    # The untrusted tool that answered first in this turn; the executor lives one turn.
    read_untrusted: list[str] = []

    async def run(name: str, raw_arguments: str) -> str:
        started = time.monotonic()
        parsed = _split_name(name)
        server_name, tool_name = parsed if parsed else ("", name)
        status, decision, stored_arguments = "refused", None, None
        flagged = None
        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError:
            arguments = None
        if isinstance(arguments, dict):
            stored_arguments = redacted_arguments(arguments)
        else:
            stored_arguments = redacted_arguments(scrub(str(raw_arguments))[:1000])

        offered = catalogue.offered.get(name)
        # The guard runs before the policy: a suspended user, then the arguments.
        threat = guard.check_arguments(arguments if isinstance(arguments, dict) else None,
                                       str(raw_arguments or ""))  # fmt: skip
        if threat == guard.SECRET_REASON:
            # Never stored, whatever the list of secrets the log redaction was started with.
            stored_arguments = json.dumps("<withheld: a configured secret>")
        if await threats.is_suspended(user_id):
            decision = Decision.SUSPENDED
        elif threat is not None:
            decision = Decision.BLOCKED
        elif offered is None:
            decision = catalogue.withheld.get(name, Decision.NOT_OFFERED)
        elif not isinstance(arguments, dict):
            decision = Decision.INVALID
        elif offered.policy == Policy.CONFIRM or (
            read_untrusted and offered.classes.get("outbound")
        ):
            after = read_untrusted[0] if read_untrusted else None
            read_then_write = bool(read_untrusted and offered.classes.get("outbound"))
            if read_then_write and _named_reread(offered, arguments):
                # A second feed or page that the task's own prompt names is a read the
                # owner approved with the task, not an address taken from untrusted content.
                read_then_write = False
            decision = _standing(name, offered, read_then_write)
            if decision is None:
                decision = await _confirm(manager, offered, stored_arguments, after)
        else:
            decision = Decision.ALLOWED

        if decision == Decision.INVALID:
            text = "error: invalid arguments: arguments must be a JSON object"
        elif decision in (Decision.ALLOWED, Decision.CONFIRMED, Decision.STANDING):
            server = manager.get(server_name)
            if server is None:
                status, text = "error", f"error: server {server_name!r} is not available"
            else:
                try:
                    text, is_error = await server.call_tool(tool_name, arguments)
                    status = "error" if is_error else "ok"
                    if not is_error:
                        # In a task's turn, the feed items it already delivered.
                        builtin_id = getattr(server.config, "builtin_id", None)
                        text = await feed_memory.filter_result(builtin_id, text)
                    if offered.classes.get("untrusted") and not read_untrusted:
                        read_untrusted.append(name)
                    flagged = guard.check_result(text) if not is_error else None
                    if flagged is not None:
                        # The model still gets it, labelled; the turn has now read untrusted
                        # content, and the use is recorded as a threat.
                        text = guard.RESULT_WARNING + text
                        status = "flagged"
                        if not read_untrusted:
                            read_untrusted.append(name)
                except McpServerError as exc:
                    status, text = "error", f"error: {exc}"
        elif decision == Decision.BLOCKED:
            text = f"refused: {REFUSALS[decision]} ({threat})"
        else:
            text = f"refused: {REFUSALS.get(decision, 'not allowed')}"
        duration_ms = int((time.monotonic() - started) * 1000)
        async with session_scope() as session:
            session.add(
                McpCall(
                    agent_id=agent_id,
                    user_id=user_id,
                    server_name=server_name,
                    tool_name=tool_name,
                    status=status,
                    decision=decision.value,
                    arguments=stored_arguments,
                    duration_ms=duration_ms,
                    result_bytes=len(text.encode()),
                )
            )
            await session.commit()
        # What the guard saw, and the user's behaviour over the last hour.
        if decision == Decision.BLOCKED:
            await threats.record_threat(user_id, agent_id, name, f"blocked: {threat}")
        elif status == "flagged":
            await threats.record_threat(user_id, agent_id, name, f"result: {flagged}")
        try:
            outbound = bool(offered and offered.classes.get("outbound")) and decision in (
                Decision.ALLOWED, Decision.CONFIRMED, Decision.STANDING,
            )  # fmt: skip
            await threats.after_call(user_id, agent_id, name, decision, outbound)
        except Exception:
            logger.exception("Checking the tool-use behaviour of user %s failed", user_id)
        return text

    return run


def _named_reread(offered: Offered, arguments: dict) -> bool:
    """A read-only tool reading addresses the running task's prompt names."""
    return offered.read_only and _named_by_task(arguments)


def _named_by_task(arguments: dict) -> bool:
    """True when the call's web addresses are all written in the running task's prompt:
    at least one address, and none the prompt does not name. False outside a task turn."""
    allowed = current_task_addresses.get()
    if not allowed:
        return False
    addresses = [
        value for value in arguments.values()
        if isinstance(value, str) and value.startswith(("http://", "https://"))
    ]  # fmt: skip
    return bool(addresses) and all(address in allowed for address in addresses)


def _standing(name: str, offered: Offered, read_then_write: bool) -> Decision | None:
    """A scheduled task's standing approval, only where nobody can be asked: STANDING
    when it covers this tool and its current approved definition, STALE when it was given for
    a definition since replaced, None otherwise (the normal confirmation, refused as no_channel
    in a task turn). Read-then-write is never approved in advance: an outbound call that
    comes after untrusted content is never STANDING. A read-only tool of a server that reaches the
    internet is outbound (its address can carry data out), so it is read again after untrusted
    content only at an address the task's prompt names (decided by `run` before this)."""
    standing = current_standing.get()
    if not standing or current_confirmer.get() is not None or read_then_write:
        return None
    if name not in standing:
        return None
    if offered.approved_hash is None or standing[name] != offered.approved_hash:
        return Decision.STALE
    return Decision.STANDING


async def _confirm(
    manager: Manager, offered: Offered, arguments_text: str, after: str | None = None
) -> Decision:
    confirmer = current_confirmer.get()
    if confirmer is None:
        return Decision.NO_CHANNEL
    server = manager.get(offered.server)
    timeout = server.config.confirm_timeout_seconds if server else 120
    try:
        answer = await confirmer(
            _question(offered.server, offered.tool, arguments_text, timeout, after), timeout
        )
    except Exception:
        logger.exception("Asking the user to confirm %s.%s failed", offered.server, offered.tool)
        return Decision.NO_CHANNEL
    if answer is None:
        return Decision.TIMEOUT
    return Decision.CONFIRMED if answer else Decision.DECLINED
