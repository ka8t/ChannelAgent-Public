"""MCP server registry service: admin CRUD for app.db.models.McpServer, the
one place this logic lives (mirrors app/admin/service.py's own docstring). Only
administrators declare servers — there is no
user-facing way to add one, and `stdio` never runs an admin-supplied command, only
a vetted built-in (app.mcp.builtin.REGISTRY).
"""

import json
import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.service import (
    ConflictError,
    InvalidInputError,
    NotFoundError,
    UserNotFoundError,
    record_admin_event,
)
from app.db.models import Agent, McpCall, McpEgress, McpGrant, McpServer, McpTransport, User
from app.mcp.builtin import BUILTIN_EGRESS, missing_setting
from app.mcp.builtin import REGISTRY as BUILTIN_REGISTRY
from app.mcp.manager import ServerConfig
from app.mcp.policy import (
    CLASSES,
    POLICIES,
    Decision,
    classes,
    default_classes,
    default_policy,
    definition_hash,
    effective_policy,
    tool_definition,
)

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}")
MAX_TOOL_NAME_LENGTH = 200
MAX_DISABLED_TOOLS = 200

TRANSPORTS = (McpTransport.STDIO.value, McpTransport.HTTP.value)
EGRESS_LABELS = (McpEgress.LOCAL.value, McpEgress.LAN.value, McpEgress.INTERNET.value)

CONFIG_FIELDS = (
    "url",
    "env_vars",
    "egress",
    "enabled",
    "timeout_seconds",
    "concurrency_limit",
    "result_max_bytes",
    "disabled_tools",
    "tool_policies",
    "shared_credentials",
    "confirm_timeout_seconds",
    "tool_classes",
)
MAX_GRANTS_PER_USER = 500


class McpServerNotFoundError(NotFoundError):
    pass


class McpServerNameTakenError(ConflictError):
    pass


class OwnerRequiredError(ValueError):
    """M5: only an owner flags a server as holding a shared credential, or grants one.
    The Admin API answers 403."""


def _clean_name(value) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise InvalidInputError("A server name is letters, digits, _ and -, up to 100 characters")
    return value


def _clean_env_vars(value) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise InvalidInputError("env_vars is an object of string to string")
    return dict(value)


def _clean_disabled_tools(value) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_DISABLED_TOOLS:
        raise InvalidInputError(f"disabled_tools is a list of at most {MAX_DISABLED_TOOLS} names")
    for name in value:
        if not isinstance(name, str) or not name or len(name) > MAX_TOOL_NAME_LENGTH:
            raise InvalidInputError("a disabled tool name is non-empty text")
    return list(dict.fromkeys(value))


def _clean_egress(value) -> str:
    if value not in EGRESS_LABELS:
        raise InvalidInputError(f"egress is one of: {', '.join(EGRESS_LABELS)}")
    return value


def _clean_timeout(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 600:
        raise InvalidInputError("timeout_seconds is a whole number from 1 to 600")
    return value


def _clean_concurrency(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 20:
        raise InvalidInputError("concurrency_limit is a whole number from 1 to 20")
    return value


def _clean_result_max_bytes(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10_000_000:
        raise InvalidInputError("result_max_bytes is a whole number from 1 to 10,000,000")
    return value


def _clean_enabled(value) -> bool:
    if not isinstance(value, bool):
        raise InvalidInputError("enabled is a boolean")
    return value


def _clean_tool_policies(value) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) > MAX_DISABLED_TOOLS:
        raise InvalidInputError(
            f"tool_policies is an object of at most {MAX_DISABLED_TOOLS} tool names"
        )
    for name, policy in value.items():
        if not name or len(name) > MAX_TOOL_NAME_LENGTH:
            raise InvalidInputError("a tool name in tool_policies is non-empty text")
        if policy not in POLICIES:
            raise InvalidInputError(f"a tool policy is one of: {', '.join(POLICIES)}")
    return dict(value)


def _clean_tool_classes(value) -> dict[str, dict[str, bool]]:
    if not isinstance(value, dict) or len(value) > MAX_DISABLED_TOOLS:
        raise InvalidInputError(
            f"tool_classes is an object of at most {MAX_DISABLED_TOOLS} tool names"
        )
    cleaned = {}
    for name, overrides in value.items():
        if not name or len(name) > MAX_TOOL_NAME_LENGTH:
            raise InvalidInputError("a tool name in tool_classes is non-empty text")
        if (
            not isinstance(overrides, dict)
            or set(overrides) - set(CLASSES)
            or not all(isinstance(v, bool) for v in overrides.values())
        ):
            raise InvalidInputError(
                f"a tool's classes are true or false for: {', '.join(CLASSES)}"
            )
        if overrides:
            cleaned[name] = dict(overrides)
    return cleaned


def _clean_confirm_timeout(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 10 <= value <= 3600:
        raise InvalidInputError("confirm_timeout_seconds is a whole number from 10 to 3600")
    return value


def _clean_url(value) -> str:
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        raise InvalidInputError("url is an http(s) URL")
    return value


_CLEANERS = {
    "url": _clean_url,
    "env_vars": _clean_env_vars,
    "egress": _clean_egress,
    "enabled": _clean_enabled,
    "timeout_seconds": _clean_timeout,
    "concurrency_limit": _clean_concurrency,
    "result_max_bytes": _clean_result_max_bytes,
    "disabled_tools": _clean_disabled_tools,
    "tool_policies": _clean_tool_policies,
    "shared_credentials": _clean_enabled,
    "confirm_timeout_seconds": _clean_confirm_timeout,
    "tool_classes": _clean_tool_classes,
}


async def _named(session: AsyncSession, name: str) -> McpServer | None:
    return (
        await session.execute(select(McpServer).where(McpServer.name == name))
    ).scalar_one_or_none()


async def get_server(session: AsyncSession, server_id: int) -> McpServer:
    server = await session.get(McpServer, server_id)
    if server is None:
        raise McpServerNotFoundError(f"No MCP server with id {server_id}")
    return server


async def list_servers(session: AsyncSession) -> list[McpServer]:
    return list((await session.execute(select(McpServer).order_by(McpServer.name))).scalars())


async def create_server(
    session: AsyncSession,
    *,
    name: str,
    protocol: str,
    builtin_id: str | None = None,
    fields: dict | None = None,
    actor: str,
    is_owner: bool = False,
) -> McpServer:
    name = _clean_name(name)
    if await _named(session, name) is not None:
        raise McpServerNameTakenError(f"A server named {name!r} already exists")
    if protocol not in TRANSPORTS:
        raise InvalidInputError(f"protocol is one of: {', '.join(TRANSPORTS)}")
    fields = dict(fields or {})
    if protocol == McpTransport.STDIO.value:
        if builtin_id not in BUILTIN_REGISTRY:
            raise InvalidInputError(f"builtin_id is one of: {', '.join(sorted(BUILTIN_REGISTRY))}")
        fields.pop("url", None)
    else:
        if "url" not in fields:
            raise InvalidInputError("an http server needs a url")
        builtin_id = None

    unknown = set(fields) - set(CONFIG_FIELDS)
    if unknown:
        raise InvalidInputError(f"Unknown server settings: {', '.join(sorted(unknown))}")
    cleaned = {key: _CLEANERS[key](value) for key, value in fields.items()}
    if cleaned.get("shared_credentials") and not is_owner:
        raise OwnerRequiredError("Only an owner flags a server as holding a shared credential")

    server = McpServer(
        name=name,
        protocol=protocol,
        builtin_id=builtin_id,
        url=cleaned.get("url"),
        env_vars=json.dumps(cleaned["env_vars"]) if cleaned.get("env_vars") else None,
        egress=cleaned.get("egress", McpEgress.LOCAL.value),
        enabled=cleaned.get("enabled", True),
        timeout_seconds=cleaned.get("timeout_seconds", 20),
        concurrency_limit=cleaned.get("concurrency_limit", 2),
        result_max_bytes=cleaned.get("result_max_bytes", 1_000_000),
        disabled_tools=cleaned.get("disabled_tools", []),
        tool_policies=cleaned.get("tool_policies", {}),
        shared_credentials=cleaned.get("shared_credentials", False),
        confirm_timeout_seconds=cleaned.get("confirm_timeout_seconds", 120),
        tool_classes=cleaned.get("tool_classes", {}),
    )
    session.add(server)
    await session.flush()
    await record_admin_event(
        session,
        actor=actor,
        action="mcp_server.create",
        target_type="mcp_server",
        target_id=server.id,
        details={"name": name, "protocol": protocol, "builtin_id": builtin_id},
    )
    return server


async def configure_server(
    session: AsyncSession, server_id: int, fields: dict, *, actor: str, is_owner: bool = False
) -> McpServer:
    server = await get_server(session, server_id)
    unknown = set(fields) - set(CONFIG_FIELDS)
    if unknown:
        raise InvalidInputError(f"Unknown server settings: {', '.join(sorted(unknown))}")
    cleaned = {key: _CLEANERS[key](value) for key, value in fields.items()}
    if (
        "shared_credentials" in cleaned
        and cleaned["shared_credentials"] != server.shared_credentials
        and not is_owner
    ):
        raise OwnerRequiredError("Only an owner changes the shared-credentials flag")
    if "url" in cleaned and server.protocol != McpTransport.HTTP:
        raise InvalidInputError("url only applies to an http server")
    for key, value in cleaned.items():
        if key == "env_vars":
            server.env_vars = json.dumps(value) if value else None
        else:
            setattr(server, key, value)
    await session.flush()
    await record_admin_event(
        session,
        actor=actor,
        action="mcp_server.configure",
        target_type="mcp_server",
        target_id=server.id,
        details={"fields": sorted(cleaned)},
    )
    return server


async def delete_server(session: AsyncSession, server_id: int, *, actor: str) -> None:
    server = await get_server(session, server_id)
    await session.delete(server)
    await session.flush()
    await record_admin_event(
        session,
        actor=actor,
        action="mcp_server.delete",
        target_type="mcp_server",
        target_id=server_id,
        details={"name": server.name},
    )


def to_config(server: McpServer) -> ServerConfig:
    """The manager's own, DB-independent view of one server's settings."""
    return ServerConfig(
        name=server.name,
        protocol=server.protocol,
        builtin_id=server.builtin_id,
        url=server.url,
        env_vars=json.loads(server.env_vars) if server.env_vars else {},
        egress=server.egress,
        timeout_seconds=server.timeout_seconds,
        concurrency_limit=server.concurrency_limit,
        result_max_bytes=server.result_max_bytes,
        disabled_tools=tuple(server.disabled_tools),
        tool_policies=dict(server.tool_policies or {}),
        approved_hashes={
            name: entry.get("sha256")
            for name, entry in (server.approved_definitions or {}).items()
        },
        has_credentials=bool(server.env_vars),
        shared_credentials=server.shared_credentials,
        confirm_timeout_seconds=server.confirm_timeout_seconds,
        tool_classes=dict(server.tool_classes or {}),
    )


async def enabled_configs(session: AsyncSession) -> list[ServerConfig]:
    servers = await list_servers(session)
    return [to_config(s) for s in servers if s.enabled]


async def recent_calls(
    session: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
    user_id: int | None = None,
    server_name: str | None = None,
    decision: str | None = None,
) -> list[McpCall]:
    stmt = select(McpCall)
    if user_id is not None:
        stmt = stmt.where(McpCall.user_id == user_id)
    if server_name is not None:
        stmt = stmt.where(McpCall.server_name == server_name)
    if decision is not None:
        if decision not in {d.value for d in Decision}:
            raise InvalidInputError(f"decision is one of: {', '.join(d.value for d in Decision)}")
        stmt = stmt.where(McpCall.decision == decision)
    stmt = stmt.order_by(McpCall.created_at.desc(), McpCall.id.desc()).limit(limit).offset(offset)
    return list((await session.execute(stmt)).scalars())


# --- grants: default deny ---


async def grants_for(session: AsyncSession, user_id: int, agent_id: int) -> frozenset:
    """(server, tool) pairs this user may reach through this agent; tool None covers
    every tool of the server. A grant with no agent covers every agent of the user."""
    rows = (
        await session.execute(
            select(McpGrant.server_name, McpGrant.tool_name).where(
                McpGrant.user_id == user_id,
                (McpGrant.agent_id.is_(None)) | (McpGrant.agent_id == agent_id),
            )
        )
    ).all()
    return frozenset((server, tool) for server, tool in rows)


async def list_grants(session: AsyncSession, *, user_id: int | None = None) -> list[McpGrant]:
    stmt = select(McpGrant)
    if user_id is not None:
        stmt = stmt.where(McpGrant.user_id == user_id)
    stmt = stmt.order_by(McpGrant.user_id, McpGrant.server_name, McpGrant.id)
    return list((await session.execute(stmt)).scalars())


def _grant_key(grant: McpGrant | dict) -> tuple:
    get = grant.get if isinstance(grant, dict) else lambda k: getattr(grant, k)
    return (get("agent_id"), get("server_name"), get("tool_name"))


async def replace_grants(
    session: AsyncSession, user_id: int, grants: list[dict], *, actor: str, is_owner: bool
) -> list[McpGrant]:
    """Replace every grant of one user with `grants` (each: server_name, tool_name or
    None, agent_id or None). Refused as a whole on any invalid entry. Granting a
    server flagged as holding a shared credential needs an owner (M5)."""
    if await session.get(User, user_id) is None:
        raise UserNotFoundError(f"No user with id {user_id}")
    if not isinstance(grants, list) or len(grants) > MAX_GRANTS_PER_USER:
        raise InvalidInputError(f"grants is a list of at most {MAX_GRANTS_PER_USER} entries")
    servers = {s.name: s for s in await list_servers(session)}
    agents = set(
        (await session.execute(select(Agent.id).where(Agent.user_id == user_id))).scalars()
    )
    wanted: dict[tuple, dict] = {}
    for entry in grants:
        server_name, tool_name = entry.get("server_name"), entry.get("tool_name")
        agent_id = entry.get("agent_id")
        if server_name not in servers:
            raise InvalidInputError(f"No MCP server named {server_name!r}")
        if tool_name is not None and (
            not isinstance(tool_name, str) or not tool_name or len(tool_name) > 200
        ):
            raise InvalidInputError("tool_name is a tool's name, or null for every tool")
        if agent_id is not None and agent_id not in agents:
            raise InvalidInputError(f"Agent {agent_id} does not belong to user {user_id}")
        if servers[server_name].shared_credentials and not is_owner:
            raise OwnerRequiredError(
                f"{server_name!r} holds a shared credential: only an owner grants it"
            )
        clean = {"agent_id": agent_id, "server_name": server_name, "tool_name": tool_name}
        wanted[_grant_key(clean)] = clean

    current = await list_grants(session, user_id=user_id)
    before = {_grant_key(g) for g in current}
    for grant in current:
        if _grant_key(grant) not in wanted:
            await session.delete(grant)
    for key, clean in wanted.items():
        if key not in before:
            session.add(McpGrant(user_id=user_id, **clean))
    await session.flush()
    added = sorted(set(wanted) - before, key=str)
    removed = sorted(before - set(wanted), key=str)
    await record_admin_event(
        session,
        actor=actor,
        action="mcp_grants.replace",
        target_type="user",
        target_id=user_id,
        details={
            "added": [list(k) for k in added],
            "removed": [list(k) for k in removed],
            "total": len(wanted),
        },
    )
    return await list_grants(session, user_id=user_id)


# --- definition pinning: a changed definition waits for approval ---


def review_tools(server: McpServer, live_tools) -> list[dict]:
    """Each live tool with its approval state (`approved`, `changed`, `new`), its
    effective and default policy, and, when changed, the approved definition next to
    the live one so an administrator reads the diff. An approved tool the server no
    longer offers is listed as `gone`."""
    approved = server.approved_definitions or {}
    rows = []
    for tool in live_tools:
        definition = tool_definition(tool)
        digest = definition_hash(definition)
        entry = approved.get(tool.name)
        state = "new" if entry is None else "approved" if entry["sha256"] == digest else "changed"
        rows.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                "enabled": tool.name not in server.disabled_tools,
                "approval": state,
                "sha256": digest,
                "policy": effective_policy(tool.name, definition, server.tool_policies or {}),
                "default_policy": default_policy(definition),
                "classes": classes(tool.name, definition, server.egress, server.tool_classes or {}),
                "default_classes": default_classes(definition, server.egress),
                "definition": definition,
                "approved_definition": entry["definition"] if state == "changed" else None,
            }
        )
    live = {tool.name for tool in live_tools}
    for name, entry in approved.items():
        if name not in live:
            rows.append(
                {
                    "name": name,
                    "description": entry["definition"].get("description", ""),
                    "enabled": name not in server.disabled_tools,
                    "approval": "gone",
                    "sha256": entry["sha256"],
                    "policy": effective_policy(
                        name, entry["definition"], server.tool_policies or {}
                    ),
                    "default_policy": default_policy(entry["definition"]),
                    "classes": classes(
                        name, entry["definition"], server.egress, server.tool_classes or {}
                    ),
                    "default_classes": default_classes(entry["definition"], server.egress),
                    "definition": entry["definition"],
                    "approved_definition": None,
                }
            )
    return rows


async def approve_definitions(
    session: AsyncSession,
    server_id: int,
    live_tools,
    *,
    tools: list[str] | None,
    actor: str,
) -> list[dict]:
    """Pin the live definitions of `tools` (every live tool when None). A name the
    server does not offer is refused, so an approval never covers a definition nobody
    saw. Approving every tool also forgets the ones the server no longer offers."""
    server = await get_server(session, server_id)
    live = {tool.name: tool for tool in live_tools}
    names = list(live) if tools is None else list(dict.fromkeys(tools))
    missing = [name for name in names if name not in live]
    if missing:
        raise InvalidInputError(f"{server.name} does not offer: {', '.join(sorted(missing))}")
    approved = {} if tools is None else dict(server.approved_definitions or {})
    changes = []
    for name in names:
        definition = tool_definition(live[name])
        digest = definition_hash(definition)
        previous = (server.approved_definitions or {}).get(name)
        if previous is None or previous["sha256"] != digest:
            changes.append({"tool": name, "from": previous and previous["sha256"], "to": digest})
        approved[name] = {"sha256": digest, "definition": definition}
    server.approved_definitions = approved
    await session.flush()
    await record_admin_event(
        session,
        actor=actor,
        action="mcp_server.approve_definitions",
        target_type="mcp_server",
        target_id=server.id,
        details={"name": server.name, "approved": sorted(names), "changed": changes},
    )
    return review_tools(server, live_tools)


# --- one command for a built-in server ---


async def enable_builtin(
    session: AsyncSession,
    builtin_id: str,
    live_tools_of,
    *,
    user_id: int,
    agent_id: int | None,
    actor: str,
) -> dict:
    """Turn a vetted built-in server on for one user, in one call, adding and never replacing:
    declare it when no server runs this built-in (named after it, with its egress label),
    enable it, approve the definitions it offers that were never approved, grant it to every
    agent of the user, and add its tools to `agent_id`'s list. A tool whose live definition
    differs from the approved one is refused (409): that change is reviewed with `list-tools`
    and `approve-definitions`, never approved here. `live_tools_of(server)` connects to the
    server and returns its tools. Returns what was done, and the required setting still empty.
    """
    if builtin_id not in BUILTIN_REGISTRY:
        raise NotFoundError(
            f"No built-in server {builtin_id!r}; one of: {', '.join(sorted(BUILTIN_REGISTRY))}"
        )
    if await session.get(User, user_id) is None:
        raise UserNotFoundError(f"No user with id {user_id}")
    agent = None
    if agent_id is not None:
        agent = await session.get(Agent, agent_id)
        if agent is None or agent.user_id != user_id:
            raise InvalidInputError(f"Agent {agent_id} does not belong to user {user_id}")

    server = (
        await session.execute(
            select(McpServer)
            .where(McpServer.protocol == McpTransport.STDIO, McpServer.builtin_id == builtin_id)
            .order_by(McpServer.id)
        )
    ).scalars().first()  # fmt: skip
    created = server is None
    if created:
        server = await create_server(
            session,
            name=builtin_id,
            protocol=McpTransport.STDIO.value,
            builtin_id=builtin_id,
            fields={"egress": BUILTIN_EGRESS[builtin_id]},
            actor=actor,
        )
    enabled = not server.enabled
    server.enabled = True
    await session.flush()

    live_tools = await live_tools_of(server)
    rows = review_tools(server, live_tools)
    changed = sorted(row["name"] for row in rows if row["approval"] == "changed")
    if changed:
        raise ConflictError(
            f"{server.name}: the definition of {', '.join(changed)} changed since it was approved;"
            " read it with list-tools, then approve-definitions"
        )
    new = [row["name"] for row in rows if row["approval"] == "new"]
    if new:
        await approve_definitions(session, server.id, live_tools, tools=new, actor=actor)

    covered = (
        await session.execute(
            select(McpGrant.id).where(
                McpGrant.user_id == user_id,
                McpGrant.agent_id.is_(None),
                McpGrant.server_name == server.name,
                McpGrant.tool_name.is_(None),
            )
        )
    ).first()  # fmt: skip
    granted = covered is None
    if granted:
        session.add(McpGrant(user_id=user_id, agent_id=None, server_name=server.name))

    added: list[str] = []
    if agent is not None:
        from app.admin.service import MAX_TOOLS

        current = list(agent.tools or [])
        offered = [t.name for t in live_tools if t.name not in (server.disabled_tools or [])]
        added = [n for n in (f"mcp__{server.name}__{t}" for t in offered) if n not in current]
        if len(current) + len(added) > MAX_TOOLS:
            raise InvalidInputError(f"An agent has at most {MAX_TOOLS} tools")
        agent.tools = current + added
    await session.flush()

    result = {
        "server_id": server.id,
        "server": server.name,
        "created": created,
        "enabled": enabled,
        "approved": new,
        "granted": granted,
        "agent_id": agent_id,
        "added_tools": added,
        "missing_setting": missing_setting(builtin_id),
    }
    await record_admin_event(
        session,
        actor=actor,
        action="mcp_builtin.enable",
        target_type="user",
        target_id=user_id,
        details={k: v for k, v in result.items() if k != "missing_setting"},
    )
    return result


# --- exposure ---


async def exposure(session: AsyncSession, agent_id: int) -> dict:
    """Per agent, what its MCP tools and its memory can do together: private data access,
    untrusted content, outbound or write reach. An agent holding all three can be steered by
    content it reads into sending private data out. Read from the approved definitions (no
    server is contacted); a tool counts only when a turn could be offered it (server enabled,
    definition approved, tool not turned off, policy not deny, and a grant for the agent's
    user)."""
    agent = await session.get(Agent, agent_id)
    if agent is None:
        raise NotFoundError(f"No agent {agent_id}")
    servers = {s.name: s for s in await list_servers(session)}
    grants = await grants_for(session, agent.user_id, agent.id)
    from app.db.models import ScheduledTask

    agent_tasks = (
        await session.execute(select(ScheduledTask).where(ScheduledTask.agent_id == agent.id))
    ).scalars().all()  # fmt: skip
    rows = []
    for name in agent.tools or []:
        server_name, sep, tool = name.removeprefix("mcp__").partition("__")
        row = {"name": name, "server": server_name, "tool": tool, "counted": False,
               "why": None, "private": None, "untrusted": None, "outbound": None,
               "overridden": [], "standing_tasks": [], "standing_stale": []}  # fmt: skip
        server = servers.get(server_name) if name.startswith("mcp__") and sep else None
        entry = (server.approved_definitions or {}).get(tool) if server else None
        # The tasks whose scheduled runs may use it without asking, and those whose
        # approval was given for a definition since replaced.
        for task in agent_tasks:
            digest = (task.standing_tools or {}).get(name)
            if digest is not None:
                current = entry is not None and entry.get("sha256") == digest
                row["standing_tasks" if current else "standing_stale"].append(task.id)
        if server is None:
            row["why"] = "no such server"
        elif entry is None:
            row["why"] = "definition not approved"
        else:
            overrides = (server.tool_classes or {}).get(tool) or {}
            row.update(classes(tool, entry["definition"], server.egress, server.tool_classes or {}))
            row["overridden"] = sorted(overrides)
            policy = effective_policy(tool, entry["definition"], server.tool_policies or {})
            if not server.enabled:
                row["why"] = "server disabled"
            elif tool in (server.disabled_tools or []):
                row["why"] = "tool turned off"
            elif policy == "deny":
                row["why"] = "policy deny"
            elif (server_name, tool) not in grants and (server_name, None) not in grants:
                row["why"] = "no grant for the agent's user"
            else:
                row["counted"] = True
        rows.append(row)
    counted = [r for r in rows if r["counted"]]
    memory = (agent.memory_mode or "off") != "off"
    result = {
        "agent_id": agent.id,
        "memory": memory,
        "private": memory or any(r["private"] for r in counted),
        "untrusted": any(r["untrusted"] for r in counted),
        "outbound": any(r["outbound"] for r in counted),
        "tools": rows,
    }
    result["all_three"] = result["private"] and result["untrusted"] and result["outbound"]
    return result
