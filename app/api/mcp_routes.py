"""MCP server registry: add, test, enable, per-tool switch. Only an
administrator ever declares a server; there is
no route a plain user reaches.
"""

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin import mcp as service
from app.admin.service import ConflictError
from app.api.actor import current_actor
from app.api.deps import get_db_session
from app.api.errors import error_responses
from app.api.schemas import (
    ExposureOut,
    McpApproveIn,
    McpCallOut,
    McpGrantOut,
    McpGrantsIn,
    McpServerIn,
    McpServerOut,
    McpServerTestOut,
    McpServerUpdate,
    McpToolOut,
    McpToolToggleIn,
)
from app.api.scopes import Principal, Scope, get_principal, require
from app.db.models import McpServer
from app.mcp.manager import ManagedServer, McpServerError

router = APIRouter()


def _out(server: McpServer) -> McpServerOut:
    return McpServerOut(
        id=server.id,
        name=server.name,
        protocol=server.protocol,
        builtin_id=server.builtin_id,
        url=server.url,
        has_env_vars=bool(server.env_vars),
        egress=server.egress,
        enabled=server.enabled,
        timeout_seconds=server.timeout_seconds,
        concurrency_limit=server.concurrency_limit,
        result_max_bytes=server.result_max_bytes,
        disabled_tools=list(server.disabled_tools),
        tool_policies=dict(server.tool_policies or {}),
        shared_credentials=server.shared_credentials,
        confirm_timeout_seconds=server.confirm_timeout_seconds,
        approved_tools=sorted(server.approved_definitions or {}),
        tool_classes=dict(server.tool_classes or {}),
        created_at=server.created_at,
    )


def _tools_out(server: McpServer, live_tools) -> list[McpToolOut]:
    return [McpToolOut(**row) for row in service.review_tools(server, live_tools)]


async def _live_tools(server: McpServer):
    tools, error = await _connect_one(server)
    if tools is None:
        raise ConflictError(f"{server.name} is not reachable: {error}")
    return tools


async def _connect_one(server: McpServer):
    """One-off connection for /test and GET .../tools, outside the shared manager
    singleton (a test must not leave a live connection other turns then reuse).
    """
    managed = ManagedServer(service.to_config(server))
    try:
        tools = await managed.list_tools()
        return tools, None
    except McpServerError as exc:
        return None, str(exc)
    finally:
        await managed.disconnect()


@router.get(
    "/mcp/servers",
    dependencies=[require(Scope.READ)],
    response_model=list[McpServerOut],
    tags=["mcp"],
    responses=error_responses(),
)
async def list_servers(session: AsyncSession = Depends(get_db_session)) -> list[McpServerOut]:
    """Every declared MCP server, enabled or not."""
    return [_out(s) for s in await service.list_servers(session)]


@router.post(
    "/mcp/servers",
    dependencies=[require(Scope.ADMIN)],
    response_model=McpServerOut,
    status_code=status.HTTP_201_CREATED,
    tags=["mcp"],
    responses=error_responses(409),
)
async def create_server(
    body: McpServerIn,
    session: AsyncSession = Depends(get_db_session),
    principal: Principal = Depends(get_principal),
) -> McpServerOut:
    """Declare a server: `stdio` names a vetted built-in, `http` an exact URL. Its
    tools are offered to no one until an administrator approves their definitions
    and grants them. `shared_credentials` needs the owner scope (403)."""
    server = await service.create_server(
        session,
        name=body.name,
        protocol=body.protocol,
        builtin_id=body.builtin_id,
        fields=body.model_dump(exclude={"name", "protocol", "builtin_id"}, exclude_none=True),
        actor=current_actor(),
        is_owner=principal.scope >= Scope.OWNER,
    )
    await session.commit()
    return _out(server)


@router.get(
    "/mcp/servers/{server_id}",
    dependencies=[require(Scope.READ)],
    response_model=McpServerOut,
    tags=["mcp"],
    responses=error_responses(404),
)
async def get_server(
    server_id: int, session: AsyncSession = Depends(get_db_session)
) -> McpServerOut:
    """One declared server by id."""
    return _out(await service.get_server(session, server_id))


@router.patch(
    "/mcp/servers/{server_id}",
    dependencies=[require(Scope.ADMIN)],
    response_model=McpServerOut,
    tags=["mcp"],
    responses=error_responses(404, 409),
)
async def update_server(
    server_id: int,
    body: McpServerUpdate,
    session: AsyncSession = Depends(get_db_session),
    principal: Principal = Depends(get_principal),
) -> McpServerOut:
    """Change one or more settings of a declared server; a field left out keeps
    its current value. Changing `shared_credentials` needs the owner scope (403)."""
    server = await service.configure_server(
        session,
        server_id,
        body.model_dump(exclude_none=True),
        actor=current_actor(),
        is_owner=principal.scope >= Scope.OWNER,
    )
    await session.commit()
    return _out(server)


@router.delete(
    "/mcp/servers/{server_id}",
    dependencies=[require(Scope.ADMIN)],
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["mcp"],
    responses=error_responses(404, 409),
)
async def delete_server(server_id: int, session: AsyncSession = Depends(get_db_session)) -> None:
    """Remove a declared server; already-recorded McpCall rows are unaffected."""
    await service.delete_server(session, server_id, actor=current_actor())
    await session.commit()


@router.post(
    "/mcp/servers/{server_id}/test",
    dependencies=[require(Scope.ADMIN)],
    response_model=McpServerTestOut,
    tags=["mcp"],
    responses=error_responses(404, 409),
)
async def test_server(
    server_id: int, session: AsyncSession = Depends(get_db_session)
) -> McpServerTestOut:
    """Connects once, lists the tools, disconnects — never joins the shared
    manager, so testing a misconfigured server cannot affect a live turn.
    """
    server = await service.get_server(session, server_id)
    tools, error = await _connect_one(server)
    if tools is None:
        return McpServerTestOut(reachable=False, tools=[], error=error)
    return McpServerTestOut(reachable=True, tools=_tools_out(server, tools))


@router.get(
    "/mcp/servers/{server_id}/tools",
    dependencies=[require(Scope.READ)],
    response_model=list[McpToolOut],
    tags=["mcp"],
    responses=error_responses(404, 409),
)
async def list_tools(
    server_id: int, session: AsyncSession = Depends(get_db_session)
) -> list[McpToolOut]:
    """A server's tools right now, connecting to it live: whether an administrator
    turned each off, its policy, and its approval state with the approved definition
    next to the live one when they differ."""
    server = await service.get_server(session, server_id)
    return _tools_out(server, await _live_tools(server))


@router.patch(
    "/mcp/servers/{server_id}/tools/{tool}",
    dependencies=[require(Scope.ADMIN)],
    response_model=McpServerOut,
    tags=["mcp"],
    responses=error_responses(404, 409),
)
async def toggle_tool(
    server_id: int,
    tool: str,
    body: McpToolToggleIn,
    session: AsyncSession = Depends(get_db_session),
) -> McpServerOut:
    """Enable or disable one of a server's tools, set its policy (`allow`,
    `confirm`, `deny`, or `default` for the one its annotations give), or override its
    classes (`private`, `untrusted`, `outbound` true or false, null for the derived
    one), without touching the others."""
    server = await service.get_server(session, server_id)
    fields: dict = {}
    if body.enabled is not None:
        disabled = set(server.disabled_tools)
        if body.enabled:
            disabled.discard(tool)
        else:
            disabled.add(tool)
        fields["disabled_tools"] = sorted(disabled)
    if body.policy is not None:
        policies = dict(server.tool_policies or {})
        if body.policy == "default":
            policies.pop(tool, None)
        else:
            policies[tool] = body.policy
        fields["tool_policies"] = policies
    if body.classes is not None:
        all_classes = {k: dict(v) for k, v in (server.tool_classes or {}).items()}
        overrides = all_classes.get(tool, {})
        for name, value in body.classes.items():
            if value is None:
                overrides.pop(name, None)
            else:
                overrides[name] = value
        all_classes[tool] = overrides
        fields["tool_classes"] = {k: v for k, v in all_classes.items() if v}
    if fields:
        server = await service.configure_server(
            session, server_id, fields, actor=current_actor()
        )
    await session.commit()
    return _out(server)


@router.post(
    "/mcp/servers/{server_id}/approve-definitions",
    dependencies=[require(Scope.ADMIN)],
    response_model=list[McpToolOut],
    tags=["mcp"],
    responses=error_responses(404, 409),
)
async def approve_definitions(
    server_id: int, body: McpApproveIn, session: AsyncSession = Depends(get_db_session)
) -> list[McpToolOut]:
    """Pin the definitions the server offers right now (all, or the named `tools`):
    read `GET .../tools` first, where a changed tool shows the approved definition
    next to the live one. Until approved, a new or changed tool is offered to no one.
    """
    server = await service.get_server(session, server_id)
    live = await _live_tools(server)
    rows = await service.approve_definitions(
        session, server_id, live, tools=body.tools, actor=current_actor()
    )
    await session.commit()
    return [McpToolOut(**row) for row in rows]


def _grant_out(grant) -> McpGrantOut:
    return McpGrantOut(
        id=grant.id,
        user_id=grant.user_id,
        agent_id=grant.agent_id,
        server_name=grant.server_name,
        tool_name=grant.tool_name,
        created_at=grant.created_at,
    )


@router.get(
    "/mcp/grants",
    dependencies=[require(Scope.READ)],
    response_model=list[McpGrantOut],
    tags=["mcp"],
    responses=error_responses(),
)
async def list_grants(
    user_id: int | None = None, session: AsyncSession = Depends(get_db_session)
) -> list[McpGrantOut]:
    """Who may use which MCP server or tool, through which agent (null: every agent
    of the user; a null tool: every tool of the server). No grant, no tool."""
    return [_grant_out(g) for g in await service.list_grants(session, user_id=user_id)]


@router.put(
    "/mcp/grants",
    dependencies=[require(Scope.ADMIN)],
    response_model=list[McpGrantOut],
    tags=["mcp"],
    responses=error_responses(404, 409),
)
async def replace_grants(
    body: McpGrantsIn,
    session: AsyncSession = Depends(get_db_session),
    principal: Principal = Depends(get_principal),
) -> list[McpGrantOut]:
    """Replace every grant of one user. A server flagged as holding a shared
    credential is granted only with the owner scope (403)."""
    grants = await service.replace_grants(
        session,
        body.user_id,
        [g.model_dump() for g in body.grants],
        actor=current_actor(),
        is_owner=principal.scope >= Scope.OWNER,
    )
    await session.commit()
    return [_grant_out(g) for g in grants]


@router.get(
    "/mcp/calls",
    dependencies=[require(Scope.ADMIN)],
    response_model=list[McpCallOut],
    tags=["mcp"],
    responses=error_responses(),
)
async def list_calls(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user_id: int | None = None,
    server_name: str | None = Query(None, max_length=100),
    decision: str | None = Query(None, max_length=16),
    session: AsyncSession = Depends(get_db_session),
) -> list[McpCallOut]:
    """Every tool call, newest first, refused ones included, with the decision taken
    before it and the arguments with secrets redacted."""
    calls = await service.recent_calls(
        session,
        limit=limit,
        offset=offset,
        user_id=user_id,
        server_name=server_name,
        decision=decision,
    )
    return [
        McpCallOut(
            id=c.id,
            created_at=c.created_at,
            user_id=c.user_id,
            agent_id=c.agent_id,
            server_name=c.server_name,
            tool_name=c.tool_name,
            decision=c.decision,
            status=c.status,
            arguments=c.arguments,
            duration_ms=c.duration_ms,
            result_bytes=c.result_bytes,
        )
        for c in calls
    ]


@router.get(
    "/agents/{agent_id}/exposure",
    dependencies=[require(Scope.READ)],
    response_model=ExposureOut,
    tags=["mcp"],
    responses=error_responses(404),
)
async def get_agent_exposure(
    agent_id: int, session: AsyncSession = Depends(get_db_session)
) -> ExposureOut:
    """What the agent's MCP tools and memory can do together: private data access,
    untrusted content, outbound or write reach, per tool and for the agent. `all_three` marks
    an agent that content it reads could steer into sending private data out: keep its
    outbound tools on `confirm`, or split it. Read from the approved definitions."""
    return ExposureOut(**await service.exposure(session, agent_id))
