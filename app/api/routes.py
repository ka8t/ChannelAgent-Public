"""Admin API routes. Every route is covered by the app-level API_SERVER_KEY
dependency (see app/api/app.py), and none holds business logic: each calls
the same app.admin.service function the admin console calls (
one-service-layer rule). Service errors become HTTP statuses in one place,
the exception handlers of app/api/app.py: NotFoundError 404, ConflictError
409, InvalidInputError 422.
"""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, Query, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin import service
from app.api.actor import current_actor
from app.api.deps import get_db_session
from app.api.errors import error_responses
from app.api.schemas import (
    AccessRequestOut,
    ActionLogOut,
    AdminEventOut,
    AgentCreate,
    AgentOut,
    AgentUpdate,
    ChannelIdentityCreate,
    ChannelIdentityOut,
    ConversationResetOut,
    IdentityAgentSet,
    PermissionGrant,
    PermissionOut,
    StorageOut,
    UserCreate,
    UserOut,
    UserUpdate,
)
from app.api.scopes import Principal, Scope, get_principal, require
from app.db.models import (
    ActionLog,
    ActionStatus,
    AdminEvent,
    Agent,
    Channel,
    ChannelIdentity,
    Direction,
    PermissionKind,
    RequestStatus,
    User,
)

router = APIRouter()

# Paging of the simple lists: the body stays an array, the total is in a header.
LIMIT = Query(default=200, ge=1, le=1000, description="At most this many items.")
OFFSET = Query(default=0, ge=0, description="Skip this many items.")


def _agent_out(agent: Agent, principal: Principal) -> AgentOut:
    """An agent as the API returns it: its system prompt only to an administrator."""
    return AgentOut(
        id=agent.id,
        user_id=agent.user_id,
        name=agent.name,
        is_active=agent.is_active,
        system_prompt=agent.system_prompt if principal.scope >= Scope.ADMIN else None,
        has_system_prompt=bool(agent.system_prompt),
        model=agent.model,
        memory_mode=agent.memory_mode,
        tools=list(agent.tools or []),
        skills=list(agent.skills or []),
        purpose=agent.purpose if principal.scope >= Scope.ADMIN else None,
    )


def _page(response: Response, items: list, limit: int, offset: int) -> list:
    response.headers["X-Total-Count"] = str(len(items))
    return items[offset : offset + limit]



# --- Users ---


@router.post(
    "/users",
    dependencies=[require(Scope.OPERATE)],
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    tags=["users"],
    responses=error_responses(409),
)
async def create_user(
    body: UserCreate, session: AsyncSession = Depends(get_db_session)
) -> User:
    """Create a user."""
    user = await service.create_user(session, body.display_name, actor=current_actor())
    await session.commit()
    return user


@router.get(
    "/users",
    dependencies=[require(Scope.READ)],
    response_model=list[UserOut],
    tags=["users"],
    responses=error_responses(),
)
async def list_users(
    response: Response,
    limit: int = LIMIT,
    offset: int = OFFSET,
    session: AsyncSession = Depends(get_db_session),
) -> list[User]:
    """List the users."""
    return _page(response, await service.list_users(session), limit, offset)


@router.get(
    "/users/{user_id}",
    dependencies=[require(Scope.READ)],
    response_model=UserOut,
    tags=["users"],
    responses=error_responses(404),
)
async def get_user(user_id: int, session: AsyncSession = Depends(get_db_session)) -> User:
    """Show one user."""
    return await service.get_user(session, user_id)


@router.patch(
    "/users/{user_id}",
    dependencies=[require(Scope.OPERATE)],
    response_model=UserOut,
    tags=["users"],
    responses=error_responses(404, 409),
)
async def update_user(
    user_id: int, body: UserUpdate, session: AsyncSession = Depends(get_db_session)
) -> User:
    """Rename a user, activate or deactivate them, or set their timezone (an IANA name,
    for the times of their scheduled tasks)."""
    user = await service.update_user(
        session,
        user_id,
        display_name=body.display_name,
        is_active=body.is_active,
        timezone=body.timezone,
        actor=current_actor(),
    )
    await session.commit()
    return user


@router.post(
    "/users/{user_id}/tools/resume",
    dependencies=[require(Scope.ADMIN)],
    response_model=UserOut,
    tags=["users"],
    responses=error_responses(404, 409),
)
async def resume_tools(user_id: int, session: AsyncSession = Depends(get_db_session)) -> User:
    """Resume the tools of a user the MCP guard suspended after repeated threats."""
    from app.mcp import threats

    user = await threats.resume(session, user_id, actor=current_actor())
    await session.commit()
    return user


@router.delete(
    "/users/{user_id}",
    dependencies=[require(Scope.OWNER)],
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["users"],
    responses=error_responses(404, 409),
)
async def delete_user(
    user_id: int,
    purge: bool = Query(
        default=False, description="Also delete the user's agents, audit trail and conversations."
    ),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    """Delete a user; with purge, also their agents, audit trail and conversations."""
    await service.delete_user(session, user_id, purge=purge, actor=current_actor())
    await session.commit()


# --- Channel identities ---


@router.post(
    "/users/{user_id}/channels",
    dependencies=[require(Scope.OPERATE)],
    response_model=ChannelIdentityOut,
    status_code=status.HTTP_201_CREATED,
    tags=["channels"],
    responses=error_responses(404, 409),
)
async def add_channel_identity(
    user_id: int, body: ChannelIdentityCreate, session: AsyncSession = Depends(get_db_session)
) -> ChannelIdentity:
    """Add a channel identity (a Telegram id, or an email address) to a user."""
    identity = await service.add_channel_identity(
        session, user_id, body.channel, body.identifier, actor=current_actor()
    )
    await session.commit()
    return identity


@router.get(
    "/users/{user_id}/channels",
    dependencies=[require(Scope.READ)],
    response_model=list[ChannelIdentityOut],
    tags=["channels"],
    responses=error_responses(404),
)
async def list_channel_identities(
    user_id: int,
    response: Response,
    limit: int = LIMIT,
    offset: int = OFFSET,
    session: AsyncSession = Depends(get_db_session),
) -> list[ChannelIdentity]:
    """List a user's channel identities."""
    items = await service.list_channel_identities(session, user_id)
    return _page(response, items, limit, offset)


@router.delete(
    "/users/{user_id}/channels/{channel_identity_id}",
    dependencies=[require(Scope.OPERATE)],
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["channels"],
    responses=error_responses(404, 409),
)
async def delete_channel_identity(
    user_id: int, channel_identity_id: int, session: AsyncSession = Depends(get_db_session)
) -> None:
    """Remove a channel identity from a user."""
    await service.remove_channel_identity(
        session, user_id, channel_identity_id, actor=current_actor()
    )
    await session.commit()


@router.put(
    "/users/{user_id}/channels/{channel_identity_id}/agent",
    dependencies=[require(Scope.OPERATE)],
    response_model=ChannelIdentityOut,
    tags=["channels"],
    responses=error_responses(404, 409),
)
async def set_identity_agent(
    user_id: int,
    channel_identity_id: int,
    body: IdentityAgentSet,
    session: AsyncSession = Depends(get_db_session),
) -> ChannelIdentity:
    """Choose which of the user's agents this channel identity talks to."""
    identity = await service.set_identity_agent(
        session, user_id, channel_identity_id, body.agent_id, actor=current_actor()
    )
    await session.commit()
    return identity


# --- Conversations ---


@router.post(
    "/users/{user_id}/conversations/reset",
    dependencies=[require(Scope.OPERATE)],
    response_model=ConversationResetOut,
    tags=["conversations"],
    responses=error_responses(404, 409),
)
async def reset_conversation(
    user_id: int,
    agent_id: int | None = Query(
        default=None, description="Only this agent's conversation. Omit for all of them."
    ),
    session: AsyncSession = Depends(get_db_session),
) -> ConversationResetOut:
    """Forget the stored history of a conversation that cannot be read. The
    next message starts a fresh one; the audit trail is kept.
    """
    threads = await service.reset_conversation(session, user_id, agent_id, actor=current_actor())
    await session.commit()
    return ConversationResetOut(threads_reset=threads)


@router.get(
    "/users/{user_id}/conversations/export",
    dependencies=[require(Scope.ADMIN)],
    tags=["conversations"],
    responses={
        **error_responses(404),
        200: {
            "description": "JSON: {about, messages}; Markdown: the conversation as a document",
            "content": {"text/markdown": {}},
        },
    },
)
async def export_conversation(
    user_id: int,
    format: Literal["json", "markdown"] = Query(default="json"),
    agent_id: int | None = Query(default=None, description="Only this agent's conversation."),
    include_text: bool = Query(
        default=False, description="decrypt and include each message's text (audited)"
    ),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """Export one user's conversation (JSON or Markdown). The text is included only when
    asked; each export is recorded as an admin event."""
    messages, about = await service.export_conversation(
        session, user_id, agent_id=agent_id, include_text=include_text, fmt=format,
        actor=current_actor(),
    )  # fmt: skip
    await session.commit()
    if format == "markdown":
        return PlainTextResponse(
            service.conversation_markdown(messages, about), media_type="text/markdown"
        )
    return JSONResponse({"about": about, "messages": messages})


# --- Permissions ---


@router.post(
    "/users/{user_id}/channels/{channel_identity_id}/permissions",
    dependencies=[require(Scope.ADMIN)],
    response_model=PermissionOut,
    status_code=status.HTTP_201_CREATED,
    tags=["permissions"],
    responses=error_responses(404, 409),
)
async def grant(
    user_id: int,
    channel_identity_id: int,
    body: PermissionGrant,
    session: AsyncSession = Depends(get_db_session),
) -> PermissionOut:
    """Grant a permission to a channel identity."""
    permission = await service.grant_identity_permission(
        session, user_id, channel_identity_id, body.kind, actor=current_actor()
    )
    await session.commit()
    return PermissionOut.model_validate(permission)


@router.get(
    "/users/{user_id}/channels/{channel_identity_id}/permissions",
    dependencies=[require(Scope.READ)],
    response_model=list[PermissionOut],
    tags=["permissions"],
    responses=error_responses(404),
)
async def list_permissions(
    user_id: int, channel_identity_id: int, session: AsyncSession = Depends(get_db_session)
) -> list[PermissionOut]:
    """List the permissions of a channel identity."""
    permissions = await service.list_identity_permissions(session, user_id, channel_identity_id)
    return [PermissionOut.model_validate(p) for p in permissions]


@router.delete(
    "/users/{user_id}/channels/{channel_identity_id}/permissions/{kind}",
    dependencies=[require(Scope.ADMIN)],
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["permissions"],
    responses=error_responses(404, 409),
)
async def revoke(
    user_id: int,
    channel_identity_id: int,
    kind: PermissionKind,
    session: AsyncSession = Depends(get_db_session),
) -> None:
    """Revoke a permission from a channel identity."""
    await service.revoke_identity_permission(
        session, user_id, channel_identity_id, kind, actor=current_actor()
    )
    await session.commit()


# --- Access requests ---


@router.get(
    "/requests",
    dependencies=[require(Scope.READ)],
    response_model=list[AccessRequestOut],
    tags=["requests"],
    responses=error_responses(),
)
async def list_requests(
    response: Response,
    status_filter: str = Query(
        default="pending", alias="status", pattern="^(pending|approved|denied|all)$"
    ),
    limit: int = LIMIT,
    offset: int = OFFSET,
    session: AsyncSession = Depends(get_db_session),
):
    """List the access requests, by default the pending ones."""
    wanted = None if status_filter == "all" else RequestStatus(status_filter)
    return _page(response, await service.list_requests(session, wanted), limit, offset)


@router.post(
    "/requests/{request_id}/approve",
    dependencies=[require(Scope.OPERATE)],
    response_model=UserOut,
    tags=["requests"],
    responses=error_responses(404, 409),
)
async def approve_request(request_id: int, session: AsyncSession = Depends(get_db_session)) -> User:
    """Approve an access request and create the user."""
    user = await service.approve_request(session, request_id, resolved_by=current_actor())
    await session.commit()
    return user


@router.post(
    "/requests/{request_id}/deny",
    dependencies=[require(Scope.OPERATE)],
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["requests"],
    responses=error_responses(404, 409),
)
async def deny_request(request_id: int, session: AsyncSession = Depends(get_db_session)) -> None:
    """Deny an access request."""
    await service.deny_request(session, request_id, resolved_by=current_actor())
    await session.commit()


# --- Agents: an admin can edit any user's agent ---


@router.get(
    "/users/{user_id}/agents",
    dependencies=[require(Scope.READ)],
    response_model=list[AgentOut],
    tags=["agents"],
    responses=error_responses(404),
)
async def list_agents(
    user_id: int,
    response: Response,
    limit: int = LIMIT,
    offset: int = OFFSET,
    session: AsyncSession = Depends(get_db_session),
    principal: Principal = Depends(get_principal),
) -> list[AgentOut]:
    """List a user's agents."""
    agents = _page(response, await service.list_agents(session, user_id), limit, offset)
    return [_agent_out(a, principal) for a in agents]


@router.post(
    "/users/{user_id}/agents",
    dependencies=[require(Scope.OPERATE)],
    response_model=AgentOut,
    status_code=status.HTTP_201_CREATED,
    tags=["agents"],
    responses=error_responses(404, 409),
)
async def create_agent(
    user_id: int,
    body: AgentCreate,
    session: AsyncSession = Depends(get_db_session),
    principal: Principal = Depends(get_principal),
) -> AgentOut:
    """Create an agent for a user, optionally with its system prompt, model, memory mode, tools."""
    settings = body.model_dump(exclude={"name"}, include=body.model_fields_set - {"name"})
    agent = await service.create_agent(
        session, user_id, body.name, actor=current_actor(), settings=settings
    )
    await session.commit()
    return _agent_out(agent, principal)


@router.get(
    "/agents/{agent_id}",
    dependencies=[require(Scope.READ)],
    response_model=AgentOut,
    tags=["agents"],
    responses=error_responses(404),
)
async def get_agent(
    agent_id: int,
    session: AsyncSession = Depends(get_db_session),
    principal: Principal = Depends(get_principal),
) -> AgentOut:
    """Show one agent."""
    return _agent_out(await service.get_agent(session, agent_id), principal)


@router.patch(
    "/agents/{agent_id}",
    dependencies=[require(Scope.OPERATE)],
    response_model=AgentOut,
    tags=["agents"],
    responses=error_responses(404, 409),
)
async def update_agent(
    agent_id: int,
    body: AgentUpdate,
    session: AsyncSession = Depends(get_db_session),
    principal: Principal = Depends(get_principal),
) -> AgentOut:
    """Rename, activate or deactivate, or set the prompt, model, memory mode and tools of any
    user's agent. Only the fields given change; a null prompt or model clears it."""
    agent = await service.get_agent(session, agent_id)
    settings = body.model_dump(include=body.model_fields_set & set(service.CONFIG_FIELDS))
    if settings:
        agent = await service.configure_agent(session, agent_id, settings, actor=current_actor())
    if body.name is not None:
        agent = await service.rename_agent(session, agent_id, body.name, actor=current_actor())
    if body.is_active is not None:
        agent = await service.set_agent_active(
            session, agent_id, body.is_active, actor=current_actor()
        )
    await session.commit()
    return _agent_out(agent, principal)


# --- Audit trail search ---


@router.get(
    "/logs",
    dependencies=[require(Scope.ADMIN)],
    response_model=list[ActionLogOut],
    tags=["audit"],
    responses=error_responses(),
)
async def search_logs(
    user_id: int | None = None,
    agent_id: int | None = None,
    channel: Channel | None = None,
    direction: Direction | None = None,
    status: ActionStatus | None = None,
    since: datetime | None = Query(default=None, description="Inclusive. Naive = UTC."),
    until: datetime | None = Query(default=None, description="Exclusive. Naive = UTC."),
    keyword: str | None = Query(default=None, description="Case-insensitive, on decrypted text."),
    limit: int = Query(default=100, ge=1, le=service.LOG_SEARCH_MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> list[ActionLog]:
    """Search the action log (conversation text) by user, agent, channel, status, date, keyword."""
    logs = await service.search_action_logs(
        session,
        actor=current_actor(),
        user_id=user_id,
        agent_id=agent_id,
        channel=channel,
        direction=direction,
        status=status,
        since=since,
        until=until,
        keyword=keyword,
        limit=limit,
        offset=offset,
    )
    await session.commit()  # a read of decrypted logs is itself recorded
    return logs


# --- What administrators did ---


@router.get(
    "/admin-events",
    dependencies=[require(Scope.ADMIN)],
    response_model=list[AdminEventOut],
    tags=["audit"],
    responses=error_responses(),
)
async def search_admin_events(
    actor: str | None = None,
    action: str | None = None,
    target_type: str | None = None,
    target_id: int | None = None,
    since: datetime | None = Query(default=None, description="Inclusive. Naive = UTC."),
    until: datetime | None = Query(default=None, description="Exclusive. Naive = UTC."),
    limit: int = Query(default=100, ge=1, le=service.LOG_SEARCH_MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> list[AdminEvent]:
    """Search what administrators did (the admin events)."""
    events = await service.search_admin_events(
        session,
        actor=actor,
        action=action,
        target_type=target_type,
        target_id=target_id,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
        reader=current_actor(),
    )
    await session.commit()
    return events


# --- Storage overview ---


@router.get(
    "/storage",
    dependencies=[require(Scope.READ)],
    response_model=StorageOut,
    tags=["system"],
    responses=error_responses(),
)
async def storage(session: AsyncSession = Depends(get_db_session)) -> StorageOut:
    """Show what the database and the checkpoints hold, and the rows that cannot be decrypted."""
    overview = await service.storage_overview(session)
    return StorageOut(
        db_size_bytes=overview.db_size_bytes,
        row_counts=overview.row_counts,
        oldest_log_at=overview.oldest_log_at,
        newest_log_at=overview.newest_log_at,
        undecryptable_rows=overview.undecryptable_rows,
        undecryptable_by_table=overview.undecryptable_by_table,
        checkpoint_size_bytes=overview.checkpoint_size_bytes,
        checkpoint_row_counts=overview.checkpoint_row_counts,
    )
