"""Admin service layer: the one place this logic lives. The
interactive CLI and the Admin API both call these functions —
neither re-implements the DB queries itself.
"""

import asyncio
import enum
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import (
    AccessRequest,
    ActionLog,
    ActionStatus,
    AdminAccount,
    AdminEvent,
    Agent,
    AgentTemplate,
    AgentTemplateVersion,
    ApiCall,
    ApiToken,
    BackupSchedule,
    Channel,
    ChannelIdentity,
    Direction,
    FeedSource,
    McpCall,
    McpGrant,
    McpServer,
    MemoryEntry,
    Permission,
    PermissionKind,
    RequestMessage,
    RequestStatus,
    RetentionConfig,
    RoutingConfig,
    RoutingRule,
    ScheduledTask,
    Skill,
    SkillVersion,
    TaskConfig,
    TaskFeedItem,
    User,
    _utcnow,
)
from app.db.session import sqlite_file_path
from app.db.types import UndecryptableText
from app.security.auth import grant_permission, revoke_permission
from app.security.hashing import channel_identifier_key

logger = logging.getLogger("channelagent")

DEFAULT_AGENT_NAME = "default"
# Recorded when a caller does not say who acts. The API and the console
# always do (tests check that none of their events is "unspecified").
DEFAULT_ACTOR = "unspecified"
MAX_ACTOR_LENGTH = 32


class NotFoundError(ValueError):
    """Something an operation refers to does not exist. All service errors are
    ValueErrors, so a front end can print any of them; the API maps
    NotFoundError to 404, ConflictError to 409 and InvalidInputError to 422.
    """


class ConflictError(ValueError):
    """The operation is valid but the current state forbids it."""


class InvalidInputError(ValueError):
    """An argument that cannot be used (empty name, too long...)."""


class UserNotFoundError(NotFoundError):
    """The user id refers to nobody."""


class IdentityNotFoundError(NotFoundError):
    pass


class AgentNotFoundError(NotFoundError):
    pass


class RequestNotFoundError(NotFoundError):
    pass


class PermissionNotHeldError(NotFoundError):
    pass


class IdentityAlreadyLinkedError(ConflictError):
    pass


class AgentNameTakenError(ConflictError):
    pass


class RequestAlreadyResolvedError(ConflictError):
    pass


class UserHasHistoryError(ConflictError):
    """Deleting was refused because the user has an audit trail."""

    def __init__(self, user_id: int, agents: int, logs: int) -> None:
        self.user_id, self.agents, self.logs = user_id, agents, logs
        entries = "log entry" if logs == 1 else "log entries"
        super().__init__(
            f"User {user_id} has {agents} agent(s) and {logs} {entries}. "
            "Deactivate the user instead, or purge to delete the history as well."
        )


async def record_admin_event(
    session: AsyncSession,
    *,
    actor: str,
    action: str,
    target_type: str,
    target_id: int | None = None,
    details: dict | None = None,
) -> AdminEvent:
    """Record one administrator action, in the caller's transaction.
    `details` must never hold a secret or an email address.
    """
    actor = (actor or "").strip()
    if not actor or len(actor) > MAX_ACTOR_LENGTH:
        raise InvalidInputError(f"An actor is 1 to {MAX_ACTOR_LENGTH} characters")
    event = AdminEvent(
        actor=actor,
        action=action,
        target_type=target_type,
        target_id=target_id,
        details=json.dumps(details, sort_keys=True, default=str) if details else None,
    )
    session.add(event)
    await session.flush()
    from app.admin import audit_chain

    await audit_chain.seal(session, event)  # Chained to the event before it
    return event


async def _require_user(session: AsyncSession, user_id: int) -> User:
    user = await session.get(User, user_id)
    if user is None:
        raise UserNotFoundError(f"No user with id {user_id}")
    return user


async def get_or_create_default_agent(session: AsyncSession, user_id: int) -> Agent:
    """A user's first Agent is created lazily, on first use — a
    single-agent user never has to think about agents at all.
    """
    await _require_user(session, user_id)
    stmt = select(Agent).where(Agent.user_id == user_id, Agent.name == DEFAULT_AGENT_NAME)
    existing = (await session.execute(stmt)).scalar_one_or_none()
    if existing is not None:
        return existing
    agent = Agent(user_id=user_id, name=DEFAULT_AGENT_NAME)
    session.add(agent)
    await session.flush()
    return agent


MAX_AGENT_NAME_LENGTH = 100


def _clean_agent_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise InvalidInputError("An agent name must not be empty")
    if len(name) > MAX_AGENT_NAME_LENGTH:
        raise InvalidInputError(f"An agent name is at most {MAX_AGENT_NAME_LENGTH} characters")
    return name


# --- per-agent configuration ---

MEMORY_MODES = ("off", "ondemand", "always", "search")
MAX_SYSTEM_PROMPT_LENGTH = 20000
MAX_TOOLS = 100
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}")
_TOOL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}")
CONFIG_FIELDS = ("system_prompt", "model", "memory_mode", "tools")


def _clean_system_prompt(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or "\x00" in value:
        raise InvalidInputError("A system prompt is text")
    value = value.strip()
    if len(value) > MAX_SYSTEM_PROMPT_LENGTH:
        raise InvalidInputError(f"A system prompt is at most {MAX_SYSTEM_PROMPT_LENGTH} characters")
    return value or None


def _clean_model(value) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str) or not _MODEL_NAME.fullmatch(value.strip()):
        raise InvalidInputError(
            "A model name is letters, digits and . _ : / -, up to 200 characters"
        )
    return value.strip()


def _clean_memory_mode(value) -> str:
    if value not in MEMORY_MODES:
        raise InvalidInputError(f"memory_mode is one of: {', '.join(MEMORY_MODES)}")
    return value


def _clean_tools(value) -> list[str]:
    if not isinstance(value, list) or len(value) > MAX_TOOLS:
        raise InvalidInputError(f"tools is a list of at most {MAX_TOOLS} tool names")
    seen: list[str] = []
    for item in value:
        if not isinstance(item, str) or not _TOOL_NAME.fullmatch(item):
            raise InvalidInputError(
                "A tool name is letters, digits and . _ : -, up to 200 characters"
            )
        if item not in seen:
            seen.append(item)
    return seen


_CLEANERS = {
    "system_prompt": _clean_system_prompt,
    "model": _clean_model,
    "memory_mode": _clean_memory_mode,
    "tools": _clean_tools,
}


def _clean_config(fields: dict) -> dict:
    unknown = set(fields) - set(CONFIG_FIELDS)
    if unknown:
        raise InvalidInputError(f"Unknown agent settings: {', '.join(sorted(unknown))}")
    return {name: _CLEANERS[name](value) for name, value in fields.items()}


async def configure_agent(
    session: AsyncSession, agent_id: int, fields: dict, *, actor: str = DEFAULT_ACTOR
) -> Agent:
    """Set the given settings of an agent, any user's. A setting that is not in `fields`
    is left alone. The event records which settings changed, never the prompt itself."""
    agent = await get_agent(session, agent_id)
    cleaned = _clean_config(fields)
    for name, value in cleaned.items():
        setattr(agent, name, value)
    await session.flush()
    details: dict = {"fields": sorted(cleaned)}
    for name in ("model", "memory_mode", "tools"):
        if name in cleaned:
            details[name] = cleaned[name]
    await record_admin_event(
        session,
        actor=actor,
        action="agent.configure",
        target_type="agent",
        target_id=agent.id,
        details=details,
    )
    return agent


async def _agent_named(session: AsyncSession, user_id: int, name: str) -> Agent | None:
    stmt = select(Agent).where(Agent.user_id == user_id, Agent.name == name)
    return (await session.execute(stmt)).scalar_one_or_none()


async def get_agent(session: AsyncSession, agent_id: int) -> Agent:
    agent = await session.get(Agent, agent_id)
    if agent is None:
        raise AgentNotFoundError(f"No agent with id {agent_id}")
    return agent


async def create_agent(
    session: AsyncSession,
    user_id: int,
    name: str,
    *,
    actor: str = DEFAULT_ACTOR,
    settings: dict | None = None,
) -> Agent:
    """`settings` may hold any of system_prompt, model, memory_mode and tools."""
    await _require_user(session, user_id)
    name = _clean_agent_name(name)
    if await _agent_named(session, user_id, name) is not None:
        raise AgentNameTakenError(f"User {user_id} already has an agent named {name!r}")
    cleaned = _clean_config(settings or {})
    agent = Agent(user_id=user_id, name=name, **cleaned)
    session.add(agent)
    await session.flush()
    await record_admin_event(
        session,
        actor=actor,
        action="agent.create",
        target_type="agent",
        target_id=agent.id,
        details={
            "name": name,
            "owner_user_id": user_id,
            **({"fields": sorted(cleaned)} if cleaned else {}),
        },
    )
    return agent


async def list_agents(session: AsyncSession, user_id: int) -> list[Agent]:
    await _require_user(session, user_id)
    stmt = select(Agent).where(Agent.user_id == user_id).order_by(Agent.id)
    return list((await session.execute(stmt)).scalars().all())


async def rename_agent(
    session: AsyncSession, agent_id: int, new_name: str, *, actor: str = DEFAULT_ACTOR
) -> Agent:
    """Admin-editable: callable against any user's Agent, not only
    the owning user's own — callers decide who's allowed to call this.
    """
    agent = await get_agent(session, agent_id)
    new_name = _clean_agent_name(new_name)
    clash = await _agent_named(session, agent.user_id, new_name)
    if clash is not None and clash.id != agent.id:
        raise AgentNameTakenError(f"User {agent.user_id} already has an agent named {new_name!r}")
    old_name = agent.name
    agent.name = new_name
    await session.flush()
    await record_admin_event(
        session,
        actor=actor,
        action="agent.rename",
        target_type="agent",
        target_id=agent.id,
        details={"old_name": old_name, "new_name": new_name},
    )
    return agent


async def set_agent_active(
    session: AsyncSession, agent_id: int, is_active: bool, *, actor: str = DEFAULT_ACTOR
) -> Agent:
    agent = await get_agent(session, agent_id)
    agent.is_active = is_active
    await session.flush()
    await record_admin_event(
        session,
        actor=actor,
        action="agent.set_active",
        target_type="agent",
        target_id=agent.id,
        details={"is_active": is_active},
    )
    return agent


async def find_agent_by_name(session: AsyncSession, user_id: int, name: str) -> Agent | None:
    """Case-insensitive exact match on one user's agents."""
    wanted = name.strip().lower()
    for agent in await list_agents(session, user_id):
        if agent.name.lower() == wanted:
            return agent
    return None


async def resolve_agent(session: AsyncSession, user_id: int, active_agent_id: int | None) -> Agent:
    """The agent a message from this identity reaches: the one the
    identity selected, if it still exists and belongs to the user, otherwise
    the user's default agent.
    """
    if active_agent_id is not None:
        agent = await session.get(Agent, active_agent_id)
        if agent is not None and agent.user_id == user_id:
            return agent
    return await get_or_create_default_agent(session, user_id)


async def set_identity_agent(
    session: AsyncSession,
    user_id: int,
    identity_id: int,
    agent_id: int | None,
    *,
    actor: str = DEFAULT_ACTOR,
) -> ChannelIdentity:
    """Choose the agent one channel identity talks to. `None` goes back to the
    default agent. The agent must belong to the same user.
    """
    identity = await _identity_of_user(session, user_id, identity_id)
    if agent_id is not None:
        agent = await session.get(Agent, agent_id)
        if agent is None or agent.user_id != user_id:
            raise AgentNotFoundError(f"No agent {agent_id} for user {user_id}")
    identity.active_agent_id = agent_id
    await session.flush()
    await record_admin_event(
        session,
        actor=actor,
        action="identity.set_agent",
        target_type="user",
        target_id=user_id,
        details={"identity_id": identity_id, "agent_id": agent_id},
    )
    return identity


async def record_action(
    session: AsyncSession,
    *,
    user_id: int,
    agent_id: int,
    channel: Channel,
    direction: Direction,
    text: str,
    status: ActionStatus = ActionStatus.OK,
    stats: dict | None = None,
) -> ActionLog:
    """`stats`: the telemetry of the turn whose reply this is (model, latency_ms,
    prompt_tokens, completion_tokens)."""
    stats = stats or {}
    entry = ActionLog(
        user_id=user_id,
        agent_id=agent_id,
        channel=channel,
        direction=direction,
        text=text,
        status=status,
        model=stats.get("model"),
        latency_ms=stats.get("latency_ms"),
        prompt_tokens=stats.get("prompt_tokens"),
        completion_tokens=stats.get("completion_tokens"),
        cached_tokens=stats.get("cached_tokens"),
        prefill_ms=round(stats["prefill_ms"]) if stats.get("prefill_ms") is not None else None,
    )
    session.add(entry)
    await session.flush()
    return entry


# The most recent messages kept per access request: a stranger cannot fill the database.
REQUEST_MESSAGES_KEEP = 50


async def record_request_message(session: AsyncSession, request: AccessRequest, text: str) -> None:
    """Keep one message of a sender who is not a user yet, with their access request,
    and only the REQUEST_MESSAGES_KEEP most recent per request."""
    session.add(RequestMessage(request_id=request.id, text=text))
    await session.flush()
    stale = (
        await session.execute(
            select(RequestMessage.id)
            .where(RequestMessage.request_id == request.id)
            .order_by(RequestMessage.id.desc())
            .offset(REQUEST_MESSAGES_KEEP)
        )
    ).scalars().all()
    if stale:
        await session.execute(delete(RequestMessage).where(RequestMessage.id.in_(stale)))


async def ensure_access_request(
    session: AsyncSession, channel: Channel, external_id: str, message_text: str
) -> tuple[AccessRequest, bool]:
    """One row per (channel, external_id), whatever its status (the table has a
    unique constraint on it), not one per message. The flag says whether admins
    should be told, that is whether the request is new or was reopened.

    - none yet: created pending, flag True.
    - pending: returned as it is, flag False.
    - denied: the denial stands, returned as it is, flag False, so a
      denied sender does not notify the admins with every message.
    - approved, and the person writes again (their identity was removed or
      their permission revoked since): reopened as pending with the new first
      message, flag True.
    """
    existing = (
        await session.execute(
            select(AccessRequest).where(
                AccessRequest.channel == channel, AccessRequest.external_id == external_id
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        if existing.status == RequestStatus.APPROVED:
            existing.status = RequestStatus.PENDING
            existing.first_message_text = message_text
            existing.resolved_at = None
            existing.resolved_by = None
            await session.flush()
            return existing, True
        return existing, False
    req = AccessRequest(channel=channel, external_id=external_id, first_message_text=message_text)
    session.add(req)
    await session.flush()
    return req, True


async def request_access(
    session: AsyncSession, channel: Channel, external_id: str, message_text: str
) -> AccessRequest:
    return (await ensure_access_request(session, channel, external_id, message_text))[0]


async def list_requests(
    session: AsyncSession, status: RequestStatus | None = RequestStatus.PENDING
) -> list[AccessRequest]:
    """Pending requests by default, oldest first. `status=None` lists all."""
    stmt = select(AccessRequest).order_by(AccessRequest.id)
    if status is not None:
        stmt = stmt.where(AccessRequest.status == status)
    return list((await session.execute(stmt)).scalars().all())


async def list_pending_requests(session: AsyncSession) -> list[AccessRequest]:
    return await list_requests(session, RequestStatus.PENDING)


async def _pending_request(session: AsyncSession, request_id: int) -> AccessRequest:
    request = await session.get(AccessRequest, request_id)
    if request is None:
        raise RequestNotFoundError(f"No request with id {request_id}")
    if request.status != RequestStatus.PENDING:
        raise RequestAlreadyResolvedError(
            f"Request {request_id} is already {request.status.value}"
        )
    return request


async def approve_request(
    session: AsyncSession, request_id: int, *, resolved_by: str | None = None
) -> User:
    """Creates the User + ChannelIdentity + CHAT permission in one step —
    an admin doesn't separately create the user then grant access. If the
    identity already exists it is granted CHAT and its user is returned.
    `resolved_by` records the actor ("api" or "console").
    """
    request = await _pending_request(session, request_id)

    # The identity may already exist: an admin can add it by hand after the
    # request was made (found on the real database). Approving then means
    # "let it chat": no second identity, no second user.
    stmt = select(ChannelIdentity).where(
        ChannelIdentity.channel == request.channel,
        ChannelIdentity.external_id == request.external_id,
    )
    identity = (await session.execute(stmt)).scalar_one_or_none()
    if identity is not None:
        user = await session.get(User, identity.user_id)
    else:
        user = User(display_name=f"Approved from {request.channel.value} request")
        session.add(user)
        await session.flush()
        identity = ChannelIdentity(
            user_id=user.id, channel=request.channel, external_id=request.external_id
        )
        session.add(identity)
        await session.flush()
    await grant_permission(session, identity, PermissionKind.CHAT)

    request.status = RequestStatus.APPROVED
    request.resolved_at = _utcnow()
    request.resolved_by = resolved_by
    await session.flush()
    await record_admin_event(
        session,
        actor=resolved_by or DEFAULT_ACTOR,
        action="request.approve",
        target_type="access_request",
        target_id=request_id,
        details={"user_id": user.id, "channel": request.channel.value},
    )
    return user


async def deny_request(
    session: AsyncSession, request_id: int, *, resolved_by: str | None = None
) -> None:
    request = await _pending_request(session, request_id)
    request.status = RequestStatus.DENIED
    request.resolved_at = _utcnow()
    request.resolved_by = resolved_by
    await session.flush()
    await record_admin_event(
        session,
        actor=resolved_by or DEFAULT_ACTOR,
        action="request.deny",
        target_type="access_request",
        target_id=request_id,
        details={"channel": request.channel.value},
    )


async def resolve_requests_for_identity(
    session: AsyncSession, channel: Channel, external_id: str, *, actor: str
) -> int:
    """Approve the pending access request of an identity that an admin has just
    created or granted a permission: the decision was made, so the request
    must not stay in the pending list. Returns how many were resolved.
    """
    pending = (
        (
            await session.execute(
                select(AccessRequest).where(
                    AccessRequest.channel == channel,
                    AccessRequest.external_id == external_id,
                    AccessRequest.status == RequestStatus.PENDING,
                )
            )
        )
        .scalars()
        .all()
    )
    for request in pending:
        request.status = RequestStatus.APPROVED
        request.resolved_at = _utcnow()
        request.resolved_by = actor
        await record_admin_event(
            session,
            actor=actor,
            action="request.auto_approve",
            target_type="access_request",
            target_id=request.id,
            details={"channel": channel.value},
        )
    if pending:
        await session.flush()
    return len(pending)


# --- Users, channel identities and permissions ---
# These used to live in the API routes, and the console ran its own queries.
# One implementation here, called by both front ends.


async def create_user(
    session: AsyncSession, display_name: str | None = None, *, actor: str = DEFAULT_ACTOR
) -> User:
    user = User(display_name=display_name)
    session.add(user)
    await session.flush()
    await record_admin_event(
        session,
        actor=actor,
        action="user.create",
        target_type="user",
        target_id=user.id,
        details={"display_name": display_name},
    )
    return user


async def list_users(session: AsyncSession) -> list[User]:
    return list((await session.execute(select(User).order_by(User.id))).scalars().all())


async def get_user(session: AsyncSession, user_id: int) -> User:
    return await _require_user(session, user_id)


async def update_user(
    session: AsyncSession,
    user_id: int,
    *,
    display_name: str | None = None,
    is_active: bool | None = None,
    timezone: str | None = None,
    actor: str = DEFAULT_ACTOR,
) -> User:
    """Only the fields that are given change. A new timezone moves the next time of
    each of the user's scheduled tasks."""
    user = await _require_user(session, user_id)
    changed: dict = {}
    if timezone is not None:
        from app.tasks import check_timezone, reschedule_user

        user.timezone = check_timezone(timezone)
        changed["timezone"] = timezone
        await session.flush()
        await reschedule_user(session, user_id)
    if display_name is not None:
        user.display_name = display_name
        changed["display_name"] = display_name
    if is_active is not None:
        user.is_active = is_active
        changed["is_active"] = is_active
    await session.flush()
    await record_admin_event(
        session,
        actor=actor,
        action="user.update",
        target_type="user",
        target_id=user_id,
        details=changed,
    )
    return user


async def _identity_of_user(
    session: AsyncSession, user_id: int, identity_id: int
) -> ChannelIdentity:
    await _require_user(session, user_id)
    identity = await session.get(ChannelIdentity, identity_id)
    if identity is None or identity.user_id != user_id:
        raise IdentityNotFoundError(f"No channel identity {identity_id} for user {user_id}")
    return identity


async def add_channel_identity(
    session: AsyncSession,
    user_id: int,
    channel: Channel,
    identifier: str,
    *,
    actor: str = DEFAULT_ACTOR,
) -> ChannelIdentity:
    """`identifier` is what an admin naturally has: a Telegram id, a Matrix
    user id, or an email address. The stored lookup key is computed here,
    with the same function the Auth Node uses, so the identity is
    immediately visible to authorize(). An email address is also kept, encrypted.
    """
    await _require_user(session, user_id)
    identifier = identifier.strip()
    if not identifier:
        raise InvalidInputError("An identifier must not be empty")
    external_id = channel_identifier_key(channel, identifier)
    stmt = select(ChannelIdentity).where(
        ChannelIdentity.channel == channel, ChannelIdentity.external_id == external_id
    )
    if (await session.execute(stmt)).scalar_one_or_none() is not None:
        raise IdentityAlreadyLinkedError("This channel identity is already linked to a user")
    identity = ChannelIdentity(
        user_id=user_id,
        channel=channel,
        external_id=external_id,
        raw_address=identifier if channel is Channel.EMAIL else None,
    )
    session.add(identity)
    await session.flush()
    # The identifier itself is not recorded: it can be an email address.
    await record_admin_event(
        session,
        actor=actor,
        action="identity.add",
        target_type="user",
        target_id=user_id,
        details={"channel": channel.value, "identity_id": identity.id},
    )
    await resolve_requests_for_identity(session, channel, external_id, actor=actor)
    return identity


async def list_channel_identities(session: AsyncSession, user_id: int) -> list[ChannelIdentity]:
    await _require_user(session, user_id)
    stmt = select(ChannelIdentity).where(ChannelIdentity.user_id == user_id).order_by(
        ChannelIdentity.id
    )
    return list((await session.execute(stmt)).scalars().all())


async def remove_channel_identity(
    session: AsyncSession, user_id: int, identity_id: int, *, actor: str = DEFAULT_ACTOR
) -> None:
    identity = await _identity_of_user(session, user_id, identity_id)
    channel = identity.channel
    # A scheduled task delivered on this identity has nowhere to go: removed with it.
    from app.graph import delete_threads
    from app.tasks import delete_tasks_of
    from app.tasks import thread_id as task_thread_id

    task_ids = await delete_tasks_of(session, identity_id=identity_id)
    await session.delete(identity)  # cascades to its permissions
    await session.flush()
    await delete_threads([task_thread_id(task_id) for task_id in task_ids])
    await record_admin_event(
        session,
        actor=actor,
        action="identity.remove",
        target_type="user",
        target_id=user_id,
        details={"channel": channel.value, "identity_id": identity_id, "tasks_deleted": task_ids},
    )


async def list_identity_permissions(
    session: AsyncSession, user_id: int, identity_id: int
) -> list[Permission]:
    await _identity_of_user(session, user_id, identity_id)
    stmt = (
        select(Permission)
        .where(Permission.channel_identity_id == identity_id)
        .order_by(Permission.id)
    )
    return list((await session.execute(stmt)).scalars().all())


async def grant_identity_permission(
    session: AsyncSession,
    user_id: int,
    identity_id: int,
    kind: PermissionKind,
    *,
    actor: str = DEFAULT_ACTOR,
) -> Permission:
    identity = await _identity_of_user(session, user_id, identity_id)
    permission = await grant_permission(session, identity, kind)
    await record_admin_event(
        session,
        actor=actor,
        action="permission.grant",
        target_type="user",
        target_id=user_id,
        details={"identity_id": identity_id, "kind": kind.value},
    )
    await resolve_requests_for_identity(
        session, identity.channel, identity.external_id, actor=actor
    )
    return permission


async def revoke_identity_permission(
    session: AsyncSession,
    user_id: int,
    identity_id: int,
    kind: PermissionKind,
    *,
    actor: str = DEFAULT_ACTOR,
) -> None:
    identity = await _identity_of_user(session, user_id, identity_id)
    if not await revoke_permission(session, identity, kind):
        raise PermissionNotHeldError(f"Permission {kind.value} was not held")
    await record_admin_event(
        session,
        actor=actor,
        action="permission.revoke",
        target_type="user",
        target_id=user_id,
        details={"identity_id": identity_id, "kind": kind.value},
    )


@dataclass(frozen=True)
class IdentityDetail:
    id: int
    channel: Channel
    external_id: str
    permissions: list[PermissionKind]
    active_agent_id: int | None = None


@dataclass(frozen=True)
class UserDetail:
    user: User
    identities: list[IdentityDetail]
    agents: list[Agent]


async def get_user_detail(session: AsyncSession, user_id: int) -> UserDetail:
    """A user with their channel identities, the permission held on each, and
    their agents: the "view detail".
    """
    user = await _require_user(session, user_id)
    identities = []
    for identity in await list_channel_identities(session, user_id):
        perms = await list_identity_permissions(session, user_id, identity.id)
        identities.append(
            IdentityDetail(
                id=identity.id,
                channel=identity.channel,
                external_id=identity.external_id,
                permissions=[p.kind for p in perms],
                active_agent_id=identity.active_agent_id,
            )
        )
    return UserDetail(user=user, identities=identities, agents=await list_agents(session, user_id))


LOG_SEARCH_MAX_LIMIT = 500
_LOG_SEARCH_BATCH = 500


def _as_utc(value: datetime) -> datetime:
    """The database stores UTC. A naive datetime is taken as UTC, an aware
    one is converted, so a caller in another timezone gets the window it
    meant instead of a silently shifted one.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


async def search_action_logs(
    session: AsyncSession,
    *,
    user_id: int | None = None,
    agent_id: int | None = None,
    channel: Channel | None = None,
    direction: Direction | None = None,
    status: ActionStatus | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    keyword: str | None = None,
    limit: int = 100,
    offset: int = 0,
    actor: str = DEFAULT_ACTOR,
) -> list[ActionLog]:
    """Search the audit trail and record that an administrator read
    it: the filters used, not the results. The caller must commit,
    read-only requests included, or the record is lost. See
    `_search_action_logs` for the filters.
    """
    filters = {
        "user_id": user_id,
        "agent_id": agent_id,
        "channel": channel,
        "direction": direction,
        "status": status,
        "since": since,
        "until": until,
        "keyword": keyword,
        "limit": limit,
        "offset": offset,
    }
    found = await _search_action_logs(session, **filters)
    await record_admin_event(
        session,
        actor=actor,
        action="logs.search",
        target_type="action_logs",
        details={
            **{
                k: v.value if isinstance(v, enum.Enum) else v
                for k, v in filters.items()
                if v is not None
            },
            "results": len(found),
        },
    )
    return found


_UNDELIVERED_LOOKBACK = 50


async def find_undelivered_answer(
    session: AsyncSession,
    user_id: int,
    agent_id: int,
    channel: Channel,
    inbound_text: str,
    not_answers: tuple[str, ...] = (),
) -> ActionLog | None:
    """The answer to this inbound message that was generated but not delivered,
    read back from the audit trail so a restart does not lose it.

    Finds the newest of the last few inbound entries with this exact text, then
    the first outbound entry after it: it is returned when its status is failed
    and its text is a real answer (not one of `not_answers`, the apology and the
    "no reply" note that a failed turn leaves). Returns None otherwise, and the
    caller runs the turn. The inbound text is encrypted, so the comparison is
    done in Python on a bounded window. Limit: two messages with the same text
    from the same sender can be mistaken for one another.
    """
    inbound = (
        await session.execute(
            select(ActionLog)
            .where(
                ActionLog.user_id == user_id,
                ActionLog.agent_id == agent_id,
                ActionLog.channel == channel,
                ActionLog.direction == Direction.INBOUND,
            )
            .order_by(ActionLog.id.desc())
            .limit(_UNDELIVERED_LOOKBACK)
        )
    ).scalars()
    entry = next((e for e in inbound if e.text == inbound_text), None)
    if entry is None:
        return None
    answer = (
        await session.execute(
            select(ActionLog)
            .where(
                ActionLog.user_id == user_id,
                ActionLog.agent_id == agent_id,
                ActionLog.channel == channel,
                ActionLog.direction == Direction.OUTBOUND,
                ActionLog.id > entry.id,
            )
            .order_by(ActionLog.id.asc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if (
        answer is None
        or answer.status != ActionStatus.FAILED
        or isinstance(answer.text, UndecryptableText)
        or answer.text in not_answers
    ):
        return None
    return answer


async def _search_action_logs(
    session: AsyncSession,
    *,
    user_id: int | None = None,
    agent_id: int | None = None,
    channel: Channel | None = None,
    direction: Direction | None = None,
    status: ActionStatus | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    keyword: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[ActionLog]:
    """Search the audit trail. Every filter is optional and they
    combine with AND. Results are most recent first (by insertion order).

    `since` is inclusive and `until` exclusive, so consecutive windows
    never overlap. `offset` skips that many *matching* rows, for paging.

    `keyword` is a case-insensitive substring match against the
    *decrypted* text. The text column holds Fernet ciphertext, which is
    different on every write, so it cannot be searched or compared in SQL:
    every other filter runs in SQL, then the keyword is applied in Python
    to each candidate row, newest first, stopping as soon as `limit`
    matches are found. That is O(n) in the number of rows passing the
    other filters, which is fine for a single-user to small-group local
    assistant. Narrow with user, agent, channel or dates first on a large
    log. No index or search infrastructure is built on purpose.
    """
    if not 1 <= limit <= LOG_SEARCH_MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {LOG_SEARCH_MAX_LIMIT}, got {limit}")
    if offset < 0:
        raise ValueError(f"offset must be 0 or more, got {offset}")

    stmt = select(ActionLog)
    if user_id is not None:
        stmt = stmt.where(ActionLog.user_id == user_id)
    if agent_id is not None:
        stmt = stmt.where(ActionLog.agent_id == agent_id)
    if channel is not None:
        stmt = stmt.where(ActionLog.channel == channel)
    if direction is not None:
        stmt = stmt.where(ActionLog.direction == direction)
    if status is not None:
        stmt = stmt.where(ActionLog.status == status)
    if since is not None:
        stmt = stmt.where(ActionLog.created_at >= _as_utc(since))
    if until is not None:
        stmt = stmt.where(ActionLog.created_at < _as_utc(until))
    stmt = stmt.order_by(ActionLog.id.desc())

    needle = keyword.strip().lower() if keyword else ""
    if not needle:
        return list((await session.execute(stmt.limit(limit).offset(offset))).scalars().all())

    matches: list[ActionLog] = []
    skipped = 0
    last_id: int | None = None
    while len(matches) < limit:
        batch_stmt = stmt if last_id is None else stmt.where(ActionLog.id < last_id)
        batch = list(
            (await session.execute(batch_stmt.limit(_LOG_SEARCH_BATCH))).scalars().all()
        )
        for entry in batch:
            if isinstance(entry.text, UndecryptableText) or needle not in entry.text.lower():
                continue
            if skipped < offset:
                skipped += 1
                continue
            matches.append(entry)
            if len(matches) == limit:
                break
        if len(batch) < _LOG_SEARCH_BATCH:
            break
        last_id = batch[-1].id
    return matches


async def search_admin_events(
    session: AsyncSession,
    *,
    actor: str | None = None,
    action: str | None = None,
    target_type: str | None = None,
    target_id: int | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
    reader: str | None = None,
) -> list[AdminEvent]:
    """Search what administrators did, newest first. Filters combine
    with AND; `since` is inclusive and `until` exclusive, like the log
    search. `reader` is who is asking: when given, the read is recorded as
    an `admin_events.search` event (after the query, so it does not appear in
    its own result) and the caller must commit.
    """
    if not 1 <= limit <= LOG_SEARCH_MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {LOG_SEARCH_MAX_LIMIT}, got {limit}")
    if offset < 0:
        raise ValueError(f"offset must be 0 or more, got {offset}")
    stmt = select(AdminEvent)
    if actor is not None:
        stmt = stmt.where(AdminEvent.actor == actor)
    if action is not None:
        stmt = stmt.where(AdminEvent.action == action)
    if target_type is not None:
        stmt = stmt.where(AdminEvent.target_type == target_type)
    if target_id is not None:
        stmt = stmt.where(AdminEvent.target_id == target_id)
    if since is not None:
        stmt = stmt.where(AdminEvent.created_at >= _as_utc(since))
    if until is not None:
        stmt = stmt.where(AdminEvent.created_at < _as_utc(until))
    stmt = stmt.order_by(AdminEvent.id.desc()).limit(limit).offset(offset)
    found = list((await session.execute(stmt)).scalars().all())
    if reader is not None:
        filters = {
            "actor": actor,
            "action": action,
            "target_type": target_type,
            "target_id": target_id,
            "since": since,
            "until": until,
            "limit": limit,
            "offset": offset,
        }
        await record_admin_event(
            session,
            actor=reader,
            action="admin_events.search",
            target_type="admin_events",
            details={**{k: v for k, v in filters.items() if v is not None}, "results": len(found)},
        )
    return found


@dataclass(frozen=True)
class StorageOverview:
    """Numbers about what the database holds."""

    db_size_bytes: int | None  # None when the database is not a SQLite file
    row_counts: dict[str, int]  # table name -> number of rows
    oldest_log_at: datetime | None  # UTC, None when there is no action log yet
    newest_log_at: datetime | None
    # Stored values that the current key cannot decrypt, per table.
    undecryptable_by_table: dict[str, int] = field(default_factory=dict)
    # The conversation checkpoints live in their own SQLite file:
    # size of the file plus its -wal and -shm files, and rows per table.
    # None / empty when that file does not exist yet or cannot be read.
    checkpoint_size_bytes: int | None = None
    checkpoint_row_counts: dict[str, int] = field(default_factory=dict)

    @property
    def undecryptable_rows(self) -> int:
        return sum(self.undecryptable_by_table.values())


_COUNTED_MODELS = (
    User,
    ChannelIdentity,
    Permission,
    AccessRequest,
    Agent,
    ActionLog,
    AdminEvent,
    RequestMessage,
    ApiCall,
    RoutingRule,
    RoutingConfig,
    McpServer,
    McpCall,
    McpGrant,
    MemoryEntry,
    BackupSchedule,
    RetentionConfig,
    Skill,
    SkillVersion,
    ScheduledTask,
    TaskConfig,
    TaskFeedItem,
    FeedSource,
    AgentTemplate,
    AgentTemplateVersion,
    AdminAccount,
    ApiToken,
)


# Tables of the main database that the overview deliberately does not count.
# A guard test fails when the schema gains a table that is in neither
# _COUNTED_MODELS nor this set, so the overview cannot drift silently.
STORAGE_EXCLUDED_TABLES = frozenset({"alembic_version"})

# Tables of the separate conversation-checkpoint file (created by LangGraph).
CHECKPOINT_TABLES = ("checkpoints", "writes")


def _checkpoint_stats() -> tuple[int | None, dict[str, int]]:
    """Size (main file + WAL + shared-memory file) and row counts of the
    checkpoint database, read without creating or changing it.
    """
    import sqlite3

    from app.checkpoints import checkpoint_db_path

    path = checkpoint_db_path()
    if not path.exists():
        return None, {}
    files = (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm"))
    size = sum(candidate.stat().st_size for candidate in files if candidate.exists())
    counts: dict[str, int] = {}
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            for table in CHECKPOINT_TABLES:
                counts[table] = con.execute(f'select count(*) from "{table}"').fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error:
        logger.warning("The checkpoint database could not be read for the storage overview")
        counts = {}
    return size, counts


# (table, encrypted column): every column that holds an encrypted value.
_ENCRYPTED_COLUMNS = (
    ("action_logs", "text"),
    ("access_requests", "first_message_text"),
    ("request_messages", "text"),
    ("channel_identities", "raw_address"),
    ("admin_events", "details"),
    ("agents", "system_prompt"),
    ("agents", "purpose"),
    ("mcp_servers", "env_vars"),
    ("mcp_calls", "detail"),
    ("mcp_calls", "arguments"),
    ("memory_entries", "title"),
    ("memory_entries", "content"),
    ("scheduled_tasks", "prompt"),
    ("admin_accounts", "totp_secret"),
)


async def count_undecryptable(session: AsyncSession) -> dict[str, int]:
    """How many stored values the current ENCRYPTION_KEY cannot decrypt, per
    table, counting only tables that have some. Reads the raw ciphertext with
    plain SQL and tries to decrypt each value itself, so it does not depend on
    the marker the ORM shows and a real message that says `<undecryptable>`
    is never counted. O(n) in the number of encrypted values.
    """
    from sqlalchemy import text

    from app.security.encryption import decrypt_value

    counts: dict[str, int] = {}
    for table, column in _ENCRYPTED_COLUMNS:
        rows = await session.execute(
            text(f'select "{column}" from "{table}" where "{column}" is not null')
        )
        bad = 0
        for (value,) in rows:
            try:
                decrypt_value(value)
            except ValueError:
                bad += 1
        if bad:
            counts[table] = bad
    return counts


async def storage_overview(session: AsyncSession) -> StorageOverview:
    """Basic visibility for the admin: the size of the SQLite file,
    the row count of every table, and the time span the audit trail
    covers. Not a storage engine or a backup tool, just numbers.

    The size is that of the main database file only. Row counts are exact
    `COUNT(*)` queries, not estimates.
    """
    counts = {
        model.__tablename__: (
            await session.execute(select(func.count()).select_from(model))
        ).scalar_one()
        for model in _COUNTED_MODELS
    }
    span = select(func.min(ActionLog.created_at), func.max(ActionLog.created_at))
    oldest, newest = (await session.execute(span)).one()

    path = sqlite_file_path(get_settings().database_url)
    size = path.stat().st_size if path is not None and path.exists() else None
    checkpoint_size, checkpoint_counts = await asyncio.to_thread(_checkpoint_stats)
    return StorageOverview(
        db_size_bytes=size,
        row_counts=counts,
        oldest_log_at=_as_utc(oldest) if oldest is not None else None,
        newest_log_at=_as_utc(newest) if newest is not None else None,
        undecryptable_by_table=await count_undecryptable(session),
        checkpoint_size_bytes=checkpoint_size,
        checkpoint_row_counts=checkpoint_counts,
    )


@dataclass(frozen=True)
class DeletionReport:
    user_id: int
    agents_deleted: int
    logs_deleted: int
    threads_deleted: int = 0  # conversation threads whose checkpoints were purged


async def _thread_ids_of_user(
    session: AsyncSession, user_id: int, agent_id: int | None = None
) -> list[str]:
    """Every conversation thread the user can have: each of their channel
    identities times each of their agents (the scheme of app.graph), or
    times one agent when `agent_id` is given.
    """
    from app.graph import thread_id_from_key

    identities = (
        (await session.execute(select(ChannelIdentity).where(ChannelIdentity.user_id == user_id)))
        .scalars()
        .all()
    )
    agent_query = select(Agent.id).where(Agent.user_id == user_id)
    if agent_id is not None:
        agent_query = agent_query.where(Agent.id == agent_id)
    agent_ids = (await session.execute(agent_query)).scalars().all()
    from app.db.models import ScheduledTask
    from app.tasks import thread_id as task_thread_id

    task_query = select(ScheduledTask.id).where(ScheduledTask.user_id == user_id)
    if agent_id is not None:
        task_query = task_query.where(ScheduledTask.agent_id == agent_id)
    task_ids = (await session.execute(task_query)).scalars().all()
    from app.builder import thread_id as builder_thread_id
    from app.graph import threads_with_history

    # The agent builder's dialogue is per identity, not per agent: part of the user's
    # threads, not of one agent's; counted only when one was opened.
    builder_threads = []
    if agent_id is None:
        for identity in identities:
            thread = builder_thread_id(identity.channel, identity.external_id)
            if await threads_with_history([thread]):
                builder_threads.append(thread)
    return (
        [
            thread_id_from_key(identity.channel, identity.external_id, agent_id)
            for identity in identities
            for agent_id in agent_ids
        ]
        + [task_thread_id(task_id) for task_id in task_ids]
        + builder_threads
    )


async def reset_conversation(
    session: AsyncSession,
    user_id: int,
    agent_id: int | None = None,
    *,
    actor: str = DEFAULT_ACTOR,
) -> int:
    """Forget the stored history of a user's conversations: one agent's,
    or all of them. This is the remedy when a thread cannot be decrypted (a
    corrupted checkpoint, a key that no longer matches) and every message on
    it fails. The next message starts a fresh conversation; the audit trail
    (`action_logs`) is not touched. Returns how many conversations had a
    stored history and were cleared.

    The checkpoints live in another database file that cannot join the
    caller's transaction: they are deleted here, the event is written in the
    caller's transaction.
    """
    await _require_user(session, user_id)
    if agent_id is not None:
        agent = await session.get(Agent, agent_id)
        if agent is None or agent.user_id != user_id:
            raise AgentNotFoundError(f"No agent {agent_id} for user {user_id}")
    from app.graph import delete_threads, threads_with_history

    thread_ids = await _thread_ids_of_user(session, user_id, agent_id)
    threads = await threads_with_history(thread_ids)
    await delete_threads(thread_ids)
    await record_admin_event(
        session,
        actor=actor,
        action="conversation.reset",
        target_type="user",
        target_id=user_id,
        details={"agent_id": agent_id, "threads": threads},
    )
    return threads


async def delete_user(
    session: AsyncSession, user_id: int, *, purge: bool = False, actor: str = DEFAULT_ACTOR
) -> DeletionReport:
    """Delete a user.

    Policy: a user who has audit-trail entries is **not** deleted unless
    `purge` is set, because the audit trail is the point and the
    normal way to stop someone is to deactivate them. With no history the
    user goes together with their agents, channel identities and
    permissions. `purge=True` also deletes the agents, every log entry and
    the conversation checkpoints. The database side runs in the
    caller's transaction: nothing persists unless the caller commits. The
    checkpoint file is a separate database and cannot join that transaction,
    so it is purged after the rows are deleted and before the commit.
    """
    user = await _require_user(session, user_id)
    agents = (
        await session.execute(
            select(func.count()).select_from(Agent).where(Agent.user_id == user_id)
        )
    ).scalar_one()
    logs = (
        await session.execute(
            select(func.count()).select_from(ActionLog).where(ActionLog.user_id == user_id)
        )
    ).scalar_one()
    if logs and not purge:
        raise UserHasHistoryError(user_id, agents, logs)

    thread_ids = await _thread_ids_of_user(session, user_id) if purge else []
    # An identity that selected an agent references it: clear that first.
    await session.execute(
        update(ChannelIdentity)
        .where(ChannelIdentity.user_id == user_id)
        .values(active_agent_id=None)
    )
    await session.execute(delete(ActionLog).where(ActionLog.user_id == user_id))
    # MCP grants reference the user and the agents: gone with them.
    await session.execute(delete(McpGrant).where(McpGrant.user_id == user_id))
    # Memory entries belong to the user and their agents: gone with them.
    await session.execute(delete(MemoryEntry).where(MemoryEntry.user_id == user_id))
    # Scheduled tasks name the user's agents and identities: gone with them. Their
    # threads are in `thread_ids` (a purge) or never ran (no history, no purge needed).
    from app.tasks import delete_tasks_of

    await delete_tasks_of(session, user_id=user_id)
    await session.execute(delete(Agent).where(Agent.user_id == user_id))
    await session.delete(user)  # cascades to channel identities and permissions
    await session.flush()
    # The conversation checkpoints live in another file: purged only
    # once the database side went through, so a failure above loses nothing.
    from app.graph import delete_threads

    threads = await delete_threads(thread_ids)
    await record_admin_event(
        session,
        actor=actor,
        action="user.delete",
        target_type="user",
        target_id=user_id,
        details={
            "purge": purge,
            "agents_deleted": agents,
            "logs_deleted": logs,
            "threads_deleted": threads,
        },
    )
    return DeletionReport(
        user_id=user_id, agents_deleted=agents, logs_deleted=logs, threads_deleted=threads
    )


# --- telemetry ---

TELEMETRY_GROUPS = ("none", "user", "agent", "model")


def _percentile(values: list[int], share: float) -> int | None:
    """Nearest-rank percentile: the smallest value with at least `share` of them at or below."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * share // 1))  # ceil without math
    return ordered[int(rank) - 1]


async def telemetry(
    session: AsyncSession,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    group_by: str = "none",
) -> list[dict]:
    """Counts over the audit trail for a period (`since` included, `until` excluded), one row
    per group: messages that reached an agent, replies, failed turns, denied messages, reply
    latency p50 and p95, and the tokens the engine reported. Computed from the columns only:
    no message text is read (nothing is decrypted). The model is known on replies only, so
    grouped by model the incoming messages fall under `null`."""
    if group_by not in TELEMETRY_GROUPS:
        raise InvalidInputError(f"group_by is one of: {', '.join(TELEMETRY_GROUPS)}")
    stmt = select(
        ActionLog.user_id,
        ActionLog.agent_id,
        ActionLog.model,
        ActionLog.direction,
        ActionLog.status,
        ActionLog.latency_ms,
        ActionLog.prompt_tokens,
        ActionLog.completion_tokens,
        ActionLog.cached_tokens,
        ActionLog.prefill_ms,
    )
    if since is not None:
        stmt = stmt.where(ActionLog.created_at >= since)
    if until is not None:
        stmt = stmt.where(ActionLog.created_at < until)
    groups: dict = {}
    for row in (await session.execute(stmt)).all():
        key = {"none": None, "user": row.user_id, "agent": row.agent_id, "model": row.model}[
            group_by
        ]
        g = groups.setdefault(
            key,
            {"messages": 0, "replies": 0, "failed_turns": 0, "denied": 0, "latencies": [],
             "prompt_tokens": 0, "completion_tokens": 0, "measured_prompt": 0,
             "cached_tokens": 0, "prefills": []},
        )  # fmt: skip
        inbound = row.direction == Direction.INBOUND
        if inbound and row.status == ActionStatus.OK:
            g["messages"] += 1
        elif inbound and row.status == ActionStatus.DENIED:
            g["denied"] += 1
        elif not inbound and row.status == ActionStatus.FAILED:
            g["failed_turns"] += 1
        elif not inbound and row.status == ActionStatus.OK:
            g["replies"] += 1
            if row.latency_ms is not None:
                g["latencies"].append(row.latency_ms)
        g["prompt_tokens"] += row.prompt_tokens or 0
        g["completion_tokens"] += row.completion_tokens or 0
        if row.cached_tokens is not None:  # Replies of the engine that reports it
            g["cached_tokens"] += row.cached_tokens
            g["measured_prompt"] += row.prompt_tokens or 0
            if row.prefill_ms is not None:
                g["prefills"].append(row.prefill_ms)
    result = []
    for key, g in sorted(groups.items(), key=lambda kv: (kv[0] is None, str(kv[0]))):
        latencies = g.pop("latencies")
        prefills, measured = g.pop("prefills"), g.pop("measured_prompt")
        g["cache_reuse_rate"] = round(g["cached_tokens"] / measured, 3) if measured else None
        g["prefill_mean_ms"] = round(sum(prefills) / len(prefills)) if prefills else None
        result.append(
            {
                "group_by": group_by,
                "key": key,
                **g,
                "latency_p50_ms": _percentile(latencies, 0.50),
                "latency_p95_ms": _percentile(latencies, 0.95),
            }
        )
    return result


# --- conversation export ---

EXPORT_FORMATS = ("json", "markdown")
EXPORT_MAX_MESSAGES = 100_000


async def export_conversation(
    session: AsyncSession,
    user_id: int,
    *,
    agent_id: int | None = None,
    include_text: bool = False,
    fmt: str = "json",
    actor: str = DEFAULT_ACTOR,
) -> tuple[list[dict], dict]:
    """Every message of one user's conversation (the audit trail of the user, or of one of
    their agents), oldest first. The text is decrypted only when `include_text` is asked;
    otherwise each message carries its length. One admin event records the export (who, whose
    conversation, how many messages, whether with text); the caller commits."""
    if fmt not in EXPORT_FORMATS:
        raise InvalidInputError(f"format is one of: {', '.join(EXPORT_FORMATS)}")
    user = await _require_user(session, user_id)
    if agent_id is not None:
        agent = await session.get(Agent, agent_id)
        if agent is None or agent.user_id != user_id:
            raise NotFoundError(f"No agent {agent_id} for user {user_id}")
    stmt = select(ActionLog).where(ActionLog.user_id == user_id)
    if agent_id is not None:
        stmt = stmt.where(ActionLog.agent_id == agent_id)
    stmt = stmt.order_by(ActionLog.created_at, ActionLog.id).limit(EXPORT_MAX_MESSAGES)
    messages = []
    for entry in (await session.execute(stmt)).scalars():
        item = {
            "id": entry.id,
            "at": entry.created_at.isoformat() if entry.created_at else None,
            "agent_id": entry.agent_id,
            "channel": entry.channel.value,
            "direction": entry.direction.value,
            "status": entry.status.value,
        }
        if include_text:
            item["text"] = entry.text
        else:
            item["length"] = len(entry.text or "")
        messages.append(item)
    about = {
        "user_id": user_id,
        "display_name": user.display_name,
        "agent_id": agent_id,
        "with_text": include_text,
        "messages": len(messages),
    }
    await record_admin_event(
        session,
        actor=actor,
        action="conversation.export",
        target_type="user",
        target_id=user_id,
        details={"agent_id": agent_id, "format": fmt, "with_text": include_text,
                 "messages": len(messages)},
    )  # fmt: skip
    return messages, about


def conversation_markdown(messages: list[dict], about: dict) -> str:
    """The export as Markdown: a header, then one section per message. The text is put in a
    fenced block, so what a channel user wrote is shown as text, never as markup."""
    who = about["display_name"] or f"user {about['user_id']}"
    agent = f", agent {about['agent_id']}" if about["agent_id"] is not None else ""
    lines = [f"# Conversation of {who}{agent}", "", f"{about['messages']} message(s).", ""]
    for m in messages:
        arrow = "from the user" if m["direction"] == "inbound" else "from the agent"
        flag = "" if m["status"] == "ok" else f" [{m['status']}]"
        lines.append(f"## {m['at']} · {m['channel']} · agent {m['agent_id']} · {arrow}{flag}")
        lines.append("")
        if "text" in m:
            fence = "````" if "```" in m["text"] else "```"
            lines += [fence + "text", m["text"], fence, ""]
        else:
            lines += [f"({m['length']} characters, text not exported)", ""]
    return "\n".join(lines)
