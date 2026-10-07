"""One timeline of who asked what and who did what.

Five sources, merged newest first:

- `message`: a message of a user or a reply to them (`action_logs`);
- `unknown_sender`: a message of a sender who is not a user yet (`request_messages`);
- `admin`: an administrative change or a sensitive read (`admin_events`);
- `tool`: an MCP tool call (`mcp_calls`);
- `api`: a request to the Admin API, refused ones included (`api_calls`).

Filters: `user_id` keeps the messages and tool calls of that user; `actor` keeps the admin events
and API calls of that actor label; `channel` keeps the messages; `kinds` keeps some sources;
`since` (inclusive) and `until` (exclusive). Paging: `limit`, `offset`, and the total of every
matching row. The text of messages is included only with `with_text`, and that read is itself
an admin event (`audit.read_text`), as a log search is.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.service import LOG_SEARCH_MAX_LIMIT, InvalidInputError, record_admin_event
from app.db.models import (
    AccessRequest,
    ActionLog,
    AdminEvent,
    ApiCall,
    Channel,
    McpCall,
    RequestMessage,
)

KINDS = ("message", "unknown_sender", "admin", "tool", "api")


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _when(stmt, column, since, until):
    if since is not None:
        stmt = stmt.where(column >= since)
    if until is not None:
        stmt = stmt.where(column < until)
    return stmt


def _sources(user_id, actor, channel, since, until) -> dict:
    """(select of rows, select of the count) per kind; a kind that a filter excludes is None."""
    by_user, by_actor, by_channel = user_id is not None, actor is not None, channel is not None
    sources: dict = {}
    if not by_actor:
        stmt = select(ActionLog)
        if by_user:
            stmt = stmt.where(ActionLog.user_id == user_id)
        if by_channel:
            stmt = stmt.where(ActionLog.channel == channel)
        sources["message"] = (_when(stmt, ActionLog.created_at, since, until), ActionLog)
    if not by_actor and not by_user:
        stmt = select(RequestMessage, AccessRequest).join(
            AccessRequest, AccessRequest.id == RequestMessage.request_id
        )
        if by_channel:
            stmt = stmt.where(AccessRequest.channel == channel)
        sources["unknown_sender"] = (
            _when(stmt, RequestMessage.created_at, since, until), RequestMessage
        )
    if not by_user and not by_channel:
        stmt = select(AdminEvent)
        if by_actor:
            stmt = stmt.where(AdminEvent.actor == actor)
        sources["admin"] = (_when(stmt, AdminEvent.created_at, since, until), AdminEvent)
    if not by_actor and not by_channel:
        stmt = select(McpCall)
        if by_user:
            stmt = stmt.where(McpCall.user_id == user_id)
        sources["tool"] = (_when(stmt, McpCall.created_at, since, until), McpCall)
    if not by_user and not by_channel:
        stmt = select(ApiCall)
        if by_actor:
            stmt = stmt.where(ApiCall.actor == actor)
        sources["api"] = (_when(stmt, ApiCall.created_at, since, until), ApiCall)
    return sources


def _row(kind: str, found, with_text: bool) -> dict:
    if kind == "message":
        log = found[0]
        detail = f"agent {log.agent_id}"
        if log.latency_ms is not None:
            detail += f", {log.model}, {log.latency_ms} ms"
        return {"at": log.created_at, "kind": kind, "who": f"user {log.user_id}",
                "what": f"{log.direction.value} {log.channel.value}", "status": log.status.value,
                "detail": detail, "text": log.text if with_text else None}  # fmt: skip
    if kind == "unknown_sender":
        message, request = found
        return {"at": message.created_at, "kind": kind,
                "who": f"{request.channel.value}/{request.external_id}",
                "what": f"message, access request #{request.id}", "status": request.status.value,
                "detail": None, "text": message.text if with_text else None}  # fmt: skip
    if kind == "admin":
        event = found[0]
        target = event.target_type + (f" {event.target_id}" if event.target_id is not None else "")
        return {"at": event.created_at, "kind": kind, "who": event.actor, "what": event.action,
                "status": None, "detail": target, "text": None}  # fmt: skip
    if kind == "tool":
        call = found[0]
        who = f"user {call.user_id}" if call.user_id is not None else "unknown"
        return {"at": call.created_at, "kind": kind, "who": who,
                "what": f"{call.server_name}.{call.tool_name}", "status": call.status,
                "detail": f"decision {call.decision}, agent {call.agent_id}, "
                          f"{call.duration_ms} ms", "text": None}  # fmt: skip
    call = found[0]
    return {"at": call.created_at, "kind": kind, "who": call.actor,
            "what": f"{call.method} {call.path}", "status": str(call.status),
            "detail": f"{call.duration_ms} ms from {call.source}", "text": None}  # fmt: skip


async def timeline(
    session: AsyncSession,
    *,
    user_id: int | None = None,
    actor: str | None = None,
    channel: Channel | None = None,
    kinds: list[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
    with_text: bool = False,
    reader: str | None = None,
) -> tuple[list[dict], int]:
    """(rows of the page, total of matching rows). With `with_text` and a `reader`, records
    an `audit.read_text` admin event; the caller commits."""
    if not 1 <= limit <= LOG_SEARCH_MAX_LIMIT:
        raise InvalidInputError(f"limit is from 1 to {LOG_SEARCH_MAX_LIMIT}")
    if offset < 0:
        raise InvalidInputError("offset is 0 or more")
    unknown = set(kinds or []) - set(KINDS)
    if unknown:
        raise InvalidInputError(f"kinds are among: {', '.join(KINDS)}")
    since, until = _utc(since), _utc(until)
    sources = _sources(user_id, actor, channel, since, until)
    if kinds:
        sources = {k: v for k, v in sources.items() if k in kinds}
    total = 0
    merged: list[dict] = []
    for kind, (stmt, model) in sources.items():
        total += (
            await session.execute(select(func.count()).select_from(stmt.subquery()))
        ).scalar_one()
        page = stmt.order_by(model.created_at.desc(), model.id.desc()).limit(limit + offset)
        for found in (await session.execute(page)).all():
            merged.append(_row(kind, found, with_text))
    merged.sort(key=lambda row: _utc(row["at"]), reverse=True)
    rows = merged[offset : offset + limit]
    if with_text and reader is not None:
        await record_admin_event(
            session, actor=reader, action="audit.read_text", target_type="audit",
            details={"user_id": user_id, "actor": actor, "kinds": kinds, "limit": limit,
                     "offset": offset, "returned": len(rows)},
        )  # fmt: skip
    return rows, total
