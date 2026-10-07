"""What the MCP guard does once it detects something: record, tell, suspend.

- `record_threat`: an admin event `mcp.threat` (actor `guard`, the user as target, the tool and
  the reason, never an argument value), a short notification to the administrators, and, at
  MCP_GUARD_SUSPEND_AFTER threats for one user in 24 hours, the suspension of their tools
  (`users.tools_suspended`, admin event `mcp.suspend`, another notification).
- `after_call`: behaviour over the last hour. Refused calls of one user (not granted, denied,
  blocked, ...) reaching MCP_GUARD_REFUSALS_PER_HOUR, and outbound calls reaching
  MCP_GUARD_OUTBOUND_PER_HOUR, are each one threat when the count reaches the threshold.
- `resume`: an administrator lifts a suspension (admin event `mcp.resume`).

Every step is best effort for the turn: a failure here is logged and the tool's answer to the
model is unchanged.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from app.db.models import AdminEvent, McpCall, User
from app.mcp.policy import Decision

logger = logging.getLogger("channelagent")

GUARD_ACTOR = "guard"
REFUSED = frozenset(
    {
        Decision.NOT_GRANTED, Decision.DENIED, Decision.UNAPPROVED, Decision.CREDENTIALS,
        Decision.NOT_OFFERED, Decision.INVALID, Decision.BLOCKED,
    }
)  # fmt: skip
HOUR = 3600.0
_outbound: dict[int, deque] = defaultdict(deque)


def _settings():
    from app.config import get_settings

    return get_settings()


async def is_suspended(user_id: int | None) -> bool:
    if user_id is None:
        return False
    from app.db.session import session_scope

    async with session_scope() as session:
        user = await session.get(User, user_id)
        return bool(user is not None and user.tools_suspended)


async def record_threat(user_id: int | None, agent_id: int | None, tool: str, reason: str) -> None:
    from app.admin.service import record_admin_event
    from app.channels.notify import notify_admins
    from app.db.session import session_scope

    try:
        async with session_scope() as session:
            await record_admin_event(
                session, actor=GUARD_ACTOR, action="mcp.threat", target_type="user",
                target_id=user_id, details={"tool": tool, "reason": reason, "agent_id": agent_id},
            )  # fmt: skip
            await session.commit()
            await notify_admins(
                session, f"Tool use flagged for user {user_id}: {tool}: {reason}."
            )
            await _maybe_suspend(session, user_id)
    except Exception:
        logger.exception("Recording a tool-use threat for user %s failed", user_id)


async def _maybe_suspend(session, user_id: int | None) -> None:
    from app.admin.service import record_admin_event
    from app.channels.notify import notify_admins

    limit = _settings().mcp_guard_suspend_after
    if user_id is None or limit <= 0:
        return
    user = await session.get(User, user_id)
    if user is None or user.tools_suspended:
        return
    since = datetime.now(UTC) - timedelta(hours=24)
    count = (
        await session.execute(
            select(func.count()).where(
                AdminEvent.action == "mcp.threat", AdminEvent.target_id == user_id,
                AdminEvent.created_at >= since,
            )
        )
    ).scalar_one()  # fmt: skip
    if count < limit:
        return
    user.tools_suspended = True
    await record_admin_event(
        session, actor=GUARD_ACTOR, action="mcp.suspend", target_type="user", target_id=user_id,
        details={"threats_in_24h": count},
    )  # fmt: skip
    await session.commit()
    await notify_admins(
        session,
        f"Tools of user {user_id} suspended after {count} flagged uses in 24 hours. "
        f"Resume them from the admin UI or: ./start.sh --admin resume-tools --user-id {user_id}",
    )


async def after_call(
    user_id: int | None, agent_id: int | None, tool: str, decision: Decision, outbound: bool,
    now: float | None = None,
) -> None:
    """Behaviour of the last hour, checked after each call (the call's row is written)."""
    if user_id is None:
        return
    settings = _settings()
    if decision in REFUSED:
        from app.db.session import session_scope

        since = datetime.now(UTC) - timedelta(hours=1)
        async with session_scope() as session:
            refused = (
                await session.execute(
                    select(func.count()).where(
                        McpCall.user_id == user_id, McpCall.created_at >= since,
                        McpCall.decision.in_([d.value for d in REFUSED]),
                    )
                )
            ).scalar_one()  # fmt: skip
        if refused == settings.mcp_guard_refusals_per_hour:
            await record_threat(user_id, agent_id, tool, f"{refused} refused tool calls in an hour")
    elif outbound:
        now = time.monotonic() if now is None else now
        calls = _outbound[user_id]
        calls.append(now)
        while calls and now - calls[0] > HOUR:
            calls.popleft()
        if len(calls) == settings.mcp_guard_outbound_per_hour:
            await record_threat(user_id, agent_id, tool, f"{len(calls)} outbound calls in an hour")


async def resume(session, user_id: int, *, actor: str) -> User:
    """Lift a suspension. The caller commits."""
    from app.admin.service import NotFoundError, record_admin_event

    user = await session.get(User, user_id)
    if user is None:
        raise NotFoundError(f"User {user_id} not found")
    was = user.tools_suspended
    user.tools_suspended = False
    await record_admin_event(
        session, actor=actor, action="mcp.resume", target_type="user", target_id=user_id,
        details={"was_suspended": was},
    )  # fmt: skip
    return user
