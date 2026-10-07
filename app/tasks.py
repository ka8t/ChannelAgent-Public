"""Scheduled tasks.

A user's task sends a prompt to one of their agents on a schedule, in a conversation of its
own (thread `task_{id}`), and delivers the reply on the channel identity the task names
(Telegram through the running adapter, email by SMTP to the identity's address).

Three schedule forms, read in the user's timezone (`users.timezone`, null = UTC):

    cron   five fields "minute hour day-of-month month day-of-week": `*`, numbers, ranges
           `a-b`, lists `a,b`, steps `*/n` and `a-b/n`; day of week 0-7, 0 and 7 are Sunday.
           When both day fields are restricted, a day matching either one counts (as cron).
    every  a period: `30m`, `2h`, `1d` (one minute to 30 days), counted from the last run.
    daily  a time `HH:MM`.

A schedule that does not parse, or never falls due, is refused (InvalidInputError, 422 in the
Admin API). The runner (`run_scheduler`, an application component like the backup scheduler)
claims a due task by moving its next time forward before running it, so a task runs once per
time; a task that falls due while the global switch is paused is moved on, not run late.

Ownership: every function takes the owning `user_id` when called for a user (the `/task`
command): a task of another user is "not found", exactly like a task that does not exist.
The Admin API calls them without it, for any user's task.
"""

import asyncio
import functools
import logging
import re
import zoneinfo
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.service import (
    AgentNotFoundError,
    ConflictError,
    IdentityNotFoundError,
    InvalidInputError,
    NotFoundError,
    _require_user,
    record_action,
    record_admin_event,
    resolve_agent,
)
from app.db.models import (
    ActionStatus,
    Agent,
    Channel,
    ChannelIdentity,
    Direction,
    PermissionKind,
    ScheduledTask,
    TaskConfig,
    User,
)

logger = logging.getLogger("channelagent")

KINDS = ("cron", "every", "daily")
MAX_PROMPT = 2000
MAX_EXPR = 100
MAX_TASKS_PER_USER = 20
MIN_EVERY_MINUTES, MAX_EVERY_MINUTES = 1, 30 * 1440
POLL_SECONDS = 30.0  # a task falls due at a minute: checked at least twice a minute
SEARCH_DAYS = 5 * 366  # a cron time is looked for over five years (29 February included)
DELIVERY_CHANNELS = (Channel.TELEGRAM, Channel.EMAIL)


class TaskNotFoundError(NotFoundError):
    pass


def thread_id(task_id: int) -> str:
    """The conversation of one task: its turns never mix with the user's own chat."""
    return f"task_{task_id}"


def _as_utc(value: datetime | None) -> datetime | None:
    # SQLite returns naive datetimes even for timezone-aware columns.
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


# --- timezone ---


@functools.lru_cache(maxsize=1)
def _known_timezones() -> frozenset[str]:
    return frozenset(zoneinfo.available_timezones())


def check_timezone(name) -> str:
    """An IANA name from the system's database ("Europe/Paris", "UTC"). Checked against the
    list, not opened: a name is never read as a path."""
    if not isinstance(name, str) or name not in _known_timezones():
        raise InvalidInputError(
            f"Unknown timezone {str(name)[:64]!r}: use an IANA name such as Europe/Paris or UTC"
        )
    return name


def guess_timezone(text) -> str | None:
    """The IANA name a user means: the exact name in any case ("europe/paris"), or a
    city that names exactly one zone ("Paris", "new york", "Sao_Paulo"). None when nothing, or
    more than one zone, matches. Only names of the system's list are returned."""
    if not isinstance(text, str):
        return None
    wanted = " ".join(text.strip().strip(".!?").split()).lower()
    if not wanted:
        return None
    zones = _known_timezones()
    by_lower = {zone.lower(): zone for zone in zones}
    if wanted in by_lower:
        return by_lower[wanted]
    city = wanted.replace(" ", "_")
    matches = [zone for zone in zones if zone.rsplit("/", 1)[-1].lower() == city]
    return matches[0] if len(matches) == 1 else None


def _zone(name: str | None) -> zoneinfo.ZoneInfo:
    return zoneinfo.ZoneInfo(name or "UTC")


# --- schedules ---

_CRON_FIELDS = (
    ("minute", 0, 59),
    ("hour", 0, 23),
    ("day of month", 1, 31),
    ("month", 1, 12),
    ("day of week", 0, 7),
)
_EVERY = re.compile(r"(\d{1,6})\s*([mhd])")
_DAILY = re.compile(r"([01]?\d|2[0-3]):([0-5]\d)")
_UNIT_MINUTES = {"m": 1, "h": 60, "d": 1440}


@dataclass(frozen=True)
class Cron:
    minutes: frozenset[int]
    hours: frozenset[int]
    days: frozenset[int]
    months: frozenset[int]
    weekdays: frozenset[int]  # 0 = Sunday
    days_restricted: bool
    weekdays_restricted: bool

    def day_matches(self, day: date) -> bool:
        if day.month not in self.months:
            return False
        in_month = day.day in self.days
        in_week = (day.isoweekday() % 7) in self.weekdays
        if self.days_restricted and self.weekdays_restricted:
            return in_month or in_week
        return in_month and in_week


def _number(text: str, name: str, low: int, high: int) -> int:
    if not text.isdigit() or not low <= int(text) <= high:
        raise InvalidInputError(f"cron {name}: {text!r} is not a number from {low} to {high}")
    return int(text)


def _cron_field(text: str, name: str, low: int, high: int) -> set[int]:
    values: set[int] = set()
    for part in text.split(","):
        base, slash, step_text = part.partition("/")
        step = _number(step_text, f"{name} step", 1, high) if slash else 1
        if base == "*":
            start, end = low, high
        elif "-" in base:
            first, _, last = base.partition("-")
            start, end = _number(first, name, low, high), _number(last, name, low, high)
            if start > end:
                raise InvalidInputError(f"cron {name}: the range {base!r} runs backwards")
        else:
            start = _number(base, name, low, high)
            end = high if slash else start
        values.update(range(start, end + 1, step))
    return values


def parse_cron(expr: str) -> Cron:
    fields = expr.split()
    if len(fields) != 5:
        raise InvalidInputError(
            "cron needs five fields: minute hour day-of-month month day-of-week"
        )
    sets = [_cron_field(f, *spec) for f, spec in zip(fields, _CRON_FIELDS, strict=True)]
    weekdays = {0 if d == 7 else d for d in sets[4]}
    return Cron(
        minutes=frozenset(sets[0]),
        hours=frozenset(sets[1]),
        days=frozenset(sets[2]),
        months=frozenset(sets[3]),
        weekdays=frozenset(weekdays),
        days_restricted=not fields[2].startswith("*"),
        weekdays_restricted=not fields[4].startswith("*"),
    )


def parse_every(expr: str) -> int:
    match = _EVERY.fullmatch(expr.strip())
    if not match:
        raise InvalidInputError("every is a number and a unit: 30m, 2h or 1d")
    minutes = int(match.group(1)) * _UNIT_MINUTES[match.group(2)]
    if not MIN_EVERY_MINUTES <= minutes <= MAX_EVERY_MINUTES:
        raise InvalidInputError("every is from 1 minute to 30 days")
    return minutes


def parse_daily(expr: str) -> Cron:
    match = _DAILY.fullmatch(expr.strip())
    if not match:
        raise InvalidInputError("daily is a time HH:MM, from 00:00 to 23:59")
    return parse_cron(f"{int(match.group(2))} {int(match.group(1))} * * *")


def _next_cron(cron: Cron, after: datetime, zone: zoneinfo.ZoneInfo) -> datetime | None:
    local = after.astimezone(zone)
    for offset in range(SEARCH_DAYS):
        day = local.date() + timedelta(days=offset)
        if not cron.day_matches(day):
            continue
        for hour in sorted(cron.hours):
            for minute in sorted(cron.minutes):
                candidate = datetime.combine(day, time(hour, minute), tzinfo=zone)
                if candidate.astimezone(UTC) > after:
                    return candidate.astimezone(UTC)
    return None


def check_schedule(kind, expr) -> tuple[str, str]:
    """The schedule, cleaned, or InvalidInputError with the reason."""
    if kind not in KINDS:
        raise InvalidInputError(f"kind is one of: {', '.join(KINDS)}")
    if not isinstance(expr, str) or not expr.strip() or len(expr) > MAX_EXPR:
        raise InvalidInputError(f"expr is a text of 1 to {MAX_EXPR} characters")
    expr = " ".join(expr.split())
    {"cron": parse_cron, "every": parse_every, "daily": parse_daily}[kind](expr)
    return kind, expr


# How many runs are looked at to find the shortest gap of a schedule: enough to see a gap
# that occurs once a day in a schedule that runs at most every 15 minutes (96 runs a day).
INTERVAL_SAMPLE_RUNS = 200


def shortest_gap_minutes(kind: str, expr: str) -> int:
    """The shortest time between two consecutive runs of the schedule, in minutes, over its next
    INTERVAL_SAMPLE_RUNS runs from a fixed day (UTC)."""
    if kind == "every":
        return parse_every(expr)
    run = next_time(kind, expr, "UTC", datetime(2026, 1, 5, tzinfo=UTC))
    shortest = None
    for _ in range(INTERVAL_SAMPLE_RUNS - 1):
        following = next_time(kind, expr, "UTC", run)
        gap = int((following - run).total_seconds() // 60)
        shortest = gap if shortest is None else min(shortest, gap)
        run = following
    return shortest


def check_self_service_interval(kind: str, expr: str) -> None:
    """A task a user schedules themselves runs at most every TASK_MIN_INTERVAL_MINUTES."""
    from app.config import get_settings

    floor = get_settings().task_min_interval_minutes
    if floor <= 0:
        return
    gap = shortest_gap_minutes(kind, expr)
    if gap < floor:
        raise InvalidInputError(
            f"a task you schedule yourself runs at most every {floor} minutes "
            f"(this schedule runs every {gap}); an administrator can set a shorter one"
        )


def next_time(kind: str, expr: str, timezone: str | None, after: datetime) -> datetime:
    """The first time strictly after `after` (UTC) the schedule falls due, in UTC."""
    after = _as_utc(after).replace(second=0, microsecond=0) if kind != "every" else after
    if kind == "every":
        return after + timedelta(minutes=parse_every(expr))
    cron = parse_cron(expr) if kind == "cron" else parse_daily(expr)
    found = _next_cron(cron, after, _zone(timezone))
    if found is None:
        raise InvalidInputError("this cron schedule never falls due (no such date)")
    return found


# --- the global switch ---


async def get_config(session: AsyncSession) -> TaskConfig:
    row = await session.get(TaskConfig, 1)
    if row is None:
        row = TaskConfig(id=1, paused=False)
        session.add(row)
        await session.flush()
    return row


async def set_paused(session: AsyncSession, paused, *, actor: str) -> TaskConfig:
    if not isinstance(paused, bool):
        raise InvalidInputError("paused is a boolean")
    row = await get_config(session)
    row.paused = paused
    await session.flush()
    await record_admin_event(
        session, actor=actor, action="tasks.pause", target_type="task_config", target_id=1,
        details={"paused": paused},
    )  # fmt: skip
    return row


# --- the service ---


async def list_tasks(session: AsyncSession, user_id: int | None = None) -> list[ScheduledTask]:
    stmt = select(ScheduledTask).order_by(ScheduledTask.id)
    if user_id is not None:
        stmt = stmt.where(ScheduledTask.user_id == user_id)
    return list((await session.execute(stmt)).scalars().all())


async def get_task(
    session: AsyncSession, task_id: int, user_id: int | None = None
) -> ScheduledTask:
    task = await session.get(ScheduledTask, task_id)
    if task is None or (user_id is not None and task.user_id != user_id):
        raise TaskNotFoundError(f"No task {task_id}")
    return task


def _clean_prompt(prompt) -> str:
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT:
        raise InvalidInputError(f"prompt is a text of 1 to {MAX_PROMPT} characters")
    return prompt.strip()


async def _identity_for(
    session: AsyncSession, user_id: int, identity_id: int | None
) -> ChannelIdentity:
    if identity_id is None:
        identities = (
            (
                await session.execute(
                    select(ChannelIdentity).where(
                        ChannelIdentity.user_id == user_id,
                        ChannelIdentity.channel.in_(DELIVERY_CHANNELS),
                    )
                )
            )
            .scalars()
            .all()
        )
        if len(identities) != 1:
            raise InvalidInputError(
                f"channel_identity_id is required: user {user_id} has "
                f"{len(identities)} Telegram or email identities"
            )
        return identities[0]
    identity = await session.get(ChannelIdentity, identity_id)
    if identity is None or identity.user_id != user_id:
        raise IdentityNotFoundError(f"No channel identity {identity_id} for user {user_id}")
    if identity.channel not in DELIVERY_CHANNELS:
        raise InvalidInputError("a task is delivered on Telegram or by email")
    return identity


async def _agent_for(
    session: AsyncSession, user_id: int, agent_id: int | None, identity: ChannelIdentity
) -> Agent:
    if agent_id is None:
        return await resolve_agent(session, user_id, identity.active_agent_id)
    agent = await session.get(Agent, agent_id)
    if agent is None or agent.user_id != user_id:
        raise AgentNotFoundError(f"No agent {agent_id} for user {user_id}")
    return agent


async def create_task(
    session: AsyncSession,
    *,
    user_id: int,
    prompt,
    kind,
    expr,
    agent_id: int | None = None,
    channel_identity_id: int | None = None,
    enabled: bool = True,
    standing_tools: list | None = None,
    actor: str,
    now: datetime | None = None,
    self_service: bool = False,
) -> ScheduledTask:
    """`self_service`: the user schedules it themselves (a channel command), so the schedule
    must respect TASK_MIN_INTERVAL_MINUTES."""
    user = await _require_user(session, user_id)
    prompt = _clean_prompt(prompt)
    kind, expr = check_schedule(kind, expr)
    if self_service:
        check_self_service_interval(kind, expr)
    if not isinstance(enabled, bool):
        raise InvalidInputError("enabled is a boolean")
    identity = await _identity_for(session, user_id, channel_identity_id)
    agent = await _agent_for(session, user_id, agent_id, identity)
    standing = await standing_approval(session, agent, standing_tools or [])
    count = (
        await session.execute(
            select(func.count()).select_from(ScheduledTask).where(ScheduledTask.user_id == user_id)
        )
    ).scalar_one()
    if count >= MAX_TASKS_PER_USER:
        raise ConflictError(f"User {user_id} already has {MAX_TASKS_PER_USER} tasks")
    now = now or datetime.now(UTC)
    task = ScheduledTask(
        user_id=user_id,
        agent_id=agent.id,
        channel_identity_id=identity.id,
        prompt=prompt,
        kind=kind,
        expr=expr,
        enabled=enabled,
        next_run_at=next_time(kind, expr, user.timezone, now) if enabled else None,
        standing_tools=standing,
    )
    session.add(task)
    await session.flush()
    await record_admin_event(
        session, actor=actor, action="task.create", target_type="task", target_id=task.id,
        details={"user_id": user_id, "kind": kind, "expr": expr, "enabled": enabled,
                 "standing_tools": sorted(standing)},
    )  # fmt: skip
    return task


UPDATABLE = (
    "prompt", "kind", "expr", "enabled", "agent_id", "channel_identity_id", "standing_tools",
)  # fmt: skip
MAX_STANDING_TOOLS = 20


async def standing_approval(session: AsyncSession, agent: Agent, names) -> dict[str, str]:
    """The standing approval for these tool names: each must be one of the agent's
    MCP tools with an approved definition, and is bound to that definition's SHA-256, so a
    definition an administrator replaces later is not covered. Grants, pinning and policies
    still apply at call time: this only answers the confirmation nobody can give in a task."""
    from app.db.models import McpServer
    from app.mcp.catalogue import _split_name

    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise InvalidInputError("standing_tools is a list of tool names")
    if len(names) > MAX_STANDING_TOOLS:
        raise InvalidInputError(f"standing_tools holds at most {MAX_STANDING_TOOLS} tools")
    approval: dict[str, str] = {}
    for name in names:
        pair = _split_name(name)
        if name not in (agent.tools or []) or pair is None:
            raise InvalidInputError(f"{name!r} is not an MCP tool of agent {agent.name!r}")
        server = (
            await session.execute(select(McpServer).where(McpServer.name == pair[0]))
        ).scalar_one_or_none()
        entry = (server.approved_definitions or {}).get(pair[1]) if server else None
        if entry is None or not entry.get("sha256"):
            raise InvalidInputError(f"{name!r} has no approved definition to agree to")
        approval[name] = entry["sha256"]
    return approval


async def approved_hashes(session: AsyncSession, names) -> dict[str, str | None]:
    """The sha256 of the approved definition of each tool name now (None when none)."""
    from app.db.models import McpServer
    from app.mcp.catalogue import _split_name

    out: dict[str, str | None] = {}
    for name in names:
        pair = _split_name(name)
        server = (
            await session.execute(select(McpServer).where(McpServer.name == pair[0]))
        ).scalar_one_or_none() if pair else None  # fmt: skip
        entry = (server.approved_definitions or {}).get(pair[1]) if server else None
        out[name] = entry.get("sha256") if entry else None
    return out


def standing_state(task: ScheduledTask, approved: dict[str, str | None]) -> dict[str, list]:
    """Which standing approvals still cover the approved definition (`current`) and which
    were given for a definition since replaced (`stale`); `approved` maps a tool name to its
    approved sha256 now."""
    current, stale = [], []
    for name, digest in sorted((task.standing_tools or {}).items()):
        (current if approved.get(name) == digest else stale).append(name)
    return {"current": current, "stale": stale}


async def update_task(
    session: AsyncSession,
    task_id: int,
    fields: dict,
    *,
    actor: str,
    user_id: int | None = None,
    now: datetime | None = None,
) -> ScheduledTask:
    """Only the fields that are given change. A new schedule, or a task enabled again,
    gets its next time counted from now."""
    unknown = set(fields) - set(UPDATABLE)
    if unknown:
        raise InvalidInputError(f"Unknown task fields: {', '.join(sorted(unknown))}")
    task = await get_task(session, task_id, user_id)
    owner = await _require_user(session, task.user_id)
    if "prompt" in fields:
        task.prompt = _clean_prompt(fields["prompt"])
    reschedule = False
    if "kind" in fields or "expr" in fields:
        task.kind, task.expr = check_schedule(
            fields.get("kind", task.kind), fields.get("expr", task.expr)
        )
        reschedule = True
    if "channel_identity_id" in fields:
        identity = await _identity_for(session, task.user_id, fields["channel_identity_id"])
        task.channel_identity_id = identity.id
    if "agent_id" in fields:
        identity = await session.get(ChannelIdentity, task.channel_identity_id)
        agent = await _agent_for(session, task.user_id, fields["agent_id"], identity)
        if agent.id != task.agent_id and "standing_tools" not in fields:
            # A standing approval was given for one agent's tools, never carried to another.
            task.standing_tools = {}
        task.agent_id = agent.id
    if "standing_tools" in fields:
        agent = await session.get(Agent, task.agent_id)
        task.standing_tools = await standing_approval(session, agent, fields["standing_tools"])
    if "enabled" in fields:
        if not isinstance(fields["enabled"], bool):
            raise InvalidInputError("enabled is a boolean")
        reschedule = reschedule or (fields["enabled"] and not task.enabled)
        task.enabled = fields["enabled"]
    if not task.enabled:
        task.next_run_at = None
    elif reschedule or task.next_run_at is None:
        task.next_run_at = next_time(task.kind, task.expr, owner.timezone, now or datetime.now(UTC))
    await session.flush()
    await record_admin_event(
        session, actor=actor, action="task.update", target_type="task", target_id=task.id,
        details={"fields": sorted(fields), "enabled": task.enabled,
                 "standing_tools": sorted(task.standing_tools or {})},
    )  # fmt: skip
    return task


async def delete_task(
    session: AsyncSession, task_id: int, *, actor: str, user_id: int | None = None
) -> None:
    task = await get_task(session, task_id, user_id)
    owner = task.user_id
    from app import feed_memory

    await feed_memory.forget(session, [task_id])
    await session.delete(task)
    await session.flush()
    from app.graph import delete_threads

    await delete_threads([thread_id(task_id)])
    await record_admin_event(
        session, actor=actor, action="task.delete", target_type="task", target_id=task_id,
        details={"user_id": owner},
    )  # fmt: skip


async def reschedule_user(session: AsyncSession, user_id: int, now: datetime | None = None) -> int:
    """After a timezone change: every enabled task of the user counted again from now."""
    user = await _require_user(session, user_id)
    tasks = [t for t in await list_tasks(session, user_id) if t.enabled]
    for task in tasks:
        task.next_run_at = next_time(task.kind, task.expr, user.timezone, now or datetime.now(UTC))
    await session.flush()
    return len(tasks)


async def delete_tasks_of(
    session: AsyncSession, *, user_id: int | None = None, identity_id: int | None = None
) -> list[int]:
    """Remove the tasks of a user being deleted, or delivered on an identity being
    removed; returns their ids (the caller deletes their threads)."""
    stmt = select(ScheduledTask.id)
    if user_id is not None:
        stmt = stmt.where(ScheduledTask.user_id == user_id)
    if identity_id is not None:
        stmt = stmt.where(ScheduledTask.channel_identity_id == identity_id)
    ids = list((await session.execute(stmt)).scalars().all())
    if ids:
        from app import feed_memory

        await feed_memory.forget(session, ids)
        await session.execute(delete(ScheduledTask).where(ScheduledTask.id.in_(ids)))
    return ids


# --- running ---


async def deliver(identity: ChannelIdentity, text: str, subject: str) -> bool:
    """Send a task's reply on the identity's channel. False when it cannot go out now (no
    running Telegram adapter, no SMTP settings, no stored address) or the send failed."""
    try:
        if identity.channel == Channel.TELEGRAM:
            from app.channels.notify import get_sender

            sender = get_sender(Channel.TELEGRAM)
            if sender is None:
                return False
            await sender(identity.external_id, text)
            return True
        if identity.channel == Channel.EMAIL:
            from app.channels import email
            from app.config import get_settings

            if not get_settings().email_smtp_host or not identity.raw_address:
                return False
            await asyncio.to_thread(email._send_reply_sync, identity.raw_address, subject, text)
            return True
    except Exception:
        logger.exception("Delivering scheduled task output on %s failed", identity.channel.value)
    return False


def _allowed(user: User | None, agent: Agent | None, identity: ChannelIdentity | None) -> bool:
    if user is None or agent is None or identity is None:
        return False
    kinds = {p.kind for p in identity.permissions}
    chat = PermissionKind.CHAT in kinds or PermissionKind.ADMIN in kinds
    return user.is_active and agent.is_active and chat


async def execute(task_id: int, *, trigger: str = "schedule") -> dict:
    """Run one task now: the turn, the log rows, the delivery, the task's last-run fields.
    Never raises for a failed turn or delivery: the outcome is in the result and the task."""
    from sqlalchemy.orm import selectinload

    from app.channels.dispatch import NO_REPLY_NOTE
    from app.channels.limits import limiter
    from app.db.session import session_scope
    from app.graph import last_turn_stats, run_turn

    now = datetime.now(UTC)
    async with session_scope() as session:
        task = await session.get(ScheduledTask, task_id)
        if task is None:
            raise TaskNotFoundError(f"No task {task_id}")
        user = await session.get(User, task.user_id)
        agent = await session.get(Agent, task.agent_id)
        identity = (
            await session.execute(
                select(ChannelIdentity)
                .where(ChannelIdentity.id == task.channel_identity_id)
                .options(selectinload(ChannelIdentity.permissions))
            )
        ).scalar_one_or_none()
        channel, key = identity.channel, identity.external_id
        prompt, user_id, agent_id = task.prompt, task.user_id, task.agent_id
        identity_id = task.channel_identity_id
        standing = dict(task.standing_tools or {})
        if not _allowed(user, agent, identity):
            task.last_run_at, task.last_status, task.last_error = now, "refused", "not allowed"
            await session.commit()
            logger.info("Scheduled task %s not run: user, agent or permission inactive", task_id)
            return {"task_id": task_id, "status": "refused", "delivered": False}
        await record_action(
            session, user_id=user_id, agent_id=agent_id, channel=channel,
            direction=Direction.INBOUND, text=f"[task {task_id}] {prompt}",
        )  # fmt: skip
        await session.commit()

    from app import feed_memory
    from app.mcp.confirm import current_standing, current_task_addresses, prompt_addresses

    reply, error = None, None
    from app.config import get_settings
    from app.graph import delete_threads, llm_timeout
    from app.replies import EmptyReplyError

    # Each run starts from an empty conversation: with the previous runs in its history,
    # the owner's model repeated the last digest from memory (with altered links) although the
    # feed gave 0 new items. What an agent must keep between runs goes in its memory.
    await delete_threads([thread_id(task_id)])

    approval = current_standing.set(standing)  # Nobody can confirm in a task's turn
    named = current_task_addresses.set(prompt_addresses(prompt))
    patience = llm_timeout.set(get_settings().llm_task_timeout_seconds)  # Nobody waits
    feed_state = feed_memory.start(task_id)  # Leave out the items already delivered
    try:
        async with limiter.turn(user_id):
            reply = await run_turn(channel, key, agent_id, prompt, thread_id=thread_id(task_id))
        if not (reply or "").strip():
            error = "EmptyReply"
    except EmptyReplyError:
        error = "EmptyReply"
    except Exception as exc:  # the class name only: never text from the conversation
        logger.exception("Scheduled task %s failed", task_id)
        error = type(exc).__name__
    finally:
        current_standing.reset(approval)
        current_task_addresses.reset(named)
        llm_timeout.reset(patience)
        shown = feed_memory.current_task.get()["shown"]
        feed_memory.current_task.reset(feed_state)
    delivered = False
    if error is None:
        async with session_scope() as session:
            identity = await session.get(ChannelIdentity, identity_id)
            delivered = await deliver(
                identity, f"Scheduled task #{task_id}:\n{reply}", f"[task {task_id}] result"
            )
    status = "failed" if error else ("ok" if delivered else "undelivered")
    if delivered:
        await feed_memory.remember(task_id, shown)
    async with session_scope() as session:
        await record_action(
            session, user_id=user_id, agent_id=agent_id, channel=channel,
            direction=Direction.OUTBOUND, text=reply if error is None else NO_REPLY_NOTE,
            status=ActionStatus.OK if status == "ok" else ActionStatus.FAILED,
            stats=last_turn_stats.get() if error is None else None,
        )  # fmt: skip
        task = await session.get(ScheduledTask, task_id)
        if task is not None:
            task.last_run_at, task.last_status = now, status
            task.last_error = error if error else (None if delivered else "not delivered")
            task.run_count += 1
        await session.commit()
    logger.info("Scheduled task %s (%s): %s", task_id, trigger, status)
    return {
        "task_id": task_id,
        "status": status,
        "delivered": delivered,
        "reply_chars": len(reply or ""),
    }


async def run_due(now: datetime | None = None) -> list[int]:
    """Claim and run every task due at `now`; returns the ids run. Paused: each due task
    is moved to its next time and none runs."""
    from app.db.session import session_scope

    now = now or datetime.now(UTC)
    async with session_scope() as session:
        paused = (await get_config(session)).paused
        due = (
            (
                await session.execute(
                    select(ScheduledTask).where(
                        ScheduledTask.enabled.is_(True), ScheduledTask.next_run_at <= now
                    )
                )
            )
            .scalars()
            .all()
        )
        claimed = []
        for task in due:
            owner = await session.get(User, task.user_id)
            try:
                task.next_run_at = next_time(task.kind, task.expr, owner.timezone, now)
            except InvalidInputError:
                task.enabled, task.next_run_at, task.last_error = False, None, "schedule ended"
                continue
            if paused:
                task.last_status = "skipped"
            else:
                claimed.append(task.id)
        await session.commit()
    for task_id in claimed:
        await execute(task_id)
    return claimed


async def next_due(session: AsyncSession) -> datetime | None:
    value = (
        await session.execute(
            select(func.min(ScheduledTask.next_run_at)).where(ScheduledTask.enabled.is_(True))
        )
    ).scalar_one()
    return _as_utc(value)


async def run_scheduler(poll_seconds: float = POLL_SECONDS) -> None:
    """The application component (app/main.py): runs the due tasks, and re-reads the
    tasks at least every `poll_seconds`, so a change through the API or `/task` applies
    without a restart. Runs until cancelled."""
    from app.db.session import session_scope

    while True:
        planned = None
        try:
            await run_due()
            async with session_scope() as session:
                planned = await next_due(session)
        except Exception:  # a locked database, a full disk: the next pass tries again
            logger.exception("Scheduled tasks: one pass failed, retrying in %ss", poll_seconds)
        now = datetime.now(UTC)
        wait = poll_seconds
        if planned is not None:
            wait = min(poll_seconds, (planned - now).total_seconds())
        await asyncio.sleep(max(wait, 0.5))
