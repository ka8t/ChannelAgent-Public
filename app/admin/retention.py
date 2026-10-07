"""Retention of stored messages, tool-call logs and conversation histories.

Three periods in days, null = kept for ever; all null by default (nothing is
deleted until an administrator sets a number of days). The admin events are never deleted by
retention (owner): who did what stays.

- messages (`action_logs`, and the messages of senders who are not users yet,
  `request_messages`): rows older than `messages_days`;
- the Admin API call trail (`api_calls`): rows older than `api_calls_days`;
- tool calls (`mcp_calls`): rows older than `tool_calls_days`;
- conversations (the checkpoints file): the history of a user's agent whose last message is
  older than `conversations_days` (a conversation with no logged message is kept: its age is
  not known).

A dry run counts what would go, per table. A real run first backs up the database and the
checkpoints file (verified copies in `backups/`), then deletes exactly the rows older than the
same cutoffs, records one admin event with the counts and the backups, and keeps the counts as
the last run.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.service import InvalidInputError, _thread_ids_of_user, record_admin_event
from app.db.models import ActionLog, ApiCall, McpCall, RequestMessage, RetentionConfig

CONFIG_ID = 1
FIELDS = ("messages_days", "tool_calls_days", "conversations_days", "api_calls_days")
MAX_DAYS = 3650
BACKUP_LABEL = "retention"


async def get_config(session: AsyncSession) -> RetentionConfig:
    row = await session.get(RetentionConfig, CONFIG_ID)
    if row is None:
        row = RetentionConfig(id=CONFIG_ID)
        session.add(row)
        await session.flush()
    return row


async def set_config(session: AsyncSession, values: dict, *, actor: str) -> RetentionConfig:
    """Set the periods given in `values` (a missing key keeps its value, null turns it off)."""
    row = await get_config(session)
    changed = {}
    for name in FIELDS:
        if name not in values:
            continue
        days = values[name]
        if days is not None and (isinstance(days, bool) or not isinstance(days, int)
                                 or not 1 <= days <= MAX_DAYS):  # fmt: skip
            raise InvalidInputError(f"{name} is a number of days from 1 to {MAX_DAYS}, or null")
        setattr(row, name, days)
        changed[name] = days
    await record_admin_event(
        session, actor=actor, action="retention.set", target_type="retention", details=changed
    )
    return row


def _cutoff(now: datetime, days: int | None) -> datetime | None:
    return now - timedelta(days=days) if days else None


async def _stale_threads(session: AsyncSession, cutoff: datetime) -> list[str]:
    """The conversation threads of every (user, agent) whose last logged message is older than
    `cutoff`, and that have a stored history."""
    from app.graph import threads_with_history

    pairs = (
        await session.execute(
            select(ActionLog.user_id, ActionLog.agent_id, func.max(ActionLog.created_at)).group_by(
                ActionLog.user_id, ActionLog.agent_id
            )
        )
    ).all()
    stale: list[str] = []
    for user_id, agent_id, last in pairs:
        if last is None:
            continue
        last = last if last.tzinfo else last.replace(tzinfo=UTC)
        if last < cutoff:
            for thread_id in await _thread_ids_of_user(session, user_id, agent_id):
                if await threads_with_history([thread_id]):
                    stale.append(thread_id)
    return stale


async def plan(session: AsyncSession, now: datetime | None = None) -> dict:
    """What a run would delete now: the counts per table, and the cutoffs used."""
    row = await get_config(session)
    now = now or datetime.now(UTC)
    cutoffs = {name: _cutoff(now, getattr(row, name)) for name in FIELDS}
    counts = {"action_logs": 0, "request_messages": 0, "mcp_calls": 0, "api_calls": 0,
              "conversations": 0}  # fmt: skip
    if cutoffs["messages_days"]:
        counts["action_logs"] = (
            await session.execute(
                select(func.count()).where(ActionLog.created_at < cutoffs["messages_days"])
            )
        ).scalar_one()
        counts["request_messages"] = (
            await session.execute(
                select(func.count()).where(RequestMessage.created_at < cutoffs["messages_days"])
            )
        ).scalar_one()
    if cutoffs["api_calls_days"]:
        counts["api_calls"] = (
            await session.execute(
                select(func.count()).where(ApiCall.created_at < cutoffs["api_calls_days"])
            )
        ).scalar_one()
    if cutoffs["tool_calls_days"]:
        counts["mcp_calls"] = (
            await session.execute(
                select(func.count()).where(McpCall.created_at < cutoffs["tool_calls_days"])
            )
        ).scalar_one()
    threads: list[str] = []
    if cutoffs["conversations_days"]:
        threads = await _stale_threads(session, cutoffs["conversations_days"])
        counts["conversations"] = len(threads)
    return {
        "counts": counts,
        "cutoffs": {k: v.isoformat() if v else None for k, v in cutoffs.items()},
        "threads": threads,
    }


def _backup() -> list[str]:
    from app.admin.backup_schedule import _files
    from app.db.backup import make_backup

    return [make_backup(path, BACKUP_LABEL).name for path in _files()]


async def run(session: AsyncSession, *, dry_run: bool, actor: str) -> dict:
    """A dry run returns the counts; a real run backs up, deletes and records. The caller
    commits."""
    from app.graph import delete_threads

    now = datetime.now(UTC)
    found = await plan(session, now)
    result = {"dry_run": dry_run, "would_delete" if dry_run else "deleted": found["counts"],
              "cutoffs": found["cutoffs"]}  # fmt: skip
    if dry_run:
        return result
    backups = await asyncio.to_thread(_backup)
    cutoffs = {k: datetime.fromisoformat(v) if v else None for k, v in found["cutoffs"].items()}
    # The conversations first: their age comes from the messages about to be deleted.
    if found["threads"]:
        await delete_threads(found["threads"])
    if cutoffs["messages_days"]:
        await session.execute(
            delete(ActionLog).where(ActionLog.created_at < cutoffs["messages_days"])
        )
        await session.execute(
            delete(RequestMessage).where(RequestMessage.created_at < cutoffs["messages_days"])
        )
    if cutoffs["api_calls_days"]:
        await session.execute(
            delete(ApiCall).where(ApiCall.created_at < cutoffs["api_calls_days"])
        )
    if cutoffs["tool_calls_days"]:
        await session.execute(
            delete(McpCall).where(McpCall.created_at < cutoffs["tool_calls_days"])
        )
    result["backups"] = backups
    row = await get_config(session)
    row.last_run_at = now
    row.last_run = {"deleted": found["counts"], "backups": backups}
    await record_admin_event(
        session,
        actor=actor,
        action="retention.run",
        target_type="retention",
        details={"deleted": found["counts"], "backups": backups},
    )
    return result
