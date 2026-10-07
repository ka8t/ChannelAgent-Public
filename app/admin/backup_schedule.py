"""Scheduled backup of the database and the conversation checkpoints.

Each run copies the database, and the checkpoints file when there is one, with SQLite's
online backup API (`app.db.backup.make_backup`: consistent while the application writes,
integrity check and row counts verified), into `backups/` next to each file, labelled
"scheduled", then keeps the `keep` newest of that label. What the last run did is stored
(`BackupSchedule`), so the Admin API and `start.sh --status` show it across restarts. A
failure notifies the administrators once, when the failure streak starts, not at
every attempt.

The settings (enabled, interval, keep) live in the database and are changed through the
Admin API (`PUT /backups/schedule`), the one place the UI and `start.sh --admin` share.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.service import InvalidInputError, record_admin_event
from app.db.backup import SCHEDULED_LABEL, _prune, make_backup
from app.db.models import BackupSchedule

logger = logging.getLogger("channelagent")

MIN_INTERVAL, MAX_INTERVAL = 1, 10080  # minutes: one minute to one week
MIN_KEEP, MAX_KEEP = 1, 365
POLL_SECONDS = 60.0  # a changed schedule takes effect within this delay


def _as_utc(value: datetime | None) -> datetime | None:
    # SQLite returns naive datetimes even for timezone-aware columns.
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


async def get_schedule(session: AsyncSession) -> BackupSchedule:
    row = await session.get(BackupSchedule, 1)
    if row is None:
        row = BackupSchedule(id=1, enabled=True, interval_minutes=1440, keep=7, last_files=[])
        session.add(row)
        await session.flush()
    return row


def next_run_at(row: BackupSchedule) -> datetime | None:
    """When the next backup is planned: the last run plus the interval. None when the
    schedule is disabled, or has never run (the scheduler then runs one at its next check,
    within POLL_SECONDS). A stored time, not "now": two reads of the schedule agree."""
    last = _as_utc(row.last_run_at)
    if not row.enabled or last is None:
        return None
    return last + timedelta(minutes=row.interval_minutes)


def is_due(row: BackupSchedule, now: datetime) -> bool:
    if not row.enabled:
        return False
    planned = next_run_at(row)
    return planned is None or planned <= now


def _clean_int(name: str, value, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise InvalidInputError(f"{name} is a whole number from {low} to {high}")
    return value


async def set_schedule(session: AsyncSession, fields: dict, *, actor: str) -> BackupSchedule:
    unknown = set(fields) - {"enabled", "interval_minutes", "keep"}
    if unknown:
        raise InvalidInputError(f"Unknown schedule settings: {', '.join(sorted(unknown))}")
    row = await get_schedule(session)
    if "enabled" in fields:
        if not isinstance(fields["enabled"], bool):
            raise InvalidInputError("enabled is a boolean")
        row.enabled = fields["enabled"]
    if "interval_minutes" in fields:
        row.interval_minutes = _clean_int(
            "interval_minutes", fields["interval_minutes"], MIN_INTERVAL, MAX_INTERVAL
        )
    if "keep" in fields:
        row.keep = _clean_int("keep", fields["keep"], MIN_KEEP, MAX_KEEP)
    await session.flush()
    await record_admin_event(
        session,
        actor=actor,
        action="backup_schedule.set",
        target_type="backup_schedule",
        target_id=1,
        details={k: fields[k] for k in sorted(fields)},
    )
    return row


def _files() -> list[Path]:
    """The SQLite files a scheduled backup copies: the database, and the checkpoints
    file when it exists (a fresh installation has none until the first conversation)."""
    from app.checkpoints import checkpoint_db_path
    from app.config import get_settings
    from app.db.session import sqlite_file_path

    database = sqlite_file_path(get_settings().database_url)
    files = [database] if database is not None else []
    checkpoints = checkpoint_db_path()
    if checkpoints.exists() and checkpoints.stat().st_size > 0:
        files.append(checkpoints)
    return files


def _backup_all(keep: int) -> list[str]:
    """Sync work: copy and verify every file, then rotate. All or nothing: when one copy
    fails, the copies of this run are removed and the older backups stay untouched."""
    sources = _files()
    made: list[Path] = []
    try:
        for path in sources:
            made.append(make_backup(path, SCHEDULED_LABEL))
    except Exception:
        for copy in made:
            copy.unlink(missing_ok=True)
        raise
    for copy, source in zip(made, sources, strict=True):
        _prune(copy.parent, source.stem, keep, label=SCHEDULED_LABEL)
    return [copy.name for copy in made]


async def run_once(*, actor: str = "scheduler") -> BackupSchedule:
    """One scheduled backup now, whatever the interval; stores the outcome."""
    from app.channels.notify import notify_admins
    from app.db.session import session_scope

    async with session_scope() as session:
        keep = (await get_schedule(session)).keep
    now = datetime.now(UTC)
    try:
        names = await asyncio.to_thread(_backup_all, keep)
        error = None
    except Exception as exc:  # a full disk, a read-only directory, a failed check
        names, error = [], f"{type(exc).__name__}: {exc}"[:500]
        logger.error("Scheduled backup failed: %s", error)
    async with session_scope() as session:
        row = await get_schedule(session)
        row.last_run_at = now
        starts_streak = error is not None and not row.failing
        if error is None:
            row.last_success_at, row.last_error, row.last_files, row.failing = (
                now,
                None,
                names,
                False,
            )
            logger.info("Scheduled backup done: %s", ", ".join(names))
        else:
            row.last_error, row.failing = error, True
        await record_admin_event(
            session,
            actor=actor,
            action="backup.scheduled",
            target_type="backup",
            details={"ok": error is None, "files": names, "error": error},
        )
        await session.commit()
        if starts_streak:
            await notify_admins(
                session,
                "The scheduled backup failed: "
                f"{error}. Check ./start.sh --status and the backups directory.",
            )
        await session.refresh(row)
        return row


async def run_scheduler(poll_seconds: float = POLL_SECONDS) -> None:
    """The application component (app/main.py): runs a backup whenever one is due, and
    re-reads the schedule at least every `poll_seconds`, so a change through the API
    applies without a restart. Runs until cancelled."""
    from app.db.session import session_scope

    while True:
        now = datetime.now(UTC)
        async with session_scope() as session:
            row = await get_schedule(session)
            due, planned = is_due(row, now), next_run_at(row)
            await session.commit()
        if due:
            await run_once()
            continue
        wait = (
            poll_seconds if planned is None else min(poll_seconds, (planned - now).total_seconds())
        )
        await asyncio.sleep(max(wait, 0.05))
