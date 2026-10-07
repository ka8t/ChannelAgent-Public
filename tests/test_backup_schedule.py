"""Tests: the scheduled backup of the database and the checkpoints, its rotation,
its status, the notification when it fails, a restore from one of its copies, the Admin
API routes, and the migration rotation that no longer deletes manual or scheduled copies.
"""

import asyncio
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.admin import backup_schedule
from app.db.backup import BACKUP_DIR_NAME, _prune

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"


@pytest.fixture
async def world(fresh_db, tmp_path, monkeypatch):
    """A database with a user, and a checkpoints file with one table and 3 rows."""
    from app.admin import service
    from app.config import get_settings
    from app.db.session import init_db, session_scope, sqlite_file_path

    checkpoints = tmp_path / "checkpoints.db"
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(checkpoints))
    get_settings.cache_clear()
    await init_db()
    async with session_scope() as session:
        await service.create_user(session, "Sam")
        await session.commit()
    with sqlite3.connect(checkpoints) as con:
        con.execute("create table checkpoints (id integer primary key, blob text)")
        con.executemany("insert into checkpoints (blob) values (?)", [("a",), ("b",), ("c",)])
    database = sqlite_file_path(get_settings().database_url)
    return {"db": database, "cp": checkpoints, "dir": database.parent / BACKUP_DIR_NAME}


def _scheduled(directory: Path, stem: str) -> list[Path]:
    return sorted(directory.glob(f"{stem}-scheduled-*.db"))


async def _set(**fields):
    from app.db.session import session_scope

    async with session_scope() as session:
        row = await backup_schedule.set_schedule(session, fields, actor="test")
        await session.commit()
        return row


# --- a run, the rotation, the check ---


async def test_a_run_copies_the_database_and_the_checkpoints_and_both_check_ok(world):
    row = await backup_schedule.run_once()
    assert not row.failing and row.last_error is None
    assert len(row.last_files) == 2
    for copy in _scheduled(world["dir"], world["db"].stem) + _scheduled(
        world["dir"], "checkpoints"
    ):
        with sqlite3.connect(copy) as con:
            assert con.execute("pragma integrity_check").fetchall() == [("ok",)]
    with sqlite3.connect(_scheduled(world["dir"], "checkpoints")[0]) as con:
        assert con.execute("select count(*) from checkpoints").fetchone()[0] == 3
    assert oct(world["dir"].stat().st_mode & 0o777) == "0o700"


async def test_the_rotation_keeps_the_newest_of_each_file(world):
    await _set(keep=3)
    for _ in range(5):
        await backup_schedule.run_once()
    assert len(_scheduled(world["dir"], world["db"].stem)) == 3
    assert len(_scheduled(world["dir"], "checkpoints")) == 3


async def test_the_scheduler_runs_every_interval_and_prunes(world):
    """Interval 1 minute: the clock is moved by setting the last run back."""
    from app.db.models import BackupSchedule
    from app.db.session import session_scope

    await _set(interval_minutes=1, keep=2)
    task = asyncio.create_task(backup_schedule.run_scheduler(poll_seconds=0.05))
    try:
        for expected in (1, 2, 3, 4):
            for _ in range(200):
                if len(list(world["dir"].glob("*-scheduled-*.db"))) >= min(expected, 2) * 2:
                    async with session_scope() as session:
                        row = await session.get(BackupSchedule, 1)
                        if row.last_run_at is not None:
                            break
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.2)  # not due again yet: no extra run
            async with session_scope() as session:
                row = await session.get(BackupSchedule, 1)
                row.last_run_at = datetime.now(UTC) - timedelta(minutes=2)
                await session.commit()
        await asyncio.sleep(0.5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    from app.db.models import AdminEvent

    async with session_scope() as session:
        from sqlalchemy import func, select

        runs = (
            await session.execute(
                select(func.count())
                .select_from(AdminEvent)
                .where(AdminEvent.action == "backup.scheduled")
            )
        ).scalar_one()
    assert runs >= 4
    assert len(_scheduled(world["dir"], world["db"].stem)) == 2
    assert len(_scheduled(world["dir"], "checkpoints")) == 2


async def test_a_disabled_schedule_does_not_run(world):
    await _set(enabled=False)
    task = asyncio.create_task(backup_schedule.run_scheduler(poll_seconds=0.05))
    await asyncio.sleep(0.4)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not world["dir"].exists() or list(world["dir"].glob("*-scheduled-*")) == []


# --- a failure ---


async def test_a_failure_is_shown_and_notified_once_per_streak(world, monkeypatch):
    sent = []

    async def notify(session, text):
        sent.append(text)
        return 1

    monkeypatch.setattr("app.channels.notify.notify_admins", notify)
    first = await backup_schedule.run_once()
    before = sorted(p.name for p in world["dir"].iterdir())
    os.chmod(world["dir"], 0o500)  # read-only target
    try:
        failed = await backup_schedule.run_once()
        again = await backup_schedule.run_once()
    finally:
        os.chmod(world["dir"], 0o700)
    assert failed.failing and "PermissionError" in failed.last_error
    assert again.failing and len(sent) == 1, "one notification per failure streak"
    assert "scheduled backup failed" in sent[0]
    assert sorted(p.name for p in world["dir"].iterdir()) == before, "older copies untouched"
    assert failed.last_success_at == first.last_success_at
    ok = await backup_schedule.run_once()
    assert not ok.failing and ok.last_error is None
    await backup_schedule.run_once()
    os.chmod(world["dir"], 0o500)
    try:
        await backup_schedule.run_once()
    finally:
        os.chmod(world["dir"], 0o700)
    assert len(sent) == 2, "a new streak notifies again"


# --- restore ---


async def test_a_restore_from_a_scheduled_backup_keeps_every_row_count(world, monkeypatch):
    from app.admin import restore
    from app.admin.restore import row_counts, run_restore
    from app.db.session import get_engine

    # The refusal looks at running `channelagent` containers: the owner's live one must not
    # change this result (it did on 2026-09-28, as in test_restore.py on 2026-09-21).
    monkeypatch.setattr(restore.shutil, "which", lambda _name: None)

    await backup_schedule.run_once()
    copy = _scheduled(world["dir"], world["db"].stem)[0]
    expected = row_counts(copy)
    await get_engine().dispose()
    run_restore(
        world["db"],
        copy,
        encryption_key=os.environ["ENCRYPTION_KEY"],
        api_port=1,
        yes=True,
        out=lambda _line: None,
    )
    assert row_counts(world["db"]) == expected
    assert expected["users"] == 1


# --- the migration rotation leaves manual and scheduled copies alone ---


def test_the_migration_rotation_keeps_manual_and_scheduled_copies(tmp_path):
    names = [f"db-abc123-2026092{i}T000000000000Z.db" for i in range(6)]
    names += ["db-manual-20260901T000000000000Z.db", "db-scheduled-20260901T000000000000Z.db"]
    for name in names:
        (tmp_path / name).write_text("x")
    _prune(tmp_path, "db", keep=2)
    left = sorted(p.name for p in tmp_path.iterdir())
    assert "db-manual-20260901T000000000000Z.db" in left
    assert "db-scheduled-20260901T000000000000Z.db" in left
    assert len([n for n in left if "-abc123-" in n]) == 2


def test_the_scheduled_rotation_touches_only_scheduled_copies(tmp_path):
    names = [f"db-scheduled-2026092{i}T000000000000Z.db" for i in range(4)]
    names += ["db-manual-20260901T000000000000Z.db", "db-abc123-20260901T000000000000Z.db"]
    for name in names:
        (tmp_path / name).write_text("x")
    _prune(tmp_path, "db", keep=1, label="scheduled")
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == [
        "db-abc123-20260901T000000000000Z.db",
        "db-manual-20260901T000000000000Z.db",
        "db-scheduled-20260923T000000000000Z.db",
    ]


# --- the Admin API ---


@pytest.fixture
async def api(world, monkeypatch):
    import httpx

    from app.admin.jobs import registry
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    registry.clear()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    headers = {"Authorization": f"Bearer {KEY}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as c:
        c.app = app
        yield c
    app.dependency_overrides.clear()
    deps.reset_failure_state()
    registry.clear()


async def test_the_api_reads_changes_and_runs_the_schedule(api, world):
    first = (await api.get("/backups/schedule")).json()
    assert (first["enabled"], first["interval_minutes"], first["keep"]) == (True, 1440, 7)
    assert first["last_run_at"] is None and first["next_run_at"] is None, "never run: due now"
    changed = await api.put("/backups/schedule", json={"interval_minutes": 60, "keep": 3})
    assert changed.status_code == 200 and changed.json()["keep"] == 3
    for bad in ({"interval_minutes": 0}, {"keep": 0}, {"keep": 366}, {"other": 1}):
        assert (await api.put("/backups/schedule", json=bad)).status_code == 422, bad
    job = (await api.post("/backups/schedule/run")).json()
    for _ in range(100):
        state = (await api.get(f"/jobs/{job['id']}")).json()
        if state["status"] in ("done", "failed"):
            break
        await asyncio.sleep(0.05)
    assert state["status"] == "done", state
    after = (await api.get("/backups/schedule")).json()
    assert len(after["last_files"]) == 2 and not after["failing"]
    assert datetime.fromisoformat(after["next_run_at"]) - datetime.fromisoformat(
        after["last_run_at"]
    ) == timedelta(minutes=60)


async def test_the_schedule_routes_scopes(api, world):
    from app.api.scopes import Principal, Scope, get_principal

    api.app.dependency_overrides[get_principal] = lambda: Principal("r", Scope.READ)
    assert (await api.get("/backups/schedule")).status_code == 200
    assert (await api.put("/backups/schedule", json={"keep": 2})).status_code == 403
    assert (await api.post("/backups/schedule/run")).status_code == 403


async def test_when_the_second_copy_fails_the_first_one_is_removed(world):
    """All or nothing: a corrupt checkpoints file fails the run after the database copy."""
    world["cp"].write_bytes(b"this is not a SQLite database" * 100)
    row = await backup_schedule.run_once()
    assert row.failing
    assert _scheduled(world["dir"], world["db"].stem) == []


async def test_the_service_bounds_the_settings_and_audits_them(world):
    from sqlalchemy import select

    from app.admin.service import InvalidInputError
    from app.db.models import AdminEvent
    from app.db.session import session_scope

    for bad in ({"keep": 0}, {"keep": 366}, {"interval_minutes": 10081}, {"enabled": "yes"}):
        with pytest.raises(InvalidInputError):
            await _set(**bad)
    await _set(keep=4, interval_minutes=30)
    async with session_scope() as session:
        events = list(
            (
                await session.execute(
                    select(AdminEvent).where(AdminEvent.action == "backup_schedule.set")
                )
            ).scalars()
        )
    import json

    assert [json.loads(e.details) for e in events] == [{"interval_minutes": 30, "keep": 4}]


async def test_a_schedule_that_never_ran_backs_up_at_once(world):
    task = asyncio.create_task(backup_schedule.run_scheduler(poll_seconds=0.05))
    try:
        for _ in range(100):
            if _scheduled(world["dir"], world["db"].stem) if world["dir"].exists() else []:
                break
            await asyncio.sleep(0.02)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(_scheduled(world["dir"], world["db"].stem)) == 1
