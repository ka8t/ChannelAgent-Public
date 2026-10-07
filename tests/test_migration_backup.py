"""Tests: a migration never runs against the only copy of the database.

Found on 2026-09-20: starting the console applied two migrations to the real
database with no backup. Now a database that is behind head is copied first,
the copy is verified, and if it cannot be made the migration does not run.
"""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
OLD_REVISION = "417b3835e5c8"  # before the two 2026-09-20 migrations


def _db(tmp_path) -> Path:
    return tmp_path / "test.db"


def _alembic(*args: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO,
        env={**os.environ},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-500:]


def _revision(path: Path) -> str | None:
    con = sqlite3.connect(path)
    try:
        row = con.execute("select version_num from alembic_version").fetchone()
        return row[0] if row else None
    finally:
        con.close()


def _count(path: Path, table: str) -> int:
    con = sqlite3.connect(path)
    try:
        return con.execute(f"select count(*) from {table}").fetchone()[0]
    finally:
        con.close()


def _backups(tmp_path) -> list[Path]:
    return sorted((tmp_path / "backups").glob("*.db")) if (tmp_path / "backups").exists() else []


def _head() -> str:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(REPO / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO / "alembic"))
    return ScriptDirectory.from_config(cfg).get_current_head()


@pytest.fixture
def old_db(fresh_db, tmp_path):
    """A database one migration step behind head, with data in it."""
    path = _db(tmp_path)
    _alembic("upgrade", OLD_REVISION)
    con = sqlite3.connect(path)
    con.execute(
        "insert into users (display_name, is_active, created_at) values ('Alice', 1, '2026-01-01')"
    )
    con.execute(
        "insert into users (display_name, is_active, created_at) values ('Bob', 1, '2026-01-02')"
    )
    con.commit()
    con.close()
    assert _revision(path) == OLD_REVISION
    return path


async def test_a_database_behind_head_is_backed_up_before_it_is_migrated(old_db, tmp_path):
    from app.db.session import init_db

    await init_db()
    assert _revision(old_db) == _head(), "the migration did run"
    backups = _backups(tmp_path)
    assert len(backups) == 1
    assert _revision(backups[0]) == OLD_REVISION, "the copy is the state before the migration"
    assert _count(backups[0], "users") == 2 == _count(old_db, "users")
    con = sqlite3.connect(backups[0])
    assert con.execute("pragma integrity_check").fetchall() == [("ok",)]
    con.close()
    assert OLD_REVISION in backups[0].name and backups[0].name.startswith("test-")


async def test_the_backup_is_independent_of_the_live_database(old_db, tmp_path):
    from app.db.session import init_db

    await init_db()
    con = sqlite3.connect(old_db)
    con.execute("delete from users")
    con.commit()
    con.close()
    assert _count(_backups(tmp_path)[0], "users") == 2


async def test_a_database_already_at_head_gets_no_backup(fresh_db, tmp_path):
    from app.db.session import init_db

    await init_db()  # fresh file: nothing to protect
    assert _backups(tmp_path) == []
    await init_db()  # at head now
    await init_db()
    assert _backups(tmp_path) == []


async def test_a_database_that_is_behind_gets_exactly_one_backup_per_start(old_db, tmp_path):
    from app.db.session import init_db

    await init_db()
    await init_db()  # already migrated: no second backup
    assert len(_backups(tmp_path)) == 1


async def test_only_the_newest_backups_are_kept(old_db, tmp_path, monkeypatch):
    from app.config import get_settings
    from app.db.session import init_db

    monkeypatch.setenv("MIGRATION_BACKUPS_KEEP", "3")
    get_settings.cache_clear()
    names = []
    for _ in range(6):
        await init_db()  # migrates to head, backing up first
        names.append(sorted(p.name for p in _backups(tmp_path)))
        _alembic("downgrade", OLD_REVISION)  # behind again
    kept = _backups(tmp_path)
    assert len(kept) == 3
    assert [p.name for p in kept] == sorted(p.name for p in kept)
    assert kept[-1].name == max(p.name for p in kept), "the newest one is among the kept"
    assert len({p.name for p in kept}) == 3, "names are unique even within one second"


async def test_keeping_zero_disables_backups(old_db, tmp_path, monkeypatch):
    from app.config import get_settings
    from app.db.session import init_db

    monkeypatch.setenv("MIGRATION_BACKUPS_KEEP", "0")
    get_settings.cache_clear()
    await init_db()
    assert _backups(tmp_path) == []
    assert _revision(old_db) == _head()


async def test_if_the_backup_cannot_be_verified_the_migration_does_not_run(
    old_db, tmp_path, monkeypatch
):
    from app.db import backup
    from app.db.session import init_db

    def corrupt(target, expected):
        raise backup.BackupError("copy does not match")

    monkeypatch.setattr(backup, "_verify", corrupt)
    with pytest.raises(backup.BackupError):
        await init_db()
    assert _revision(old_db) == OLD_REVISION, "the database was left untouched"
    assert _count(old_db, "users") == 2
    assert _backups(tmp_path) == [], "the bad copy is removed"


async def test_a_database_with_tables_but_no_alembic_table_is_also_backed_up(fresh_db, tmp_path):
    """A file from before Alembic, or one Alembic never touched: it has data,
    so it is copied first, and the copy shows the state before.
    """
    from app.db.session import init_db

    path = _db(tmp_path)
    con = sqlite3.connect(path)
    con.execute("create table something (id integer)")
    con.execute("insert into something values (1), (2), (3)")
    con.commit()
    con.close()
    await init_db()
    (backup,) = _backups(tmp_path)
    assert "-none-" in backup.name
    assert _count(backup, "something") == 3
    con = sqlite3.connect(backup)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type = 'table'")}
    con.close()
    assert tables == {"something"}, "the copy is the state before any migration"
    assert _revision(path) == _head()


def test_no_backup_is_attempted_for_a_non_sqlite_url(tmp_path):
    from app.db.backup import backup_before_migration

    assert backup_before_migration("postgresql+asyncpg://u:p@h/db", "abc", 5) is None


def test_the_default_number_of_backups_kept_is_five(monkeypatch):
    from app.config import Settings

    monkeypatch.delenv("MIGRATION_BACKUPS_KEEP", raising=False)
    assert Settings(_env_file=None).migration_backups_keep == 5
