"""Async engine/session setup and startup table creation.

Single place the rest of the app (Auth Node, Admin API) gets a DB
session from, so nothing else constructs its own engine.
"""

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def sqlite_file_path(database_url: str) -> Path | None:
    """The file behind a SQLite URL, or None for an in-memory or non-SQLite
    database (there is no file whose size could be reported).
    """
    url = make_url(database_url)
    if url.drivername.startswith("sqlite") and url.database and url.database != ":memory:":
        return Path(url.database)
    return None


def _ensure_sqlite_dir_exists(database_url: str) -> None:
    """SQLite refuses to create a DB file inside a missing directory —
    ``OperationalError: unable to open database file``, not a clearer
    "no such directory". The Docker image happens to avoid this
    (Dockerfile's mkdir + docker-compose's volume both pre-create
    data/), which is exactly why running natively (start.sh --native)
    was the first thing to actually hit it.
    """
    path = sqlite_file_path(database_url)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(path.parent, os.W_OK | os.X_OK):
            # The image runs as uid 10001, not root: a data directory
            # mounted from the host must belong to that user.
            raise PermissionError(
                f"The data directory {path.parent} is not writable by uid {os.getuid()}. "
                "With Docker on Linux, give the mounted directory to the container user: "
                "chown -R 10001:10001 <host data directory>."
            )


def _enable_sqlite_foreign_keys(engine: AsyncEngine) -> None:
    """SQLite ignores foreign keys unless every connection asks for them:
    without this an agent could be created for a user that does not
    exist, and deleting a user left orphan rows behind. Applied to the
    application's engine only. Alembic's own engine (alembic/env.py) keeps
    the default, because SQLite's copy-and-move table rebuild in a batch
    migration is not meant to run with enforcement on.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _on_connect(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


@lru_cache
def get_engine() -> AsyncEngine:
    database_url = get_settings().database_url
    _ensure_sqlite_dir_exists(database_url)
    engine = create_async_engine(database_url)
    if make_url(database_url).drivername.startswith("sqlite"):
        _enable_sqlite_foreign_keys(engine)
    return engine


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


def _run_migrations_sync(database_url: str) -> None:
    # alembic's Config/command API is synchronous, and env.py's own
    # run_migrations_online() calls asyncio.run() internally — calling
    # it directly from inside app/main.py's already-running event loop
    # would raise "asyncio.run() cannot be called from a running event
    # loop". init_db() below runs this in a separate thread instead.
    from alembic.command import upgrade
    from alembic.config import Config

    cfg = Config(str(_REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_REPO_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    # Keep the application's logging configuration: alembic/env.py would
    # otherwise call fileConfig() and disable the app's loggers.
    cfg.attributes["configure_logger"] = False
    # Copy the database first if a migration is about to change it.
    from alembic.script import ScriptDirectory

    from app.db.backup import backup_before_migration

    head = ScriptDirectory.from_config(cfg).get_current_head()
    backup_before_migration(database_url, head, get_settings().migration_backups_keep)
    upgrade(cfg, "head")


async def init_db() -> None:
    """Applies all Alembic migrations up to head. Safe to call on
    every startup — Alembic no-ops if the DB is already current.
    Replaced calling Base.metadata.create_all directly: schema changes
    now go through real migrations (alembic/versions/), tracked by
    revision, instead of requiring the DB to be dropped and recreated.
    """
    database_url = get_settings().database_url
    _ensure_sqlite_dir_exists(database_url)
    await asyncio.to_thread(_run_migrations_sync, database_url)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        yield session
