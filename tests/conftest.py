"""Shared pytest fixtures.

The ENCRYPTION_KEY below is a throwaway key used by the tests only. It has never
protected real data: the real key lives in .env and must never appear in a
tracked file (tests/test_no_committed_secrets.py fails if a Fernet-shaped key
appears anywhere else).

ENCRYPTION_KEY must be set before app.config is imported anywhere
(get_settings() is required, no default) — set at collection time,
before any test module imports app.* code.
"""

import os

os.environ.setdefault("ENCRYPTION_KEY", "PmKTledxEc-gdO4tty5QO4PjB48zp_GqWMVIpigdwEg=")
# The API answers only to its own names; the tests reach it as "t" and "testserver".
os.environ.setdefault("ALLOWED_HOSTS", "t,testserver,localhost,127.0.0.1")
# No test writes the repository's logs/ (the application's LOG_FILE default); a test of the log
# file sets its own path.
os.environ.setdefault("LOG_FILE", "")
# The engine cache on disk asks the engine for its model and slots: off unless a test of it
# turns it on (tests/test_slot_cache.py), so the mock engines of other tests need no slot API.
os.environ.setdefault("SLOT_CACHE_MAX_MB", "0")

import pytest


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Points DATABASE_URL at a fresh, empty SQLite file for one test,
    and clears the lru_cache'd Settings/engine/sessionmaker so the new
    value actually takes effect instead of reusing a previous test's.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")

    from app.config import get_settings
    from app.db.session import get_engine, get_sessionmaker

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()

    yield

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


@pytest.fixture(autouse=True)
async def isolated_checkpoints(tmp_path, monkeypatch):
    """Every test gets its own checkpoint file, so no test can write
    into the real data/ directory, and the checkpoint connection, which
    belongs to the test's event loop, is closed when the test ends.
    """
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.db"))
    from app import graph as graph_module
    from app.api import deps
    from app.config import get_settings
    from app.db import types

    get_settings.cache_clear()
    types.reset_warning_state()
    deps.reset_failure_state()
    graph_module._token_cache.clear()
    graph_module._tokenizer_down_until = 0.0
    from app.channels.limits import limiter

    limiter.clear()  # the per-user limits are process-wide: none carries over a test
    from app import slot_cache

    slot_cache.reset()  # the engine and slot owners it knows are process-wide too
    yield
    from app import graph

    await graph.close_graph()


@pytest.fixture(autouse=True)
def isolated_heartbeat(tmp_path, monkeypatch):
    """The healthcheck heartbeat file lives in the test's own directory,
    never in the machine's temp directory.
    """
    from app import health

    monkeypatch.setattr(health, "HEARTBEAT_PATH", tmp_path / "channelagent.heartbeat")
    health._components.clear()
