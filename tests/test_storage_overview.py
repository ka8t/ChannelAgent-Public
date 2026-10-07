"""Tests: the storage overview.

The numbers are checked against independent sources, as the issue
requires: row counts against a direct sqlite3 query, the file size
against the filesystem, the log span against MIN/MAX in raw SQL. A test
that only checked "the endpoint returns 200" would prove nothing.
"""

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.db.models import Channel, Direction
from tests._console import command, run_console

TABLES = (
    "users",
    "channel_identities",
    "permissions",
    "access_requests",
    "agents",
    "action_logs",
    "admin_events",
    "request_messages",
    "api_calls",
    "routing_rules",
    "routing_config",
    "mcp_servers",
    "mcp_calls",
    "mcp_grants",
    "memory_entries",
    "backup_schedule",
    "retention_config",
    "skills",
    "skill_versions",
    "scheduled_tasks",
    "task_config",
    "task_feed_items",
    "feed_sources",
    "agent_templates",
    "agent_template_versions",
    "admin_accounts",
    "api_tokens",
)
OLDEST = datetime(2026, 9, 10, 8, 0, 0, tzinfo=UTC)
NEWEST = datetime(2026, 9, 20, 10, 0, 0, tzinfo=UTC)
LOG_TIMES = [
    OLDEST,
    datetime(2026, 9, 10, 8, 0, 5, tzinfo=UTC),
    datetime(2026, 9, 12, 9, 30, 0, tzinfo=UTC),
    datetime(2026, 9, 15, 12, 0, 0, tzinfo=UTC),
    datetime(2026, 9, 15, 12, 0, 3, tzinfo=UTC),
    datetime(2026, 9, 20, 7, 0, 0, tzinfo=UTC),
    NEWEST,
]


def _db_file() -> Path:
    from app.config import get_settings

    return Path(get_settings().database_url.split("///", 1)[1])


def _without_trail(counts: dict[str, int]) -> dict[str, int]:
    """The counts without `api_calls`: every API request adds a row to it, so two reads
    taken a request apart differ there by nature."""
    return {table: n for table, n in counts.items() if table != "api_calls"}


def _direct_counts() -> dict[str, int]:
    con = sqlite3.connect(_db_file())
    try:
        return {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in TABLES}
    finally:
        con.close()


@pytest.fixture
async def empty_db(fresh_db):
    from app.db.session import init_db

    await init_db()


@pytest.fixture
async def populated(empty_db):
    from app.admin.service import request_access
    from app.db.models import ActionLog, Agent, ChannelIdentity, PermissionKind, User
    from app.db.session import session_scope
    from app.security.auth import grant_permission

    async with session_scope() as session:
        alice, bob, carol = (User(display_name=n) for n in ("Alice", "Bob", "Carol"))
        session.add_all([alice, bob, carol])
        await session.flush()
        identities = [
            ChannelIdentity(user_id=alice.id, channel=Channel.TELEGRAM, external_id="1"),
            ChannelIdentity(user_id=alice.id, channel=Channel.EMAIL, external_id="hash-a"),
            ChannelIdentity(user_id=bob.id, channel=Channel.TELEGRAM, external_id="2"),
            ChannelIdentity(user_id=carol.id, channel=Channel.MATRIX, external_id="@c:x"),
        ]
        session.add_all(identities)
        await session.flush()
        await grant_permission(session, identities[0], PermissionKind.ADMIN)
        await grant_permission(session, identities[0], PermissionKind.CHAT)
        await grant_permission(session, identities[1], PermissionKind.CHAT)
        await grant_permission(session, identities[2], PermissionKind.CHAT)
        await grant_permission(session, identities[3], PermissionKind.CHAT)
        await request_access(session, Channel.TELEGRAM, "900", "hello")
        await request_access(session, Channel.EMAIL, "hash-z", "hi")
        agents = [
            Agent(user_id=alice.id, name="default"),
            Agent(user_id=alice.id, name="work"),
            Agent(user_id=bob.id, name="default"),
        ]
        session.add_all(agents)
        await session.flush()
        for i, moment in enumerate(LOG_TIMES):
            session.add(
                ActionLog(
                    user_id=alice.id,
                    agent_id=agents[0].id,
                    channel=Channel.TELEGRAM,
                    direction=Direction.INBOUND if i % 2 == 0 else Direction.OUTBOUND,
                    text=f"message {i}",
                    created_at=moment,
                )
            )
        await session.commit()


async def _overview():
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as session:
        return await service.storage_overview(session)


# --- service function ---


async def test_row_counts_match_a_direct_sql_query(populated):
    overview = await _overview()
    expected = {
        "users": 3,
        "channel_identities": 4,
        "permissions": 5,
        "access_requests": 2,
        "agents": 3,
        "action_logs": 7,
        "admin_events": 0,
        "request_messages": 0,
        "api_calls": 0,
        "routing_rules": 0,
        "routing_config": 0,
        "mcp_servers": 0,
        "mcp_calls": 0,
        "mcp_grants": 0,
        "memory_entries": 0,
        "backup_schedule": 0,
        "retention_config": 0,
        "skills": 0,
        "skill_versions": 0,
        "scheduled_tasks": 0,
        "task_config": 0,
        "task_feed_items": 0,
        "feed_sources": 0,
        "agent_templates": 0,
        "agent_template_versions": 0,
        "admin_accounts": 0,
        "api_tokens": 0,
    }
    assert overview.row_counts == expected, "known dataset"
    assert overview.row_counts == _direct_counts(), "independent direct query"


async def test_size_matches_the_filesystem(populated):
    overview = await _overview()
    assert overview.db_size_bytes == os.path.getsize(_db_file())
    assert overview.db_size_bytes > 0


async def test_log_span_matches_min_and_max_in_raw_sql(populated):
    overview = await _overview()
    assert overview.oldest_log_at == OLDEST
    assert overview.newest_log_at == NEWEST
    assert overview.oldest_log_at.tzinfo is not None
    con = sqlite3.connect(_db_file())
    low, high = con.execute("select min(created_at), max(created_at) from action_logs").fetchone()
    con.close()
    assert low.startswith("2026-09-10 08:00:00") and high.startswith("2026-09-20 10:00:00")


async def test_empty_database(empty_db):
    overview = await _overview()
    assert overview.row_counts == dict.fromkeys(TABLES, 0)
    assert overview.oldest_log_at is None and overview.newest_log_at is None
    assert overview.db_size_bytes == os.path.getsize(_db_file()) > 0


async def test_counts_follow_the_data(populated):
    from app.db.models import User
    from app.db.session import session_scope

    async with session_scope() as session:
        session.add_all([User(display_name="Dave"), User(display_name="Eve")])
        await session.commit()
    overview = await _overview()
    assert overview.row_counts["users"] == 5
    assert overview.row_counts == _direct_counts()


async def test_database_that_is_not_a_sqlite_file_has_no_size(populated, monkeypatch):
    from app.admin import service

    monkeypatch.setattr(
        service,
        "get_settings",
        lambda: SimpleNamespace(database_url="postgresql+asyncpg://u:p@h/db"),
    )
    overview = await _overview()
    assert overview.db_size_bytes is None
    assert overview.row_counts["action_logs"] == 7


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("sqlite+aiosqlite:///./data/channelagent.db", Path("./data/channelagent.db")),
        ("sqlite+aiosqlite:////abs/path/x.db", Path("/abs/path/x.db")),
        ("sqlite+aiosqlite:///:memory:", None),
        ("sqlite+aiosqlite://", None),
        ("postgresql+asyncpg://u:p@h/db", None),
    ],
)
def test_sqlite_file_path(url, expected):
    from app.db.session import sqlite_file_path

    assert sqlite_file_path(url) == expected


# --- Admin API: GET /storage ---


@pytest.fixture
async def api(populated, monkeypatch):
    from app.api.app import app
    from app.config import get_settings

    monkeypatch.setenv("API_SERVER_KEY", "test-key")
    get_settings.cache_clear()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


AUTH = {"Authorization": "Bearer test-key"}


async def test_api_requires_the_key(api):
    assert (await api.get("/storage")).status_code == 401
    assert (await api.get("/storage", headers={"Authorization": "Bearer wrong"})).status_code == 401


async def test_api_numbers_match_sql_and_filesystem(api):
    r = await api.get("/storage", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    # The API call trail gains a row for this very request after it answers.
    assert _without_trail(body["row_counts"]) == _without_trail(_direct_counts())
    assert body["db_size_bytes"] == os.path.getsize(_db_file())
    assert body["oldest_log_at"].startswith("2026-09-10T08:00:00")
    assert body["newest_log_at"].startswith("2026-09-20T10:00:00")
    assert set(body) == {
        "db_size_bytes",
        "row_counts",
        "oldest_log_at",
        "newest_log_at",
        "undecryptable_rows",
        "undecryptable_by_table",
        "checkpoint_size_bytes",
        "checkpoint_row_counts",
    }
    assert body["undecryptable_rows"] == 0 and body["undecryptable_by_table"] == {}


async def test_api_does_not_reveal_the_file_path(api):
    text = (await api.get("/storage", headers=AUTH)).text
    assert str(_db_file().parent) not in text and _db_file().name not in text


# --- Admin console ---


def _shown(out: str) -> dict:
    return json.loads(out[out.index("{") : out.rindex("}") + 1])


async def test_console_prints_the_same_numbers(populated, monkeypatch):
    shown = _shown(await run_console(monkeypatch, *command("storage")))
    assert shown["db_size_bytes"] == os.path.getsize(_db_file())
    assert _without_trail(shown["row_counts"]) == _without_trail(_direct_counts())
    assert shown["oldest_log_at"].startswith("2026-09-10T08:00:00")
    assert shown["newest_log_at"].startswith("2026-09-20T10:00:00")


async def test_console_on_an_empty_database(empty_db, monkeypatch):
    shown = _shown(await run_console(monkeypatch, *command("storage")))
    assert shown["oldest_log_at"] is None and shown["newest_log_at"] is None
    assert all(shown["row_counts"][t] == 0 for t in TABLES)


# --- one service function behind both front ends ---


async def test_api_and_console_call_the_same_service_function(api, monkeypatch):
    from app.admin import service

    calls = 0
    real = service.storage_overview

    async def spy(session):
        nonlocal calls
        calls += 1
        return await real(session)

    monkeypatch.setattr(service, "storage_overview", spy)
    assert (await api.get("/storage", headers=AUTH)).status_code == 200
    await run_console(monkeypatch, *command("storage"))
    assert calls == 2, "exactly one service call per front end (the console via the API)"
