"""Tests: one undecryptable row must not make log search, the console
or the API fail. It is shown with a stable marker, reported in the storage
overview, never matched by a keyword search and never written back.
"""

import logging
import os
import sqlite3

import httpx
import pytest
from cryptography.fernet import Fernet, InvalidToken

from app.admin import service
from app.db.models import Channel, Direction
from app.db.types import UNDECRYPTABLE_MARKER, UndecryptableText
from tests._console import command, run_console

KEY = "k" * 32
AUTH = {"Authorization": f"Bearer {KEY}"}


def _db() -> str:
    from app.config import get_settings

    return get_settings().database_url.split("///", 1)[1]


def _sql(query, *args):
    con = sqlite3.connect(_db())
    try:
        return con.execute(query, args).fetchall()
    finally:
        con.close()


def _exec(query, *args):
    con = sqlite3.connect(_db())
    con.execute(query, args)
    con.commit()
    con.close()


@pytest.fixture
async def world(fresh_db):
    """3 log rows (alpha secret / beta secret / gamma), 1 access request and 1
    email identity. Row 2 of the log, the request text and the email address
    are then corrupted with a value that is not a Fernet token.
    """
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as s:
        user = await service.create_user(s, "Alice")
        identity = await service.add_channel_identity(s, user.id, Channel.EMAIL, "a@example.com")
        agent = await service.create_agent(s, user.id, "default")
        for text in ("alpha secret", "beta secret", "gamma"):
            await service.record_action(
                s,
                user_id=user.id,
                agent_id=agent.id,
                channel=Channel.TELEGRAM,
                direction=Direction.INBOUND,
                text=text,
            )
        await service.request_access(s, Channel.TELEGRAM, "777", "let me in")
        await s.commit()
        identity_id = identity.id
    _exec("update action_logs set text = 'not-a-token' where id = 2")
    _exec("update access_requests set first_message_text = 'garbage' where id = 1")
    _exec("update channel_identities set raw_address = 'junk' where id = ?", identity_id)


async def _search(**kw):
    from app.db.session import session_scope

    async with session_scope() as s:
        return await service.search_action_logs(s, **kw)


async def test_search_returns_every_row_and_marks_the_unreadable_one(world):
    rows = await _search()
    assert [r.id for r in rows] == [3, 2, 1]
    assert [r.text for r in rows] == ["gamma", UNDECRYPTABLE_MARKER, "alpha secret"]
    assert isinstance(rows[1].text, UndecryptableText)
    assert not isinstance(rows[0].text, UndecryptableText)


async def test_a_keyword_never_matches_the_marker(world):
    assert [r.id for r in await _search(keyword="secret")] == [1], "row 2 is unreadable"
    assert await _search(keyword="undecryptable") == []
    assert await _search(keyword="<undecryptable>") == []


async def test_a_real_message_that_says_the_marker_is_still_searchable(world):
    from app.db.session import session_scope

    async with session_scope() as s:
        await service.record_action(
            s,
            user_id=1,
            agent_id=1,
            channel=Channel.TELEGRAM,
            direction=Direction.INBOUND,
            text="<undecryptable>",
        )
        await s.commit()
    assert [r.id for r in await _search(keyword="undecryptable")] == [4]


async def test_a_warning_is_logged_and_nothing_is_raised(world, caplog):
    caplog.set_level(logging.WARNING, logger="channelagent")
    await _search()
    assert any("Cannot decrypt a stored value" in r.message for r in caplog.records)
    assert not any("not-a-token" in r.message for r in caplog.records), "no ciphertext in logs"


async def test_the_marker_is_never_written_back_to_the_database(world):
    from sqlalchemy import select

    from app.db.models import ActionLog
    from app.db.session import session_scope

    async with session_scope() as s:
        row = (await s.execute(select(ActionLog).where(ActionLog.id == 2))).scalar_one()
        row.status = row.status.FAILED
        await s.commit()
    assert _sql("select text, status from action_logs where id = 2") == [("not-a-token", "failed")]


# --- API and console ---


@pytest.fixture
async def api(world, monkeypatch):
    from app.api.app import app
    from app.config import get_settings

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_api_logs_answer_200_with_the_marker(api):
    r = await api.get("/logs", headers=AUTH)
    assert r.status_code == 200
    assert [e["text"] for e in r.json()] == ["gamma", UNDECRYPTABLE_MARKER, "alpha secret"]
    r = await api.get("/logs", params={"keyword": "undecryptable"}, headers=AUTH)
    assert r.status_code == 200 and r.json() == []


async def test_api_requests_answer_200_and_the_request_can_still_be_approved(api):
    r = await api.get("/requests", headers=AUTH)
    assert r.status_code == 200 and r.json()[0]["first_message_text"] == UNDECRYPTABLE_MARKER
    assert (await api.post("/requests/1/approve", headers=AUTH)).status_code == 200


async def test_console_search_and_requests_print_the_marker(world, monkeypatch):
    logs = await run_console(monkeypatch, *command("search-logs"))
    requests = await run_console(monkeypatch, *command("list-requests"))
    assert UNDECRYPTABLE_MARKER in logs and UNDECRYPTABLE_MARKER in requests


# --- the count ---


def _independent_count() -> dict[str, int]:
    """Try every stored value with the test key, straight from SQLite."""
    fernet = Fernet(os.environ["ENCRYPTION_KEY"].encode())
    out = {}
    for table, column in (
        ("action_logs", "text"),
        ("access_requests", "first_message_text"),
        ("channel_identities", "raw_address"),
    ):
        bad = 0
        for (value,) in _sql(f"select {column} from {table} where {column} is not null"):
            try:
                fernet.decrypt(value.encode())
            except InvalidToken:
                bad += 1
        out[table] = bad
    return out


async def test_the_storage_overview_counts_unreadable_values_per_table(world):
    from app.db.session import session_scope

    async with session_scope() as s:
        overview = await service.storage_overview(s)
    assert _independent_count() == {
        "action_logs": 1,
        "access_requests": 1,
        "channel_identities": 1,
    }
    assert overview.undecryptable_by_table == _independent_count()
    assert overview.undecryptable_rows == 3


async def test_api_and_console_report_the_count(api, monkeypatch):

    body = (await api.get("/storage", headers=AUTH)).json()
    assert body["undecryptable_rows"] == 3
    assert body["undecryptable_by_table"] == {
        "action_logs": 1,
        "access_requests": 1,
        "channel_identities": 1,
    }
    out = await run_console(monkeypatch, *command("storage"))
    assert '"undecryptable_rows": 3' in out and '"action_logs": 1' in out


async def test_a_healthy_database_reports_zero(fresh_db):
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as s:
        await service.create_user(s, "Alice")
        overview = await service.storage_overview(s)
    assert overview.undecryptable_rows == 0 and overview.undecryptable_by_table == {}


async def test_with_the_wrong_key_every_value_is_reported_and_nothing_raises(
    world, monkeypatch, caplog
):
    from app.config import get_settings
    from app.db.session import session_scope
    from app.security import encryption

    # Re-write the three log rows properly with the good key, then change the key.
    from app.security.encryption import encrypt_value

    for i, text in enumerate(("a", "b", "c"), start=1):
        _exec("update action_logs set text = ? where id = ?", encrypt_value(text), i)
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    get_settings.cache_clear()
    encryption._fernet.cache_clear()
    try:
        rows = await _search()
        assert [r.text for r in rows] == [UNDECRYPTABLE_MARKER] * 3
        async with session_scope() as s:
            overview = await service.storage_overview(s)
        assert overview.undecryptable_by_table["action_logs"] == 3
    finally:
        encryption._fernet.cache_clear()


async def test_a_wrong_key_logs_one_warning_per_minute_not_one_per_value(
    world, caplog, monkeypatch
):
    from app.db import types

    caplog.set_level(logging.WARNING, logger="channelagent")
    clock = [1000.0]
    monkeypatch.setattr(types.time, "monotonic", lambda: clock[0])
    types.reset_warning_state()
    _exec("update action_logs set text = 'bad'")  # 3 unreadable log rows
    await _search()
    await _search()
    first = [r for r in caplog.records if "Cannot decrypt" in r.message]
    assert len(first) == 1, "6 unreadable values read, 1 warning"
    clock[0] += types.WARNING_INTERVAL_SECONDS + 1
    await _search()
    later = [r for r in caplog.records if "Cannot decrypt" in r.message]
    assert len(later) == 2 and "5 similar messages suppressed" in later[1].message
