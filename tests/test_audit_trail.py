"""Tests: one trail of who asked what and who did what.

- Every message of a sender who is not a user yet is kept (the first too), encrypted, capped.
- Every Admin API request is recorded, refused ones included, without body or query string.
- A failing trail never changes the API's answer.
- The timeline merges the five sources, filters, pages with a total, and gives message text
  only on request, that read itself audited.
- Retention and the backup check know the new tables.
"""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from app.admin import service
from app.db.models import Channel, Direction

KEY = "audit-test-key-" + "a" * 20
AUTH = {"Authorization": f"Bearer {KEY}"}
STRANGER = "70001"


def _db() -> Path:
    from app.config import get_settings

    return Path(get_settings().database_url.split("///", 1)[1])


def _sql(query, *args):
    con = sqlite3.connect(_db())
    try:
        return con.execute(query, args).fetchall()
    finally:
        con.close()


@pytest.fixture
async def world(fresh_db, monkeypatch):
    """Alice (Telegram 11, chat), Bob (Telegram 22, no permission: a known user)."""
    from app.api import deps
    from app.config import get_settings
    from app.db.models import PermissionKind
    from app.db.session import init_db, session_scope

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    async with session_scope() as s:
        alice = await service.create_user(s, "Alice")
        ident = await service.add_channel_identity(s, alice.id, Channel.TELEGRAM, "11")
        await service.grant_identity_permission(s, alice.id, ident.id, PermissionKind.CHAT)
        bob = await service.create_user(s, "Bob")
        await service.add_channel_identity(s, bob.id, Channel.TELEGRAM, "22")
        await s.commit()
    yield {"alice": alice.id, "bob": bob.id}
    get_settings.cache_clear()


@pytest.fixture
async def api(world):
    from app.api.app import app

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        yield client


async def _denied(external_id: str, text: str):
    from app.channels.dispatch import dispatch_event
    from app.channels.schema import NormalizedEvent
    from app.db.session import session_scope

    async def reply(_text):
        pass

    async with session_scope() as s:
        return await dispatch_event(s, NormalizedEvent(external_id, Channel.TELEGRAM, text, reply))


# --- messages of senders who are not users yet ---


async def test_every_message_of_an_unknown_sender_is_kept_encrypted(world):
    for text in ("hello", "are you there?", "please let me in"):
        await _denied(STRANGER, text)
    assert _sql("select count(*) from access_requests")[0][0] == 1
    rows = _sql("select text from request_messages order by id")
    assert len(rows) == 3, "the first message too"
    assert all("hello" not in r[0] and "please" not in r[0] for r in rows), "encrypted at rest"
    from app.db.models import RequestMessage
    from app.db.session import session_scope

    async with session_scope() as s:
        from sqlalchemy import select

        texts = [m.text for m in (await s.execute(select(RequestMessage))).scalars()]
    assert texts == ["hello", "are you there?", "please let me in"]


async def test_a_stranger_cannot_fill_the_database(world, monkeypatch):
    monkeypatch.setattr(service, "REQUEST_MESSAGES_KEEP", 3)
    for n in range(7):
        await _denied(STRANGER, f"message {n}")
    assert _sql("select count(*) from request_messages")[0][0] == 3


async def test_a_known_user_without_permission_is_logged_once_with_the_user(world):
    await _denied("22", "hi, it's Bob")
    assert _sql("select count(*) from request_messages")[0][0] == 0
    assert _sql("select count(*) from action_logs where user_id = ?", world["bob"])[0][0] == 1


# --- the Admin API call trail ---


async def test_every_api_request_is_recorded_refused_ones_included(api):
    assert (await api.get("/users", headers=AUTH)).status_code == 200
    assert (await api.get("/users", headers={**AUTH, "X-Client": "ui:web"})).status_code == 200
    assert (await api.get("/users")).status_code == 401
    assert (await api.get("/users", headers={**AUTH, "Host": "evil.example"})).status_code == 421
    rows = _sql("select actor, method, path, status from api_calls order by id")
    assert rows == [
        ("api", "GET", "/users", 200),
        ("ui:web", "GET", "/users", 200),
        ("unauthenticated", "GET", "/users", 401),
        ("unauthenticated", "GET", "/users", 421),
    ]
    assert all(s == "127.0.0.1" for (s,) in _sql("select source from api_calls"))


async def test_neither_the_query_string_nor_the_body_is_recorded(api):
    await api.get("/logs", params={"keyword": "my-private-search"}, headers=AUTH)
    await api.post("/users", json={"display_name": "Private Name"}, headers=AUTH)
    dump = "\n".join("|".join(map(str, r)) for r in _sql("select * from api_calls"))
    assert "my-private-search" not in dump and "Private Name" not in dump
    assert ("GET", "/logs") in _sql("select method, path from api_calls")


async def test_a_failing_trail_never_changes_the_answer(api, monkeypatch, caplog):
    from app.api import audit

    def broken(**_row):
        raise RuntimeError("database gone")

    # Only the trail's write fails: the authentication reads the same database.
    monkeypatch.setattr(audit, "ApiCall", broken)
    response = await api.get("/users", headers=AUTH)
    assert response.status_code == 200
    assert "trail could not be written" in caplog.text
    assert audit.UNAUTHENTICATED == "unauthenticated"


# --- the timeline ---


async def _scene(world):
    """A stranger writes twice, Bob (no permission) once, Alice chats (in and out), a tool
    call, an admin change."""
    from app.db.models import McpCall
    from app.db.session import session_scope

    await _denied(STRANGER, "hello")
    await _denied(STRANGER, "anyone?")
    await _denied("22", "Bob has no permission")  # another user's message: filtered out by user
    async with session_scope() as s:
        agent = await service.get_or_create_default_agent(s, world["alice"])
        for direction in (Direction.INBOUND, Direction.OUTBOUND):
            await service.record_action(
                s, user_id=world["alice"], agent_id=agent.id, channel=Channel.TELEGRAM,
                direction=direction, text=f"secret {direction.value}",
            )  # fmt: skip
        s.add(McpCall(agent_id=agent.id, server_name="time", tool_name="get_time", status="ok",
                      duration_ms=3, user_id=world["alice"], decision="allow"))  # fmt: skip
        await service.update_user(s, world["bob"], display_name="Robert", actor="cli:owner")
        await s.commit()


def _direct_total() -> int:
    tables = ("action_logs", "request_messages", "admin_events", "mcp_calls", "api_calls")
    return sum(_sql(f"select count(*) from {t}")[0][0] for t in tables)


async def test_the_timeline_lists_every_step_and_its_total_matches_sql(api, world):
    await _scene(world)
    before = _direct_total()
    response = await api.get("/audit/timeline", params={"limit": 200}, headers=AUTH)
    assert response.status_code == 200
    rows = response.json()
    assert int(response.headers["X-Total-Count"]) == before == len(rows)
    # The scene went through the services, not the API; this request is recorded after it answers.
    assert {r["kind"] for r in rows} == {"message", "unknown_sender", "admin", "tool"}
    ats = [r["at"] for r in rows]
    assert ats == sorted(ats, reverse=True), "newest first"
    assert all(r["text"] is None for r in rows), "no text unless asked"
    stranger = [r for r in rows if r["kind"] == "unknown_sender"]
    assert len(stranger) == 2 and stranger[0]["who"] == f"telegram/{STRANGER}"


async def test_filters_by_user_actor_kind_and_time(api, world):
    await _scene(world)
    by_user = (
        await api.get("/audit/timeline", params={"user_id": world["alice"]}, headers=AUTH)
    ).json()
    assert {r["kind"] for r in by_user} == {"message", "tool"}
    assert all(r["who"] == f"user {world['alice']}" for r in by_user)
    by_actor = (
        await api.get("/audit/timeline", params={"actor": "cli:owner"}, headers=AUTH)
    ).json()
    assert [r["what"] for r in by_actor] == ["user.update"]
    only_tools = (await api.get("/audit/timeline", params={"kinds": "tool"}, headers=AUTH)).json()
    assert [r["what"] for r in only_tools] == ["time.get_time"]
    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    later = await api.get("/audit/timeline", params={"since": future}, headers=AUTH)
    assert later.json() == [] and later.headers["X-Total-Count"] == "0"
    bad = await api.get("/audit/timeline", params={"kinds": "everything"}, headers=AUTH)
    assert bad.status_code == 422


async def test_paging_keeps_the_order_and_the_total(api, world):
    await _scene(world)
    kinds = ["message", "unknown_sender"]
    whole = (await api.get("/audit/timeline", params={"kinds": kinds}, headers=AUTH)).json()
    page = {"kinds": kinds, "limit": 2}
    first = await api.get("/audit/timeline", params=page, headers=AUTH)
    second = await api.get("/audit/timeline", params={**page, "offset": 2}, headers=AUTH)
    assert first.headers["X-Total-Count"] == second.headers["X-Total-Count"] == "5"
    assert first.json() + second.json() == whole[:4]


async def test_text_only_on_request_and_that_read_is_audited(api, world):
    await _scene(world)
    rows = (
        await api.get(
            "/audit/timeline", params={"kinds": "message", "with_text": "true"},
            headers={**AUTH, "X-Client": "ui:web"},
        )
    ).json()  # fmt: skip
    assert sorted(r["text"] for r in rows) == [
        "Bob has no permission", "secret inbound", "secret outbound",
    ]  # fmt: skip
    events = _sql("select actor from admin_events where action = 'audit.read_text'")
    assert events == [("ui:web",)]


async def test_the_timeline_needs_the_admin_scope(api):
    from app.api.app import app
    from app.api.scopes import Principal, Scope, get_principal

    app.dependency_overrides[get_principal] = lambda: Principal(actor="api", scope=Scope.OPERATE)
    try:
        assert (await api.get("/audit/timeline", headers=AUTH)).status_code == 403
    finally:
        app.dependency_overrides.clear()
    assert (await api.get("/audit/timeline")).status_code == 401


# --- retention and backups know the new tables ---


async def test_retention_deletes_old_trail_rows_and_old_stranger_messages(world):
    from app.admin import retention
    from app.db.models import ApiCall
    from app.db.session import session_scope

    await _denied(STRANGER, "old message")
    old = datetime.now(UTC) - timedelta(days=40)
    async with session_scope() as s:
        s.add(ApiCall(actor="api", source="127.0.0.1", method="GET", path="/users", status=200,
                      duration_ms=1, created_at=old))  # fmt: skip
        s.add(ApiCall(actor="api", source="127.0.0.1", method="GET", path="/users", status=200,
                      duration_ms=1))  # fmt: skip
        await s.commit()
    con = sqlite3.connect(_db())
    con.execute("update request_messages set created_at = ?", (old.isoformat(),))
    con.commit()
    con.close()
    async with session_scope() as s:
        await retention.set_config(s, {"messages_days": 30, "api_calls_days": 30}, actor="test")
        dry = await retention.run(s, dry_run=True, actor="test")
        await s.commit()
    assert dry["would_delete"]["api_calls"] == 1 and dry["would_delete"]["request_messages"] == 1
    async with session_scope() as s:
        await retention.run(s, dry_run=False, actor="test")
        await s.commit()
    assert _sql("select count(*) from api_calls")[0][0] == 1
    assert _sql("select count(*) from request_messages")[0][0] == 0


def test_a_backup_matches_the_source_as_it_was_copied_whatever_is_written_meanwhile(
    tmp_path, monkeypatch
):
    """Rows written while a backup runs (the API call trail; 2026-10-04: task_config)
    do not fail it: the copy is checked against the counts taken in its own read transaction."""
    from app.db import backup

    tables = ("api_calls", "action_logs", "task_config")
    source = tmp_path / "s.db"
    con = sqlite3.connect(source)
    con.execute("pragma journal_mode=wal")
    for table in tables:
        con.execute(f"create table {table} (id integer)")
        con.execute(f"insert into {table} values (1)")
    con.commit()
    con.close()
    def write():
        writer = sqlite3.connect(source)
        for table in tables:
            writer.execute(f"insert into {table} values (2)")
        writer.commit()
        writer.close()

    real_counts, real_verify = backup._row_counts, backup._verify

    def counts_then_write(con):  # a row lands between the count and the copy
        counted = real_counts(con)
        if con.execute("pragma database_list").fetchone()[2] == str(source):
            write()
        return counted

    def write_then_verify(target, expected):  # and another one before the check
        write()
        real_verify(target, expected)

    monkeypatch.setattr(backup, "_row_counts", counts_then_write)
    monkeypatch.setattr(backup, "_verify", write_then_verify)
    copy = backup.make_backup(source, "manual")
    con = sqlite3.connect(copy)
    counts = {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in tables}
    con.close()
    assert counts == {"api_calls": 1, "action_logs": 1, "task_config": 1}


def test_a_backup_with_a_missing_or_extra_row_fails(tmp_path):
    from app.db import backup

    copy = tmp_path / "c.db"
    con = sqlite3.connect(copy)
    con.execute("create table api_calls (id integer)")
    con.execute("create table action_logs (id integer)")
    con.executemany("insert into api_calls values (?)", [(i,) for i in range(3)])
    con.executemany("insert into action_logs values (?)", [(i,) for i in range(2)])
    con.commit()
    con.close()
    backup._verify(copy, {"api_calls": 3, "action_logs": 2})
    with pytest.raises(backup.BackupError, match="action_logs"):
        backup._verify(copy, {"api_calls": 3, "action_logs": 3})
    with pytest.raises(backup.BackupError, match="api_calls"):
        backup._verify(copy, {"api_calls": 4, "action_logs": 2})
    with pytest.raises(backup.BackupError, match="api_calls"):
        backup._verify(copy, {"api_calls": 2, "action_logs": 2})
    with pytest.raises(backup.BackupError, match="task_config"):
        backup._verify(copy, {"api_calls": 3, "action_logs": 2, "task_config": 0})


def test_a_backup_that_fails_the_integrity_check_is_refused(tmp_path):
    from app.db import backup

    copy = tmp_path / "c.db"
    con = sqlite3.connect(copy)
    con.execute("create table t (x integer)")
    con.execute("create index i on t (x)")
    con.executemany("insert into t values (?)", [(i,) for i in range(50)])
    con.commit()
    root = con.execute("select rootpage from sqlite_master where name = 'i'").fetchone()[0]
    page = con.execute("pragma page_size").fetchone()[0]
    con.close()
    with open(copy, "r+b") as f:  # wipe the index entries: the rows stay readable
        f.seek(root * page - 200)
        f.write(b"\x00" * 150)
    with pytest.raises(backup.BackupError, match="integrity check"):
        backup._verify(copy, {"t": 50})
