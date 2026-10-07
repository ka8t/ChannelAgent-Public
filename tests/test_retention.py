"""Tests: retention of stored messages, tool-call logs and conversation histories.

- Off by default (every period null): nothing counted, nothing deleted.
- A dry run counts per table what a real run then deletes, exactly (before and after counts).
- A backup of the database exists before the real run, with the rows about to go; 1 admin event.
- A conversation idle for longer than its period loses its history; an active one keeps it.
- Admin events are never deleted; the owner scope is needed to set or run; bad periods 422.
"""

import asyncio
import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
import pytest

from app.admin import service
from app.db.models import ActionLog, Channel, Direction, McpCall

KEY = "Rq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
AUTH = {"Authorization": f"Bearer {KEY}"}
NOW = datetime.now(UTC)


def _db() -> Path:
    from app.config import get_settings

    return Path(get_settings().database_url.split("///", 1)[1])


def _sql(query, *args, path=None):
    con = sqlite3.connect(path or _db())
    try:
        return con.execute(query, args).fetchall()
    finally:
        con.close()


class _Engine(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        data = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
        self.send_response(200 if self.path == "/v1/chat/completions" else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data.encode())

    def do_GET(self):
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
async def api(fresh_db, monkeypatch):
    """User 1 (Telegram "41", agent 1): messages 400, 100 and 1 days old, a conversation.
    User 2 (Telegram "42", agent 2): a message 1 day old, a conversation. Two tool calls,
    300 and 5 days old."""
    from app import graph
    from app.admin.jobs import registry
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db, session_scope

    server = HTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    registry.clear()
    await init_db()
    async with session_scope() as s:
        for name, ext in (("Old", "41"), ("New", "42")):
            user = await service.create_user(s, name)
            await service.add_channel_identity(s, user.id, Channel.TELEGRAM, ext)
            await service.create_agent(s, user.id, "default")
        for user, days in ((1, 400), (1, 100), (1, 1), (2, 1)):
            s.add(ActionLog(user_id=user, agent_id=user, channel=Channel.TELEGRAM,
                            direction=Direction.INBOUND, text="t",
                            created_at=NOW - timedelta(days=days)))  # fmt: skip
        for days in (300, 5):
            s.add(McpCall(server_name="clock", tool_name="now", status="ok", duration_ms=3,
                          created_at=NOW - timedelta(days=days)))  # fmt: skip
        await s.commit()
    for ext, agent in (("41", 1), ("42", 2)):
        await graph.run_turn(Channel.TELEGRAM, ext, agent, "hello")
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=AUTH) as c:
        yield c
    server.shutdown()
    registry.clear()
    get_settings.cache_clear()


async def _job(api, dry_run: bool) -> dict:
    job = (await api.post("/retention/run", json={"dry_run": dry_run})).json()
    for _ in range(200):
        job = (await api.get(f"/jobs/{job['id']}")).json()
        if job["status"] in ("done", "failed"):
            return job
        await asyncio.sleep(0.02)
    raise AssertionError(job)


async def _threads() -> list[str]:
    from app import graph

    return [t for t in ("telegram_41_1", "telegram_42_2")
            if await graph.threads_with_history([graph.thread_id_from_key(
                Channel.TELEGRAM, t.split("_")[1], int(t.split("_")[2]))])]  # fmt: skip


async def test_off_by_default_nothing_is_counted_or_deleted(api):
    body = (await api.get("/retention")).json()
    assert [body[k] for k in ("messages_days", "tool_calls_days", "conversations_days")] == [
        None, None, None
    ]  # fmt: skip
    before = _sql("select count(*) from action_logs")[0][0]
    job = await _job(api, dry_run=False)
    assert job["result"]["deleted"] == {
        "action_logs": 0, "request_messages": 0, "mcp_calls": 0, "api_calls": 0,
        "conversations": 0,
    }  # fmt: skip
    assert _sql("select count(*) from action_logs")[0][0] == before == 4


async def test_the_dry_run_counts_what_the_run_then_deletes_exactly(api):
    await api.put(
        "/retention", json={"messages_days": 90, "tool_calls_days": 30, "conversations_days": 30}
    )
    old_logs = _sql("select count(*) from action_logs where created_at < ?",
                    (NOW - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S"))[0][0]  # fmt: skip
    dry = await _job(api, dry_run=True)
    assert dry["result"]["would_delete"] == {"action_logs": 2, "request_messages": 0,
                                             "mcp_calls": 1, "api_calls": 0, "conversations": 0}
    assert old_logs == 2 and _sql("select count(*) from action_logs")[0][0] == 4, "dry: kept"
    events_before = _sql("select count(*) from admin_events")[0][0]
    real = await _job(api, dry_run=False)
    assert real["result"]["deleted"] == dry["result"]["would_delete"]
    assert _sql("select count(*) from action_logs")[0][0] == 4 - 2
    cutoff = (NOW - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
    assert _sql("select count(*) from action_logs where created_at < ?", cutoff) == [(0,)], (
        "the old rows went, not the recent ones"
    )
    assert _sql("select count(*) from mcp_calls")[0][0] == 2 - 1
    events = _sql("select action from admin_events order by id")
    assert len(events) == events_before + 1 and events[-1] == ("retention.run",)


async def test_a_backup_holding_the_rows_exists_before_the_deletion(api):
    await api.put("/retention", json={"messages_days": 90})
    job = await _job(api, dry_run=False)
    backups = job["result"]["backups"]
    database = [b for b in backups if b.startswith(_db().stem + "-retention-")]
    assert len(database) == 1 and len(backups) == 2, "the database and the checkpoints file"
    assert any(b.startswith("checkpoints-retention-") for b in backups)
    copy = _db().parent / "backups" / database[0]
    assert _sql("select count(*) from action_logs", path=copy) == [(4,)], "taken before"
    assert _sql("select count(*) from action_logs") == [(2,)]


async def test_an_idle_conversation_loses_its_history_an_active_one_keeps_it(api):
    assert await _threads() == ["telegram_41_1", "telegram_42_2"]
    await api.put("/retention", json={"conversations_days": 30})
    await api.post("/users/1/agents", json={"name": "x"})  # an event, not a message
    # User 1's last message is 1 day old: move every one of its messages past 30 days.
    old = (NOW - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S.000000")
    _sql_write = sqlite3.connect(_db())
    _sql_write.execute("update action_logs set created_at = ? where user_id = 1", (old,))
    _sql_write.commit()
    _sql_write.close()
    dry = await _job(api, dry_run=True)
    assert dry["result"]["would_delete"]["conversations"] == 1
    await _job(api, dry_run=False)
    assert await _threads() == ["telegram_42_2"]
    assert _sql("select count(*) from action_logs where user_id = 1")[0][0] == 3, "logs kept"


async def test_admin_events_are_never_deleted(api):
    await api.put("/retention", json={"messages_days": 1, "tool_calls_days": 1})
    old = (NOW - timedelta(days=500)).strftime("%Y-%m-%d %H:%M:%S.000000")
    con = sqlite3.connect(_db())
    con.execute("update admin_events set created_at = ?", (old,))
    con.commit()
    con.close()
    before = _sql("select count(*) from admin_events")[0][0]
    await _job(api, dry_run=False)
    assert _sql("select count(*) from admin_events")[0][0] == before + 1


async def test_periods_are_checked_kept_when_left_out_and_turned_off_by_null(api):
    assert (await api.put("/retention", json={"messages_days": 0})).status_code == 422
    assert (await api.put("/retention", json={"messages_days": 3651})).status_code == 422
    assert (await api.put("/retention", json={"days": 3})).status_code == 422
    await api.put("/retention", json={"messages_days": 30, "tool_calls_days": 60})
    body = (await api.put("/retention", json={"tool_calls_days": None})).json()
    assert (body["messages_days"], body["tool_calls_days"]) == (30, None)


async def test_the_owner_scope_is_needed_to_set_or_run(api):
    from app.api.app import app
    from app.api.scopes import Principal, Scope, get_principal

    app.dependency_overrides[get_principal] = lambda: Principal("api", Scope.ADMIN)
    try:
        assert (await api.get("/retention")).status_code == 200
        assert (await api.put("/retention", json={"messages_days": 1})).status_code == 403
        assert (await api.post("/retention/run", json={"dry_run": False})).status_code == 403
    finally:
        app.dependency_overrides.clear()
