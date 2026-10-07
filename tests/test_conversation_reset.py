"""Tests: an administrator can reset a conversation whose stored
history cannot be read, from the service, the API and the console, without
touching SQL. The reset is recorded and leaves other conversations
alone.
"""

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from app.db.models import Channel
from tests._console import command, run_console

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
AUTH = {"Authorization": f"Bearer {KEY}"}


class _CountingLLM(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        if self.path != "/v1/chat/completions":  # no tokenizer on this mock
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        data = json.dumps(
            {"choices": [{"message": {"content": f"N={len(body['messages'])}"}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def llm(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _CountingLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    server.shutdown()


def _sql(path_kind: str, query: str, *args):
    from app.config import get_settings

    settings = get_settings()
    if path_kind == "app":
        path = settings.database_url.split("///", 1)[1]
    else:
        path = settings.checkpoint_db_path
    con = sqlite3.connect(path)
    try:
        return con.execute(query, args).fetchall()
    finally:
        con.close()


def _threads():
    return sorted(
        r[0] for r in _sql("cp", "select distinct thread_id from checkpoints")
    )


def _events():
    return _sql("app", "select actor, action, target_type, target_id from admin_events order by id")


@pytest.fixture
async def world(fresh_db, llm, monkeypatch):
    """User 1 (telegram 42) with agents 1 "default" and 2 "second"; user 2
    (telegram 43) with agent 3. One turn on each of the three conversations,
    then a second turn on each, so every thread holds 4 messages.
    """
    from app.admin import service
    from app.config import get_settings
    from app.db.session import init_db, session_scope
    from app.graph import run_turn

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    await init_db()
    async with session_scope() as s:
        u1 = await service.create_user(s, "Alice")
        u2 = await service.create_user(s, "Bob")
        await service.add_channel_identity(s, u1.id, Channel.TELEGRAM, "42")
        await service.add_channel_identity(s, u2.id, Channel.TELEGRAM, "43")
        await service.create_agent(s, u1.id, "default")
        await service.create_agent(s, u1.id, "second")
        await service.create_agent(s, u2.id, "default")
        await s.commit()
    for user, agent in (("42", 1), ("42", 2), ("43", 3)):
        await run_turn(Channel.TELEGRAM, user, agent, "one")
        await run_turn(Channel.TELEGRAM, user, agent, "two")
    async with session_scope() as s:
        from sqlalchemy import text

        await s.execute(text("delete from admin_events"))
        await s.commit()
    assert _threads() == ["telegram_42_1", "telegram_42_2", "telegram_43_3"]


def _corrupt(thread_id: str):
    from app.config import get_settings

    con = sqlite3.connect(get_settings().checkpoint_db_path)
    con.execute(
        "update checkpoints set checkpoint = ? where thread_id = ?", (b"garbage", thread_id)
    )
    con.commit()
    con.close()


async def _next(user, agent):
    from app.graph import run_turn

    return (await run_turn(Channel.TELEGRAM, user, agent, "again")).split()[0]


# --- the service function ---


async def test_reset_one_agent_of_a_corrupted_thread_restarts_only_that_thread(world):
    from app.admin import service
    from app.db.session import session_scope

    _corrupt("telegram_42_1")
    with pytest.raises(ValueError):
        await _next("42", 1)

    async with session_scope() as s:
        count = await service.reset_conversation(s, 1, 1, actor="t")
        await s.commit()
    assert count == 1
    assert _threads() == ["telegram_42_2", "telegram_43_3"]  # direct SQL
    assert await _next("42", 1) == "N=1"  # a fresh conversation
    assert await _next("42", 2) == "N=5"  # untouched: 4 messages + 1
    assert await _next("43", 3) == "N=5"


async def test_reset_without_an_agent_resets_every_conversation_of_the_user(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        count = await service.reset_conversation(s, 1, actor="t")
        await s.commit()
    assert count == 2
    assert _threads() == ["telegram_43_3"]
    assert await _next("42", 1) == "N=1" and await _next("42", 2) == "N=1"
    assert await _next("43", 3) == "N=5"


async def test_reset_covers_every_channel_identity_of_the_user(world):
    from app.admin import service
    from app.db.session import session_scope
    from app.graph import run_turn

    async with session_scope() as s:
        await service.add_channel_identity(s, 1, Channel.MATRIX, "@alice:example.org")
        await s.commit()
    await run_turn(Channel.MATRIX, "@alice:example.org", 1, "hi")
    assert "matrix_@alice:example.org_1" in _threads() or any(
        t.startswith("matrix_") for t in _threads()
    )
    async with session_scope() as s:
        count = await service.reset_conversation(s, 1, 1, actor="t")
        await s.commit()
    assert count == 2  # telegram and matrix, agent 1
    assert not any(t.startswith("matrix_") for t in _threads())
    assert "telegram_42_1" not in _threads()


async def test_reset_refuses_an_unknown_user_or_an_agent_of_someone_else(world):
    from app.admin import service
    from app.db.session import session_scope

    before = _threads()
    async with session_scope() as s:
        with pytest.raises(service.UserNotFoundError):
            await service.reset_conversation(s, 999, actor="t")
        with pytest.raises(service.AgentNotFoundError):
            await service.reset_conversation(s, 1, 3, actor="t")  # agent 3 is Bob's
        with pytest.raises(service.AgentNotFoundError):
            await service.reset_conversation(s, 1, 999, actor="t")
        await s.commit()
    assert _threads() == before
    assert _events() == []


async def test_reset_writes_exactly_one_event_with_the_counts(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        await service.reset_conversation(s, 1, actor="t")
        await s.commit()
    assert _events() == [("t", "conversation.reset", "user", 1)]
    async with session_scope() as s:
        [event] = await service.search_admin_events(s, action="conversation.reset")
    assert json.loads(event.details) == {"agent_id": None, "threads": 2}


async def test_reset_counts_only_conversations_that_had_a_history(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        assert await service.reset_conversation(s, 1, 1, actor="t") == 1
        assert await service.reset_conversation(s, 1, 1, actor="t") == 0  # already empty
        assert await service.reset_conversation(s, 1, actor="t") == 1  # only agent 2 is left
        await s.commit()


async def test_reset_of_a_user_with_no_conversation_is_not_an_error(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        carol = await service.create_user(s, "Carol")
        await s.commit()
    async with session_scope() as s:
        assert await service.reset_conversation(s, carol.id, actor="t") == 0
        await s.commit()
    assert _threads() == ["telegram_42_1", "telegram_42_2", "telegram_43_3"]


# --- the API ---


@pytest.fixture
async def api(world):
    from app.api.app import app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_api_reset_needs_the_key(api):
    assert (await api.post("/users/1/conversations/reset")).status_code == 401
    assert "telegram_42_1" in _threads()


async def test_api_reset_one_agent_then_all(api):
    r = await api.post("/users/1/conversations/reset", params={"agent_id": 1}, headers=AUTH)
    assert (r.status_code, r.json()) == (200, {"threads_reset": 1})
    assert _threads() == ["telegram_42_2", "telegram_43_3"]
    r = await api.post("/users/1/conversations/reset", headers=AUTH)
    assert (r.status_code, r.json()) == (200, {"threads_reset": 1})  # agent 1 was already empty
    assert _threads() == ["telegram_43_3"]
    assert [e[:3] for e in _events()] == [("api", "conversation.reset", "user")] * 2


async def test_api_reset_errors(api):
    assert (await api.post("/users/999/conversations/reset", headers=AUTH)).status_code == 404
    r = await api.post("/users/1/conversations/reset", params={"agent_id": 3}, headers=AUTH)
    assert r.status_code == 404
    assert (
        await api.post("/users/1/conversations/reset", params={"agent_id": "x"}, headers=AUTH)
    ).status_code == 422
    assert _threads() == ["telegram_42_1", "telegram_42_2", "telegram_43_3"]


# --- the console ---


async def test_console_reset_one_agent_and_all(world, monkeypatch):
    from app.admin.client import client_label

    out = await run_console(monkeypatch, *command("reset-conversation", user_id=1, agent_id=1))
    assert _threads() == ["telegram_42_2", "telegram_43_3"] and "1" in out
    await run_console(monkeypatch, *command("reset-conversation", user_id=1))
    assert _threads() == ["telegram_43_3"]
    assert [e[:2] for e in _events()] == [(client_label(), "conversation.reset")] * 2


async def test_console_reset_with_bad_input_changes_nothing(world, monkeypatch):
    out = await run_console(
        monkeypatch,
        *command("reset-conversation", user_id="abc"),
        *command("reset-conversation", user_id=1, agent_id="xyz"),
        *command("reset-conversation", user_id=999),
        *command("reset-conversation", user_id=1, agent_id=3),
    )
    assert out.count("Invalid input, nothing sent.") == 2 and out.count("Not done (HTTP 404)") == 2
    assert _threads() == ["telegram_42_1", "telegram_42_2", "telegram_43_3"]
    assert _events() == []


async def test_api_and_console_call_the_same_service_function(world, api, monkeypatch):
    from app.admin import service
    from app.admin.client import client_label

    calls = []
    real = service.reset_conversation

    async def spy(session, user_id, agent_id=None, **kw):
        calls.append((user_id, agent_id, kw.get("actor")))
        return await real(session, user_id, agent_id, **kw)

    monkeypatch.setattr(service, "reset_conversation", spy)
    await api.post("/users/1/conversations/reset", params={"agent_id": 1}, headers=AUTH)
    await run_console(monkeypatch, *command("reset-conversation", user_id=2, agent_id=3))
    assert calls == [(1, 1, "api"), (2, 3, client_label())]


async def test_api_and_console_record_the_same_event(world, api, monkeypatch):
    from app.admin import service
    from app.admin.client import client_label
    from app.db.session import session_scope

    await api.post("/users/1/conversations/reset", params={"agent_id": 1}, headers=AUTH)
    await run_console(monkeypatch, *command("reset-conversation", user_id=2, agent_id=3))
    async with session_scope() as s:
        events = await service.search_admin_events(s, action="conversation.reset")
    by_actor = {e.actor: json.loads(e.details) for e in events}
    assert by_actor == {
        "api": {"agent_id": 1, "threads": 1},
        client_label(): {"agent_id": 3, "threads": 1},
    }


# --- a failed turn points at the remedy ---


async def test_a_turn_that_cannot_read_its_history_logs_where_to_reset_it(world, caplog):
    from sqlalchemy import select

    from app.channels.dispatch import APOLOGY_MESSAGE, DispatchOutcome, dispatch_event
    from app.channels.schema import NormalizedEvent
    from app.db.models import ChannelIdentity, PermissionKind
    from app.db.session import session_scope
    from app.security.auth import grant_permission

    async with session_scope() as s:
        ident = (
            await s.execute(select(ChannelIdentity).where(ChannelIdentity.external_id == "42"))
        ).scalar_one()
        await grant_permission(s, ident, PermissionKind.CHAT)
        await s.commit()
    _corrupt("telegram_42_1")
    sent: list[str] = []

    async def reply(text):
        sent.append(text)

    with caplog.at_level("ERROR", logger="channelagent"):
        async with session_scope() as s:
            outcome = await dispatch_event(s, NormalizedEvent("42", Channel.TELEGRAM, "hi", reply))
    assert outcome is DispatchOutcome.FAILED and sent == [APOLOGY_MESSAGE]
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "reset-conversation" in text and "conversations/reset" in text
    assert "user 1" in text and "agent 1" in text


async def test_an_ordinary_failure_does_not_suggest_a_reset(world, caplog, monkeypatch):
    from sqlalchemy import select

    from app.channels.dispatch import DispatchOutcome, dispatch_event
    from app.channels.schema import NormalizedEvent
    from app.config import get_settings
    from app.db.models import ChannelIdentity, PermissionKind
    from app.db.session import session_scope
    from app.security.auth import grant_permission

    async with session_scope() as s:
        ident = (
            await s.execute(select(ChannelIdentity).where(ChannelIdentity.external_id == "42"))
        ).scalar_one()
        await grant_permission(s, ident, PermissionKind.CHAT)
        await s.commit()
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:1")
    get_settings.cache_clear()

    async def reply(text):
        pass

    with caplog.at_level("ERROR", logger="channelagent"):
        async with session_scope() as s:
            outcome = await dispatch_event(s, NormalizedEvent("42", Channel.TELEGRAM, "hi", reply))
    assert outcome is DispatchOutcome.FAILED
    assert "reset-conversation" not in " ".join(r.getMessage() for r in caplog.records)
