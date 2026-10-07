"""Tests: how a message picks its agent. Each channel identity stores
the agent it talks to (default when none), chosen with /agent <name> or by an
admin, and every agent keeps its own conversation.
"""

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import httpx
import pytest

from app.admin import service
from app.channels import dispatch
from app.channels.dispatch import DENIED_MESSAGE, DispatchOutcome, handle_agent_command
from app.channels.schema import NormalizedEvent
from app.db.models import Channel, PermissionKind
from tests._console import command, run_console

KEY = "k" * 32
AUTH = {"Authorization": f"Bearer {KEY}"}


def _sql(query, *args):
    from app.config import get_settings

    con = sqlite3.connect(get_settings().database_url.split("///", 1)[1])
    try:
        return con.execute(query, args).fetchall()
    finally:
        con.close()


@pytest.fixture
async def world(fresh_db):
    """User 1 Alice, telegram identity 1 (chat): agents 1 default, 2 work,
    3 old (deactivated). User 2 Bob, telegram identity 2 (chat): agent 4
    default. User 3 Fay: telegram identity 3 with no permission.
    """
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as s:
        alice = await service.create_user(s, "Alice")
        bob = await service.create_user(s, "Bob")
        fay = await service.create_user(s, "Fay")
        for user, ext in ((alice, "1"), (bob, "2")):
            identity = await service.add_channel_identity(s, user.id, Channel.TELEGRAM, ext)
            await service.grant_identity_permission(s, user.id, identity.id, PermissionKind.CHAT)
        await service.add_channel_identity(s, fay.id, Channel.TELEGRAM, "3")
        await service.create_agent(s, alice.id, "default")
        await service.create_agent(s, alice.id, "work")
        old = await service.create_agent(s, alice.id, "old")
        await service.set_agent_active(s, old.id, False)
        await service.create_agent(s, bob.id, "default")
        await s.commit()


@pytest.fixture
def turns(monkeypatch):
    calls: list[tuple[int, str]] = []

    async def fake_turn(channel, user_id, agent_id, text, **_kwargs):
        calls.append((agent_id, text))
        return f"answer from agent {agent_id}"

    monkeypatch.setattr(dispatch, "run_turn", fake_turn)
    return calls


async def _send(text, user="1", channel=Channel.TELEGRAM):
    from app.db.session import session_scope

    replies: list[str] = []

    async def reply(t):
        replies.append(t)

    async with session_scope() as s:
        if text.startswith("/agent"):
            outcome = await handle_agent_command(
                s, NormalizedEvent(user, channel, text, reply), text[len("/agent") :]
            )
        else:
            outcome = await dispatch.dispatch_event(s, NormalizedEvent(user, channel, text, reply))
    return outcome, replies


# --- service ---


async def test_resolve_agent_defaults_and_ignores_a_foreign_or_missing_selection(world):
    from app.db.session import session_scope

    async with session_scope() as s:
        assert (await service.resolve_agent(s, 1, None)).id == 1
        assert (await service.resolve_agent(s, 1, 2)).id == 2
        assert (await service.resolve_agent(s, 1, 4)).id == 1, "Bob's agent is not Alice's"
        assert (await service.resolve_agent(s, 1, 999)).id == 1


async def test_set_identity_agent_checks_ownership_and_none_resets(world):
    from app.db.session import session_scope

    async with session_scope() as s:
        await service.set_identity_agent(s, 1, 1, 2)
        await s.commit()
    assert _sql("select active_agent_id from channel_identities where id = 1") == [(2,)]
    async with session_scope() as s:
        for bad in (4, 999):
            with pytest.raises(service.AgentNotFoundError):
                await service.set_identity_agent(s, 1, 1, bad)
        with pytest.raises(service.IdentityNotFoundError):
            await service.set_identity_agent(s, 1, 2, 1)  # identity 2 is Bob's
        with pytest.raises(service.UserNotFoundError):
            await service.set_identity_agent(s, 999, 1, 1)
        await service.set_identity_agent(s, 1, 1, None)
        await s.commit()
    assert _sql("select active_agent_id from channel_identities where id = 1") == [(None,)]


async def test_find_agent_by_name_is_case_insensitive_and_per_user(world):
    from app.db.session import session_scope

    async with session_scope() as s:
        assert (await service.find_agent_by_name(s, 1, "  WORK ")).id == 2
        assert await service.find_agent_by_name(s, 1, "nope") is None
        assert (await service.find_agent_by_name(s, 2, "default")).id == 4


async def test_a_user_who_selected_an_agent_can_still_be_deleted_and_purged(world):
    from app.db.session import session_scope

    async with session_scope() as s:
        await service.set_identity_agent(s, 1, 1, 2)
        await service.set_identity_agent(s, 2, 2, 4)
        await s.commit()
    async with session_scope() as s:
        report = await service.delete_user(s, 2)  # no history: agents go, the selection is cleared
        await s.commit()
    assert report.agents_deleted == 1 and _sql("select count(*) from users where id = 2") == [(0,)]
    async with session_scope() as s:
        await service.delete_user(s, 1, purge=True)
        await s.commit()
    assert _sql("select count(*) from agents where user_id in (1, 2)") == [(0,)]
    assert _sql("pragma foreign_key_check") == []


# --- which agent a message reaches ---


async def test_without_a_selection_the_default_agent_answers(world, turns):
    await _send("hello")
    assert turns == [(1, "hello")]


async def test_after_selecting_an_agent_messages_reach_it_and_the_log_says_so(world, turns):
    outcome, replies = await _send("/agent work")
    assert outcome is DispatchOutcome.OK and replies == ["You are now talking to agent 'work'."]
    await _send("plan my week")
    assert turns == [(2, "plan my week")]
    assert _sql("select agent_id, direction from action_logs order by id") == [
        (2, "inbound"),
        (2, "outbound"),
        (2, "inbound"),
        (2, "outbound"),
    ]


async def test_switching_back_and_forth(world, turns):
    await _send("/agent work")
    await _send("one")
    await _send("/agent DEFAULT")
    await _send("two")
    assert turns == [(2, "one"), (1, "two")]


async def test_a_selected_agent_that_gets_deactivated_refuses_without_an_llm_call(world, turns):
    from app.db.session import session_scope

    await _send("/agent work")
    async with session_scope() as s:
        await service.set_agent_active(s, 2, False)
        await s.commit()
    outcome, replies = await _send("still there?")
    assert outcome is DispatchOutcome.DENIED and replies == [dispatch.AGENT_DISABLED_MESSAGE]
    assert turns == [], "no LLM call"
    outcome, replies = await _send("/agent default")
    assert outcome is DispatchOutcome.OK
    await _send("back")
    assert turns == [(1, "back")]


async def test_the_selection_survives_a_new_session(world, turns):
    await _send("/agent work")
    assert _sql("select active_agent_id from channel_identities where id = 1") == [(2,)]
    await _send("later")
    assert turns[-1] == (2, "later")


async def test_two_users_choose_independently(world, turns):
    await _send("/agent work", user="1")
    await _send("hi", user="2")
    await _send("hi", user="1")
    assert turns == [(4, "hi"), (2, "hi")]


async def test_each_agent_keeps_its_own_conversation_with_the_real_graph(world):
    class Counting(BaseHTTPRequestHandler):
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
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer(("127.0.0.1", 0), Counting)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    import os

    os.environ["LLAMA_SERVER_URL"] = f"http://127.0.0.1:{server.server_port}"
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        answers = []
        for text in ("a", "b", "/agent work", "c", "/agent default", "d"):
            _, replies = await _send(text)
            answers.append(replies[-1])
    finally:
        server.shutdown()
        os.environ.pop("LLAMA_SERVER_URL", None)
        get_settings.cache_clear()
    # default: a (N=1), b (N=3) | work: c (N=1) | default again: d (N=5)
    assert [a for a in answers if a.startswith("N=")] == ["N=1", "N=3", "N=1", "N=5"]


# --- the /agent command ---


async def test_agent_lists_the_users_agents_marking_the_current_and_disabled_ones(world, turns):
    outcome, replies = await _send("/agent")
    assert replies == [
        "Your agents: default*, work, old (disabled). * is the one you are talking to. "
        "Use /agent <name> to switch."
    ]
    await _send("/agent work")
    _, replies = await _send("/agent")
    assert "default, work*, old (disabled)" in replies[0]


async def test_an_unknown_or_disabled_agent_is_refused_and_the_selection_is_kept(world, turns):
    _, replies = await _send("/agent nope")
    assert replies[0].startswith("No agent named 'nope'. Your agents: default*")
    _, replies = await _send("/agent old")
    assert replies[0].startswith("Agent 'old' is disabled.")
    assert _sql("select active_agent_id from channel_identities where id = 1") == [(None,)]


async def test_another_users_agent_name_is_not_reachable(world, turns):
    await _send("/agent work", user="2")  # Bob has no "work"
    assert _sql("select active_agent_id from channel_identities where id = 2") == [(None,)]


async def test_an_unauthorized_sender_gets_the_denial_and_a_request_not_a_selection(world, turns):
    outcome, replies = await _send("/agent work", user="777")
    assert outcome is DispatchOutcome.DENIED and replies == [DENIED_MESSAGE]
    assert _sql("select count(*) from access_requests") == [(1,)]
    outcome, replies = await _send("/agent work", user="3")  # known, no permission
    assert outcome is DispatchOutcome.DENIED and replies == [DENIED_MESSAGE]
    assert _sql("select active_agent_id from channel_identities where id = 3") == [(None,)]


async def test_the_command_is_recorded_in_the_audit_trail(world, turns):
    await _send("/agent work")
    assert _sql("select agent_id, direction, status from action_logs order by id") == [
        (2, "inbound", "ok"),
        (2, "outbound", "ok"),
    ]


# --- the Telegram handler ---


async def test_the_telegram_agent_handler_switches_and_answers_in_the_same_chat(world, turns):
    from app.channels import telegram

    sent = []

    class Bot:
        async def send_message(self, chat_id, text):
            sent.append((chat_id, text))

    update = SimpleNamespace(
        message=SimpleNamespace(text="/agent work"),
        effective_user=SimpleNamespace(id=1),
        effective_chat=SimpleNamespace(id=555),
    )
    await telegram._on_agent(update, SimpleNamespace(bot=Bot(), args=["work"]))
    assert sent == [(555, "You are now talking to agent 'work'.")]
    assert _sql("select active_agent_id from channel_identities where id = 1") == [(2,)]
    await telegram._on_agent(update, SimpleNamespace(bot=Bot(), args=[]))
    assert sent[-1][1].startswith("Your agents: default, work*")


async def test_the_telegram_agent_handler_ignores_updates_without_message_or_user(world, turns):
    from app.channels import telegram

    for update in (
        SimpleNamespace(message=None, effective_user=SimpleNamespace(id=1), effective_chat=None),
        SimpleNamespace(
            message=SimpleNamespace(text="/agent"), effective_user=None, effective_chat=None
        ),
    ):
        await telegram._on_agent(update, SimpleNamespace(bot=None, args=[]))
    assert _sql("select count(*) from action_logs") == [(0,)]


# --- Admin API and console ---


@pytest.fixture
async def api(world, monkeypatch):
    from app.api.app import app
    from app.config import get_settings

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_api_sets_and_resets_the_agent_of_an_identity(api):
    r = await api.put("/users/1/channels/1/agent", json={"agent_id": 2}, headers=AUTH)
    assert r.status_code == 200 and r.json()["active_agent_id"] == 2
    listed = (await api.get("/users/1/channels", headers=AUTH)).json()
    assert listed[0]["active_agent_id"] == 2
    r = await api.put("/users/1/channels/1/agent", json={"agent_id": None}, headers=AUTH)
    assert r.status_code == 200 and r.json()["active_agent_id"] is None


async def test_api_refuses_a_foreign_agent_an_unknown_identity_and_a_missing_key(api):
    assert (
        await api.put("/users/1/channels/1/agent", json={"agent_id": 4}, headers=AUTH)
    ).status_code == 404
    assert (
        await api.put("/users/1/channels/2/agent", json={"agent_id": 1}, headers=AUTH)
    ).status_code == 404
    assert (await api.put("/users/1/channels/1/agent", json={"agent_id": 1})).status_code == 401
    assert (await api.put("/users/1/channels/1/agent", json={}, headers=AUTH)).status_code == 422
    assert _sql("select active_agent_id from channel_identities where id = 1") == [(None,)]


async def test_console_sets_the_agent_and_back_to_the_default(world, monkeypatch):
    """Through the API; `null` sets the default agent again, as `agent_id: null`."""
    out = await run_console(
        monkeypatch, *command("set-identity-agent", user_id=1, channel_identity_id=1, agent_id=2)
    )
    assert '"active_agent_id": 2' in out
    assert _sql("select active_agent_id from channel_identities where id = 1") == [(2,)]
    out = await run_console(
        monkeypatch,
        *command("set-identity-agent", user_id=1, channel_identity_id=1, agent_id="null"),
    )
    assert _sql("select active_agent_id from channel_identities where id = 1") == [(None,)]
    out = await run_console(
        monkeypatch,
        *command("set-identity-agent", user_id=1, channel_identity_id=1, agent_id=999),
        *command("set-identity-agent", user_id=1, channel_identity_id=1, agent_id="abc"),
    )
    assert "Not done (HTTP 404)" in out and "No agent 999 for user 1" in out
    assert "Invalid input, nothing sent." in out
