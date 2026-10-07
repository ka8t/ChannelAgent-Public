"""Tests: a reply that was generated but could not be delivered is
delivered again on the retry, without running the turn a second time.

Real dispatch, real graph and checkpoints; a counting mock LLM proves the
model is called once, and the checkpoint proves the history has one turn.
"""

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app import graph
from app.channels.dispatch import (
    APOLOGY_MESSAGE,
    NO_REPLY_NOTE,
    DispatchOutcome,
    dispatch_event,
)
from app.channels.schema import NormalizedEvent
from app.db.models import Channel

SENDER = "sender@example.com"


class _CountingLLM(BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        if self.path != "/v1/chat/completions":  # no tokenizer on this mock
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.rfile.read(int(self.headers["Content-Length"]))
        type(self).calls += 1
        data = json.dumps(
            {"choices": [{"message": {"content": f"answer #{type(self).calls}"}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def llm(monkeypatch):
    _CountingLLM.calls = 0
    server = HTTPServer(("127.0.0.1", 0), _CountingLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    from app.config import get_settings

    get_settings.cache_clear()
    yield _CountingLLM
    server.shutdown()
    get_settings.cache_clear()


@pytest.fixture
async def sender(fresh_db):
    from app.channels.dispatch import channel_identifier_key
    from app.db.models import ChannelIdentity, PermissionKind, User
    from app.db.session import init_db, session_scope
    from app.security.auth import grant_permission

    await init_db()
    async with session_scope() as s:
        user = User(display_name="Sender")
        s.add(user)
        await s.flush()
        ident = ChannelIdentity(
            user_id=user.id,
            channel=Channel.EMAIL,
            external_id=channel_identifier_key(Channel.EMAIL, SENDER),
        )
        s.add(ident)
        await s.flush()
        await grant_permission(s, ident, PermissionKind.CHAT)
        await s.commit()


class _Smtp:
    """A fake SMTP that fails a set number of times, then works."""

    def __init__(self, failures: int):
        self.failures = failures
        self.attempts = 0
        self.sent: list[str] = []

    async def reply(self, text: str) -> None:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise RuntimeError("SMTP down")
        self.sent.append(text)


async def _dispatch(smtp: _Smtp, text="question", retry=False):
    from app.db.session import session_scope

    event = NormalizedEvent(SENDER, Channel.EMAIL, text, smtp.reply)
    async with session_scope() as session:
        return await dispatch_event(session, event, apologize=False, retry=retry)


def _rows():
    from app.config import get_settings

    con = sqlite3.connect(get_settings().database_url.split("///", 1)[1])
    try:
        return con.execute("select direction, status from action_logs order by id").fetchall()
    finally:
        con.close()


async def _history_length() -> int:
    from app.db.models import ChannelIdentity  # noqa: F401 - keeps models registered
    from app.security.hashing import channel_identifier_key

    compiled = await graph.get_graph()
    key = channel_identifier_key(Channel.EMAIL, SENDER)
    thread = graph.thread_id_from_key(Channel.EMAIL, key, 1)
    state = await compiled.aget_state({"configurable": {"thread_id": thread}})
    return len(state.values["messages"])


async def test_smtp_failing_twice_then_working_runs_the_turn_once(sender, llm):
    smtp = _Smtp(failures=2)

    first = await _dispatch(smtp)
    second = await _dispatch(smtp, retry=True)
    third = await _dispatch(smtp, retry=True)

    assert [first, second, third] == [
        DispatchOutcome.UNDELIVERED,
        DispatchOutcome.UNDELIVERED,
        DispatchOutcome.OK,
    ]
    assert llm.calls == 1, "the model is called once"
    assert await _history_length() == 2, "one turn: one question, one answer"
    assert smtp.attempts == 3 and smtp.sent == ["answer #1"]
    assert _rows() == [("inbound", "ok"), ("outbound", "ok")], "one outbound entry, ending ok"


async def test_a_permanent_delivery_failure_never_calls_the_model_again(sender, llm):
    smtp = _Smtp(failures=99)
    await _dispatch(smtp)
    for _ in range(5):
        assert await _dispatch(smtp, retry=True) is DispatchOutcome.UNDELIVERED
    assert llm.calls == 1
    assert await _history_length() == 2
    assert _rows() == [("inbound", "ok"), ("outbound", "failed")]


async def test_the_kept_answer_survives_a_restart(sender, llm):
    """Nothing is kept in memory: closing the graph and the caches, as a new
    process would start, the retry still finds the answer in the audit trail.
    """
    smtp = _Smtp(failures=1)
    await _dispatch(smtp)
    await graph.close_graph()

    assert await _dispatch(smtp, retry=True) is DispatchOutcome.OK
    assert smtp.sent == ["answer #1"] and llm.calls == 1


async def test_a_failed_turn_is_still_rerun_on_retry(sender, monkeypatch, llm):
    """No answer was produced, so there is nothing to redeliver: the turn runs."""
    import os

    from app.config import get_settings

    working_url = os.environ["LLAMA_SERVER_URL"]
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:1")
    get_settings.cache_clear()
    smtp = _Smtp(failures=0)
    assert await _dispatch(smtp) is DispatchOutcome.FAILED
    assert _rows() == [("inbound", "ok"), ("outbound", "failed")]

    monkeypatch.setenv("LLAMA_SERVER_URL", working_url)
    get_settings.cache_clear()
    assert await _dispatch(smtp, retry=True) is DispatchOutcome.OK
    assert llm.calls == 1 and smtp.sent == ["answer #1"]
    assert _rows() == [("inbound", "ok"), ("outbound", "failed"), ("outbound", "ok")]
    assert await _history_length() == 2, "The user message is not stored twice"


async def test_a_retry_of_a_different_message_still_appends_it(sender, llm):
    """The guard only skips an identical last user message."""
    smtp = _Smtp(failures=0)
    assert await _dispatch(smtp, text="first") is DispatchOutcome.OK
    assert await _dispatch(smtp, text="second", retry=True) is DispatchOutcome.OK
    assert await _history_length() == 4


async def test_the_apology_and_the_no_reply_note_are_never_redelivered(sender, llm):
    from app.admin import service
    from app.db.models import ActionStatus, Direction
    from app.db.session import session_scope

    async with session_scope() as session:
        user_id = (await service.list_users(session))[0].id
        agent = await service.get_or_create_default_agent(session, user_id)
        agent_id = agent.id
        await service.record_action(
            session, user_id=user_id, agent_id=agent_id, channel=Channel.EMAIL,
            direction=Direction.INBOUND, text="question",
        )
        await service.record_action(
            session, user_id=user_id, agent_id=agent_id, channel=Channel.EMAIL,
            direction=Direction.OUTBOUND, text=NO_REPLY_NOTE, status=ActionStatus.FAILED,
        )
        await session.commit()
        found = await service.find_undelivered_answer(
            session, user_id, agent_id, Channel.EMAIL, "question",
            not_answers=(APOLOGY_MESSAGE, NO_REPLY_NOTE),
        )
    assert found is None


async def test_an_answer_that_was_delivered_is_not_sent_again(sender, llm):
    smtp = _Smtp(failures=0)
    assert await _dispatch(smtp) is DispatchOutcome.OK
    assert llm.calls == 1
    # A retry of a message that already succeeded runs a turn (nothing pending).
    assert await _dispatch(smtp, retry=True) is DispatchOutcome.OK
    assert llm.calls == 2


async def test_another_senders_pending_answer_is_not_used(sender, llm):
    """The lookup is scoped to the user, agent and channel of the message."""
    from app.admin import service
    from app.db.models import ChannelIdentity, PermissionKind, User
    from app.db.session import session_scope
    from app.security.auth import grant_permission
    from app.security.hashing import channel_identifier_key

    await _dispatch(_Smtp(failures=99))
    async with session_scope() as s:
        other = User(display_name="Other")
        s.add(other)
        await s.flush()
        ident = ChannelIdentity(
            user_id=other.id,
            channel=Channel.EMAIL,
            external_id=channel_identifier_key(Channel.EMAIL, "other@example.com"),
        )
        s.add(ident)
        await s.flush()
        await grant_permission(s, ident, PermissionKind.CHAT)
        await s.commit()
        agent = await service.get_or_create_default_agent(s, other.id)
        found = await service.find_undelivered_answer(
            s, other.id, agent.id, Channel.EMAIL, "question"
        )
    assert found is None


async def test_the_pending_answer_is_the_one_of_that_message_not_an_older_one(sender, llm):
    """An earlier message was answered and delivered; the later one is pending.
    The retry of the later one must find its own answer.
    """
    assert await _dispatch(_Smtp(failures=0), text="old question") is DispatchOutcome.OK
    smtp = _Smtp(failures=1)
    assert await _dispatch(smtp, text="new question") is DispatchOutcome.UNDELIVERED
    assert await _dispatch(smtp, text="new question", retry=True) is DispatchOutcome.OK
    assert llm.calls == 2, "one call per message, none for the retry"
    assert smtp.sent == ["answer #2"]
    assert _rows() == [("inbound", "ok"), ("outbound", "ok")] * 2
