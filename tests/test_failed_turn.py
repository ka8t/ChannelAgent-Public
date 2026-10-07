"""Tests: a failed turn must be visible to the user, recorded in
the audit trail, and must never raise out of dispatch_event.
"""

import sqlite3

import pytest

from app.channels.dispatch import (
    APOLOGY_MESSAGE,
    DENIED_MESSAGE,
    NO_REPLY_NOTE,
    DispatchOutcome,
    dispatch_event,
)
from app.channels.schema import NormalizedEvent
from app.db.models import Channel


def _rows():
    from app.config import get_settings

    db = get_settings().database_url.split("///", 1)[1]
    con = sqlite3.connect(db)
    try:
        return con.execute(
            "select direction, status from action_logs order by id"
        ).fetchall()
    finally:
        con.close()


@pytest.fixture
async def authorized(fresh_db):
    from app.db.models import ChannelIdentity, PermissionKind, User
    from app.db.session import init_db, session_scope
    from app.security.auth import grant_permission

    await init_db()
    async with session_scope() as s:
        u = User(display_name="Alice")
        s.add(u)
        await s.flush()
        ident = ChannelIdentity(user_id=u.id, channel=Channel.TELEGRAM, external_id="55")
        s.add(ident)
        await s.flush()
        await grant_permission(s, ident, PermissionKind.CHAT)
        await s.commit()


class _Sink:
    def __init__(self, fail=False):
        self.sent: list[str] = []
        self.fail = fail

    async def reply(self, text: str) -> None:
        if self.fail:
            raise RuntimeError("channel down")
        self.sent.append(text)


async def _dispatch(user="55", channel=Channel.TELEGRAM, sink=None, **kwargs):
    from app.db.session import session_scope

    sink = sink or _Sink()
    event = NormalizedEvent(user, channel, "hello", sink.reply)
    async with session_scope() as session:
        outcome = await dispatch_event(session, event, **kwargs)
    return outcome, sink


@pytest.fixture
def llm_down(monkeypatch):
    """A real connection error: nothing listens on port 1."""
    from app.config import get_settings

    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:1")
    get_settings.cache_clear()


@pytest.fixture
def llm_up(monkeypatch):
    async def fake(channel, user_id, agent_id, text, **_kwargs):
        return "the answer"

    monkeypatch.setattr("app.channels.dispatch.run_turn", fake)


async def test_llm_unreachable_apologizes_and_keeps_the_inbound_message(authorized, llm_down):
    outcome, sink = await _dispatch()
    assert outcome is DispatchOutcome.FAILED
    assert sink.sent == [APOLOGY_MESSAGE], "exactly one apology, no error detail"
    assert _rows() == [("inbound", "ok"), ("outbound", "failed")]


async def test_the_apology_never_contains_error_details(authorized, llm_down):
    _, sink = await _dispatch()
    assert "127.0.0.1" not in sink.sent[0] and "Connect" not in sink.sent[0]


async def test_apologize_false_sends_nothing_but_still_records_the_failure(authorized, llm_down):
    from app.admin import service
    from app.db.session import session_scope

    outcome, sink = await _dispatch(apologize=False)
    assert outcome is DispatchOutcome.FAILED
    assert sink.sent == []
    assert _rows() == [("inbound", "ok"), ("outbound", "failed")]
    async with session_scope() as s:
        texts = [e.text for e in await service.search_action_logs(s)]
    assert NO_REPLY_NOTE in texts


async def test_a_retry_does_not_write_the_inbound_message_twice(authorized, llm_down):
    await _dispatch(apologize=False)
    await _dispatch(apologize=False, retry=True)
    await _dispatch(apologize=False, retry=True)
    assert _rows() == [
        ("inbound", "ok"), ("outbound", "failed"), ("outbound", "failed"), ("outbound", "failed"),
    ]


async def test_failing_apology_delivery_does_not_raise(authorized, llm_down):
    outcome, _ = await _dispatch(sink=_Sink(fail=True))
    assert outcome is DispatchOutcome.FAILED
    assert _rows() == [("inbound", "ok"), ("outbound", "failed")]


async def test_a_successful_turn_is_ok_ok(authorized, llm_up):
    outcome, sink = await _dispatch()
    assert outcome is DispatchOutcome.OK
    assert sink.sent == ["the answer"]
    assert _rows() == [("inbound", "ok"), ("outbound", "ok")]


async def test_delivery_failure_is_undelivered_and_keeps_the_generated_answer(authorized, llm_up):
    from app.admin import service
    from app.db.models import ActionStatus
    from app.db.session import session_scope

    outcome, _ = await _dispatch(sink=_Sink(fail=True))
    assert outcome is DispatchOutcome.UNDELIVERED
    assert _rows() == [("inbound", "ok"), ("outbound", "failed")]
    async with session_scope() as s:
        failed = await service.search_action_logs(s, status=ActionStatus.FAILED)
    assert [e.text for e in failed] == ["the answer"], "the undelivered answer is not lost"


async def test_denied_known_identity_is_recorded_as_denied(fresh_db):
    from app.db.models import ChannelIdentity, User
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as s:
        u = User(display_name="No permission")
        s.add(u)
        await s.flush()
        s.add(ChannelIdentity(user_id=u.id, channel=Channel.TELEGRAM, external_id="66"))
        await s.commit()
    outcome, sink = await _dispatch(user="66")
    assert outcome is DispatchOutcome.DENIED
    assert sink.sent == [DENIED_MESSAGE]
    assert _rows() == [("inbound", "denied")]


async def test_unknown_identity_is_denied_without_a_log_row(fresh_db):
    from app.db.session import init_db

    await init_db()
    outcome, sink = await _dispatch(user="777")
    assert outcome is DispatchOutcome.DENIED
    assert sink.sent == [DENIED_MESSAGE]
    assert _rows() == []


async def test_unknown_email_sender_gets_no_reply(fresh_db):
    from app.db.session import init_db

    await init_db()
    outcome, sink = await _dispatch(user="stranger@example.com", channel=Channel.EMAIL)
    assert outcome is DispatchOutcome.DENIED
    assert sink.sent == []


async def test_telegram_adapter_path_survives_a_failed_turn(authorized, llm_down):
    """The Telegram handler calls dispatch_event with defaults: the user
    gets the apology and nothing is raised into python-telegram-bot.
    """
    from types import SimpleNamespace

    from app.channels import telegram

    sent = []

    class Bot:
        async def send_message(self, chat_id, text):
            sent.append((chat_id, text))

    update = SimpleNamespace(
        message=SimpleNamespace(text="hello"),
        effective_user=SimpleNamespace(id=55),
        effective_chat=SimpleNamespace(id=999),
    )
    await telegram._on_message(update, SimpleNamespace(bot=Bot()))
    assert sent == [(999, APOLOGY_MESSAGE)]
    assert _rows() == [("inbound", "ok"), ("outbound", "failed")]


@pytest.mark.parametrize("empty", ["", "  \n "])
async def test_an_empty_model_reply_is_a_failed_turn_with_the_apology(
    authorized, monkeypatch, empty
):
    """A reasoning model ended a turn with no visible text (2026-09-28): sent as is,
    Telegram refused it ("Message text is empty") and the user got nothing at all."""
    from app.channels import dispatch

    async def empty_turn(*args, **kwargs):
        return empty

    monkeypatch.setattr(dispatch, "run_turn", empty_turn)
    outcome, sink = await _dispatch()
    assert outcome is DispatchOutcome.FAILED
    assert sink.sent == [APOLOGY_MESSAGE]
    assert _rows() == [("inbound", "ok"), ("outbound", "failed")]


async def test_an_empty_reply_logs_the_engine_figures_and_never_the_text(caplog):
    """The cause of an empty reply was not measurable afterwards."""
    import httpx

    from app.graph import _chat

    def engine(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "", "reasoning_content": "secret-thought " * 10},
                    }
                ],
                "usage": {"completion_tokens": 512},
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(engine), base_url="http://engine"
    ) as client:
        assert await _chat(client, [{"role": "user", "content": "hi"}]) == ""
    assert (
        "finish_reason=length, completion_tokens=512, reasoning_chars=150" in caplog.text
    )
    assert "secret-thought" not in caplog.text


async def test_an_empty_reply_is_not_stored_so_a_retry_does_not_repeat_the_question(
    authorized, monkeypatch
):
    """2026-10-04: three failed attempts of one email left question, empty answer, question,
    empty answer, question in the conversation; the empty answer is no longer stored."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from langchain_core.messages import HumanMessage
    from sqlalchemy import select

    from app.config import get_settings
    from app.db.models import Agent
    from app.db.session import session_scope
    from app.graph import build_thread_id, close_graph, get_graph

    class Empty(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            data = json.dumps(
                {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    httpd = HTTPServer(("127.0.0.1", 0), Empty)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{httpd.server_port}")
    get_settings.cache_clear()
    try:
        outcomes = [
            (await _dispatch(apologize=False, retry=attempt > 0))[0] for attempt in range(3)
        ]
        assert outcomes == [DispatchOutcome.FAILED] * 3
        async with session_scope() as s:
            agent_id = (await s.execute(select(Agent.id))).scalar_one()
        graph = await get_graph()
        thread = build_thread_id(Channel.TELEGRAM, "55", agent_id)
        stored = (await graph.aget_state({"configurable": {"thread_id": thread}})).values
        assert [(type(m), m.content) for m in stored["messages"]] == [(HumanMessage, "hello")]
    finally:
        await close_graph()
        httpd.shutdown()
        get_settings.cache_clear()
