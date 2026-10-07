"""Tests: the answer is read from the engine while it is written and shown on the
channel (typing, then a Telegram draft or an edited message), never failing the turn.
"""

import asyncio
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from app import engine_stream, tools
from app.channels import telegram
from app.channels.schema import NormalizedEvent, ReplyProgress
from app.db.models import Channel


def _sse(chunks: list[dict]) -> bytes:
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"


def _content(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}, "finish_reason": None}]}


ANSWER_CHUNKS = [
    {"choices": [{"delta": {"reasoning_content": "The user wants"}}]},
    _content("RAM is "),
    _content("main memory."),
    {"choices": [{"delta": {}, "finish_reason": "stop"}], "model": "/m/x.gguf"},
    {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 7}},
]


def _mock_client(bodies: list[bytes], seen: list | None = None, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(
            status, content=bodies.pop(0), headers={"content-type": "text/event-stream"}
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://e")


# --- reading the stream ---


async def test_the_stream_is_assembled_and_only_the_visible_text_is_shown():
    shown: list[str] = []

    async def on_text(text):
        shown.append(text)

    seen: list = []
    async with _mock_client([_sse(ANSWER_CHUNKS)], seen) as client:
        body = await engine_stream.complete(client, {"messages": []}, on_text)
    assert seen[0]["stream"] is True and seen[0]["stream_options"] == {"include_usage": True}
    assert shown == ["RAM is", "RAM is main memory."]
    message = body["choices"][0]["message"]
    assert message["content"] == "RAM is main memory."
    assert message["reasoning_content"] == "The user wants"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 12, "completion_tokens": 7}
    assert body["model"] == "/m/x.gguf"


async def test_a_thinking_block_written_in_the_content_is_never_shown():
    shown: list[str] = []

    async def on_text(text):
        shown.append(text)

    chunks = [_content("<think>let me"), _content(" see</think>"), _content("Answer.")]
    async with _mock_client([_sse(chunks)]) as client:
        body = await engine_stream.complete(client, {"messages": []}, on_text)
    assert shown == ["Answer."]
    assert body["choices"][0]["message"]["content"] == "<think>let me see</think>Answer."


async def test_tool_call_fragments_are_joined():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "get_", "arguments": '{"tz"'}}
        ]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "time", "arguments": ': "UTC"}'}}
        ]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]  # fmt: skip
    async with _mock_client([_sse(chunks)]) as client:
        body = await engine_stream.complete(client, {"messages": []}, None)
    call = body["choices"][0]["message"]["tool_calls"][0]
    assert call["id"] == "c1"
    assert call["function"] == {"name": "get_time", "arguments": '{"tz": "UTC"}'}


async def test_a_failing_display_stops_showing_but_the_answer_is_complete():
    calls = 0

    async def broken(text):
        nonlocal calls
        calls += 1
        raise RuntimeError("telegram down")

    async with _mock_client([_sse(ANSWER_CHUNKS)]) as client:
        body = await engine_stream.complete(client, {"messages": []}, broken)
    assert calls == 1
    assert body["choices"][0]["message"]["content"] == "RAM is main memory."


async def test_an_engine_error_is_raised_with_its_status():
    async with _mock_client([b'{"error": "no such model"}'], status=400) as client:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await engine_stream.complete(client, {"messages": []}, None)
    assert caught.value.response.status_code == 400


async def test_the_tool_loop_streams_every_round_and_returns_the_final_answer():
    tool_round = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "t", "arguments": "{}"}}
        ]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
    ]  # fmt: skip
    shown: list[str] = []
    usages: list[dict] = []

    async def on_text(text):
        shown.append(text)

    async def executor(name, arguments):
        return "12:00"

    seen: list = []
    async with _mock_client([_sse(tool_round), _sse(ANSWER_CHUNKS)], seen) as client:
        reply, rounds = await tools.run_tool_loop(
            client, [{"role": "user", "content": "time?"}], [{"type": "function"}], executor,
            on_text=on_text, on_usage=usages.append,
        )  # fmt: skip
    assert (reply, rounds) == ("RAM is main memory.", 2)
    assert all(request["stream"] is True for request in seen)
    assert seen[1]["messages"][-1]["role"] == "tool"
    assert shown[-1] == "RAM is main memory."
    assert [u["usage"]["prompt_tokens"] for u in usages] == [5, 12]


# --- a whole turn through dispatch ---


class _Engine(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if not self.path.endswith("/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return
        type(self).requests.append(body)
        if body.get("stream"):
            data, kind = _sse(ANSWER_CHUNKS), "text/event-stream"
        else:
            data = json.dumps({
                "choices": [{"message": {"content": "RAM is main memory."}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7},
            }).encode()  # fmt: skip
            kind = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
async def engine(fresh_db, monkeypatch):
    from app.config import get_settings
    from app.db.models import ChannelIdentity, PermissionKind, User
    from app.db.session import init_db, session_scope
    from app.graph import close_graph
    from app.security.auth import grant_permission

    _Engine.requests = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{httpd.server_port}")
    get_settings.cache_clear()
    await init_db()
    async with session_scope() as s:
        user = User(display_name="Alice")
        s.add(user)
        await s.flush()
        ident = ChannelIdentity(user_id=user.id, channel=Channel.TELEGRAM, external_id="55")
        s.add(ident)
        await s.flush()
        await grant_permission(s, ident, PermissionKind.CHAT)
        await s.commit()
    yield _Engine
    await close_graph()
    httpd.shutdown()
    get_settings.cache_clear()


class _Progress(ReplyProgress):
    def __init__(self, log):
        self.log = log

    async def start(self):
        self.log.append("start")

    async def update(self, text):
        self.log.append(f"update:{text}")

    async def stop(self):
        self.log.append("stop")


def _outbound_tokens():
    from app.config import get_settings

    con = sqlite3.connect(get_settings().database_url.split("///", 1)[1])
    try:
        return con.execute(
            "select prompt_tokens, completion_tokens from action_logs "
            "where direction = 'outbound'"
        ).fetchall()
    finally:
        con.close()


async def _dispatch(event):
    from app.channels.dispatch import dispatch_event
    from app.db.session import session_scope

    async with session_scope() as session:
        return await dispatch_event(session, event)


async def test_a_turn_shows_progress_then_delivers_the_final_reply(engine):
    log: list[str] = []

    async def reply(text):
        log.append(f"reply:{text}")

    event = NormalizedEvent("55", Channel.TELEGRAM, "RAM?", reply, progress=_Progress(log))
    await _dispatch(event)
    assert log == [
        "start", "update:RAM is", "update:RAM is main memory.", "stop",
        "reply:RAM is main memory.",
    ]  # fmt: skip
    assert engine.requests[-1]["stream"] is True
    # The usage of a streamed answer is still counted.
    assert _outbound_tokens() == [(12, 7)]


async def test_a_channel_without_progress_is_not_streamed(engine):
    sent: list[str] = []

    async def reply(text):
        sent.append(text)

    await _dispatch(NormalizedEvent("55", Channel.TELEGRAM, "RAM?", reply))
    assert sent == ["RAM is main memory."]
    assert engine.requests[-1]["stream"] is False
    assert _outbound_tokens() == [(12, 7)]


async def test_progress_stops_before_the_apology_of_a_failed_turn(engine, monkeypatch):
    import app.channels.dispatch as dispatch

    async def down(*a, **k):
        raise RuntimeError("engine down")

    monkeypatch.setattr(dispatch, "run_turn", down)
    log: list[str] = []

    async def reply(text):
        log.append(f"reply:{text}")

    await _dispatch(NormalizedEvent("55", Channel.TELEGRAM, "RAM?", reply, progress=_Progress(log)))
    assert log[0] == "start"
    assert log.index("stop") < log.index(f"reply:{dispatch.APOLOGY_MESSAGE}")


# --- Telegram ---


class _Bot:
    def __init__(self, fail_drafts=False):
        self.calls: list[tuple] = []
        self.fail_drafts = fail_drafts

    async def send_message_draft(self, chat_id, draft_id, text=None):
        if self.fail_drafts:
            self.calls.append(("refused draft", text))
            raise RuntimeError("draft refused")
        self.calls.append(("draft", text))

    async def send_message(self, chat_id, text, **kw):
        self.calls.append(("send", text))
        return SimpleNamespace(message_id=77)

    async def edit_message_text(self, chat_id, message_id, text):
        self.calls.append(("edit", message_id, text))

    async def send_chat_action(self, chat_id, action):
        self.calls.append(("action", action))


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


async def test_a_private_chat_gets_drafts_at_most_once_a_second():
    bot, clock = _Bot(), _Clock()
    progress = telegram.TelegramProgress(bot, 1, private=True, clock=clock)
    await progress.update("RAM")
    clock.now += 0.4
    await progress.update("RAM is")
    clock.now += 0.7
    await progress.update("RAM is main")
    clock.now += 5
    await progress.update("   ")
    assert bot.calls == [("draft", "RAM"), ("draft", "RAM is main")]
    assert progress.take_message() is None


async def test_a_group_gets_one_message_edited_in_place():
    bot, clock = _Bot(), _Clock()
    progress = telegram.TelegramProgress(bot, 1, private=False, clock=clock)
    await progress.update("RAM")
    clock.now += 2
    await progress.update("RAM is")
    assert bot.calls == [("send", "RAM"), ("edit", 77, "RAM is")]
    assert progress.take_message() == 77
    assert progress.take_message() is None


async def test_typing_and_the_thinking_placeholder_are_sent_until_stop(monkeypatch):
    monkeypatch.setattr(telegram, "TYPING_EVERY", 0.01)
    bot = _Bot()
    progress = telegram.TelegramProgress(bot, 1, private=True)
    await progress.start()
    await asyncio.sleep(0.05)
    await progress.stop()
    count = len(bot.calls)
    await asyncio.sleep(0.03)
    assert len(bot.calls) == count, "nothing is sent after stop()"
    assert bot.calls[0] == ("action", "typing")
    assert ("draft", "") in bot.calls
    assert sum(1 for c in bot.calls if c == ("action", "typing")) >= 2


async def test_a_refused_draft_never_fails_and_stops_the_drafts():
    bot = _Bot(fail_drafts=True)
    clock = _Clock()
    progress = telegram.TelegramProgress(bot, 1, private=True, clock=clock)
    await progress.update("RAM")
    clock.now += 5
    await progress.update("RAM is main memory")
    assert bot.calls == [("refused draft", "RAM")], "one refusal, then no more drafts"


def _update(chat_type):
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=1, type=chat_type),
        effective_user=SimpleNamespace(id=55),
    )


async def test_the_final_reply_in_a_group_edits_the_streamed_message():
    bot = _Bot()
    event = telegram._event(_update("group"), SimpleNamespace(bot=bot), "RAM?")
    await event.progress.update("RAM is")
    await event.reply("RAM is main memory.")
    assert bot.calls == [("send", "RAM is"), ("edit", 77, "RAM is main memory.")]


async def test_the_final_reply_in_a_group_is_not_edited_when_unchanged():
    bot = _Bot()
    event = telegram._event(_update("supergroup"), SimpleNamespace(bot=bot), "RAM?")
    await event.progress.update("RAM is main memory.")
    await event.reply("RAM is main memory.")
    assert bot.calls == [("send", "RAM is main memory.")]


async def test_the_final_reply_in_a_private_chat_is_a_new_message():
    bot = _Bot()
    event = telegram._event(_update("private"), SimpleNamespace(bot=bot), "RAM?")
    await event.progress.update("RAM is")
    await event.reply("RAM is main memory.")
    assert bot.calls == [("draft", "RAM is"), ("send", "RAM is main memory.")]


async def test_the_usage_hook_never_reads_a_stream():
    """Reading a streamed response in the hook would deliver the whole answer before any
    of it is shown."""
    from app import graph

    class _Unreadable(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise AssertionError("the hook read the stream")
            yield b""

    request = httpx.Request("POST", "http://e/v1/chat/completions")
    response = httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=_Unreadable(), request=request
    )
    token = graph._turn_stats.set({"model": None, "prompt_tokens": 0, "completion_tokens": 0})
    try:
        await graph._count_usage(response)
    finally:
        graph._turn_stats.reset(token)
