"""Tests for the `/model` chat command ("la construire").

`/model` lists the engine's models without calling a model; `/model <name> <message>`
answers that one message with that model, through the normal pipeline (authorization, the
conversation gets the message, not the command); a name the engine does not offer, a missing
message or an engine that does not answer get a short reply and no turn.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from app.admin import service
from app.channels import dispatch
from app.channels.dispatch import DENIED_MESSAGE, DispatchOutcome, handle_model_command
from app.channels.schema import NormalizedEvent
from app.db.models import Channel, PermissionKind

MODELS = ["big.gguf", "small.gguf"]


@pytest.fixture
async def world(fresh_db):
    """User 1 Alice, Telegram "1", allowed to chat, agent "default". Fay, Telegram "3",
    no permission."""
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as s:
        alice = await service.create_user(s, "Alice")
        fay = await service.create_user(s, "Fay")
        identity = await service.add_channel_identity(s, alice.id, Channel.TELEGRAM, "1")
        await service.grant_identity_permission(s, alice.id, identity.id, PermissionKind.CHAT)
        await service.add_channel_identity(s, fay.id, Channel.TELEGRAM, "3")
        await service.create_agent(s, alice.id, "default")
        await s.commit()


@pytest.fixture
def turns(monkeypatch):
    calls: list[dict] = []

    async def fake_turn(channel, user_id, agent_id, text, **kwargs):
        calls.append({"text": text, "model": kwargs.get("model")})
        return "an answer"

    monkeypatch.setattr(dispatch, "run_turn", fake_turn)
    return calls


@pytest.fixture
def engine(monkeypatch):
    state = SimpleNamespace(models=MODELS, asked=0)

    async def fake_models():
        state.asked += 1
        return state.models

    monkeypatch.setattr(dispatch, "engine_models", fake_models)
    return state


def _event(user="1"):
    replies: list[str] = []

    async def reply(text):
        replies.append(text)

    return NormalizedEvent(user_id=user, channel=Channel.TELEGRAM, text="/model", reply=reply), (
        replies
    )


async def _logs() -> list[str]:
    from app.db.session import session_scope

    async with session_scope() as s:
        return [e.text for e in await service.search_action_logs(s)][::-1]


async def _run(name, message, user="1"):
    from app.db.session import session_scope

    event, replies = _event(user)
    async with session_scope() as s:
        outcome = await handle_model_command(s, event, name, message)
    return outcome, replies


async def test_model_alone_lists_the_engines_models_without_a_turn(world, turns, engine):
    outcome, replies = await _run("", "")
    assert outcome == DispatchOutcome.OK and turns == []
    assert replies == ["Models: big.gguf, small.gguf. Use /model <name> <message>."]
    assert set(await _logs()) == {"/model", replies[0]}


async def test_a_message_is_answered_by_the_chosen_model(world, turns, engine):
    outcome, replies = await _run("small.gguf", "line one\nline two")
    assert outcome == DispatchOutcome.OK and replies == ["an answer"]
    assert turns == [{"text": "line one\nline two", "model": "small.gguf"}]
    logs = await _logs()
    assert "line one\nline two" in logs and not any(t.startswith("/model") for t in logs)


async def test_a_name_the_engine_does_not_offer_gets_no_turn(world, turns, engine):
    outcome, replies = await _run("../../etc/passwd", "hello")
    assert turns == [] and replies[0].startswith("No model named '../../etc/passwd'.")


async def test_a_name_without_a_message_gets_a_hint(world, turns, engine):
    _, replies = await _run("big.gguf", "   ")
    assert turns == [] and replies == ["Add the message after the name: /model big.gguf <message>."]


async def test_an_engine_that_does_not_answer_gets_no_turn(world, turns, engine):
    engine.models = None
    _, replies = await _run("big.gguf", "hello")
    assert turns == [] and "not available" in replies[0]


async def test_an_unauthorized_sender_gets_the_usual_denial(world, turns, engine):
    from app.db.session import session_scope

    outcome, replies = await _run("big.gguf", "hello", user="3")
    assert outcome == DispatchOutcome.DENIED and replies == [DENIED_MESSAGE]
    assert turns == [] and engine.asked == 0
    async with session_scope() as s:
        assert len(await service.list_pending_requests(s)) == 1


async def test_the_engines_model_list_is_read_from_v1_models(monkeypatch):
    from app.config import get_settings

    class Engine(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"data": [{"id": "b.gguf"}, {"id": "a.gguf"}, {"x": 1}]}).encode()
            self.send_response(200 if self.path == "/v1/models" else 404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Engine)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
        get_settings.cache_clear()
        assert await dispatch.engine_models() == ["a.gguf", "b.gguf"]
        monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:9")
        get_settings.cache_clear()
        assert await dispatch.engine_models() is None
    finally:
        server.shutdown()
        get_settings.cache_clear()
