"""Tests: the engine's bearer key. An engine that answers 401 without it (a real
HTTPServer, like llama-server started with LLAMA_API_KEY) is reached by a turn, the status
route and the model list with the key, and refused without it; the key follows the rules of
the other keys, is masked by `--show-config` and redacted from the logs.
"""

import json
import logging
import secrets
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.config import engine_headers, get_settings

KEY = secrets.token_hex(24)


class _LockedEngine(BaseHTTPRequestHandler):
    """/health is public (as in llama-server); everything else needs the bearer key."""

    seen: list = []

    def log_message(self, *a):
        pass

    def _answer(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        type(self).seen.append((self.path, self.headers.get("Authorization")))
        if self.path == "/health" or self.headers.get("Authorization") == f"Bearer {KEY}":
            return True
        self._answer(401, {"error": {"message": "Invalid API Key", "type": "authentication_error"}})
        return False

    def do_GET(self):
        if not self._authorized():
            return
        if self.path == "/v1/models":
            self._answer(200, {"data": [{"id": "m.gguf"}]})
        elif self.path == "/props":
            self._answer(200, {"model_path": "/models/m.gguf",
                               "default_generation_settings": {"n_ctx": 4096}})  # fmt: skip
        elif self.path == "/slots":
            self._answer(200, [{"id": 0, "is_processing": False}])
        else:
            self._answer(200, {"status": "ok"})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if not self._authorized():
            return
        if self.path == "/tokenize":
            self._answer(200, {"tokens": [1, 2, 3]})
        else:
            self._answer(200, {"choices": [{"message": {"role": "assistant", "content": "hi"}}]})


@pytest.fixture
def engine(monkeypatch):
    _LockedEngine.seen = []
    server = HTTPServer(("127.0.0.1", 0), _LockedEngine)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    get_settings.cache_clear()
    yield server
    server.shutdown()
    get_settings.cache_clear()


def _with_key(monkeypatch, key):
    if key is None:
        monkeypatch.delenv("LLAMA_SERVER_API_KEY", raising=False)
        monkeypatch.setenv("LLAMA_SERVER_API_KEY", "")
    else:
        monkeypatch.setenv("LLAMA_SERVER_API_KEY", key)
    get_settings.cache_clear()


def test_the_header_is_sent_only_when_a_key_is_set(monkeypatch):
    _with_key(monkeypatch, None)
    assert engine_headers() == {}
    _with_key(monkeypatch, KEY)
    assert engine_headers() == {"Authorization": f"Bearer {KEY}"}


# --- every client of the engine ---


async def test_a_turn_reaches_the_locked_engine_with_the_key_and_not_without(
    fresh_db, engine, monkeypatch
):
    from app.admin import service
    from app.db.models import Channel
    from app.db.session import init_db, session_scope
    from app.graph import close_graph, run_turn

    await init_db()
    async with session_scope() as session:
        user = await service.create_user(session, "Sam")
        agent = await service.create_agent(session, user.id, "main")
        await service.add_channel_identity(session, user.id, Channel.TELEGRAM, "111")
        await session.commit()
    try:
        _with_key(monkeypatch, KEY)
        assert await run_turn(Channel.TELEGRAM, "111", agent.id, "hello") == "hi"
        chat = [auth for path, auth in _LockedEngine.seen if path == "/v1/chat/completions"]
        assert chat == [f"Bearer {KEY}"]
        _with_key(monkeypatch, None)
        with pytest.raises(Exception, match="401"):
            await run_turn(Channel.TELEGRAM, "111", agent.id, "again")
    finally:
        await close_graph()


async def test_the_status_route_and_the_model_list_send_the_key(engine, monkeypatch):
    from app.api.status import _engine
    from app.channels.dispatch import engine_models

    _with_key(monkeypatch, KEY)
    status = await _engine()
    assert (status.reachable, status.model, status.slots_total) == (True, "m.gguf", 1)
    assert await engine_models() == ["m.gguf"]
    _with_key(monkeypatch, None)
    assert (await _engine()).reachable is False
    assert await engine_models() is None


# --- the key as a setting ---


@pytest.mark.parametrize("value", ["short", "a" * 40, "changeme-changeme-changeme"])
def test_a_weak_key_is_refused_by_the_rules(value):
    from app.settings_rules import validate

    assert validate("LLAMA_SERVER_API_KEY", value) is not None


def test_a_good_key_or_none_is_accepted_and_the_key_is_masked():
    from app.settings_rules import SENSITIVE, validate

    assert validate("LLAMA_SERVER_API_KEY", KEY) is None
    assert validate("LLAMA_SERVER_API_KEY", "") is None
    assert "LLAMA_SERVER_API_KEY" in SENSITIVE


def test_the_key_is_redacted_from_every_log_line(monkeypatch, caplog):
    from app.logging_setup import REDACTED, configured_secrets, install_redaction

    _with_key(monkeypatch, KEY)
    assert KEY in configured_secrets()
    install_redaction()
    try:
        logging.getLogger("channelagent").warning("calling the engine with %s", KEY)
        assert KEY not in caplog.text and REDACTED in caplog.text
    finally:
        install_redaction([])
