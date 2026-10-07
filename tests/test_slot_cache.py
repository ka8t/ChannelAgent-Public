"""A conversation's engine cache saved on disk and restored before its next turn.

A fake engine renders a conversation as `<role>content` lines and counts one token per
character, saves a slot as a file in the slot folder and restores it from there, as llama-server
does with `--slot-save-path`. The real engine was measured
separately.
"""

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from app import slot_cache
from app.config import get_settings


class FakeEngine:
    def __init__(self, models: int = 1, slots: int = 2, refuse_save: bool = False):
        self.models, self.slots, self.refuse_save = models, slots, refuse_save
        self.calls: list[tuple[str, str]] = []
        self.size = 1000

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        action = request.url.params.get("action", "")
        self.calls.append((path, action))
        body = json.loads(request.content or b"{}")
        if path == "/v1/models":
            data = [{"id": f"m{i}", "meta": {"size": self.size, "n_params": 7}}
                    for i in range(self.models)]  # fmt: skip
            return httpx.Response(200, json={"data": data})
        if path == "/props":
            return httpx.Response(200, json={"total_slots": self.slots})
        if path == "/apply-template":
            text = "".join(f"<{m['role']}>{m['content']}\n" for m in body["messages"])
            return httpx.Response(200, json={"prompt": text + "<assistant>"})
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": [ord(c) for c in body["content"]]})
        if path == "/completion":
            return httpx.Response(200, json={"timings": {}})
        if path.startswith("/slots/"):
            target = slot_cache.slot_dir() / body.get("filename", "")
            if action == "save":
                if self.refuse_save:
                    return httpx.Response(501, json={"error": "slot save path not set"})
                target.write_bytes(b"x" * 100)
                return httpx.Response(200, json={"n_saved": 1})
            if action == "restore":
                if not target.is_file():
                    return httpx.Response(400, json={"error": "no file"})
                return httpx.Response(200, json={"n_restored": 1})
            return httpx.Response(200, json={})
        return httpx.Response(404)

    def count(self, path: str, action: str = "") -> int:
        return sum(1 for p, a in self.calls if p == path and a == action)


@pytest.fixture
def cache_on(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'app.db'}")
    monkeypatch.setenv("SLOT_CACHE_MAX_MB", "1")
    get_settings.cache_clear()
    slot_cache.reset()
    folder = slot_cache.slot_dir()
    folder.mkdir()
    return folder


def _client(engine: FakeEngine) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(engine.handler), base_url="http://e")


def _history(turns: int, word: str = "h") -> list[dict]:
    """A conversation of `turns` exchanges of about 100 characters: it grows at its end."""
    pairs = [
        ({"role": "user", "content": f"{word * 50} {i}"},
         {"role": "assistant", "content": word * 50})
        for i in range(turns)
    ]  # fmt: skip
    return [m for pair in pairs for m in pair]


def _then(history: list[dict], turns: int = 1) -> list[dict]:
    """The history of a later turn: this turn's question and a reply, `turns` times."""
    pair = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "ok " * 30}]
    return history + pair * turns


def _bodies(history: list[dict], memory: str = "", turn: str = "question"):
    """The turn's request (memory hits in the new message) and its stable shape."""
    head = [{"role": "system", "content": "You are kind."}, *history]
    sent = [*head, {"role": "user", "content": memory + turn}]
    stable = [
        *head, {"role": "user", "content": turn},
        {"role": "assistant", "content": "ok"}, {"role": "user", "content": "next"},
    ]  # fmt: skip
    return {"messages": sent}, {"messages": stable}


async def _turn(engine: FakeEngine, thread: str, history: list[dict], memory: str = "") -> dict:
    body, stable = _bodies(history, memory)
    async with _client(engine) as client, slot_cache.pinned(client, thread, body, stable) as pin:
        return pin


async def test_off_when_the_variable_is_0_or_the_engine_has_several_models(cache_on, monkeypatch):
    engine = FakeEngine(models=2)
    assert await _turn(engine, "t1", _history(20)) == {}
    assert engine.count("/completion") == 0
    monkeypatch.setenv("SLOT_CACHE_MAX_MB", "0")
    get_settings.cache_clear()
    slot_cache.reset()
    engine = FakeEngine()
    assert await _turn(engine, "t1", _history(20)) == {}
    assert engine.calls == []


async def test_a_long_turn_saves_its_stable_prefix_once_per_step(cache_on):
    engine = FakeEngine()
    pin = await _turn(engine, "t1", _history(20), memory="MEMORY HITS ")
    assert pin == {"id_slot": 0}
    assert engine.count("/completion") == 1 and engine.count("/slots/0", "save") == 1
    (meta_file,) = cache_on.glob("*.json")
    meta = json.loads(meta_file.read_text())
    # The prefix stops where the memory hits start: what the next turn sends again.
    sent = _bodies(_history(20))[0]["messages"]
    rendered = "".join(f"<{m['role']}>{m['content']}\n" for m in sent)
    assert meta["n"] == len(rendered) - len("question\n")
    assert oct(os.stat(meta_file).st_mode & 0o777) == "0o600"
    # Same thread, still in its slot, prefix grown by less than a step: nothing more.
    await _turn(engine, "t1", _then(_history(20)))
    assert engine.count("/slots/0", "restore") == 0 and engine.count("/slots/0", "save") == 1
    # Grown by a step: read and saved again.
    await _turn(engine, "t1", _then(_history(20), slot_cache.SAVE_STEP // 100 + 1))
    assert engine.count("/slots/0", "save") == 2 and engine.count("/slots/0", "restore") == 0


async def test_after_a_restart_the_file_is_restored_when_it_is_a_prefix(cache_on):
    engine = FakeEngine()
    await _turn(engine, "t1", _history(20))
    slot_cache.reset()  # the application (and so the engine) restarted
    engine = FakeEngine()
    assert await _turn(engine, "t1", _then(_history(20))) == {"id_slot": 0}
    assert engine.count("/slots/0", "restore") == 1 and engine.count("/slots/0", "save") == 0
    # The conversation changed before the saved point (the window moved), and is longer than
    # the saved prefix: not restored, saved again.
    slot_cache.reset()
    engine = FakeEngine()
    changed = _then(_history(20), 3)
    changed[0] = {"role": "user", "content": "z" * len(changed[0]["content"])}
    await _turn(engine, "t1", changed)
    assert engine.count("/slots/0", "restore") == 0 and engine.count("/slots/0", "save") == 1


async def test_a_short_conversation_is_not_saved(cache_on):
    engine = FakeEngine()
    assert await _turn(engine, "t1", []) == {"id_slot": 0}
    assert engine.count("/completion") == 0 and list(cache_on.iterdir()) == []


async def test_two_conversations_at_once_get_two_slots_and_a_third_waits(cache_on):
    engine = FakeEngine(slots=2)
    order = []
    async with _client(engine) as client:
        b, s = _bodies([])
        async with slot_cache.pinned(client, "a", b, s) as pa, slot_cache.pinned(
            client, "b", b, s
        ) as pb:
            assert {pa["id_slot"], pb["id_slot"]} == {0, 1}

            async def third():
                async with slot_cache.pinned(client, "c", b, s) as pc:
                    order.append(("c", pc["id_slot"]))

            task = asyncio.create_task(third())
            await asyncio.sleep(0.01)
            assert order == []  # both slots are taken: the third turn waits
        await task
    assert order == [("c", 0)]  # the least recently used slot


async def test_the_engine_refusing_the_slot_api_turns_the_cache_off(cache_on):
    engine = FakeEngine(refuse_save=True)
    assert await _turn(engine, "t1", _history(20)) == {"id_slot": 0}  # the turn goes on
    assert await _turn(engine, "t1", _history(20)) == {}
    assert list(cache_on.glob("*.json")) == []


async def test_forgetting_a_thread_deletes_its_files_and_erases_its_slot(cache_on):
    engine = FakeEngine()
    await _turn(engine, "gone", _history(20))
    await _turn(engine, "kept", _history(20, "k"))
    assert len(list(cache_on.glob("*.bin"))) == 2
    async with _client(engine) as client:
        assert await slot_cache.forget(client, ["gone"]) == 1
    assert len(list(cache_on.glob("*.bin"))) == 1 and len(list(cache_on.glob("*.json"))) == 1
    assert engine.count("/slots/0", "erase") == 1


async def test_delete_threads_forgets_the_engine_cache(cache_on):
    from app.graph import delete_threads

    name = f"abc123abc123-{slot_cache._thread_key('telegram_1_1')}.bin"
    (cache_on / name).write_bytes(b"x")
    (cache_on / name).with_suffix(".json").write_text("{}")
    await delete_threads(["telegram_1_1"])
    assert list(cache_on.iterdir()) == []


async def test_the_oldest_files_go_above_the_cap_and_another_models_files_go(cache_on):
    old = cache_on / "abc123abc123-oldthread.bin"
    old.write_bytes(b"x")  # small: only the model key can remove it, not the cap
    os.utime(old, (1, 1))
    engine = FakeEngine()
    await _turn(engine, "t1", _history(20))
    assert not old.exists(), "another model's file is deleted when the engine is found"
    key = slot_cache._state["engine"][0]
    big = cache_on / f"{key}-big.bin"
    big.write_bytes(b"x" * (2**20))
    os.utime(big, (2, 2))
    await _turn(engine, "t2", _history(20, "j"))
    assert not big.exists(), "above SLOT_CACHE_MAX_MB the oldest file is deleted"
    assert len(list(cache_on.glob("*.bin"))) == 2


class _HttpEngine(BaseHTTPRequestHandler):
    """The fake engine over HTTP, for a real turn: it also answers the chat completion and
    records the slot each one was pinned to."""

    fake: FakeEngine
    pinned: list = []

    def log_message(self, *a):
        pass

    def _forward(self, method: str, body: bytes = b"") -> None:
        request = httpx.Request(method, f"http://e{self.path}", content=body)
        if self.path == "/v1/chat/completions":
            type(self).pinned.append(json.loads(body).get("id_slot"))
            response = httpx.Response(
                200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
            )
        else:
            response = self.fake.handler(request)
        self.send_response(response.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response.content)))
        self.end_headers()
        self.wfile.write(response.content)

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST", self.rfile.read(int(self.headers.get("Content-Length", 0))))


async def test_a_turn_is_pinned_saved_and_restored_after_a_restart(fresh_db, monkeypatch):
    from app.admin import service
    from app.db.models import Channel
    from app.db.session import init_db, session_scope
    from app.graph import close_graph, run_turn

    monkeypatch.setenv("SLOT_CACHE_MAX_MB", "64")
    _HttpEngine.fake, _HttpEngine.pinned = FakeEngine(), []
    server = HTTPServer(("127.0.0.1", 0), _HttpEngine)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    get_settings.cache_clear()
    slot_cache.slot_dir().mkdir(parents=True)
    await init_db()
    async with session_scope() as session:
        user = await service.create_user(session, "Sam")
        agent = await service.create_agent(session, user.id, "main")
        await service.add_channel_identity(session, user.id, Channel.TELEGRAM, "111")
        await session.commit()
    try:
        for i in range(3):
            assert await run_turn(Channel.TELEGRAM, "111", agent.id, f"{i} " + "q" * 400) == "hi"
        fake = _HttpEngine.fake
        assert _HttpEngine.pinned == [0, 0, 0]
        assert fake.count("/slots/0", "save") == 1 and fake.count("/slots/0", "restore") == 0
        assert len(list(slot_cache.slot_dir().glob("*.bin"))) == 1
        slot_cache.reset()  # a restart: the next turn restores the saved prefix
        assert await run_turn(Channel.TELEGRAM, "111", agent.id, "after") == "hi"
        assert fake.count("/slots/0", "restore") == 1 and _HttpEngine.pinned[-1] == 0
    finally:
        await close_graph()
        server.shutdown()
        get_settings.cache_clear()
