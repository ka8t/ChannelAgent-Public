"""Tests: live engine status and usage telemetry.

- A turn records, on its reply, the model the engine reported, the latency and the tokens of
  every completion of the turn.
- GET /telemetry: messages, replies, failed turns, denied, latency p50/p95 and tokens per
  period, in total or per user, agent or model; the counts equal a direct SQL count; no
  message text is read.
"""

import json
import sqlite3
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from app.admin import service
from app.channels.dispatch import dispatch_event
from app.channels.schema import NormalizedEvent
from app.db.models import ActionLog, ActionStatus, Channel, Direction, PermissionKind

KEY = "Tq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
AUTH = {"Authorization": f"Bearer {KEY}"}


def _sql(query, *args):
    from app.config import get_settings

    con = sqlite3.connect(get_settings().database_url.split("///", 1)[1])
    try:
        return con.execute(query, args).fetchall()
    finally:
        con.close()


class _Engine(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = json.dumps(
            {
                "model": "reported-model.gguf",
                "choices": [{"message": {"role": "assistant", "content": "hi there"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
                "timings": {"cache_n": 8, "prompt_n": 3, "prompt_ms": 123.4},
            }
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


@pytest.fixture
async def world(fresh_db, monkeypatch):
    from app.api import deps
    from app.config import get_settings
    from app.db.session import init_db, session_scope

    server = HTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    async with session_scope() as s:
        user = await service.create_user(s, "Alice")
        identity = await service.add_channel_identity(s, user.id, Channel.TELEGRAM, "42")
        await service.grant_identity_permission(s, user.id, identity.id, PermissionKind.CHAT)
        await service.create_agent(s, user.id, "default")
        await s.commit()
    yield
    server.shutdown()
    get_settings.cache_clear()


async def test_a_turn_records_its_model_latency_and_tokens_on_the_reply(world):
    from app.db.session import session_scope

    async def reply(_text):
        pass

    event = NormalizedEvent(user_id="42", channel=Channel.TELEGRAM, text="hello", reply=reply)
    async with session_scope() as s:
        await dispatch_event(s, event)
    rows = _sql(
        "select direction, model, latency_ms, prompt_tokens, completion_tokens, cached_tokens, "
        "prefill_ms from action_logs order by id"
    )
    assert rows[0] == ("inbound", None, None, None, None, None, None)
    direction, model, latency, prompt, completion, cached, prefill = rows[1]
    assert (direction, model, prompt, completion) == ("outbound", "reported-model.gguf", 11, 3)
    assert (cached, prefill) == (8, 123), "The engine's cache figures"
    assert latency is not None and latency >= 0


# --- GET /telemetry ---

ROWS = [
    # user, agent, direction, status, model, latency, prompt, completion, day
    (1, 1, "inbound", "ok", None, None, None, None, 10),
    (1, 1, "outbound", "ok", "a.gguf", 100, 10, 2, 10),
    (1, 1, "inbound", "ok", None, None, None, None, 11),
    (1, 1, "outbound", "ok", "a.gguf", 300, 20, 4, 11),
    (1, 2, "inbound", "ok", None, None, None, None, 12),
    (1, 2, "outbound", "failed", None, None, None, None, 12),
    (2, 3, "inbound", "denied", None, None, None, None, 12),
    (2, 3, "inbound", "ok", None, None, None, None, 20),
    (2, 3, "outbound", "ok", "b.gguf", 900, 30, 6, 20),
]


@pytest.fixture
async def filled(fresh_db, monkeypatch):
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db, session_scope

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    async with session_scope() as s:
        a, b = await service.create_user(s, "A"), await service.create_user(s, "B")
        await service.create_agent(s, a.id, "one")
        await service.create_agent(s, a.id, "two")
        await service.create_agent(s, b.id, "three")
        for user, agent, direction, status, model, latency, prompt, completion, day in ROWS:
            s.add(
                ActionLog(
                    user_id=user, agent_id=agent, channel=Channel.TELEGRAM,
                    direction=Direction(direction), status=ActionStatus(status), text="x",
                    model=model, latency_ms=latency, prompt_tokens=prompt,
                    completion_tokens=completion, created_at=datetime(2026, 9, day, tzinfo=UTC),
                )
            )  # fmt: skip
        await s.commit()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=AUTH) as c:
        yield c
    get_settings.cache_clear()


def _direct(where: str = "1 = 1") -> dict:
    """The same counts, straight from SQLite."""

    def one(q):
        return _sql(f"select {q} from action_logs where {where}")[0][0]

    return {
        "messages": one("count(*) filter (where direction = 'inbound' and status = 'ok')"),
        "replies": one("count(*) filter (where direction = 'outbound' and status = 'ok')"),
        "failed_turns": one("count(*) filter (where direction = 'outbound' and status = 'failed')"),
        "denied": one("count(*) filter (where direction = 'inbound' and status = 'denied')"),
        "prompt_tokens": one("coalesce(sum(prompt_tokens), 0)"),
        "completion_tokens": one("coalesce(sum(completion_tokens), 0)"),
    }


def _counts(row: dict) -> dict:
    return {k: row[k] for k in ("messages", "replies", "failed_turns", "denied",
                                "prompt_tokens", "completion_tokens")}  # fmt: skip


async def test_the_total_equals_a_direct_sql_count(filled):
    rows = (await filled.get("/telemetry")).json()
    assert len(rows) == 1 and rows[0]["key"] is None
    assert _counts(rows[0]) == _direct() == {
        "messages": 4, "replies": 3, "failed_turns": 1, "denied": 1,
        "prompt_tokens": 60, "completion_tokens": 12,
    }  # fmt: skip
    assert rows[0]["latency_p50_ms"] == 300 and rows[0]["latency_p95_ms"] == 900


@pytest.mark.parametrize("group_by, column", [("user", "user_id"), ("agent", "agent_id")])
async def test_each_group_equals_a_direct_sql_count(filled, group_by, column):
    rows = (await filled.get("/telemetry", params={"group_by": group_by})).json()
    keys = [r[0] for r in _sql(f"select distinct {column} from action_logs order by 1")]
    assert [r["key"] for r in rows] == keys
    for row in rows:
        assert _counts(row) == _direct(f"{column} = {row['key']}"), row


async def test_per_model_counts_the_replies_of_each_model(filled):
    rows = {
        r["key"]: r for r in (await filled.get("/telemetry", params={"group_by": "model"})).json()
    }
    assert rows["a.gguf"]["replies"] == 2 and rows["a.gguf"]["latency_p50_ms"] == 100
    assert rows["a.gguf"]["latency_p95_ms"] == 300 and rows["b.gguf"]["prompt_tokens"] == 30
    assert rows[None]["messages"] == 4 and rows[None]["failed_turns"] == 1


async def test_the_period_includes_since_and_excludes_until(filled):
    rows = (
        await filled.get(
            "/telemetry", params={"since": "2026-09-11T00:00:00Z", "until": "2026-09-20T00:00:00Z"}
        )
    ).json()
    where = "created_at >= '2026-09-11' and created_at < '2026-09-20'"
    assert _counts(rows[0]) == _direct(where) and rows[0]["messages"] == 2


async def test_a_bad_group_and_the_read_scope(filled):
    from app.api.app import app
    from app.api.scopes import Principal, Scope, get_principal

    assert (await filled.get("/telemetry", params={"group_by": "channel"})).status_code == 422
    app.dependency_overrides[get_principal] = lambda: Principal("api", Scope.READ)
    try:
        assert (await filled.get("/telemetry")).status_code == 200
    finally:
        app.dependency_overrides.clear()


async def test_no_message_text_is_read(filled, monkeypatch):
    """Telemetry needs no text: an undecryptable text changes nothing and is never decrypted."""
    from app.db import types

    calls = []
    real = types.decrypt_value

    def spy(value):
        calls.append(value)
        return real(value)

    monkeypatch.setattr(types, "decrypt_value", spy)  # where the column type decrypts
    assert (await filled.get("/telemetry", params={"group_by": "agent"})).status_code == 200
    assert calls == []
    assert (await filled.get("/logs", params={"limit": 1})).status_code == 200
    assert calls, "the spy sees a read that does decrypt"


def test_the_percentile_is_the_nearest_rank():
    assert service._percentile([], 0.5) is None
    assert service._percentile([5], 0.95) == 5
    assert service._percentile([1, 2, 3, 4], 0.5) == 2
    assert service._percentile(list(range(1, 101)), 0.95) == 95


async def test_the_usage_of_every_completion_of_a_turn_is_added_up():
    """A turn with tool rounds or a summary makes several completions: all are counted."""
    from app import graph

    stats = {"model": None, "prompt_tokens": 0, "completion_tokens": 0}
    graph._turn_stats.set(stats)
    request = httpx.Request("POST", "http://engine/v1/chat/completions")
    for body in (
        {"model": "m.gguf", "usage": {"prompt_tokens": 7, "completion_tokens": 2}},
        {"model": "m.gguf", "usage": {"prompt_tokens": 9, "completion_tokens": 4}},
    ):
        await graph._count_usage(httpx.Response(200, json=body, request=request))
    path_model = {"model": "/private/dir/models/p.gguf", "usage": {"completion_tokens": 1}}
    await graph._count_usage(httpx.Response(200, json=path_model, request=request))
    assert stats["model"] == "p.gguf", "the file name, never the directory"
    stats["model"] = "m.gguf"
    stats["completion_tokens"] -= 1
    other = httpx.Request("POST", "http://engine/tokenize")
    await graph._count_usage(
        httpx.Response(200, json={"usage": {"prompt_tokens": 99}}, request=other)
    )
    assert stats == {"model": "m.gguf", "prompt_tokens": 16, "completion_tokens": 6,
                     "cached_tokens": 0, "prefill_ms": 0.0}  # fmt: skip



def test_the_cache_figures_are_added_up_from_timings_or_usage():
    """Llama-server's `timings.cache_n` and `prompt_ms` (build b10735); the OpenAI field
    `usage.prompt_tokens_details.cached_tokens` when there are no timings."""
    from app import graph

    stats = {"model": None, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
             "prefill_ms": 0.0}  # fmt: skip
    graph._turn_stats.set(stats)
    timings = {"cache_n": 259, "prompt_ms": 298.4}
    graph._add_usage({"usage": {"prompt_tokens": 263}, "timings": timings})
    graph._add_usage({"usage": {"prompt_tokens": 100,
                                "prompt_tokens_details": {"cached_tokens": 60}}})  # fmt: skip
    graph._add_usage({"usage": {"prompt_tokens": 5}})
    assert (stats["prompt_tokens"], stats["cached_tokens"]) == (368, 319)
    assert stats["prefill_ms"] == pytest.approx(298.4)


async def test_a_streamed_completion_keeps_its_timings():
    from app import engine_stream

    chunks = [
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 9},
         "timings": {"cache_n": 7, "prompt_ms": 12.5}},
    ]  # fmt: skip
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"

    def handler(request):
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://e") as c:
        assembled = await engine_stream.complete(c, {"messages": []}, None)
    assert assembled["timings"] == {"cache_n": 7, "prompt_ms": 12.5}


async def test_telemetry_gives_the_reuse_rate_and_the_mean_prefill(filled):
    from app.db.session import session_scope

    async with session_scope() as s:
        # The third reply comes from an engine that reports no cache figures.
        for prompt, cached, prefill in ((100, 80, 200), (300, 270, 400), (50, None, None)):
            s.add(ActionLog(
                user_id=2, agent_id=3, channel=Channel.TELEGRAM, direction=Direction.OUTBOUND,
                status=ActionStatus.OK, text="x", model="b.gguf", latency_ms=10,
                prompt_tokens=prompt, completion_tokens=1, cached_tokens=cached,
                prefill_ms=prefill, created_at=datetime(2026, 9, 25, tzinfo=UTC),
            ))  # fmt: skip
        await s.commit()
    (row,) = (await filled.get("/telemetry", params={"since": "2026-09-25T00:00:00Z"})).json()
    assert (row["cached_tokens"], row["cache_reuse_rate"], row["prefill_mean_ms"]) == (
        350, 0.875, 300,
    )
    (old,) = (await filled.get("/telemetry", params={"until": "2026-09-25T00:00:00Z"})).json()
    assert (old["cached_tokens"], old["cache_reuse_rate"], old["prefill_mean_ms"]) == (
        0, None, None,
    ), "replies without the figures do not count as 0% reuse"
