"""Tests: routing a turn to a model by intent (rules, then a default),
llama-server's own router mode replacing llama-swap. Decision order: the agent's
own model first; only when it has none are the rules, then the default,
consulted. No classifier (D7): rules only.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from app.admin.routing import select_model
from app.api.scopes import Principal, Scope, get_principal
from app.db.models import Channel

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
GOOD = {"Authorization": f"Bearer {KEY}"}


# --- select_model: the pure decision function, no database ---


def test_no_rule_and_no_default_selects_nothing():
    assert select_model(text="hello", rules=[], default_model=None) is None


def test_a_matching_rule_wins_over_the_default():
    rules = [{"match_type": "min_length", "match_value": "10", "model": "big"}]
    assert select_model(text="a short one", rules=rules, default_model="small") == "big"


def test_no_matching_rule_falls_back_to_the_default():
    rules = [{"match_type": "min_length", "match_value": "500", "model": "big"}]
    assert select_model(text="short", rules=rules, default_model="small") == "small"


def test_the_first_matching_rule_wins():
    rules = [
        {"match_type": "command_prefix", "match_value": "/code", "model": "coder"},
        {"match_type": "min_length", "match_value": "1", "model": "catch-all"},
    ]
    assert select_model(text="/code fix this", rules=rules, default_model=None) == "coder"


def test_a_command_prefix_rule_matches_the_start_of_the_message():
    rules = [{"match_type": "command_prefix", "match_value": "/code", "model": "coder"}]
    assert select_model(text="/code fix this", rules=rules, default_model=None) == "coder"
    assert select_model(text="not /code fix this", rules=rules, default_model=None) is None


# --- the API: GET/PUT /routing ---


@pytest.fixture
async def api(fresh_db, monkeypatch):
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:9")
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=GOOD) as c:
        yield c
    app.dependency_overrides.clear()
    deps.reset_failure_state()


async def test_routing_starts_empty(api):
    body = (await api.get("/routing")).json()
    assert body == {
        "default_model": None, "rules": [], "model_ctx_sizes": {},
        "max_tools": 5, "tool_rules": [], "tool_model": None,
    }  # fmt: skip


async def test_put_sets_rules_default_and_ctx_sizes(api):
    payload = {
        "default_model": "big-model",
        "rules": [{"match_type": "min_length", "match_value": 50, "model": "coder-model"}],
        "model_ctx_sizes": {"coder-model": 8192, "big-model": 32768},
    }
    response = await api.put("/routing", json=payload)
    assert response.status_code == 200, response.text
    assert response.json() == {
        "default_model": "big-model",
        "rules": [{"match_type": "min_length", "match_value": "50", "model": "coder-model"}],
        "model_ctx_sizes": {"coder-model": 8192, "big-model": 32768},
        "max_tools": 5,
        "tool_rules": [],
        "tool_model": None,
    }
    assert (await api.get("/routing")).json() == response.json()


async def test_put_replaces_the_whole_table(api):
    first = {"default_model": "a", "rules": [], "model_ctx_sizes": {}}
    await api.put("/routing", json=first)
    second = {
        "default_model": "b",
        "rules": [{"match_type": "command_prefix", "match_value": "/x", "model": "c"}],
        "model_ctx_sizes": {},
    }
    response = await api.put("/routing", json=second)
    assert response.json()["default_model"] == "b"
    assert response.json()["rules"] == [
        {"match_type": "command_prefix", "match_value": "/x", "model": "c"}
    ]


async def test_put_records_an_admin_event(api):
    await api.put("/routing", json={"default_model": "m", "rules": [], "model_ctx_sizes": {}})
    events = (await api.get("/admin-events")).json()
    assert any(e["action"] == "routing.set" for e in events)


@pytest.mark.parametrize(
    "rule",
    [
        {"match_type": "not-a-type", "match_value": "x", "model": "m"},
        {"match_type": "min_length", "match_value": "not-an-int", "model": "m"},
        {"match_type": "min_length", "match_value": -1, "model": "m"},
        {"match_type": "command_prefix", "match_value": "", "model": "m"},
        {"match_type": "min_length", "match_value": 5, "model": ""},
        {"match_type": "min_length", "match_value": 5, "model": "bad name!"},
    ],
)
async def test_an_invalid_rule_is_refused(api, rule):
    response = await api.put(
        "/routing", json={"default_model": None, "rules": [rule], "model_ctx_sizes": {}}
    )
    assert response.status_code == 422, response.text
    assert (await api.get("/routing")).json()["rules"] == []


async def test_an_invalid_model_ctx_size_is_refused(api):
    response = await api.put(
        "/routing",
        json={"default_model": None, "rules": [], "model_ctx_sizes": {"m": -1}},
    )
    assert response.status_code == 422


async def test_more_than_the_rule_limit_is_refused(api):
    from app.admin.routing import MAX_RULES

    too_many = [
        {"match_type": "min_length", "match_value": i + 1, "model": "m"}
        for i in range(MAX_RULES + 1)
    ]
    response = await api.put(
        "/routing", json={"default_model": None, "rules": too_many, "model_ctx_sizes": {}}
    )
    assert response.status_code == 422
    assert (await api.get("/routing")).json()["rules"] == []


def test_the_service_layer_itself_enforces_the_rule_limit():
    """Not just the API schema's own list length: app.admin.routing is the real
    boundary, reachable by anything that calls it directly, not only the API."""
    from app.admin.routing import MAX_RULES, InvalidInputError, _clean_rules

    too_many = [
        {"match_type": "min_length", "match_value": i + 1, "model": "m"}
        for i in range(MAX_RULES + 1)
    ]
    with pytest.raises(InvalidInputError):
        _clean_rules(too_many)


async def test_reading_needs_only_read_scope(api):
    from app.api.app import app

    app.dependency_overrides[get_principal] = lambda: Principal("r", Scope.READ)
    assert (await api.get("/routing")).status_code == 200
    assert (await api.put("/routing", json={})).status_code == 403


# --- a turn's model choice (two messages, two models) ---


class _Recorder(BaseHTTPRequestHandler):
    requests: list = []
    refuse_model: str | None = None

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        type(self).requests.append(body)
        if body.get("model") == type(self).refuse_model:
            data = json.dumps(
                {"error": {"code": 400, "message": "model not found", "type": "invalid_request"}}
            ).encode()
            self.send_response(400)
        else:
            data = json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
            ).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
async def env(fresh_db, monkeypatch):
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    _Recorder.requests = []
    _Recorder.refuse_model = None
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("LLAMA_CTX_SIZE", "2000")
    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=GOOD) as api:
        user = (await api.post("/users", json={"display_name": "Sam"})).json()["id"]
        api.user = user
        yield api
    app.dependency_overrides.clear()
    server.shutdown()
    get_settings.cache_clear()
    deps.reset_failure_state()


async def _agent(api, name, **settings):
    response = await api.post(f"/users/{api.user}/agents", json={"name": name, **settings})
    assert response.status_code == 201, response.text
    return response.json()


async def _turn(agent_id, text="hello"):
    from app.graph import run_turn

    _Recorder.requests.clear()
    await run_turn(Channel.TELEGRAM, "555", agent_id, text)
    return _Recorder.requests[-1]


async def test_two_messages_route_to_different_models_by_rule(env):
    await env.put(
        "/routing",
        json={
            "default_model": "small-model",
            "rules": [{"match_type": "min_length", "match_value": 50, "model": "big-model"}],
            "model_ctx_sizes": {},
        },
    )
    agent = await _agent(env, "routed")
    short = await _turn(agent["id"], "hi")
    long = await _turn(agent["id"], "x" * 60)
    assert short["model"] == "small-model"
    assert long["model"] == "big-model"
    assert short["model"] != long["model"]


async def test_a_command_prefix_rule_picks_its_model(env):
    rule = {"match_type": "command_prefix", "match_value": "/code", "model": "coder-model"}
    await env.put(
        "/routing",
        json={"default_model": "chat-model", "rules": [rule], "model_ctx_sizes": {}},
    )
    agent = await _agent(env, "routed")
    assert (await _turn(agent["id"], "/code fix this bug"))["model"] == "coder-model"
    assert (await _turn(agent["id"], "just chatting"))["model"] == "chat-model"


async def test_the_agents_own_model_wins_over_any_rule(env):
    await env.put(
        "/routing",
        json={
            "default_model": "default-model",
            "rules": [{"match_type": "min_length", "match_value": "1", "model": "rule-model"}],
            "model_ctx_sizes": {},
        },
    )
    agent = await _agent(env, "pinned", model="agents-own-model")
    assert (await _turn(agent["id"], "anything"))["model"] == "agents-own-model"


# --- an unknown model falls back to the default and it is logged ---


async def test_an_unknown_model_falls_back_to_the_default(env, caplog):
    import logging

    _Recorder.refuse_model = "ghost-model"
    await env.put(
        "/routing", json={"default_model": "real-model", "rules": [], "model_ctx_sizes": {}}
    )
    agent = await _agent(env, "ghost-agent", model="ghost-model")
    with caplog.at_level(logging.WARNING, logger="channelagent"):
        request = await _turn(agent["id"], "hello")
    assert request["model"] == "real-model"
    assert len(_Recorder.requests) == 2  # the refused attempt, then the fallback
    assert _Recorder.requests[0]["model"] == "ghost-model"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "ghost-model" in warnings[0].getMessage() and "real-model" in warnings[0].getMessage()


async def test_no_fallback_when_the_default_is_the_one_that_failed(env):
    """Otherwise the same failing request would be sent twice for nothing."""
    _Recorder.refuse_model = "only-model"
    await env.put(
        "/routing", json={"default_model": "only-model", "rules": [], "model_ctx_sizes": {}}
    )
    agent = await _agent(env, "solo", model="only-model")
    from app.graph import run_turn

    _Recorder.requests.clear()
    with pytest.raises(httpx.HTTPStatusError):
        await run_turn(Channel.TELEGRAM, "555", agent["id"], "hello")
    assert len(_Recorder.requests) == 1


# --- each model's history budget uses its own context size ---


async def test_each_models_history_budget_uses_its_own_context_size(env):
    from app.graph import history_token_budget

    await env.put(
        "/routing",
        json={
            "default_model": None,
            "rules": [],
            "model_ctx_sizes": {"small-ctx-model": 400, "big-ctx-model": 4000},
        },
    )
    small = await _agent(env, "small", model="small-ctx-model")
    big = await _agent(env, "big", model="big-ctx-model")

    small_last = None
    big_last = None
    for i in range(20):
        small_last = await _turn(small["id"], f"message {i} " + "q" * 200)
        big_last = await _turn(big["id"], f"message {i} " + "q" * 200)

    small_tokens = count_tokens_approximately(
        [
            (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
            for m in small_last["messages"]
        ]
    )
    big_tokens = count_tokens_approximately(
        [
            (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
            for m in big_last["messages"]
        ]
    )
    assert small_tokens <= history_token_budget(400)
    assert big_tokens <= history_token_budget(4000)
    assert len(small_last["messages"]) < len(big_last["messages"])


async def test_a_configured_ctx_size_above_llama_ctx_size_is_capped(env):
    """model_ctx_sizes only ever narrows LLAMA_CTX_SIZE (the engine's real cap for
    every router-loaded model), never widens it."""
    from app.graph import history_token_budget

    await env.put(
        "/routing",
        json={"default_model": None, "rules": [], "model_ctx_sizes": {"capped-model": 999999}},
    )
    agent = await _agent(env, "capped", model="capped-model")
    # Large enough that an uncapped 999999-token budget would never trim it, while the
    # real cap (LLAMA_CTX_SIZE=2000 from the env fixture, budget 1500) trims it well before.
    for i in range(40):
        last = await _turn(agent["id"], f"message {i} " + "q" * 300)
    tokens = count_tokens_approximately(
        [
            (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
            for m in last["messages"]
        ]
    )
    assert tokens <= history_token_budget(2000)  # LLAMA_CTX_SIZE set by the env fixture


async def test_the_senders_model_for_one_message_wins_over_the_agents(env):
    """`/model <name> <message>`: first in the decision order, for
    that message only."""
    from app.graph import run_turn

    agent = await _agent(env, "pinned", model="agents-own-model")
    _Recorder.requests.clear()
    await run_turn(Channel.TELEGRAM, "555", agent["id"], "hi", model="chosen-model")
    await run_turn(Channel.TELEGRAM, "555", agent["id"], "and now?")
    assert [r["model"] for r in _Recorder.requests] == ["chosen-model", "agents-own-model"]
