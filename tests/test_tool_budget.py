"""Tests: the tool budget and the model of tool turns.

- `select_tools`: at most `max_tools` (0 = all), the tools of a matching tool rule first, then
  by the words the message shares with a tool's name and description.
- `/routing`: `max_tools` (5 when left out), `tool_rules`, `tool_model`; bad values 422; the
  admin event records them.
- A real turn with 20 tools allowed shows at most T of them, the right one included, and a call
  to a tool left out is refused as not offered; an agent that uses MCP tools is answered by
  `tool_model` unless the agent has its own model.
"""

import json
import sqlite3
import threading
from http.server import HTTPServer

import httpx
import pytest

from app import tools as tools_module
from app.admin.routing import select_tools
from app.db.models import Channel
from app.mcp import catalogue as mcp_catalogue
from app.mcp.policy import Policy
from tests.test_mcp_graph_wiring import _message, _Scripted

KEY = "Tb8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"

# 20 tools, as the benchmark's.
DESCRIPTIONS = {
    "get_time": "Get the current time in a timezone.",
    "get_weather": "Get the current weather in a city.",
    "calculator": "Evaluate an arithmetic expression.",
    "translate": "Translate text into a target language.",
    "search_web": "Search the web for a query.",
    "convert_currency": "Convert an amount from one currency to another.",
    "set_reminder": "Set a reminder for a time.",
    "lookup_word": "Look up the definition of a word.",
    "roll_dice": "Roll a die with a number of sides.",
    "create_calendar_event": "Create a calendar event.",
    "get_stock_price": "Get the current price of a stock ticker.",
    "send_email": "Send an email.",
    "list_files": "List files in a directory.",
    "play_music": "Play a song.",
    "book_flight": "Book a flight.",
    "order_pizza": "Order a pizza.",
    "water_plants": "Water the plants.",
    "turn_on_lights": "Turn on the lights.",
    "lock_door": "Lock a door.",
    "start_timer": "Start a countdown timer.",
}


def _tool(name, server="bench"):
    return {
        "type": "function",
        "function": {
            "name": f"mcp__{server}__{name}",
            "description": DESCRIPTIONS[name],
            "parameters": {"type": "object", "properties": {}},
        },
    }


TOOLS = [_tool(name) for name in DESCRIPTIONS]


def _names(tools):
    return [t["function"]["name"].removeprefix("mcp__bench__") for t in tools]


# --- select_tools ---


def test_under_the_cap_or_without_one_every_tool_is_shown():
    assert select_tools(text="hi", tools=TOOLS[:5], rules=[], max_tools=5) == TOOLS[:5]
    assert select_tools(text="hi", tools=TOOLS, rules=[], max_tools=0) == TOOLS


def test_the_tools_sharing_words_with_the_message_come_first():
    shown = select_tools(text="What's the weather like in Berlin?", tools=TOOLS, rules=[],
                         max_tools=3)  # fmt: skip
    assert len(shown) == 3 and "get_weather" in _names(shown)
    shown = select_tools(text="Convert 100 dollars to euros", tools=TOOLS, rules=[], max_tools=1)
    assert _names(shown) == ["convert_currency"]


def test_a_word_that_begins_another_counts_and_short_words_do_not():
    shown = select_tools(text="Remind me to call my dentist", tools=TOOLS, rules=[], max_tools=1)
    assert _names(shown) == ["set_reminder"]
    shown = select_tools(text="the pizza for dinner", tools=TOOLS, rules=[], max_tools=1)
    assert _names(shown) == ["order_pizza"], "'the' and 'for' in many descriptions do not count"


def test_a_matching_tool_rule_wins_over_the_words():
    rules = [{"keyword": "DICE", "tools": ["mcp__bench__roll_*", "mcp__bench__start_timer"]}]
    shown = select_tools(text="roll the dice for the weather", tools=TOOLS, rules=rules,
                         max_tools=2)  # fmt: skip
    assert _names(shown) == ["roll_dice", "start_timer"], "keyword case-insensitive, * matches"
    shown = select_tools(text="weather in Paris", tools=TOOLS, rules=rules, max_tools=2)
    assert "roll_dice" not in _names(shown), "a rule whose keyword is absent chooses nothing"


def test_the_shown_tools_keep_the_catalogue_order():
    shown = select_tools(text="lock the door and turn on the lights", tools=TOOLS, rules=[],
                         max_tools=2)  # fmt: skip
    assert _names(shown) == ["turn_on_lights", "lock_door"]


def test_narrow_refuses_a_tool_left_out():
    built = mcp_catalogue.Catalogue()
    for tool in TOOLS[:3]:
        name = tool["function"]["name"]
        built.tools.append(tool)
        built.offered[name] = mcp_catalogue.Offered("bench", name.split("__")[-1], Policy.ALLOW)
    mcp_catalogue.narrow(built, TOOLS[:1])
    assert _names(built.tools) == ["get_time"] and list(built.offered) == ["mcp__bench__get_time"]
    assert set(built.withheld) == {"mcp__bench__get_weather", "mcp__bench__calculator"}
    assert all(d == mcp_catalogue.Decision.NOT_OFFERED for d in built.withheld.values())


# --- the /routing fields ---


@pytest.fixture
async def api(fresh_db, monkeypatch):
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    _Scripted.requests = []
    _Scripted.responses = []
    server = HTTPServer(("127.0.0.1", 0), _Scripted)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    tools_module.reset_capability_check()
    await init_db()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    headers = {"Authorization": f"Bearer {KEY}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as c:
        c.user = (await c.post("/users", json={"display_name": "Sam"})).json()["id"]
        yield c
    app.dependency_overrides.clear()
    server.shutdown()
    get_settings.cache_clear()
    deps.reset_failure_state()
    tools_module.reset_capability_check()


def _events():
    from app.config import get_settings

    con = sqlite3.connect(get_settings().database_url.split("///", 1)[1])
    try:
        query = "select count(*) from admin_events where action = 'routing.set'"
        return con.execute(query).fetchone()[0]
    finally:
        con.close()


async def test_the_tool_budget_is_set_and_read_through_routing(api):
    rule = {"keyword": "dice", "tools": ["mcp__bench__roll_dice"]}
    body = {"max_tools": 3, "tool_rules": [rule], "tool_model": "big.gguf"}
    response = await api.put("/routing", json=body)
    assert response.status_code == 200, response.text
    got = (await api.get("/routing")).json()
    assert (got["max_tools"], got["tool_rules"], got["tool_model"]) == (3, [rule], "big.gguf")
    assert (await api.put("/routing", json={})).json()["max_tools"] == 5, "5 when left out"
    assert _events() == 2
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as db:  # the details are encrypted at rest: read them decrypted
        events = await service.search_admin_events(db, action="routing.set")
    details = sorted((json.loads(e.details) for e in events), key=lambda d: d["max_tools"])
    assert [(d["max_tools"], d["tool_rules"], d["tool_model"]) for d in details] == [
        (3, 1, "big.gguf"), (5, 0, None)
    ]  # fmt: skip


async def test_bad_tool_budgets_are_refused(api):
    for body in (
        {"max_tools": 101},
        {"max_tools": -1},
        {"tool_rules": [{"keyword": "x", "tools": []}]},
        {"tool_rules": [{"keyword": "", "tools": ["a"]}]},
        {"tool_rules": [{"keyword": "x", "tools": ["a"], "model": "m"}]},
        {"tool_model": "../escape"},
    ):
        assert (await api.put("/routing", json=body)).status_code == 422, body
    assert _events() == 0


def test_the_service_refuses_what_the_schema_would_let_through():
    from app.admin.routing import _clean_max_tools, _clean_tool_rules
    from app.admin.service import InvalidInputError

    for bad in (True, 1.5, 101):
        with pytest.raises(InvalidInputError):
            _clean_max_tools(bad)
    for bad in ([{"keyword": "x"}], [{"keyword": "x", "tools": "a"}], "x", [1]):
        with pytest.raises(InvalidInputError):
            _clean_tool_rules(bad)


# --- a real turn ---


@pytest.fixture
def twenty_tools(monkeypatch):
    async def fake_build(manager, allowed, grants=frozenset()):
        built = mcp_catalogue.Catalogue()
        for tool in TOOLS:
            name = tool["function"]["name"]
            if name in allowed:
                built.tools.append(tool)
                built.offered[name] = mcp_catalogue.Offered("bench", name.split("__")[-1],
                                                            Policy.ALLOW)  # fmt: skip
        return built

    monkeypatch.setattr(mcp_catalogue, "build_tools", fake_build)


async def _agent(api, **settings):
    names = [t["function"]["name"] for t in TOOLS]
    response = await api.post(f"/users/{api.user}/agents",
                              json={"name": "a", "tools": names, **settings})  # fmt: skip
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _turn_requests():
    return [r for r in _Scripted.requests if r.get("tool_choice") is None]


async def test_a_turn_with_20_tools_shows_at_most_the_budget(api, twenty_tools):
    from app.graph import run_turn

    agent = await _agent(api)
    _Scripted.responses = [_message(content="ok"), _message(content="ok")]
    await run_turn(Channel.TELEGRAM, "555", agent, "What's the weather like in Berlin today?")
    (request,) = _turn_requests()
    shown = _names(request["tools"])
    assert len(shown) == 5 and "get_weather" in shown
    await api.put("/routing", json={"max_tools": 0})
    await run_turn(Channel.TELEGRAM, "555", agent, "What's the weather like in Berlin today?")
    assert len(_turn_requests()[-1]["tools"]) == 20
    assert len(json.dumps(request["tools"])) < len(json.dumps(_turn_requests()[-1]["tools"]))


async def test_a_call_to_a_tool_left_out_is_refused(api, twenty_tools):
    from app.graph import run_turn

    agent = await _agent(api)
    call = {"id": "c1", "type": "function",
            "function": {"name": "mcp__bench__order_pizza", "arguments": "{}"}}  # fmt: skip
    _Scripted.responses = [_message(tool_calls=[call]), _message(content="done")]
    await run_turn(Channel.TELEGRAM, "555", agent, "What's the weather like in Berlin today?")
    first, second = _turn_requests()
    assert "order_pizza" not in _names(first["tools"])
    (answer,) = [m["content"] for m in second["messages"] if m["role"] == "tool"]
    assert "not available to this agent" in answer


async def test_tool_turns_use_the_tool_model_unless_the_agent_has_its_own(api, twenty_tools):
    from app.graph import run_turn

    await api.put("/routing", json={"default_model": "small.gguf", "tool_model": "big.gguf"})
    with_tools = await _agent(api)
    own = (await api.post(f"/users/{api.user}/agents",
                          json={"name": "own", "model": "own.gguf",
                                "tools": ["mcp__bench__get_time"]})).json()["id"]  # fmt: skip
    plain = (await api.post(f"/users/{api.user}/agents", json={"name": "plain"})).json()["id"]
    _Scripted.responses = [_message(content="ok") for _ in range(3)]
    for agent in (with_tools, own, plain):
        await run_turn(Channel.TELEGRAM, "555", agent, "hello")
    assert [r.get("model") for r in _turn_requests()] == ["big.gguf", "own.gguf", "small.gguf"]


async def test_the_ui_sets_the_intent_to_tool_mapping(api):
    """"Admin UI": the tool rules (keyword to tools), the budget and the tool model are set
    from the UI's routing form, a client of the same API."""
    import re

    from app.server import root
    from app.ui import security

    security.sessions.clear()
    security.login_limiter.clear()
    transport = httpx.ASGITransport(app=root, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="https://t") as ui:
        page = await ui.get("/ui/login")
        csrf = re.search(r'name="csrf" value="([^"]+)"', page.text).group(1)
        await ui.post("/ui/login", data={"csrf": csrf, "key": KEY})
        home = (await ui.get("/ui/")).text
        csrf = re.search(r'name="csrf" value="([^"]+)"', home).group(1)
        assert 'href="/ui/op/get-routing"' in home
        rules = json.dumps([{"keyword": "weather", "tools": ["mcp__bench__get_weather"]}])
        sent = await ui.post(
            "/ui/op/set-routing",
            data={"csrf": csrf, "rules": "[]", "model_ctx_sizes": "{}", "max_tools": "3",
                  "tool_rules": rules, "tool_model": "big.gguf"},
        )  # fmt: skip
        assert sent.status_code in (200, 303), sent.text
        shown = (await ui.get("/ui/op/get-routing")).text
        assert "mcp__bench__get_weather" in shown
    security.sessions.clear()
    got = (await api.get("/routing")).json()
    assert (got["max_tools"], got["tool_model"]) == (3, "big.gguf")
    assert got["tool_rules"] == [{"keyword": "weather", "tools": ["mcp__bench__get_weather"]}]
