"""End-to-end test: a turn whose agent is allowed one MCP
tool calls the real built-in "time" server and answers from the result, with
exactly one McpCall row recorded. The chat model is mocked (a real HTTPServer, per
this repo's test conventions); the MCP server is the real stdio subprocess.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app import tools as tools_module
from app.db.models import Channel, McpCall


class _Scripted(BaseHTTPRequestHandler):
    """Replies to /v1/chat/completions from a queue; the probe request (
    tool_choice=required) is answered separately, matched by that field, so the
    queue only needs the turn's own rounds in order.
    """

    responses: list = []
    requests: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path != "/v1/chat/completions":  # no tokenizer on this mock (falls back)
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.loads(raw)
        type(self).requests.append(body)
        if body.get("tool_choice") == "required":
            # The capability probe: always answer with a tool call.
            payload = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "probe",
                                    "type": "function",
                                    "function": {"name": "_capability_probe", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ]
            }
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


@pytest.fixture
async def env(fresh_db, monkeypatch):
    import httpx

    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    _Scripted.requests = []
    _Scripted.responses = []
    server = HTTPServer(("127.0.0.1", 0), _Scripted)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("API_SERVER_KEY", "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA")
    get_settings.cache_clear()
    deps.reset_failure_state()
    tools_module.reset_capability_check()
    await init_db()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    headers = {"Authorization": "Bearer Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"}
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as api:
        user = (await api.post("/users", json={"display_name": "Sam"})).json()["id"]
        api.user = user
        yield api
    from app.mcp.manager import manager as shared_manager

    await shared_manager.reset()
    app.dependency_overrides.clear()
    server.shutdown()
    get_settings.cache_clear()
    deps.reset_failure_state()
    tools_module.reset_capability_check()


async def test_a_turn_calls_the_built_in_server_and_answers_from_the_result(env):
    mcp_server = await env.post(
        "/mcp/servers", json={"name": "time", "protocol": "stdio", "builtin_id": "time"}
    )
    assert mcp_server.status_code == 201, mcp_server.text

    agent = await env.post(
        f"/users/{env.user}/agents",
        json={"name": "helper", "tools": ["mcp__time__get_time"]},
    )
    assert agent.status_code == 201, agent.text
    agent_id = agent.json()["id"]
    # Nothing is offered before the definitions are approved and a grant exists.
    server_id = mcp_server.json()["id"]
    approved = await env.post(f"/mcp/servers/{server_id}/approve-definitions", json={})
    assert approved.status_code == 200, approved.text
    granted = await env.put(
        "/mcp/grants", json={"user_id": env.user, "grants": [{"server_name": "time"}]}
    )
    assert granted.status_code == 200, granted.text

    _Scripted.responses = [
        _message(
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "mcp__time__get_time",
                        "arguments": '{"timezone": "UTC"}',
                    },
                }
            ]
        ),
        _message(content="It's a good time to check the clock."),
    ]

    from app.graph import run_turn

    reply = await run_turn(Channel.TELEGRAM, "555", agent_id, "what time is it?")

    assert reply == "It's a good time to check the clock."

    # The tool call really reached the built-in server: the message sent back for
    # the final round carries its real (non-mocked) answer, framed as data.
    final_request = _Scripted.requests[-1]
    tool_message = next(m for m in final_request["messages"] if m.get("role") == "tool")
    assert "untrusted data" in tool_message["content"]
    assert "T" in tool_message["content"]  # the real ISO 8601 datetime from the built-in

    # Exactly one McpCall row for the one call the turn made.
    from sqlalchemy import select

    from app.db.session import session_scope

    async with session_scope() as session:
        rows = list((await session.execute(select(McpCall))).scalars())
    assert len(rows) == 1
    row = rows[0]
    assert (row.agent_id, row.user_id, row.server_name, row.tool_name, row.status) == (
        agent_id,
        env.user,
        "time",
        "get_time",
        "ok",
    )
    assert row.decision == "allowed"  # read-only, closed-world: "allow" by default (M4)


async def test_without_a_grant_the_turn_is_offered_no_tool(env):
    """Default deny: the approved tool of an agent is still not offered to a
    user who holds no grant, so the model is called without tools."""
    server_id = (
        await env.post(
            "/mcp/servers", json={"name": "time", "protocol": "stdio", "builtin_id": "time"}
        )
    ).json()["id"]
    await env.post(f"/mcp/servers/{server_id}/approve-definitions", json={})
    agent = await env.post(
        f"/users/{env.user}/agents", json={"name": "helper", "tools": ["mcp__time__get_time"]}
    )
    _Scripted.responses = [_message(content="no tools here")]

    from app.graph import run_turn

    reply = await run_turn(Channel.TELEGRAM, "555", agent.json()["id"], "what time is it?")

    assert reply == "no tools here"
    turn = [r for r in _Scripted.requests if r.get("tool_choice") is None]
    assert len(turn) == 1 and "tools" not in turn[0]


async def test_an_agent_with_no_tools_never_probes_or_uses_the_loop(env):
    agent = await env.post(f"/users/{env.user}/agents", json={"name": "plain"})
    agent_id = agent.json()["id"]
    _Scripted.responses = [_message(content="plain reply")]

    from app.graph import run_turn

    reply = await run_turn(Channel.TELEGRAM, "555", agent_id, "hello")

    assert reply == "plain reply"
    # No probe request: the tool_choice=required branch in _Scripted was never hit.
    assert all(r.get("tool_choice") is None for r in _Scripted.requests)


# --- memory through a real turn ---


async def test_a_turn_in_always_mode_sees_the_index_and_can_save(env):
    agent = await env.post(f"/users/{env.user}/agents", json={"name": "mem"})
    agent_id = agent.json()["id"]
    await env.patch(f"/agents/{agent_id}", json={"memory_mode": "always"})
    base = f"/users/{env.user}/agents/{agent_id}/memory"
    created = await env.post(base, json={"title": "Pet", "content": "dog Biscuit"})
    assert created.status_code == 201
    _Scripted.responses = [
        _message(
            tool_calls=[
                {
                    "id": "m1",
                    "type": "function",
                    "function": {
                        "name": "memory_add",
                        "arguments": '{"title": "Job", "content": "nurse"}',
                    },
                }
            ]
        ),
        _message(content="Noted."),
    ]

    from app.graph import run_turn

    reply = await run_turn(Channel.TELEGRAM, "555", agent_id, "I work as a nurse.")

    assert reply == "Noted."
    first = [r for r in _Scripted.requests if r.get("tool_choice") is None][0]
    system, latest = first["messages"][0], first["messages"][-1]
    assert system["role"] == "system" and "memory_" in system["content"]
    # The index changes between turns, so it goes with the new message, not the system.
    assert "#1 Pet" not in system["content"]
    assert latest["role"] == "user" and "#1 Pet" in latest["content"]
    assert latest["content"].endswith("I work as a nurse.")
    assert "Biscuit" not in system["content"], "always mode injects the index only"
    names = [t["function"]["name"] for t in first["tools"]]
    assert "memory_add" in names and len(names) == 5
    titles = [e["title"] for e in (await env.get(base)).json()]
    assert sorted(titles) == ["Job", "Pet"]


async def test_a_turn_in_off_mode_has_no_memory(env):
    agent = await env.post(f"/users/{env.user}/agents", json={"name": "plain2"})
    agent_id = agent.json()["id"]
    base = f"/users/{env.user}/agents/{agent_id}/memory"
    await env.post(base, json={"title": "Pet", "content": "dog Biscuit"})
    _Scripted.responses = [_message(content="hi")]

    from app.graph import run_turn

    assert await run_turn(Channel.TELEGRAM, "555", agent_id, "hello") == "hi"
    turn = [r for r in _Scripted.requests if r.get("tool_choice") is None]
    assert len(turn) == 1 and "tools" not in turn[0]
    assert all("Pet" not in str(m.get("content")) for m in turn[0]["messages"])


async def test_a_server_credential_reaches_no_prompt_log_or_stderr(env, caplog, capfd):
    """Epic"no secret in prompts, logs or server stderr capture": a credential
    stored on the real built-in server is passed to its process only; after a turn that calls
    its tool, the secret is found 0 times in what the engine received, in the application's
    logs, in the captured stdout and stderr (the server's stderr included) and in the API."""
    import logging

    secret = "cred-" + "Q7w" * 10
    caplog.set_level(logging.DEBUG)
    server = await env.post("/mcp/servers", json={
        "name": "time", "protocol": "stdio", "builtin_id": "time",
        "env_vars": {"TIME_API_TOKEN": secret}, "shared_credentials": True})  # fmt: skip
    assert server.status_code == 201, server.text
    server_id = server.json()["id"]
    await env.post(f"/mcp/servers/{server_id}/approve-definitions", json={})
    await env.put("/mcp/grants", json={"user_id": env.user, "grants": [{"server_name": "time"}]})
    agent = await env.post(
        f"/users/{env.user}/agents", json={"name": "helper", "tools": ["mcp__time__get_time"]}
    )
    function = {"name": "mcp__time__get_time", "arguments": '{"timezone": "UTC"}'}
    call = {"id": "c1", "type": "function", "function": function}
    _Scripted.responses = [_message(tool_calls=[call]), _message(content="It is noon.")]

    from app.graph import run_turn

    assert await run_turn(Channel.TELEGRAM, "555", agent.json()["id"], "time?") == "It is noon."
    tool_messages = [m for r in _Scripted.requests for m in r["messages"] if m["role"] == "tool"]
    assert tool_messages, "the tool really ran"
    api_text = (await env.get("/mcp/servers")).text + (await env.get("/mcp/calls")).text
    captured = capfd.readouterr()
    places = {
        "engine requests": json.dumps(_Scripted.requests),
        "logs": caplog.text,
        "stdout": captured.out,
        "stderr": captured.err,
        "api": api_text,
    }
    assert {name: text.count(secret) for name, text in places.items()} == dict.fromkeys(places, 0)
