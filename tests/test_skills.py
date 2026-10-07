"""Tests: local skills (decision M3) and MCP prompts as the `/prompt` command.

- The skill routes: create, read (the list without bodies), change (a new description or body
  is a new version, the old one kept), delete, import a `SKILL.md` folder of SKILLS_DIR (a
  path, a symbolic link or a name that is not the folder's is refused), grant to an agent;
  each change is one admin event; a scope below admin gets 403.
- A turn: only the names and descriptions of the granted skills are in the prompt (0 bodies),
  a body enters the conversation only through `load_skill`, and a skill not granted is neither
  listed nor loadable.
- 100 skills at the maximum description length stay under a stated size.
- `/prompt`: lists the prompts of the servers the user may reach through the agent (the real
  built-in time server), runs one with its arguments as a normal turn, refuses the others.
"""

import json
import sqlite3
import threading
from http.server import HTTPServer

import httpx
import pytest

from app import tools as tools_module
from app.admin import service, skills
from app.channels import dispatch
from app.channels.dispatch import DispatchOutcome, handle_prompt_command
from app.channels.schema import NormalizedEvent
from app.db.models import Channel, PermissionKind
from tests.test_mcp_graph_wiring import _message, _Scripted

KEY = "Sk8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
AUTH = {"Authorization": f"Bearer {KEY}"}
ALPHA_BODY = "ALPHA-BODY: always answer in exactly three words."
BETA_BODY = "BETA-BODY: never shown to this agent."


def _sql(query, *args):
    from app.config import get_settings

    con = sqlite3.connect(get_settings().database_url.split("///", 1)[1])
    try:
        return con.execute(query, args).fetchall()
    finally:
        con.close()


def _events(action):
    return _sql("select count(*) from admin_events where action = ?", action)[0][0]


@pytest.fixture
async def api(fresh_db, monkeypatch, tmp_path):
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
    monkeypatch.setenv("SKILLS_DIR", str(tmp_path / "skills"))
    (tmp_path / "skills").mkdir()
    get_settings.cache_clear()
    deps.reset_failure_state()
    tools_module.reset_capability_check()
    await init_db()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=AUTH) as c:
        c.skills_dir = tmp_path / "skills"
        c.user = (await c.post("/users", json={"display_name": "Sam"})).json()["id"]
        yield c
    from app.mcp.manager import manager as shared_manager

    await shared_manager.reset()
    app.dependency_overrides.clear()
    server.shutdown()
    get_settings.cache_clear()
    deps.reset_failure_state()
    tools_module.reset_capability_check()


async def _skill(api, name, body, description=None):
    response = await api.post(
        "/skills", json={"name": name, "description": description or f"{name} helper", "body": body}
    )
    assert response.status_code == 201, response.text
    return response.json()


# --- the routes ---


async def test_create_read_change_and_version_a_skill(api):
    created = await _skill(api, "alpha", ALPHA_BODY)
    assert (created["version"], created["source"], created["body"]) == (1, "api", ALPHA_BODY)
    assert (await api.post("/skills", json={"name": "alpha", "description": "x", "body": "y"})
            ).status_code == 409  # fmt: skip
    listed = (await api.get("/skills")).json()
    assert [s["name"] for s in listed] == ["alpha"] and "body" not in listed[0]

    only_tools = await api.patch("/skills/alpha", json={"tools": ["mcp__time__get_time"]})
    assert only_tools.json()["version"] == 1, "a change of tools alone is not a new version"
    changed = await api.patch("/skills/alpha", json={"body": "ALPHA-BODY v2"})
    assert changed.json()["version"] == 2 and changed.json()["body"] == "ALPHA-BODY v2"
    history = (await api.get("/skills/alpha/versions")).json()
    assert [(v["version"], v["body"]) for v in history] == [(1, ALPHA_BODY), (2, "ALPHA-BODY v2")]
    assert _events("skill.create") == 1 and _events("skill.update") == 2


async def test_bad_input_and_unknown_names(api):
    for body in (
        {"name": "Bad Name", "description": "d", "body": "b"},
        {"name": "ok", "description": "d" * 121, "body": "b"},
        {"name": "ok", "description": "d", "body": ""},
        {"name": "ok", "description": "d", "body": "b", "extra": 1},
    ):
        assert (await api.post("/skills", json=body)).status_code == 422, body
    assert (await api.get("/skills/nope")).status_code == 404
    assert (await api.patch("/skills/nope", json={"body": "x"})).status_code == 404
    assert (await api.delete("/skills/nope")).status_code == 404
    assert _sql("select count(*) from skills")[0][0] == 0


async def test_delete_removes_the_versions_and_every_grant(api):
    await _skill(api, "alpha", ALPHA_BODY)
    await api.patch("/skills/alpha", json={"body": "v2"})
    agent = (await api.post(f"/users/{api.user}/agents", json={"name": "a"})).json()["id"]
    assert (await api.put(f"/agents/{agent}/skills", json={"skills": ["alpha"]})).json() == {
        "agent_id": agent, "skills": ["alpha"]
    }  # fmt: skip
    assert (await api.delete("/skills/alpha")).status_code == 204
    assert _sql("select count(*) from skill_versions")[0][0] == 0
    assert (await api.get(f"/agents/{agent}")).json()["skills"] == []
    assert _events("skill.delete") == 1


async def test_granting_to_an_agent(api):
    await _skill(api, "alpha", ALPHA_BODY)
    agent = (await api.post(f"/users/{api.user}/agents", json={"name": "a"})).json()["id"]
    assert (await api.get(f"/agents/{agent}")).json()["skills"] == [], "none by default"
    unknown = await api.put(f"/agents/{agent}/skills", json={"skills": ["alpha", "ghost"]})
    assert unknown.status_code == 422 and "ghost" in unknown.text
    assert (await api.put("/agents/99/skills", json={"skills": []})).status_code == 404
    await api.put(f"/agents/{agent}/skills", json={"skills": ["alpha", "alpha"]})
    assert (await api.get(f"/agents/{agent}")).json()["skills"] == ["alpha"]
    assert _events("agent.skills") == 1


async def test_a_scope_below_admin_cannot_change_skills(api):
    from app.api.app import app
    from app.api.scopes import Principal, Scope, get_principal

    await _skill(api, "alpha", ALPHA_BODY)
    app.dependency_overrides[get_principal] = lambda: Principal("api", Scope.OPERATE)
    try:
        assert (await api.get("/skills/alpha")).status_code == 200
        assert (await api.post("/skills", json={"name": "b", "description": "d", "body": "b"})
                ).status_code == 403  # fmt: skip
        assert (await api.patch("/skills/alpha", json={"body": "x"})).status_code == 403
        assert (await api.delete("/skills/alpha")).status_code == 403
        assert (await api.post("/skills/import", json={"folder": "alpha"})).status_code == 403
        assert (await api.put("/agents/1/skills", json={"skills": []})).status_code == 403
    finally:
        app.dependency_overrides.clear()
    assert (await api.get("/skills/alpha")).json()["body"] == ALPHA_BODY


# --- import of a SKILL.md folder ---


def _write(api, folder, name=None, body="Step one.\nStep two.", description="Does things"):
    path = api.skills_dir / folder
    path.mkdir(exist_ok=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name or folder}\ndescription: {description}\ntools: [mcp__time__get_time]\n"
        f"---\n{body}\n"
    )
    return path


async def test_import_creates_then_versions_a_folder(api):
    _write(api, "reminder")
    first = await api.post("/skills/import", json={"folder": "reminder"})
    assert first.status_code == 200, first.text
    assert (first.json()["version"], first.json()["source"], first.json()["tools"]) == (
        1, "folder:reminder", ["mcp__time__get_time"]
    )  # fmt: skip
    assert first.json()["body"] == "Step one.\nStep two."
    _write(api, "reminder", body="Step one, revised.")
    second = (await api.post("/skills/import", json={"folder": "reminder"})).json()
    assert second["version"] == 2 and second["body"] == "Step one, revised."
    again = (await api.post("/skills/import", json={"folder": "reminder"})).json()
    assert again["version"] == 2, "the same text is not a new version"


async def test_import_refuses_paths_links_and_mismatched_names(api, tmp_path):
    for folder in ("../etc", "a/b", "/abs", ".."):
        assert (await api.post("/skills/import", json={"folder": folder})).status_code == 422
    assert (await api.post("/skills/import", json={"folder": "missing"})).status_code == 404
    _write(api, "named", name="other")
    mismatch = await api.post("/skills/import", json={"folder": "named"})
    assert mismatch.status_code == 422 and "folder" in mismatch.text
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text("---\nname: linked\ndescription: d\n---\nsecret\n")
    (api.skills_dir / "linked").symlink_to(outside)
    assert (await api.post("/skills/import", json={"folder": "linked"})).status_code == 404
    (api.skills_dir / "filelink").mkdir()
    (api.skills_dir / "filelink" / "SKILL.md").symlink_to(outside / "SKILL.md")
    assert (await api.post("/skills/import", json={"folder": "filelink"})).status_code == 404
    (api.skills_dir / "broken").mkdir()
    (api.skills_dir / "broken" / "SKILL.md").write_text("no front matter")
    assert (await api.post("/skills/import", json={"folder": "broken"})).status_code == 422
    assert _sql("select count(*) from skills")[0][0] == 0


def test_parse_skill_md_rejects_bad_yaml():
    from app.admin.service import InvalidInputError

    with pytest.raises(InvalidInputError):
        skills.parse_skill_md("---\nname: [unclosed\n---\nbody\n")
    with pytest.raises(InvalidInputError):
        skills.parse_skill_md("---\n- a list\n---\nbody\n")


# --- what an agent sees ---


def _system(request) -> str:
    return next((m["content"] for m in request["messages"] if m["role"] == "system"), "")


def _turn_requests():
    return [r for r in _Scripted.requests if r.get("tool_choice") is None]


def _load(call_id, name):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": skills.LOAD_TOOL, "arguments": json.dumps({"name": name})},
    }


async def _agent_with(api, granted):
    await _skill(api, "alpha", ALPHA_BODY, "Answers briefly")
    await _skill(api, "beta", BETA_BODY, "Something else")
    agent = (await api.post(f"/users/{api.user}/agents", json={"name": "a"})).json()["id"]
    await api.put(f"/agents/{agent}/skills", json={"skills": granted})
    return agent


async def test_the_prompt_lists_granted_skills_and_holds_no_body(api):
    from app.graph import run_turn

    agent = await _agent_with(api, ["alpha"])
    _Scripted.responses = [_message(content="done")]
    assert await run_turn(Channel.TELEGRAM, "555", agent, "hello") == "done"
    (request,) = _turn_requests()
    system = _system(request)
    assert "- alpha: Answers briefly" in system and skills.INDEX_HEADER in system
    assert "beta" not in system, "a skill not granted is not listed"
    everything = json.dumps(request)
    assert "ALPHA-BODY" not in everything and "BETA-BODY" not in everything, "0 bodies"
    assert [t["function"]["name"] for t in request["tools"]] == [skills.LOAD_TOOL]


async def test_a_body_enters_only_through_load_skill_and_only_if_granted(api):
    from app.graph import run_turn

    agent = await _agent_with(api, ["alpha"])
    _Scripted.responses = [
        _message(tool_calls=[_load("c1", "beta")]),
        _message(tool_calls=[_load("c2", "alpha")]),
        _message(content="Three words here."),
    ]
    assert await run_turn(Channel.TELEGRAM, "555", agent, "hi") == "Three words here."
    first, second, third = _turn_requests()
    assert "ALPHA-BODY" not in json.dumps(first)
    refused = [m["content"] for m in second["messages"] if m["role"] == "tool"]
    assert len(refused) == 1 and "no skill named 'beta'" in refused[0]
    assert "BETA-BODY" not in json.dumps(third), "a skill not granted is not loadable"
    loaded = [m["content"] for m in third["messages"] if m["role"] == "tool"]
    assert any(ALPHA_BODY in text for text in loaded)


async def test_an_agent_without_skills_gets_no_index_and_no_tool(api):
    from app.graph import run_turn

    agent = await _agent_with(api, [])
    _Scripted.responses = [_message(content="plain")]
    assert await run_turn(Channel.TELEGRAM, "555", agent, "hello") == "plain"
    (request,) = _turn_requests()
    assert skills.INDEX_HEADER not in _system(request) and "tools" not in request


async def test_a_grant_withdrawn_mid_conversation_is_read_again_at_the_call(api):
    executor = skills.make_executor(await _agent_with(api, ["alpha"]))
    assert ALPHA_BODY in await executor(skills.LOAD_TOOL, '{"name": "alpha"}')
    await api.put("/agents/1/skills", json={"skills": []})
    assert "no skill named" in await executor(skills.LOAD_TOOL, '{"name": "alpha"}')
    assert "not JSON" in await executor(skills.LOAD_TOOL, "{")
    assert "no skill named" in await executor(skills.LOAD_TOOL, '{"name": 3}')


INDEX_MAX_CHARS = 17_000


def test_100_skills_at_the_longest_description_stay_small():
    """The token count is measured with the real engine's tokenizer
    separately; this bounds the text that N is measured on."""
    entries = [(f"skill-{i:03d}-" + "x" * 30, "d" * skills.MAX_DESCRIPTION) for i in range(100)]
    text = skills.index(entries)
    assert text.count("\n- ") == 100 and len(text) <= INDEX_MAX_CHARS, len(text)
    assert skills.index([]) == ""


# --- /prompt ---


@pytest.fixture
async def prompt_world(api, monkeypatch):
    """The real built-in time server, approved, granted to Sam; Sam's Telegram "555" may chat;
    agent "helper" uses its get_time tool."""
    from app.db.session import session_scope

    server = (
        await api.post(
            "/mcp/servers", json={"name": "time", "protocol": "stdio", "builtin_id": "time"}
        )
    ).json()["id"]
    await api.post(f"/mcp/servers/{server}/approve-definitions", json={})
    await api.put("/mcp/grants", json={"user_id": api.user, "grants": [{"server_name": "time"}]})
    agent = await api.post(
        f"/users/{api.user}/agents", json={"name": "helper", "tools": ["mcp__time__get_time"]}
    )
    async with session_scope() as s:
        identity = await service.add_channel_identity(s, api.user, Channel.TELEGRAM, "555")
        await service.grant_identity_permission(s, api.user, identity.id, PermissionKind.CHAT)
        identity.active_agent_id = agent.json()["id"]
        await s.commit()
    turns: list[str] = []

    async def fake_turn(channel, user_id, agent_id, text, **kwargs):
        turns.append(text)
        return "an answer"

    monkeypatch.setattr(dispatch, "run_turn", fake_turn)
    api.turns, api.server_id, api.agent = turns, server, agent.json()["id"]
    return api


async def _prompt(name, arguments=""):
    from app.db.session import session_scope

    replies: list[str] = []

    async def reply(text):
        replies.append(text)

    event = NormalizedEvent(user_id="555", channel=Channel.TELEGRAM, text="/prompt", reply=reply)
    async with session_scope() as s:
        outcome = await handle_prompt_command(s, event, name, arguments)
    return outcome, replies


async def test_prompt_alone_lists_the_servers_prompts(prompt_world):
    outcome, replies = await _prompt("")
    assert outcome == DispatchOutcome.OK and prompt_world.turns == []
    assert replies == [
        "Prompts:\n/prompt time-in — Ask what time it is in a city, using the time tool."
    ]


async def test_a_prompt_runs_as_a_normal_turn_with_its_arguments(prompt_world):
    outcome, replies = await _prompt("time-in", "city='New York'")
    assert outcome == DispatchOutcome.OK and replies == ["an answer"]
    (text,) = prompt_world.turns
    assert text.startswith("What time is it now in New York?")
    await _prompt("time/time-in", "city=Paris")
    assert "Paris" in prompt_world.turns[-1], "server/name is accepted"


async def test_a_prompt_with_bad_arguments_or_an_unknown_name_is_not_run(prompt_world):
    for name, arguments, expected in (
        ("time-in", "", "Usage: /prompt time-in city=..."),
        ("time-in", "Paris", "Usage: /prompt time-in city=..."),
        ("time-in", "city='unclosed", "Usage: /prompt time-in city=..."),
        ("ghost", "", "No prompt named 'ghost'."),
    ):
        outcome, replies = await _prompt(name, arguments)
        assert outcome == DispatchOutcome.OK and replies[0].startswith(expected), replies
    assert prompt_world.turns == []


async def test_no_prompt_without_the_grant_the_tool_or_the_approval(prompt_world):
    api = prompt_world
    await api.put("/mcp/grants", json={"user_id": api.user, "grants": []})
    assert (await _prompt(""))[1] == ["No prompt is available to you."]
    assert (await _prompt("time-in", "city=Paris"))[1][0].startswith("No prompt named")
    await api.put("/mcp/grants", json={"user_id": api.user, "grants": [{"server_name": "time"}]})
    await api.patch(f"/agents/{api.agent}", json={"tools": []})
    assert (await _prompt(""))[1] == ["No prompt is available to you."]
    await api.patch(f"/agents/{api.agent}", json={"tools": ["mcp__other__tool"]})
    assert (await _prompt(""))[1] == ["No prompt is available to you."], "another server's tool"
    await api.patch(f"/agents/{api.agent}", json={"tools": ["mcp__time__get_time"]})
    await api.patch(f"/mcp/servers/{api.server_id}", json={"enabled": False})
    assert (await _prompt(""))[1] == ["No prompt is available to you."]
    assert api.turns == []


async def test_an_unauthorized_sender_gets_the_normal_denial(prompt_world):
    from app.db.session import session_scope

    replies: list[str] = []

    async def reply(text):
        replies.append(text)

    event = NormalizedEvent(user_id="999", channel=Channel.TELEGRAM, text="/prompt", reply=reply)
    async with session_scope() as s:
        await handle_prompt_command(s, event, "time-in", "city=Paris")
    assert replies == [dispatch.DENIED_MESSAGE] and prompt_world.turns == []


async def test_the_ui_creates_versions_and_grants_skills(api):
    """"Admin UI": create, version and grant skills from the UI (a client of the API)."""
    import re

    from app.server import root
    from app.ui import security

    security.sessions.clear()
    security.login_limiter.clear()
    agent = (await api.post(f"/users/{api.user}/agents", json={"name": "a"})).json()["id"]
    transport = httpx.ASGITransport(app=root, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="https://t") as ui:
        page = await ui.get("/ui/login")
        csrf = re.search(r'name="csrf" value="([^"]+)"', page.text).group(1)
        await ui.post("/ui/login", data={"csrf": csrf, "key": KEY})
        home = (await ui.get("/ui/")).text
        csrf = re.search(r'name="csrf" value="([^"]+)"', home).group(1)
        assert 'href="/ui/op/list-skills"' in home
        created = await ui.post(
            "/ui/op/create-skill",
            data={"csrf": csrf, "name": "alpha", "description": "Answers briefly",
                  "body": ALPHA_BODY, "tools": "[]"},
        )  # fmt: skip
        assert created.status_code in (200, 303), created.text
        await ui.post("/ui/op/update-skill", data={"csrf": csrf, "name": "alpha", "body": "v2"})
        await ui.post(
            "/ui/op/set-agent-skills",
            data={"csrf": csrf, "agent_id": str(agent), "skills": '["alpha"]'},
        )
        listing = (await ui.get("/ui/op/list-skills")).text
        assert 'href="/ui/op/list-skill-versions?name=alpha"' in listing
        assert "/ui/op/set-agent-skills?agent_id=" in (
            await ui.get("/ui/op/list-agents", params={"user_id": api.user})
        ).text
    security.sessions.clear()
    assert [v["version"] for v in (await api.get("/skills/alpha/versions")).json()] == [1, 2]
    assert (await api.get(f"/agents/{agent}")).json()["skills"] == ["alpha"]
    assert _sql("select actor from admin_events where action = 'agent.skills'") == [("ui:web",)]
