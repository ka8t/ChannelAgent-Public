"""Tests: the per-agent configuration (system prompt, model, memory mode, tools)."""

import json
import os
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
import pytest

from app.api.scopes import Principal, Scope, get_principal

REPO = Path(__file__).resolve().parent.parent
KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
GOOD = {"Authorization": f"Bearer {KEY}"}
PROMPT = "You are Marie, the assistant of the Dupont bakery. Answer in French."


class _Recorder(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        _Recorder.requests.append(body)
        data = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data.encode())


@pytest.fixture
async def env(fresh_db, monkeypatch):
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    _Recorder.requests = []
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


async def _turn(agent_id, text="Bonjour"):
    from app.db.models import Channel
    from app.graph import run_turn

    _Recorder.requests.clear()
    reply = await run_turn(Channel.TELEGRAM, "555", agent_id, text)
    assert reply == "ok"
    return _Recorder.requests[-1]


# --- what a turn does with the configuration ---


async def test_two_agents_of_one_user_answer_with_different_system_prompts(env):
    marie = await _agent(env, "marie", system_prompt=PROMPT)
    plain = await _agent(env, "plain")
    other = await _agent(env, "other", system_prompt="Answer only with the word OTHER.")
    first = await _turn(marie["id"])
    second = await _turn(plain["id"])
    third = await _turn(other["id"])
    assert first["messages"][0] == {"role": "system", "content": PROMPT}
    assert [m["role"] for m in second["messages"]] == ["user"]
    assert third["messages"][0]["content"] == "Answer only with the word OTHER."
    assert PROMPT not in json.dumps(third)
    print(
        "system prompts seen by the engine:",
        [r["messages"][0]["role"] for r in (first, second, third)],
    )


async def test_the_model_is_sent_only_when_the_agent_has_one(env):
    named = await _agent(env, "named", model="qwen2.5-0.5b-instruct-q4_k_m.gguf")
    default = await _agent(env, "default")
    assert (await _turn(named["id"]))["model"] == "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    assert "model" not in await _turn(default["id"])


async def test_a_change_applies_from_the_next_turn(env):
    agent = await _agent(env, "live")
    assert [m["role"] for m in (await _turn(agent["id"]))["messages"]] == ["user"]
    patched = await env.patch(f"/agents/{agent['id']}", json={"system_prompt": PROMPT})
    assert patched.status_code == 200
    body = await _turn(agent["id"], "Encore")
    assert body["messages"][0] == {"role": "system", "content": PROMPT}
    await env.patch(f"/agents/{agent['id']}", json={"system_prompt": None})
    assert all(m["role"] != "system" for m in (await _turn(agent["id"], "Et encore"))["messages"])


async def test_the_prompt_and_the_summary_share_one_system_message():
    from app.graph import SUMMARY_HEADER, _prefix

    assert _prefix(None, "") == ""
    assert _prefix(PROMPT, "") == PROMPT
    assert _prefix(None, "earlier") == SUMMARY_HEADER + "earlier"
    both = _prefix(PROMPT, "earlier")
    assert both == PROMPT + "\n\n" + SUMMARY_HEADER + "earlier"


async def test_a_long_prompt_takes_room_from_the_history(env):
    long_prompt = "Rule: be brief. " * 60  # about 240 tokens of the 1,500 of the budget
    plain = await _agent(env, "plain")
    strict = await _agent(env, "strict", system_prompt=long_prompt)
    sizes = {}
    for agent in (plain, strict):
        for i in range(60):
            body = await _turn(agent["id"], f"message number {i} " + "word " * 20)
        sizes[agent["name"]] = len([m for m in body["messages"] if m["role"] != "system"])
    print("history messages sent:", sizes)
    assert sizes["strict"] < sizes["plain"]


# --- the settings ---


async def test_defaults_and_a_partial_update_keep_the_rest(env):
    agent = await _agent(env, "a")
    assert (agent["model"], agent["memory_mode"], agent["tools"]) == (None, "off", [])
    assert agent["system_prompt"] is None and agent["has_system_prompt"] is False
    first = await env.patch(
        f"/agents/{agent['id']}",
        json={
            "system_prompt": PROMPT,
            "model": "m.gguf",
            "memory_mode": "always",
            "tools": ["a", "b"],
        },
    )
    assert first.json()["memory_mode"] == "always" and first.json()["tools"] == ["a", "b"]
    renamed = await env.patch(f"/agents/{agent['id']}", json={"name": "renamed"})
    body = renamed.json()
    assert body["name"] == "renamed" and body["system_prompt"] == PROMPT
    assert (body["model"], body["memory_mode"], body["tools"]) == ("m.gguf", "always", ["a", "b"])
    cleared = (
        await env.patch(f"/agents/{agent['id']}", json={"model": None, "system_prompt": ""})
    ).json()
    assert cleared["model"] is None and cleared["system_prompt"] is None
    assert cleared["memory_mode"] == "always"  # not given, not changed


@pytest.mark.parametrize("mode", ["", "sometimes", "OFF", "1"])
async def test_an_invalid_memory_mode_is_refused_with_422(env, mode):
    agent = await _agent(env, "a")
    assert (
        await env.patch(f"/agents/{agent['id']}", json={"memory_mode": mode})
    ).status_code == 422
    created = await env.post(f"/users/{env.user}/agents", json={"name": "b", "memory_mode": mode})
    assert created.status_code == 422
    assert (await env.get(f"/agents/{agent['id']}")).json()["memory_mode"] == "off"


@pytest.mark.parametrize("mode", ["off", "ondemand", "always", "search"])
async def test_every_memory_mode_is_accepted(env, mode):
    agent = await _agent(env, "a", memory_mode=mode)
    assert agent["memory_mode"] == mode


@pytest.mark.parametrize(
    "settings",
    [
        {"tools": ["ok", "bad name"]},
        {"tools": ["semi;colon"]},
        {"tools": [""]},
        {"tools": ["x"] * 101},
        {"tools": "not a list"},
        {"tools": [1]},
        {"model": "bad model!"},
        {"model": "x" * 201},
        {"system_prompt": "x" * 20001},
        {"unknown": "field"},
    ],
)
async def test_bad_settings_are_refused_and_nothing_changes(env, settings):
    agent = await _agent(env, "a", system_prompt="keep", tools=["keep"])
    response = await env.patch(f"/agents/{agent['id']}", json=settings)
    assert response.status_code == 422
    kept = (await env.get(f"/agents/{agent['id']}")).json()
    assert kept["system_prompt"] == "keep" and kept["tools"] == ["keep"]


async def test_duplicate_tools_are_folded_in_order_and_names_are_trimmed(env):
    agent = await _agent(env, "a", tools=["mcp__time__now", "mcp__fs__read", "mcp__time__now"])
    assert agent["tools"] == ["mcp__time__now", "mcp__fs__read"]
    assert (await _agent(env, "b", model="  m.gguf  "))["model"] == "m.gguf"


async def test_a_service_caller_gets_the_same_checks(env):
    from app.admin import service
    from app.db.session import session_scope

    agent = await _agent(env, "a")
    async with session_scope() as session:
        with pytest.raises(service.InvalidInputError):
            await service.configure_agent(session, agent["id"], {"memory_mode": "nope"})
        with pytest.raises(service.InvalidInputError):
            await service.configure_agent(session, agent["id"], {"colour": "red"})
        with pytest.raises(service.InvalidInputError):
            await service.configure_agent(session, agent["id"], {"tools": ["t"] * 101})
        with pytest.raises(service.InvalidInputError):
            await service.configure_agent(session, agent["id"], {"system_prompt": "x" * 20001})
        with pytest.raises(service.NotFoundError):
            await service.configure_agent(session, 9999, {"model": "m.gguf"})


# --- privacy of the prompt ---


async def test_the_prompt_is_encrypted_at_rest_and_never_in_the_checkpoint(env):
    from app.config import get_settings
    from app.graph import close_graph

    agent = await _agent(env, "secret", system_prompt=PROMPT)
    await _turn(agent["id"])
    await close_graph()
    db = get_settings().database_url.split("///", 1)[1]
    con = sqlite3.connect(db)
    try:
        stored = con.execute(
            "select system_prompt from agents where id = ?", (agent["id"],)
        ).fetchone()[0]
    finally:
        con.close()
    assert stored and PROMPT not in stored and "Marie" not in stored
    checkpoint = Path(os.environ["CHECKPOINT_DB_PATH"]).read_bytes()
    print(
        "prompt occurrences: database column 0, checkpoint file "
        f"{checkpoint.count(PROMPT.encode())}"
    )
    assert PROMPT.encode() not in checkpoint and b"Dupont" not in checkpoint


async def test_only_an_administrator_reads_the_prompt_back(env):
    from app.api.app import app

    agent = await _agent(env, "a", system_prompt=PROMPT)
    for scope, sees in (
        (Scope.READ, False),
        (Scope.OPERATE, False),
        (Scope.ADMIN, True),
        (Scope.OWNER, True),
    ):
        app.dependency_overrides[get_principal] = lambda s=scope: Principal("t", s)
        for url in (f"/agents/{agent['id']}", f"/users/{env.user}/agents"):
            body = (await env.get(url)).json()
            body = body if isinstance(body, dict) else body[0]
            assert body["has_system_prompt"] is True
            assert (body["system_prompt"] == PROMPT) is sees, (scope, url)
        if scope >= Scope.OPERATE:
            patched = (await env.patch(f"/agents/{agent['id']}", json={"model": "m.gguf"})).json()
            assert (patched["system_prompt"] == PROMPT) is sees


async def test_the_event_names_the_settings_changed_never_the_prompt(env):
    agent = await _agent(env, "a", system_prompt=PROMPT, memory_mode="search")
    await env.patch(
        f"/agents/{agent['id']}", json={"system_prompt": "Another private prompt", "tools": ["t"]}
    )
    events = (await env.get("/admin-events", params={"target_type": "agent"})).json()
    text = json.dumps(events)
    assert PROMPT not in text and "Another private prompt" not in text
    actions = {e["action"] for e in events}
    assert {"agent.create", "agent.configure"} <= actions
    configure = [e for e in events if e["action"] == "agent.configure"][0]
    assert json.loads(configure["details"])["fields"] == ["system_prompt", "tools"]


# --- the migration ---


def _alembic(db: Path, *args: str):
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "DATABASE_URL": f"sqlite+aiosqlite:///{db}"},
    )


def test_the_migration_gives_existing_agents_the_defaults_and_goes_back(tmp_path):
    db = tmp_path / "old.db"
    assert _alembic(db, "upgrade", "9dbfacca49a4").returncode == 0
    con = sqlite3.connect(db)
    con.execute(
        "insert into users (id, display_name, is_active, created_at) "
        "values (1, 'Old', 1, '2026-01-01')"
    )
    con.execute(
        "insert into agents (id, user_id, name, is_active, created_at) "
        "values (1, 1, 'default', 1, '2026-01-01')"
    )
    con.commit()
    con.close()
    up = _alembic(db, "upgrade", "head")
    assert up.returncode == 0, up.stderr
    con = sqlite3.connect(db)
    row = con.execute(
        "select name, system_prompt, model, memory_mode, tools from agents"
    ).fetchone()
    assert row == ("default", None, None, "off", "[]")
    con.close()
    check = _alembic(db, "check")
    assert check.returncode == 0, check.stdout + check.stderr
    # Target the revision by name, not "-1": added a migration on top of this
    # one, so "one step back" no longer lands before this migration's own columns.
    assert _alembic(db, "downgrade", "9dbfacca49a4").returncode == 0
    con = sqlite3.connect(db)
    columns = [c[1] for c in con.execute("pragma table_info(agents)")]
    assert columns == ["id", "user_id", "name", "is_active", "created_at"]
    assert con.execute("select count(*) from agents").fetchone()[0] == 1
    con.close()
    assert _alembic(db, "upgrade", "head").returncode == 0


def test_the_generated_forms_and_commands_get_the_memory_modes_as_choices():
    from app.admin.manifest import operations
    from app.api.app import app

    for command in ("create-agent", "update-agent"):
        op = next(o for o in operations(app) if o["command"] == command)
        fields = {f["name"]: f for f in op["fields"]}
        assert fields["memory_mode"]["enum"] == ["off", "ondemand", "always", "search"], command
        assert {"system_prompt", "model", "memory_mode", "tools"} <= set(fields), command
