"""Tests for the terminal channel: a turn sent through the API as the channel `terminal`,
from the identity bound to the API account; a job followed with GET /chat/{id}; a tool's
question answered in the terminal; nothing runs without a token; the chat client's loop."""

import asyncio
import sqlite3

import httpx
import pytest

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
GOOD = {"Authorization": f"Bearer {KEY}"}


def _rows():
    from app.config import get_settings

    con = sqlite3.connect(get_settings().database_url.split("///", 1)[1])
    try:
        return con.execute(
            "select channel, direction, status from action_logs order by id"
        ).fetchall()
    finally:
        con.close()


@pytest.fixture
async def world(fresh_db, monkeypatch):
    """Sam, bound to the terminal account `owner` with chat; the agent's turn is scripted."""
    from app.admin import service
    from app.api import deps
    from app.config import get_settings
    from app.db.models import Channel, PermissionKind
    from app.db.session import init_db, session_scope

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    async with session_scope() as session:
        user = await service.create_user(session, "Sam")
        await service.create_agent(session, user.id, "default")
        identity = await service.add_channel_identity(session, user.id, Channel.TERMINAL, "owner")
        await service.grant_identity_permission(session, user.id, identity.id, PermissionKind.CHAT)
        await session.commit()
    turns: list[tuple] = []

    async def fake_turn(channel, user_id, agent_id, text, **kwargs):
        turns.append((channel, user_id, text))
        if text.startswith("tool"):
            from app.mcp.confirm import current_confirmer

            confirm = current_confirmer.get()
            answer = await confirm("Run the tool write_note?", float(text.split()[1]))
            return f"the tool was answered {answer}"
        return "hello from the agent"

    monkeypatch.setattr("app.channels.dispatch.run_turn", fake_turn)
    yield turns
    deps.reset_failure_state()
    get_settings.cache_clear()


@pytest.fixture
async def api(world):
    from app.api.app import app

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=GOOD) as c:
        yield c
    app.dependency_overrides.clear()


async def _follow(api, job, until=lambda j: j["status"] in ("done", "failed")):
    for _ in range(200):
        if until(job):
            return job
        await asyncio.sleep(0.02)
        job = (await api.get(f"/chat/{job['id']}")).json()
    raise AssertionError(f"the job did not reach the state: {job}")


async def test_a_turn_from_the_terminal_is_logged_as_the_terminal_channel(api, world):
    started = await api.post("/chat", json={"text": "hi"})
    assert started.status_code == 202
    job = await _follow(api, started.json())
    assert job["status"] == "done"
    assert job["result"]["replies"] == ["hello from the agent"]
    assert job["result"]["outcome"] == "ok"
    assert world == [("terminal", "owner", "hi")]
    assert _rows() == [("terminal", "inbound", "ok"), ("terminal", "outbound", "ok")]


async def test_without_a_token_nothing_runs(api, world):
    for headers in ({"Authorization": ""}, {"Authorization": "Bearer wrong-key-1234567890"}):
        refused = await api.post("/chat", json={"text": "hi"}, headers=headers)
        assert refused.status_code == 401
    garbage = await api.post("/chat", content=b"{not json", headers={"Authorization": ""})
    assert garbage.status_code == 401, "the token is checked before the body is read"
    await asyncio.sleep(0.05)
    assert world == [] and _rows() == []


async def test_an_account_without_a_terminal_identity_is_denied_and_runs_nothing(api, world):
    from sqlalchemy import update

    from app.db.models import ChannelIdentity
    from app.db.session import session_scope

    async with session_scope() as session:  # bound to another account: not this key's
        await session.execute(update(ChannelIdentity).values(external_id="someone-else"))
        await session.commit()
    job = await _follow(api, (await api.post("/chat", json={"text": "hi"})).json())
    assert job["result"]["outcome"] == "denied"
    assert len(job["result"]["replies"]) == 1, "the terminal is told, like Telegram"
    assert world == []


async def test_a_tool_question_is_answered_in_the_terminal(api, world):
    job = (await api.post("/chat", json={"text": "tool 5"})).json()
    asked = await _follow(api, job, until=lambda j: (j["result"] or {}).get("question"))
    assert asked["result"]["question"] == "Run the tool write_note?"
    assert asked["status"] == "running"
    answered = await api.post(f"/chat/{job['id']}/answer", json={"answer": True})
    assert answered.status_code == 200
    done = await _follow(api, job)
    assert done["result"]["replies"] == ["the tool was answered True"]
    again = await api.post(f"/chat/{job['id']}/answer", json={"answer": False})
    assert again.status_code == 409, "nothing waits any more"


async def test_no_answer_in_time_refuses_the_tool(api, world):
    job = await _follow(api, (await api.post("/chat", json={"text": "tool 0.05"})).json())
    assert job["result"]["replies"] == ["the tool was answered None"]


async def test_only_chat_jobs_are_shown_and_answered(api):
    from app.admin.jobs import registry

    async def nothing(job):
        return {"secret": "not a chat"}

    other = registry.start("backup", nothing)
    assert (await api.get(f"/chat/{other.id}")).status_code == 404
    assert (await api.post(f"/chat/{other.id}/answer", json={"answer": True})).status_code == 404
    assert (await api.get("/chat/nope")).status_code == 404


async def test_the_chat_needs_the_operate_scope(api, world):
    from app.api.app import app
    from app.api.scopes import Principal, Scope, get_principal

    app.dependency_overrides[get_principal] = lambda: Principal("r", Scope.READ)
    assert (await api.post("/chat", json={"text": "hi"})).status_code == 403
    app.dependency_overrides[get_principal] = lambda: Principal("o", Scope.OPERATE)
    assert (await api.post("/chat", json={"text": "hi"})).status_code == 202
    await asyncio.sleep(0.1)
    assert world == [("terminal", "owner", "hi")]


async def test_the_account_comes_from_the_key_not_from_a_header(api, world):
    job = (await api.post("/chat", json={"text": "hi"}, headers={"X-Client": "cli:mallory"})).json()
    await _follow(api, job)
    assert world == [("terminal", "owner", "hi")]


# --- the client ---


async def test_the_client_sends_lines_answers_questions_and_prints_replies(api, world):
    from app.admin import chat

    lines = iter(["hi", "tool 5", "/quit"])
    asked: list[str] = []
    out: list[str] = []

    def read(prompt: str) -> str:
        if prompt.startswith("Run the tool"):
            asked.append(prompt)
            return "y"
        return next(lines)

    code = await chat.loop(api, read=read, write=out.append, poll=0.01)
    assert code == 0
    assert asked == ["Run the tool write_note? [y/N] "]
    assert out == ["hello from the agent", "the tool was answered True"]
    assert [t[2] for t in world] == ["hi", "tool 5"]


async def test_the_client_switches_agent_first_and_stops_at_end_of_input(api, world):
    from app.admin import chat

    out: list[str] = []

    def read(prompt: str) -> str:
        raise EOFError

    assert await chat.loop(api, read=read, write=out.append, poll=0.01, agent="default") == 0
    assert [t[2] for t in world] == [], "/agent is a command, not a turn"
    assert out and "default" in out[0]


async def test_the_commands_of_the_other_channels_work_from_the_terminal(api, world):
    async def say(text):
        job = (await api.post("/chat", json={"text": text})).json()
        return (await _follow(api, job))["result"]["replies"][-1]

    assert (await say("/new")).startswith(("New conversation", "This is already a new"))
    assert "task" in (await say("/task")).lower()
    assert "nope" in await say("/agent nope")
    assert world == [], "commands are not agent turns"
    assert all(channel == "terminal" for channel, _d, _s in _rows())


def test_start_sh_has_the_chat_mode():
    from pathlib import Path

    script = (Path(__file__).resolve().parents[1] / "start.sh").read_text()
    assert '--chat) MODE="chat"' in script
    assert "exec python3 -m app.admin.chat" in script
    assert "api|chat|restore|rekey) remote_api && client_only=1" in script
    assert "--chat [--agent NAME]" in script


async def test_the_in_process_client_closes_the_mcp_connections_it_opened(world, monkeypatch):
    """A turn in process may open MCP connections; left to the interpreter's shutdown they
    printed an anyio "cancel scope in a different task" traceback (2026-10-04)."""
    from app.admin.client import open_client
    from app.mcp import manager as mcp_manager

    closed = []

    async def reset():
        closed.append(True)

    monkeypatch.setattr(mcp_manager.manager, "reset", reset)
    async with open_client("inprocess") as client:
        assert client.transport_name == "inprocess"
    assert closed == [True]
