"""Tests: Agent model, admin editability, and per-agent thread
isolation (two agents belonging to the SAME user must not share state
— isolation tests only covered two different users).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


class _CountingMockLlama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        if self.path != "/v1/chat/completions":  # no tokenizer on this mock
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        n = len(body["messages"])
        resp = {"choices": [{"message": {"role": "assistant", "content": f"reply#{n}"}}]}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def mock_llama(monkeypatch):
    server = HTTPServer(("127.0.0.1", 8096), _CountingMockLlama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8096")
    from app import config

    config.get_settings.cache_clear()
    yield
    server.shutdown()
    config.get_settings.cache_clear()


@pytest.mark.asyncio
async def test_two_agents_same_user_have_independent_thread_state(fresh_db, mock_llama):
    from app.admin.service import create_agent
    from app.db.models import Channel, User
    from app.db.session import init_db, session_scope
    from app.graph import run_turn

    await init_db()
    async with session_scope() as session:
        user = User(display_name="Multi-agent test")
        session.add(user)
        await session.flush()
        agent1 = await create_agent(session, user.id, "work")
        agent2 = await create_agent(session, user.id, "personal")
        await session.commit()
        a1_id, a2_id = agent1.id, agent2.id

    r1 = await run_turn(Channel.TELEGRAM, "999", a1_id, "hello agent1")
    assert r1 == "reply#1"
    r2 = await run_turn(Channel.TELEGRAM, "999", a2_id, "hello agent2")
    assert r2 == "reply#1", "a second agent must start with its own empty history"
    r3 = await run_turn(Channel.TELEGRAM, "999", a1_id, "second message to agent1")
    assert r3 == "reply#3", "agent1's history must have accumulated independently of agent2"


@pytest.mark.asyncio
async def test_admin_can_rename_and_deactivate_any_users_agent(fresh_db):
    from app.admin.service import create_agent, rename_agent, set_agent_active
    from app.db.models import User
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as session:
        user = User(display_name="Someone else")
        session.add(user)
        await session.flush()
        agent = await create_agent(session, user.id, "old-name")
        await session.commit()
        agent_id = agent.id

    async with session_scope() as session:
        renamed = await rename_agent(session, agent_id, "new-name")
        assert renamed.name == "new-name"
        deactivated = await set_agent_active(session, agent_id, False)
        assert deactivated.is_active is False
        await session.commit()


@pytest.mark.asyncio
async def test_get_or_create_default_agent_is_idempotent(fresh_db):
    from sqlalchemy import func, select

    from app.admin.service import get_or_create_default_agent
    from app.db.models import Agent, User
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as session:
        user = User(display_name="Single agent")
        session.add(user)
        await session.flush()
        a1 = await get_or_create_default_agent(session, user.id)
        a2 = await get_or_create_default_agent(session, user.id)
        await session.commit()
        assert a1.id == a2.id

    async with session_scope() as session:
        count = (await session.execute(select(func.count()).select_from(Agent))).scalar_one()
    assert count == 1
