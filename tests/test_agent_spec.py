"""Tests: an agent a user creates themselves from a specification. A valid
specification creates one agent and one task with one admin event; each right the user does not
hold (a tool granted only to another agent or not at all, a skill that is not self-service, an
identity of another user, the agent cap) refuses the whole specification and writes nothing;
the check route writes nothing; free text stays out of the event and is encrypted at rest.
"""

import json
import secrets
import sqlite3
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from app.admin import agent_spec
from app.db.models import (
    AdminEvent,
    Agent,
    Channel,
    McpGrant,
    McpServer,
    PermissionKind,
    ScheduledTask,
    Skill,
)

KEY = secrets.token_urlsafe(24)  # a throwaway key, generated at run time
PURPOSE = "Latest AI news for my commute"
TASK_PROMPT = "Give me the five most important AI news of the week, with links."


@pytest.fixture
async def world(fresh_db):
    """Sam (Telegram 111) with a default agent, a grant on every tool of `feeds` for all
    agents, a grant on `web` for the default agent only, a self-service skill and an
    administrator-only skill; Alex (Telegram 222)."""
    from app.admin import service
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as session:
        ids = {}
        for name, external in (("sam", "111"), ("alex", "222")):
            user = await service.create_user(session, name)
            agent = await service.create_agent(session, user.id, "default")
            identity = await service.add_channel_identity(
                session, user.id, Channel.TELEGRAM, external
            )
            await service.grant_identity_permission(
                session, user.id, identity.id, PermissionKind.CHAT
            )
            ids[name], ids[f"{name}_agent"], ids[f"{name}_identity"] = (
                user.id, agent.id, identity.id,
            )  # fmt: skip
        session.add(McpGrant(user_id=ids["sam"], server_name="feeds"))
        session.add(
            McpServer(
                name="feeds", protocol="stdio", builtin_id="time",
                approved_definitions={"read_feed": {"sha256": "a" * 64, "definition": {}}},
            )
        )  # fmt: skip
        session.add(McpGrant(user_id=ids["sam"], agent_id=ids["sam_agent"], server_name="web"))
        session.add(Skill(name="digest", description="Write a digest", body="b", self_service=True))
        session.add(Skill(name="admin-only", description="Private", body="b"))
        await session.commit()
    return ids


def _spec(**changes) -> dict:
    spec = {
        "name": "ai-news",
        "purpose": PURPOSE,
        "system_prompt": "You write short, sourced news digests.",
        "memory_mode": "ondemand",
        "tools": ["mcp__feeds__read_feed"],
        "skills": ["digest"],
        "task_prompt": TASK_PROMPT,
        "schedule_kind": "cron",
        "schedule_expr": "0 9 * * 4",
    }
    spec.update(changes)
    return spec


async def _counts(world) -> tuple[int, int, int]:
    from app.db.session import session_scope

    async with session_scope() as session:

        def count(model):
            return select(func.count()).select_from(model)

        of_sam = count(Agent).where(Agent.user_id == world["sam"])
        agents = (await session.execute(of_sam)).scalar_one()
        tasks = (await session.execute(count(ScheduledTask))).scalar_one()
        grants = (await session.execute(count(McpGrant))).scalar_one()
    return agents, tasks, grants


async def _create(world, spec, now=None):
    from app.db.session import session_scope

    async with session_scope() as session:
        agent, task = await agent_spec.create_agent_from_spec(
            session, world["sam"], spec, actor="test", now=now
        )
        await session.commit()
        return agent.id, task.id if task else None


# --- a valid specification ---


async def test_a_valid_spec_creates_one_agent_one_task_and_one_event(world):
    before = await _counts(world)
    now = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)  # a Monday
    agent_id, task_id = await _create(world, _spec(), now=now)
    after = await _counts(world)
    print(f"agents/tasks/grants before {before}, after {after}")
    assert after == (before[0] + 1, before[1] + 1, before[2])

    from app.db.session import session_scope

    async with session_scope() as session:
        agent = await session.get(Agent, agent_id)
        task = await session.get(ScheduledTask, task_id)
        stmt = select(AdminEvent).where(AdminEvent.action == "agent.create_from_spec")
        events = (await session.execute(stmt)).scalars().all()
    assert (agent.name, agent.purpose, agent.memory_mode) == ("ai-news", PURPOSE, "ondemand")
    assert agent.tools == ["mcp__feeds__read_feed"] and agent.skills == ["digest"]
    assert (task.agent_id, task.channel_identity_id) == (agent_id, world["sam_identity"])
    assert (task.kind, task.expr, task.prompt) == ("cron", "0 9 * * 4", TASK_PROMPT)
    # Thursday 1 October 2026, 09:00 UTC (the user has no timezone).
    assert task.next_run_at.replace(tzinfo=UTC) == datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
    assert len(events) == 1
    details = json.loads(events[0].details)
    assert details["task_id"] == task_id and details["tools"] == ["mcp__feeds__read_feed"]
    text = events[0].details
    for free_text in (PURPOSE, TASK_PROMPT, "sourced news"):
        assert free_text not in text


async def test_a_spec_without_a_schedule_creates_no_task(world):
    before = await _counts(world)
    spec = _spec()
    for field in agent_spec.SCHEDULE_FIELDS:
        spec.pop(field)
    _, task_id = await _create(world, spec)
    assert task_id is None
    assert await _counts(world) == (before[0] + 1, before[1], before[2])


async def test_purpose_is_encrypted_at_rest(world, tmp_path):
    agent_id, _ = await _create(world, _spec())
    con = sqlite3.connect(tmp_path / "test.db")
    (stored,) = con.execute("select purpose from agents where id = ?", (agent_id,)).fetchone()
    con.close()
    assert stored and PURPOSE not in stored


# --- refusals: nothing is written, every reason is given ---


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"tools": ["mcp__web__fetch_page"]}, "not granted to you for all your agents"),
        ({"tools": ["mcp__mail__send"]}, "not granted to you for all your agents"),
        ({"tools": ["load_skill"]}, "not an MCP tool name"),
        ({"skills": ["admin-only"]}, "granted by an administrator only"),
        ({"skills": ["nope"]}, "no skill named"),
        ({"name": "default"}, "already have an agent named"),
        ({"name": "  "}, "must not be empty"),
        ({"schedule_expr": "0 0 31 2 *"}, "schedule:"),
        ({"schedule_kind": None}, "schedule_kind missing"),
        ({"memory_mode": "forever"}, "memory_mode"),
        ({"purpose": "x" * 201}, "purpose is at most 200"),
        ({"surprise": 1}, "unknown fields: surprise"),
    ],
)
async def test_each_refusal_writes_nothing_and_says_why(world, changes, reason):
    before = await _counts(world)
    with pytest.raises(agent_spec.SpecRefusedError, match=reason):
        await _create(world, _spec(**changes))
    assert await _counts(world) == before


async def test_another_users_identity_is_refused(world):
    before = await _counts(world)
    with pytest.raises(agent_spec.SpecRefusedError, match="delivery_identity_id"):
        await _create(world, _spec(delivery_identity_id=world["alex_identity"]))
    assert await _counts(world) == before


async def test_an_identity_without_a_schedule_is_refused(world):
    spec = _spec(delivery_identity_id=world["sam_identity"])
    for field in agent_spec.SCHEDULE_FIELDS:
        spec.pop(field)
    with pytest.raises(agent_spec.SpecRefusedError, match="only with a schedule"):
        await _create(world, spec)


async def test_every_reason_is_listed_not_only_the_first(world):
    from app.db.session import session_scope

    spec = _spec(tools=["mcp__web__fetch_page"], skills=["admin-only"], name="default")
    async with session_scope() as session:
        _, errors = await agent_spec.check_spec(session, world["sam"], spec)
    assert len(errors) == 3, errors


async def test_the_agent_cap_counts_every_agent_of_the_user(world, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("SELF_SERVICE_MAX_AGENTS", "2")
    get_settings.cache_clear()
    await _create(world, _spec(name="second"))  # the default agent is the first
    before = await _counts(world)
    assert before[0] == 2
    with pytest.raises(agent_spec.SpecRefusedError, match="already have 2 agents"):
        await _create(world, _spec(name="third"))
    assert await _counts(world) == before


async def test_a_cap_of_zero_turns_self_service_creation_off(world, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("SELF_SERVICE_MAX_AGENTS", "0")
    get_settings.cache_clear()
    with pytest.raises(agent_spec.SpecRefusedError, match="SELF_SERVICE_MAX_AGENTS=0"):
        await _create(world, _spec())


async def test_the_task_cap_is_kept(world, monkeypatch):
    from app import tasks

    monkeypatch.setattr(tasks, "MAX_TASKS_PER_USER", 0)
    before = await _counts(world)
    with pytest.raises(agent_spec.SpecRefusedError, match="already have 0 tasks"):
        await _create(world, _spec())
    assert await _counts(world) == before


# --- the Admin API ---


@pytest.fixture
async def api(world, monkeypatch):
    import httpx

    from app.api import deps
    from app.api.app import app
    from app.config import get_settings

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    headers = {"Authorization": f"Bearer {KEY}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as c:
        c.ids = world
        yield c
    app.dependency_overrides.clear()


async def test_api_creates_the_agent_and_its_task(api):
    response = await api.post(f"/users/{api.ids['sam']}/agent-specs", json=_spec())
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["agent"]["name"] == "ai-news" and body["agent"]["purpose"] == PURPOSE
    assert isinstance(body["task_id"], int)


async def test_api_refuses_with_422_and_writes_nothing(api):
    before = await _counts(api.ids)
    response = await api.post(
        f"/users/{api.ids['sam']}/agent-specs", json=_spec(tools=["mcp__web__fetch_page"])
    )
    assert response.status_code == 422
    assert "not granted" in response.json()["detail"]
    assert await _counts(api.ids) == before


async def test_api_check_lists_the_reasons_and_writes_nothing(api):
    before = await _counts(api.ids)
    ok = await api.post(f"/users/{api.ids['sam']}/agent-specs/check", json=_spec())
    bad = await api.post(
        f"/users/{api.ids['sam']}/agent-specs/check",
        json=_spec(skills=["admin-only"], tools=["mcp__mail__send"]),
    )
    assert ok.status_code == bad.status_code == 200
    assert ok.json() == {"valid": True, "errors": []}
    assert bad.json()["valid"] is False and len(bad.json()["errors"]) == 2
    assert await _counts(api.ids) == before


async def test_api_unknown_user_is_404(api):
    response = await api.post("/users/999/agent-specs", json=_spec())
    assert response.status_code == 404


async def test_api_routes_need_the_admin_scope(api):
    from app.api.app import app
    from app.api.scopes import Principal, Scope, get_principal

    app.dependency_overrides[get_principal] = lambda: Principal("op", Scope.OPERATE)
    for path in ("agent-specs", "agent-specs/check"):
        response = await api.post(f"/users/{api.ids['sam']}/{path}", json=_spec())
        assert response.status_code == 403


async def test_skill_self_service_flag_through_the_api(api):
    created = await api.post(
        "/skills",
        json={"name": "notes", "description": "d", "body": "b", "self_service": True},
    )
    assert created.status_code == 201 and created.json()["self_service"] is True
    changed = await api.patch("/skills/notes", json={"self_service": False})
    assert changed.status_code == 200 and changed.json()["self_service"] is False
    listed = {s["name"]: s["self_service"] for s in (await api.get("/skills")).json()}
    assert listed == {"digest": True, "admin-only": False, "notes": False}


# --- standing approval ---


async def test_unattended_tools_become_the_tasks_standing_approval(world):
    from app.db.session import session_scope

    _agent_id, task_id = await _create(world, _spec(unattended=["mcp__feeds__read_feed"]))
    async with session_scope() as session:
        task = await session.get(ScheduledTask, task_id)
        assert task.standing_tools == {"mcp__feeds__read_feed": "a" * 64}
        stmt = select(AdminEvent).where(AdminEvent.action == "agent.create_from_spec")
        event = (await session.execute(stmt)).scalar_one()
    assert json.loads(event.details)["standing_tools"] == ["mcp__feeds__read_feed"]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"unattended": ["mcp__web__fetch_page"]}, "is not one of the agent's tools"),
        (
            {"tools": ["mcp__feeds__read_feed", "mcp__feeds__other"],
             "unattended": ["mcp__feeds__other"]},
            "has no approved definition",
        ),
        ({"unattended": "mcp__feeds__read_feed"}, "unattended: a list of tool names"),
    ],
)  # fmt: skip
async def test_unattended_refusals_write_nothing(world, changes, reason):
    before = await _counts(world)
    with pytest.raises(agent_spec.SpecRefusedError, match=reason):
        await _create(world, _spec(**changes))
    assert await _counts(world) == before


async def test_unattended_needs_a_schedule(world):
    spec = _spec(unattended=["mcp__feeds__read_feed"])
    for field in agent_spec.SCHEDULE_FIELDS:
        spec.pop(field)
    with pytest.raises(agent_spec.SpecRefusedError, match="unattended: only with a schedule"):
        await _create(world, spec)


async def test_the_builder_cannot_schedule_under_the_floor(world):
    from app.db.session import session_scope

    async with session_scope() as session:
        _, errors = await agent_spec.check_spec(
            session, world["sam"], _spec(schedule_kind="every", schedule_expr="5m")
        )
    assert any(e.startswith("schedule: a task you schedule yourself runs at most") for e in errors)
