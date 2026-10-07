"""Tests: a user's own agents. Each operation (overview, pause, resume, retire, run,
change from a specification) on the user's own agent changes exactly the expected rows, and on
another user's agent changes 0 rows and answers like a missing one; the same through the Admin
API, the `/myagents` command and the `/editagent` dialogue (scripted model).
"""

import json
import secrets
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from app import builder
from app.admin import agent_spec, my_agents
from app.admin.service import AgentNotFoundError
from app.channels.schema import NormalizedEvent
from app.db.models import (
    AdminEvent,
    Agent,
    Channel,
    McpGrant,
    McpServer,
    PermissionKind,
    ScheduledTask,
    Skill,
    User,
)

KEY = secrets.token_urlsafe(24)  # a throwaway key, generated at run time
MONDAY = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
TOOL = "mcp__feeds__read_feed"


def spec(**changes) -> dict:
    data = {
        "name": "ai-news",
        "purpose": "AI news",
        "system_prompt": "You write short digests.",
        "memory_mode": "off",
        "tools": [TOOL],
        "skills": [],
        "task_prompt": "The AI news of the week.",
        "schedule_kind": "cron",
        "schedule_expr": "0 9 * * 4",
        "unattended": [TOOL],
    }
    data.update(changes)
    return data


@pytest.fixture
async def world(fresh_db):
    """Sam (Telegram 111, Europe/Paris) with agent ai-news and its task; Alex (Telegram 222)
    with agent alexbot and its task. The feeds tool is approved and granted to both."""
    from app.admin import service
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as session:
        ids = {}
        session.add(McpServer(name="feeds", protocol="stdio", builtin_id="feeds",
                              approved_definitions={"read_feed": {"sha256": "h",
                                                                  "definition": {}}}))  # fmt: skip
        for name, external in (("sam", "111"), ("alex", "222")):
            user = await service.create_user(session, name)
            await service.create_agent(session, user.id, "default")
            identity = await service.add_channel_identity(
                session, user.id, Channel.TELEGRAM, external
            )
            await service.grant_identity_permission(
                session, user.id, identity.id, PermissionKind.CHAT
            )
            session.add(McpGrant(user_id=user.id, server_name="feeds"))
            ids[name], ids[f"{name}_identity"] = user.id, identity.id
        (await session.get(User, ids["sam"])).timezone = "Europe/Paris"
        await session.flush()
        for who, agent_name in (("sam", "ai-news"), ("alex", "alexbot")):
            agent, task = await agent_spec.create_agent_from_spec(
                session, ids[who], spec(name=agent_name), actor="test", now=MONDAY
            )
            ids[f"{who}_agent"], ids[f"{who}_task"] = agent.id, task.id
        await session.commit()
    return ids


async def snapshot() -> dict:
    """Every agent and task row, to count what an operation changed."""
    from app.db.session import session_scope

    async with session_scope() as session:
        agents = {a.id: (a.name, a.is_active, a.system_prompt, tuple(a.tools))
                  for a in (await session.execute(select(Agent))).scalars()}  # fmt: skip
        tasks = {t.id: (t.agent_id, t.kind, t.expr, t.enabled, str(t.next_run_at), t.prompt)
                 for t in (await session.execute(select(ScheduledTask))).scalars()}  # fmt: skip
    return {"agents": agents, "tasks": tasks}


def changed(before: dict, after: dict) -> set:
    rows = set()
    for table in ("agents", "tasks"):
        for key in set(before[table]) | set(after[table]):
            if before[table].get(key) != after[table].get(key):
                rows.add((table, key))
    return rows


# --- the service ---


async def run(operation):
    from app.db.session import session_scope

    async with session_scope() as session:
        result = await operation(session)
        await session.commit()
        return result


async def test_overview_lists_the_users_agents_and_tasks(world):
    from app.db.session import session_scope

    async with session_scope() as session:
        rows = await my_agents.overview(session, world["sam"])
    assert [r["name"] for r in rows] == ["default", "ai-news"]
    task = rows[1]["tasks"][0]
    assert (task["task_id"], task["kind"], task["expr"], task["enabled"]) == (
        world["sam_task"], "cron", "0 9 * * 4", True,
    )  # fmt: skip
    assert task["next_run_at"] == datetime(2026, 10, 1, 7, 0, tzinfo=UTC)


async def test_pause_and_resume_change_exactly_the_agents_task(world):
    before = await snapshot()
    ids = await run(lambda s: my_agents.set_paused(s, world["sam"], world["sam_agent"], True,
                                                   actor="t"))  # fmt: skip
    after = await snapshot()
    assert ids == [world["sam_task"]]
    assert changed(before, after) == {("tasks", world["sam_task"])}
    assert after["tasks"][world["sam_task"]][3:5] == (False, "None")
    await run(lambda s: my_agents.set_paused(s, world["sam"], world["sam_agent"], False,
                                             actor="t"))  # fmt: skip
    again = await snapshot()
    assert again["tasks"][world["sam_task"]][3] is True
    assert again["tasks"][world["sam_task"]][4] != "None"


async def test_retire_deletes_the_tasks_and_disables_only_that_agent(world):
    before = await snapshot()
    await run(lambda s: my_agents.retire(s, world["sam"], world["sam_agent"], actor="t"))
    after = await snapshot()
    assert changed(before, after) == {("agents", world["sam_agent"]), ("tasks", world["sam_task"])}
    assert after["agents"][world["sam_agent"]][1] is False
    assert world["sam_task"] not in after["tasks"]


@pytest.mark.parametrize("operation", ["pause", "resume", "retire", "task_ids", "update", "spec"])
async def test_another_users_agent_changes_nothing_and_reads_as_missing(world, operation):
    alex_agent = world["alex_agent"]
    calls = {
        "pause": lambda s: my_agents.set_paused(s, world["sam"], alex_agent, True, actor="t"),
        "resume": lambda s: my_agents.set_paused(s, world["sam"], alex_agent, False, actor="t"),
        "retire": lambda s: my_agents.retire(s, world["sam"], alex_agent, actor="t"),
        "task_ids": lambda s: my_agents.task_ids(s, world["sam"], alex_agent),
        "update": lambda s: agent_spec.update_agent_from_spec(
            s, world["sam"], alex_agent, spec(name="stolen"), actor="t"),
        "spec": lambda s: agent_spec.own_agent(s, world["sam"], alex_agent),
    }  # fmt: skip
    before = await snapshot()
    with pytest.raises(AgentNotFoundError, match=f"No agent {alex_agent}$"):
        await run(calls[operation])
    assert changed(before, await snapshot()) == set()


async def test_update_changes_the_schedule_and_keeps_the_rest(world):
    before = await snapshot()
    agent, task = await run(lambda s: agent_spec.update_agent_from_spec(
        s, world["sam"], world["sam_agent"], spec(schedule_expr="0 8 * * 4"), actor="t",
        now=MONDAY))  # fmt: skip
    after = await snapshot()
    assert changed(before, after) == {("tasks", world["sam_task"])}
    assert task.id == world["sam_task"] and after["tasks"][task.id][2] == "0 8 * * 4"
    assert after["tasks"][task.id][4].startswith("2026-10-01 06:00")  # 08:00 in Paris
    from app.db.session import session_scope

    async with session_scope() as session:
        event = (await session.execute(select(AdminEvent).where(
            AdminEvent.action == "agent.update_from_spec"))).scalar_one()  # fmt: skip
    assert json.loads(event.details)["fields"] == ["schedule_expr"]


async def test_update_without_a_schedule_deletes_the_task_and_with_one_creates_it(world):
    no_schedule = spec(task_prompt=None, schedule_kind=None, schedule_expr=None, unattended=[])
    _agent, task = await run(lambda s: agent_spec.update_agent_from_spec(
        s, world["sam"], world["sam_agent"], no_schedule, actor="t"))  # fmt: skip
    assert task is None and world["sam_task"] not in (await snapshot())["tasks"]
    _agent, task = await run(lambda s: agent_spec.update_agent_from_spec(
        s, world["sam"], world["sam_agent"], spec(), actor="t", now=MONDAY))  # fmt: skip
    assert task is not None and task.id != world["sam_task"]
    assert task.standing_tools == {TOOL: "h"}


async def test_update_keeps_what_an_administrator_set_but_checks_additions(world):
    from app.db.session import session_scope

    async with session_scope() as session:
        agent = await session.get(Agent, world["sam_agent"])
        agent.tools = [TOOL, "mcp__admin__only"]  # set by an administrator
        agent.skills = ["admin-skill"]
        session.add(Skill(name="admin-skill", description="d", body="b"))
        session.add(Skill(name="private", description="d", body="b"))
        await session.commit()
    kept = spec(tools=[TOOL, "mcp__admin__only"], skills=["admin-skill"], system_prompt="New.")
    await run(lambda s: agent_spec.update_agent_from_spec(
        s, world["sam"], world["sam_agent"], kept, actor="t", now=MONDAY))  # fmt: skip
    added = spec(tools=[TOOL, "mcp__other__tool"], skills=["admin-skill", "private"])
    before = await snapshot()
    with pytest.raises(agent_spec.SpecRefusedError) as refused:
        await run(lambda s: agent_spec.update_agent_from_spec(
            s, world["sam"], world["sam_agent"], added, actor="t"))  # fmt: skip
    assert len(refused.value.errors) == 2 and changed(before, await snapshot()) == set()


async def test_update_refuses_another_agents_name_but_keeps_its_own(world):
    await run(lambda s: agent_spec.update_agent_from_spec(
        s, world["sam"], world["sam_agent"], spec(), actor="t", now=MONDAY))  # fmt: skip
    with pytest.raises(agent_spec.SpecRefusedError, match="already have an agent named 'default'"):
        await run(lambda s: agent_spec.update_agent_from_spec(
            s, world["sam"], world["sam_agent"], spec(name="default"), actor="t"))  # fmt: skip


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


async def test_api_own_agent_routes(api, monkeypatch):
    from app import tasks

    ids, sam = api.ids, api.ids["sam"]
    mine = f"/users/{sam}/agents/{ids['sam_agent']}"
    overview = (await api.get(f"/users/{sam}/agent-overview")).json()
    assert [a["name"] for a in overview] == ["default", "ai-news"]
    current = (await api.get(mine + "/spec")).json()
    assert current["schedule_expr"] == "0 9 * * 4" and current["unattended"] == [TOOL]
    changed_spec = await api.put(mine + "/spec", json={**current, "schedule_expr": "0 8 * * 4"})
    assert changed_spec.status_code == 200 and changed_spec.json()["task_id"] == ids["sam_task"]
    paused = (await api.post(mine + "/pause")).json()
    assert paused == {"agent_id": ids["sam_agent"], "changed_task_ids": [ids["sam_task"]]}
    assert (await api.post(mine + "/resume")).json()["changed_task_ids"] == [ids["sam_task"]]

    async def fake_execute(task_id, *, trigger):
        return {"task_id": task_id, "status": "ok"}

    monkeypatch.setattr(tasks, "execute", fake_execute)
    job = await api.post(mine + "/run")
    assert job.status_code == 202
    retired = await api.post(mine + "/retire")
    assert retired.status_code == 200 and retired.json()["is_active"] is False
    no_task = await api.post(mine + "/run")
    assert no_task.status_code == 422 and "no scheduled task" in no_task.json()["detail"]


@pytest.mark.parametrize("method,suffix", [
    ("get", "/spec"), ("put", "/spec"), ("post", "/pause"), ("post", "/resume"),
    ("post", "/retire"), ("post", "/run"),
])  # fmt: skip
async def test_api_another_users_agent_is_404_and_changes_nothing(api, method, suffix):
    path = f"/users/{api.ids['sam']}/agents/{api.ids['alex_agent']}{suffix}"
    before = await snapshot()
    kwargs = {"json": spec(name="x")} if method == "put" else {}
    response = await getattr(api, method)(path, **kwargs)
    assert response.status_code == 404
    assert changed(before, await snapshot()) == set()


async def test_api_scopes_of_the_own_agent_routes(api):
    from app.api.app import app
    from app.api.scopes import Principal, Scope, get_principal

    mine = f"/users/{api.ids['sam']}/agents/{api.ids['sam_agent']}"
    app.dependency_overrides[get_principal] = lambda: Principal("reader", Scope.READ)
    assert (await api.post(mine + "/pause")).status_code == 403
    app.dependency_overrides[get_principal] = lambda: Principal("op", Scope.OPERATE)
    assert (await api.post(mine + "/pause")).status_code == 200
    assert (await api.get(mine + "/spec")).status_code == 403
    assert (await api.get(f"/users/{api.ids['sam']}/agent-overview")).status_code == 403


# --- /myagents ---


class _Sink:
    def __init__(self):
        self.sent: list[str] = []
        self.asked: list[str] = []

    async def reply(self, text):
        self.sent.append(text)


async def say(text, external="111", confirm="none"):
    """`confirm`: True, False or None (no answer) for a channel that asks; "none" for one that
    cannot ask."""
    from app.channels.dispatch import dispatch_event
    from app.db.session import session_scope

    sink = _Sink()

    async def ask(question, timeout):
        sink.asked.append(question)
        return confirm

    event = NormalizedEvent(external, Channel.TELEGRAM, text, sink.reply,
                            confirm=None if confirm == "none" else ask)  # fmt: skip
    async with session_scope() as session:
        await dispatch_event(session, event)
    return sink


async def test_myagents_lists_in_the_users_timezone(world):
    answer = (await say("/myagents")).sent[-1]
    print(answer)
    assert answer.startswith("Your agents (times in Europe/Paris):")
    assert "- ai-news [on]: AI news" in answer
    assert f"task #{world['sam_task']} cron 0 9 * * 4 (on), next Thu 2026-10-01 09:00" in answer
    assert "alexbot" not in answer


async def test_myagents_pause_resume_and_the_name_of_another_users_agent(world):
    assert (await say("/myagents pause ai-news")).sent[-1] == (
        "Agent 'ai-news': 1 scheduled task(s) paused."
    )
    assert "(paused)" in (await say("/myagents")).sent[-1]
    assert (await say("/myagents resume ai-news")).sent[-1].endswith("1 scheduled task(s) resumed.")
    before = await snapshot()
    answer = (await say("/myagents pause alexbot")).sent[-1]
    assert answer.startswith("No agent named 'alexbot'.")
    assert changed(before, await snapshot()) == set()


async def test_myagents_run_runs_the_agents_tasks(world, monkeypatch):
    from app import tasks

    ran = []

    async def fake_execute(task_id, *, trigger):
        ran.append(task_id)
        return {"task_id": task_id, "status": "ok"}

    monkeypatch.setattr(tasks, "execute", fake_execute)
    answer = (await say("/myagents run ai-news")).sent[-1]
    assert ran == [world["sam_task"]] and answer == f"Agent 'ai-news' ran: task #{ran[0]} ok"


@pytest.mark.parametrize("said", [False, None])
async def test_myagents_delete_without_a_yes_keeps_the_agent(world, said):
    before = await snapshot()
    sink = await say("/myagents delete ai-news", confirm=said)
    assert sink.asked and sink.sent[-1] == "Agent 'ai-news' kept."
    assert changed(before, await snapshot()) == set()


async def test_myagents_delete_on_yes_retires_the_agent(world):
    sink = await say("/myagents delete ai-news", confirm=True)
    assert sink.asked[0].startswith("Delete agent 'ai-news'? Its 1 scheduled task(s)")
    assert sink.sent[-1] == "Agent 'ai-news' disabled and its 1 scheduled task(s) deleted."
    after = await snapshot()
    assert (
        after["agents"][world["sam_agent"]][1] is False and world["sam_task"] not in after["tasks"]
    )


async def test_myagents_delete_on_a_channel_that_cannot_ask_needs_the_word_confirm(world):
    first = (await say("/myagents delete ai-news")).sent[-1]
    assert first.endswith("Send /myagents delete ai-news confirm to do it.")
    assert world["sam_task"] in (await snapshot())["tasks"]
    await say("/myagents delete ai-news confirm")
    assert world["sam_task"] not in (await snapshot())["tasks"]


# --- /editagent ---


@pytest.fixture
def model(monkeypatch):
    state = {"outputs": [], "calls": []}

    async def fake(messages, schema):
        state["calls"].append({"messages": messages, "schema": schema})
        return dict(state["outputs"].pop(0))

    monkeypatch.setattr(builder, "ask_model", fake)
    return state


def model_output(**changes) -> dict:
    data = {k: v for k, v in spec(**changes).items() if k != "unattended"}
    data.setdefault("question", None)
    return data


async def test_editagent_shows_the_change_and_applies_it_on_yes(world, model):
    model["outputs"] = [model_output(schedule_expr="0 8 * * 4")]
    shown = (await say("/editagent ai-news move it to 8:00")).sent[-1]
    print(shown)
    assert shown.startswith("Changes to agent 'ai-news':")
    assert "Schedule: cron 0 9 * * 4 -> cron 0 8 * * 4 (Europe/Paris; next runs Thu" in shown
    assert "08:00" in shown and "Instructions" not in shown
    first_prompt = model["calls"][0]["messages"][1]["content"]
    assert "Change to this agent: move it to 8:00" in first_prompt
    assert '"schedule_expr": "0 9 * * 4"' in first_prompt
    assert TOOL in json.dumps(model["calls"][0]["schema"])
    before = await snapshot()
    done = (await say("yes")).sent[-1]
    # The edit is applied at the real current time: the next Thursday 08:00 in Paris after now.
    paris = ZoneInfo("Europe/Paris")
    day = datetime.now(paris)
    nxt = day.replace(hour=8, minute=0, second=0, microsecond=0)
    nxt += timedelta(days=(3 - day.weekday()) % 7)
    if nxt <= day:
        nxt += timedelta(days=7)
    expected = f"Agent 'ai-news' updated. Next run: Thu {nxt:%Y-%m-%d} 08:00 (Europe/Paris)"
    assert done.startswith(expected), done
    after = await snapshot()
    assert changed(before, after) == {("tasks", world["sam_task"])}
    assert after["tasks"][world["sam_task"]][2] == "0 8 * * 4"


async def test_editagent_no_changes_nothing(world, model):
    model["outputs"] = [model_output(schedule_expr="0 8 * * 4")]
    await say("/editagent ai-news move it to 8:00")
    before = await snapshot()
    assert (await say("no")).sent[-1] == builder.EDIT_CANCELLED
    assert changed(before, await snapshot()) == set()


async def test_editagent_of_another_users_agent_or_without_a_change(world, model):
    before = await snapshot()
    assert (await say("/editagent alexbot move it")).sent[-1] == "No agent named 'alexbot'."
    assert (await say("/editagent ai-news")).sent[-1] == builder.EDIT_USAGE
    assert model["calls"] == [] and changed(before, await snapshot()) == set()


async def test_editagent_keeps_the_tasks_delivery_channel(world, model):
    from app.db.session import session_scope

    model["outputs"] = [model_output(system_prompt="Shorter digests.")]
    await say("/editagent ai-news shorter summaries")
    await say("yes")
    async with session_scope() as session:
        task = await session.get(ScheduledTask, world["sam_task"])
        agent = await session.get(Agent, world["sam_agent"])
    assert task.channel_identity_id == world["sam_identity"]
    assert agent.system_prompt == "Shorter digests." and task.expr == "0 9 * * 4"


async def test_myagents_and_editagent_by_email(world, model):
    from app.admin import service
    from app.channels.dispatch import dispatch_event
    from app.db.session import session_scope

    async with session_scope() as session:
        identity = await service.add_channel_identity(
            session, world["alex"], Channel.EMAIL, "alex@example.org"
        )
        await service.grant_identity_permission(
            session, world["alex"], identity.id, PermissionKind.CHAT
        )
        await session.commit()

    async def mail(text):
        sink = _Sink()
        event = NormalizedEvent("alex@example.org", Channel.EMAIL, text, sink.reply)
        async with session_scope() as session:
            await dispatch_event(session, event)
        return sink.sent[-1]

    assert "- alexbot [on]" in await mail("/myagents")
    assert (await mail("/myagents pause alexbot")).endswith("1 scheduled task(s) paused.")
    model["outputs"] = [model_output(name="alexbot", schedule_expr="0 7 * * 4")]
    assert (await mail("/editagent alexbot at 7")) == builder.TIMEZONE_QUESTION  # Alex has none
    assert "cron 0 9 * * 4 -> cron 0 7 * * 4" in await mail("UTC")
    assert (await mail("yes")).startswith("Agent 'alexbot' updated.")
    async with session_scope() as session:
        task = await session.get(ScheduledTask, world["alex_task"])
    assert task.channel_identity_id == world["alex_identity"], "still delivered on Telegram"


async def test_pausing_twice_changes_nothing_the_second_time(world):
    first = await run(lambda s: my_agents.set_paused(s, world["sam"], world["sam_agent"], True,
                                                     actor="t"))  # fmt: skip
    before = await snapshot()
    second = await run(lambda s: my_agents.set_paused(s, world["sam"], world["sam_agent"], True,
                                                      actor="t"))  # fmt: skip
    assert (first, second) == ([world["sam_task"]], [])
    assert changed(before, await snapshot()) == set()


async def test_a_change_is_not_limited_by_the_agent_cap(world, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("SELF_SERVICE_MAX_AGENTS", "1")  # Sam already has 2 agents
    get_settings.cache_clear()
    agent, _task = await run(lambda s: agent_spec.update_agent_from_spec(
        s, world["sam"], world["sam_agent"], spec(system_prompt="Changed."), actor="t",
        now=MONDAY))  # fmt: skip
    assert agent.system_prompt == "Changed."


def test_the_word_null_is_an_empty_field():
    data = {**model_output(), "purpose": "null", "system_prompt": " None ", "model": "null"}
    spec_out, _question = builder.normalise(data)
    assert spec_out["purpose"] is spec_out["system_prompt"] is spec_out["model"] is None


async def test_an_edit_keeps_the_approval_and_proposes_only_added_tools(world, model):
    from app.db.session import session_scope

    async with session_scope() as session:
        task = await session.get(ScheduledTask, world["sam_task"])
        task.standing_tools = {}  # the user never approved the feeds tool for this task
        await session.commit()
    model["outputs"] = [model_output(schedule_expr="0 8 * * 4")]
    shown = (await say("/editagent ai-news move it to 8:00")).sent[-1]
    assert "Used without asking" not in shown, "an existing tool is not proposed again"
    await say("yes")
    async with session_scope() as session:
        assert (await session.get(ScheduledTask, world["sam_task"])).standing_tools == {}
