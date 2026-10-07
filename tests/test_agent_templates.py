"""Tests: agent templates. An administrator's data (validated, versioned, audited,
read and changed through the API with their scopes), starter templates installed once, and the
builder that picks a template, fixes what it fixes without widening a right, and asks only its
questions."""

import json

import httpx
import pytest
from sqlalchemy import select

from app import builder
from app.admin import agent_templates as templates
from app.admin.service import ConflictError, InvalidInputError
from app.db.models import AdminEvent, AgentTemplate, AgentTemplateVersion
from tests.test_builder import complete, model, say, world  # noqa: F401  (fixtures)

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
DIGEST = {
    "name": "news-digest",
    "description": "A digest of the news of feeds on a topic, sent on a schedule.",
    "guidance": "Name the feed addresses in task_prompt.",
    "agent_instructions": "Copy every link exactly.",
    "memory_mode": "off",
    "tools": ["get_time", "fetch_page"],
    "needs_schedule": True,
    "questions": {"purpose": "Which topic?", "schedule": "When should I send it?"},
}


async def _sql(query, *args):
    from app.db.session import session_scope

    async with session_scope() as session:
        return (await session.execute(query, *args)).scalars().all()


# --- the service ---


async def test_a_template_is_created_versioned_and_audited(fresh_db):
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as session:
        created = await templates.create_template(session, DIGEST, actor="admin")
        assert (created.version, created.tools) == (1, ["fetch_page", "get_time"])
        same = await templates.update_template(session, "news-digest", {"memory_mode": "off"},
                                               actor="admin")  # fmt: skip
        assert same.version == 1, "no real change, no new version"
        changed = await templates.update_template(
            session, "news-digest", {"memory_mode": "ondemand", "enabled": False}, actor="admin"
        )
        assert changed.version == 2
        await session.commit()
    versions = await _sql(select(AgentTemplateVersion).order_by(AgentTemplateVersion.version))
    assert [(v.version, v.data["memory_mode"], v.data["enabled"]) for v in versions] == [
        (1, "off", True), (2, "ondemand", False),
    ]  # fmt: skip
    events = await _sql(select(AdminEvent.action).order_by(AdminEvent.id))
    assert events == ["agent_template.create", "agent_template.update", "agent_template.update"]
    async with session_scope() as session:
        await templates.delete_template(session, "news-digest", actor="admin")
        await session.commit()
    assert await _sql(select(AgentTemplate)) == []
    assert await _sql(select(AgentTemplateVersion)) == [], "versions go with the template"


@pytest.mark.parametrize(
    "fields, message",
    [
        ({**DIGEST, "name": "News Digest"}, "lowercase"),
        ({**DIGEST, "description": " "}, "needs a description"),
        ({**DIGEST, "description": "x" * 201}, "at most 200"),
        ({**DIGEST, "memory_mode": "forever"}, "memory_mode"),
        ({**DIGEST, "tools": ["mcp__web__fetch page"]}, "tool names"),
        ({**DIGEST, "tools": [f"t{i}" for i in range(21)]}, "at most 20"),
        ({**DIGEST, "questions": {"tools": "Which tools?"}}, "not tools"),
        ({**DIGEST, "questions": {"purpose": " "}}, "purpose question"),
        ({**DIGEST, "questions": {"purpose": "x" * 301}}, "at most 300"),
        ({**DIGEST, "needs_schedule": "yes"}, "boolean"),
        ({**DIGEST, "guidance": "x" * 4001}, "at most 4000"),
        ({**DIGEST, "agent_instructions": "x" * 2001}, "at most 2000"),
        ({**DIGEST, "system_prompt": "x"}, "Unknown template field"),
    ],
)
def test_an_invalid_template_is_refused_with_the_field(fields, message):
    with pytest.raises(InvalidInputError, match=message):
        templates.clean(fields, creating=True)


async def test_names_are_unique_and_kept(fresh_db):
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as session:
        await templates.create_template(session, DIGEST, actor="admin")
        with pytest.raises(ConflictError):
            await templates.create_template(session, DIGEST, actor="admin")
        with pytest.raises(InvalidInputError, match="keeps its name"):
            await templates.update_template(session, "news-digest", {"name": "x"}, actor="a")


async def test_starters_are_data_installed_once(fresh_db):
    from app.db.session import init_db, session_scope

    await init_db()
    starters = templates.starter_templates()
    assert [t["name"] for t in starters] == ["news-digest", "page-watch", "reminder",
                                             "daily-summary"]  # fmt: skip
    assert templates.STARTER_FILE.suffix == ".json"
    async with session_scope() as session:
        assert await templates.install_starters(session, actor="admin") == {
            "created": ["news-digest", "page-watch", "reminder", "daily-summary"], "kept": [],
        }  # fmt: skip
        await templates.update_template(session, "reminder", {"enabled": False}, actor="admin")
        again = await templates.install_starters(session, actor="admin")
        await session.commit()
    assert again == {"created": [], "kept": ["news-digest", "page-watch", "reminder",
                                             "daily-summary"]}  # fmt: skip
    rows = {t.name: (t.source, t.version, t.enabled) for t in await _sql(select(AgentTemplate))}
    assert rows["reminder"] == ("starter", 2, False), "a present template is left as it is"


# --- the API ---


@pytest.fixture
async def api(fresh_db, monkeypatch):
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t",
                                 headers={"Authorization": f"Bearer {KEY}"}) as client:  # fmt: skip
        yield client
    deps.reset_failure_state()
    get_settings.cache_clear()


async def test_the_api_creates_reads_versions_changes_and_deletes(api):
    created = await api.post("/agent-templates", json=DIGEST)
    assert created.status_code == 201 and created.json()["version"] == 1
    assert (await api.post("/agent-templates", json=DIGEST)).status_code == 409
    listed = await api.get("/agent-templates")
    assert [t["name"] for t in listed.json()] == ["news-digest"]
    changed = await api.patch("/agent-templates/news-digest", json={"needs_schedule": False})
    assert (changed.status_code, changed.json()["version"]) == (200, 2)
    versions = (await api.get("/agent-templates/news-digest/versions")).json()
    shown = [(v["version"], v["data"]["needs_schedule"]) for v in versions]
    assert shown == [(1, True), (2, False)]
    refused = await api.patch("/agent-templates/news-digest", json={"memory_mode": "forever"})
    assert refused.status_code in (400, 422)
    starters = await api.post("/agent-templates/starters")
    assert starters.json() == {"created": ["page-watch", "reminder", "daily-summary"],
                               "kept": ["news-digest"]}  # fmt: skip
    assert (await api.delete("/agent-templates/news-digest")).status_code == 204
    assert (await api.get("/agent-templates/news-digest")).status_code == 404


async def test_reading_needs_read_and_a_change_needs_admin(api):
    from app.api.app import app
    from app.api.scopes import Principal, Scope, get_principal

    await api.post("/agent-templates", json=DIGEST)
    for scope in (Scope.READ, Scope.OPERATE):
        app.dependency_overrides[get_principal] = lambda s=scope: Principal("t", s)
        try:
            assert (await api.get("/agent-templates")).status_code == 200
            assert (await api.post("/agent-templates/starters")).status_code == 403
            assert (await api.patch("/agent-templates/news-digest",
                                    json={"enabled": False})).status_code == 403  # fmt: skip
            assert (await api.delete("/agent-templates/news-digest")).status_code == 403
        finally:
            app.dependency_overrides.clear()
    no_token = await api.get("/agent-templates", headers={"Authorization": ""})
    assert no_token.status_code == 401


# --- the builder ---


def test_a_template_fixes_its_fields_without_widening_a_right():
    offered = [{"name": "mcp__clock__get_time"}, {"name": "mcp__notes__read"}]
    spec = {"memory_mode": "always", "tools": ["mcp__notes__read"], "system_prompt": "Be brief."}
    applied = builder.apply_template(dict(spec), DIGEST, offered)
    assert applied["memory_mode"] == "off"
    assert applied["tools"] == ["mcp__clock__get_time", "mcp__notes__read"], (
        "get_time offered and attached; fetch_page not offered to this user, so not attached"
    )
    assert applied["system_prompt"] == "Be brief.\n\nCopy every link exactly."
    again = builder.apply_template(dict(applied), DIGEST, offered)
    assert again["system_prompt"].count("Copy every link exactly.") == 1
    assert builder.apply_template(dict(spec), {}, offered) == spec


def test_only_the_templates_questions_are_asked_for_what_is_missing():
    scheduled = {"purpose": "AI news", "schedule_kind": "daily", "schedule_expr": "08:00",
                 "task_prompt": "Digest"}  # fmt: skip
    assert builder.template_question(scheduled, DIGEST) is None
    assert builder.template_question({**scheduled, "purpose": None}, DIGEST) == "Which topic?"
    unscheduled = {**scheduled, "schedule_kind": None, "schedule_expr": None}
    assert builder.template_question(unscheduled, DIGEST) == "When should I send it?"
    assert builder.template_question(unscheduled, {**DIGEST, "needs_schedule": False}) is None
    no_prompt = {**scheduled, "task_prompt": None}
    assert builder.template_question(no_prompt, DIGEST) is None, "no question written for it"
    assert builder.template_question(no_prompt, {**DIGEST, "questions": {
        "task_prompt": "What should each run produce?"}}) == "What should each run produce?"


async def test_the_pick_is_an_enumeration_of_the_templates_and_none(monkeypatch):
    seen = {}

    async def fake(messages, schema):
        seen["schema"], seen["messages"] = schema, messages
        return {"template": "news-digest"}

    monkeypatch.setattr(builder, "ask_model", fake)
    other = {**DIGEST, "name": "reminder", "description": "Remind the user."}
    assert (await builder.pick_template("AI news daily", [DIGEST, other]))["name"] == "news-digest"
    assert seen["schema"]["properties"]["template"]["enum"] == ["news-digest", "reminder", "none"]
    assert "Request: AI news daily" in seen["messages"][1]["content"]

    async def none(messages, schema):
        return {"template": "none"}

    monkeypatch.setattr(builder, "ask_model", none)
    assert await builder.pick_template("x", [DIGEST]) == {}
    assert await builder.pick_template("x", []) == {}


async def test_the_builder_follows_the_picked_template(world, model):  # noqa: F811
    from app.db.session import session_scope

    async with session_scope() as session:
        await templates.create_template(session, DIGEST, actor="admin")
        await session.commit()
    model["outputs"] = [
        {"template": "news-digest"},
        complete(question="Which language?", schedule_kind=None, schedule_expr=None,
                 task_prompt=None, memory_mode="always", tools=[]),  # fmt: skip
    ]
    first = (await say("/newagent AI news digest")).sent[-1]
    assert first == "When should I send it?", "the template's question, not the model's"
    pick, extraction = model["calls"]
    assert pick["schema"]["properties"]["template"]["enum"] == ["news-digest", "none"]
    prompt = extraction["messages"][1]["content"]
    assert "Template news-digest:" in prompt and "Name the feed addresses" in prompt
    model["outputs"] = [complete(memory_mode="always", tools=[], schedule_kind="daily",
                                 schedule_expr="08:00")]  # fmt: skip
    summary = (await say("every day at 8")).sent[-1]
    assert len(model["calls"]) == 3, "no second pick"
    assert "mcp__clock__get_time" in summary, "the template's tool, offered to Sam"
    assert "fetch_page" not in summary, "not offered to Sam: not attached"
    assert "Copy every link exactly." in summary or "copy every link" in summary.lower()


async def test_a_disabled_template_is_not_offered_and_an_edit_never_picks(world, model):  # noqa: F811
    from app.db.session import session_scope

    async with session_scope() as session:
        await templates.create_template(session, {**DIGEST, "enabled": False}, actor="admin")
        await session.commit()
    model["outputs"] = [complete()]
    await say("/newagent AI news digest")
    assert len(model["calls"]) == 1, "no template enabled: no pick call"
    assert "Template" not in json.dumps(model["calls"][0]["messages"])
