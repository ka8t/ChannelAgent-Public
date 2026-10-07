"""Tests: the agent builder dialogue. The model is scripted (its real behaviour is
measured on a real engine): what the user may attach, the repairs of
the model's output, the dialogue through the shared pipeline (question, summary with the next
run times and buttons, yes / no / test / change), /cancel, the question limit, a restart in the
middle of a dialogue, an abandoned dialogue, and the audit trail.
"""

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from app import builder
from app.channels.schema import NormalizedEvent
from app.db.models import (
    ActionLog,
    Agent,
    Channel,
    McpGrant,
    McpServer,
    PermissionKind,
    ScheduledTask,
    Skill,
)

TASK_PROMPT = "Give me the five most important AI news of the week, with links."
DEFINITION = {"name": "get_time", "description": "Current time in a timezone"}


def complete(**changes) -> dict:
    """A complete, valid output of the model."""
    data = {
        "name": "ai-news",
        "purpose": "Latest AI news",
        "system_prompt": "You write short, sourced news digests.",
        "memory_mode": "off",
        "tools": ["mcp__clock__get_time"],
        "skills": [],
        "schedule_kind": "cron",
        "schedule_expr": "0 9 * * 4",
        "task_prompt": TASK_PROMPT,
        "question": None,
    }
    data.update(changes)
    return data


@pytest.fixture
async def world(fresh_db):
    """Sam (Telegram 111, Europe/Paris) and Alex (Telegram 222). Sam may attach the tool
    clock.get_time (a grant for all agents, approved) and the self-service skill digest."""
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
            ids[name], ids[f"{name}_agent"] = user.id, agent.id
        (await session.get(type(user), ids["sam"])).timezone = "Europe/Paris"
        session.add(
            McpServer(
                name="clock",
                protocol="stdio",
                builtin_id="time",
                approved_definitions={
                    "get_time": {"sha256": "x", "definition": DEFINITION},
                    "set_alarm": {"sha256": "y", "definition": {"description": "Set an alarm"}},
                    "secret": {"sha256": "z", "definition": {"description": "Not approved?"}},
                },
                disabled_tools=["secret"],
                tool_policies={"set_alarm": "deny"},
            )
        )
        session.add(McpGrant(user_id=ids["sam"], server_name="clock"))

        def server(name, tools, **extra):
            approved = {t: {"sha256": t, "definition": {"description": t}} for t in tools}
            session.add(McpServer(name=name, protocol="stdio", builtin_id="time",
                                  approved_definitions=approved, **extra))  # fmt: skip

        # Not offered: a grant bound to the default agent only, a disabled server, a server
        # holding a credential not flagged as shared; notes: only the tool granted by name.
        server("web", ["fetch_page"])
        session.add(McpGrant(user_id=ids["sam"], agent_id=ids["sam_agent"], server_name="web"))
        server("off", ["ping"], enabled=False)
        session.add(McpGrant(user_id=ids["sam"], server_name="off"))
        server("creds", ["send"], env_vars='{"TOKEN": "t"}')
        session.add(McpGrant(user_id=ids["sam"], server_name="creds"))
        server("notes", ["read", "write"])
        session.add(McpGrant(user_id=ids["sam"], server_name="notes", tool_name="read"))
        session.add(Skill(name="digest", description="Write a digest", body="b", self_service=True))
        session.add(Skill(name="admin-only", description="Private", body="b"))
        await session.commit()
    return ids


@pytest.fixture
def model(monkeypatch):
    """The scripted model: each call pops the next output; the messages it got are kept."""
    state = {"outputs": [], "calls": []}

    async def fake(messages, schema):
        state["calls"].append({"messages": messages, "schema": schema})
        if not state["outputs"]:
            raise AssertionError("the model was called more often than scripted")
        return dict(state["outputs"].pop(0))

    monkeypatch.setattr(builder, "ask_model", fake)
    return state


@pytest.fixture
def turns(monkeypatch):
    """The agent's own turns (for messages that are not for the builder)."""
    import app.channels.dispatch as dispatch

    calls = []

    async def fake_run_turn(channel, user_id, agent_id, text, **kwargs):
        calls.append(text)
        return "agent reply"

    monkeypatch.setattr(dispatch, "run_turn", fake_run_turn)
    return calls


class _Sink:
    def __init__(self):
        self.sent: list[str] = []
        self.choices: list[tuple] = []

    async def reply(self, text):
        self.sent.append(text)

    async def reply_choices(self, text, choices, nonce):
        self.sent.append(text)
        self.choices.append((choices, nonce))


async def say(text: str, external="111", sink=None) -> _Sink:
    from app.channels.dispatch import dispatch_event
    from app.db.session import session_scope

    sink = sink or _Sink()
    event = NormalizedEvent(
        external, Channel.TELEGRAM, text, sink.reply, reply_choices=sink.reply_choices
    )
    async with session_scope() as session:
        await dispatch_event(session, event)
    return sink


async def counts(world) -> tuple[int, int]:
    from app.db.session import session_scope

    async with session_scope() as session:
        agents = (
            await session.execute(
                select(func.count()).select_from(Agent).where(Agent.user_id == world["sam"])
            )
        ).scalar_one()
        tasks = (
            await session.execute(select(func.count()).select_from(ScheduledTask))
        ).scalar_one()
    return agents, tasks


# --- pure parts ---


def test_normalise_repairs_the_measured_mistakes():
    spec, question = builder.normalise(complete(schedule_kind="daily", schedule_expr="0 9 * * 4"))
    assert (spec["schedule_kind"], spec["schedule_expr"], question) == ("cron", "0 9 * * 4", None)
    spec, _ = builder.normalise(complete(schedule_kind="cron", schedule_expr="08:30"))
    assert (spec["schedule_kind"], spec["schedule_expr"]) == ("daily", "08:30")
    spec, _ = builder.normalise(complete(name="AI News Digest!"))
    assert spec["name"] == "ai-news-digest"
    spec, _ = builder.normalise(complete(schedule_kind=None, schedule_expr=None))
    assert spec["schedule_kind"] is spec["schedule_expr"] is spec["task_prompt"] is None
    spec, _ = builder.normalise(complete(task_prompt=None))
    assert (spec["schedule_kind"], spec["schedule_expr"], spec["task_prompt"]) == (
        "cron", "0 9 * * 4", None,
    ), "a schedule without its prompt is kept for the check and the repair"
    _, question = builder.normalise(complete(question="  Which topic?  "))
    assert question == "Which topic?"
    assert "question" not in builder.normalise(complete())[0]


def test_the_schema_enumerates_only_what_the_user_may_attach():
    tools = [{"name": "mcp__clock__get_time", "description": ""}]
    schema = builder.output_schema(tools, [])
    assert schema["properties"]["tools"]["items"]["enum"] == ["mcp__clock__get_time"]
    assert schema["properties"]["skills"] == {"type": "array", "maxItems": 0}
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_classify_answers_in_english_and_french():
    assert [builder.classify(w) for w in ("Yes", "oui!", "no", "Non.", "test", "tester")] == [
        "yes", "yes", "no", "no", "test", "test",
    ]  # fmt: skip
    assert builder.classify("move it to 8:00") is None


def test_next_runs_are_in_the_users_timezone():
    now = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)  # a Monday
    runs = builder.next_runs("cron", "0 9 * * 4", "Europe/Paris", now)
    shown = [t.strftime("%a %d %H:%M") for t in runs]
    assert shown == ["Thu 01 09:00", "Thu 08 09:00", "Thu 15 09:00"]


async def test_choices_are_the_grants_for_all_agents_approved_on_and_not_denied(world):
    from app.db.session import session_scope

    async with session_scope() as session:
        tools, skills = await builder._choices(session, world["sam"])
        alex_tools, _ = await builder._choices(session, world["alex"])
    assert sorted(t["name"] for t in tools) == ["mcp__clock__get_time", "mcp__notes__read"]
    assert tools[0]["description"] == "Current time in a timezone"
    assert [s["name"] for s in skills] == ["digest"]
    assert alex_tools == []


# --- the dialogue ---


async def test_question_then_summary_then_yes_creates_the_agent_and_its_task(world, model):
    model["outputs"] = [
        complete(question="Which topics: research, products, or open source?"),
        complete(),
    ]
    first = await say("/newagent an agent that gives me the AI news every Thursday at 9:00")
    assert first.sent == ["Which topics: research, products, or open source?"]
    assert await counts(world) == (1, 0)

    second = await say("open source")
    summary = second.sent[-1]
    print(summary)
    assert "Name: ai-news" in summary and "cron 0 9 * * 4 (Europe/Paris)" in summary
    assert "Next runs: Thu" in summary and "09:00" in summary
    assert "Tools: mcp__clock__get_time" in summary and "Delivered on: telegram" in summary
    choices, nonce = second.choices[-1]
    assert choices == ["yes", "test", "no"] and nonce
    pending = await builder.waiting(builder.thread_id(Channel.TELEGRAM, "111"))
    assert pending["nonce"] == nonce, "a button is valid only for the question waiting"

    assert (
        "Used without asking in its scheduled runs (nobody can confirm then): "
        "mcp__clock__get_time" in summary
    ), "the clock has no annotations, so it asks (confirm): approved by the yes"
    done = await say("yes")
    assert done.sent[-1].startswith("Agent 'ai-news' created. Its task is #1")
    assert await counts(world) == (2, 1)
    from app.db.session import session_scope

    async with session_scope() as session:
        task = (await session.execute(select(ScheduledTask))).scalar_one()
        assert task.standing_tools == {"mcp__clock__get_time": "x"}
    assert await builder.waiting(builder.thread_id(Channel.TELEGRAM, "111")) is None
    # The answer to the question reached the model with the dialogue.
    assert "open source" in model["calls"][1]["messages"][1]["content"]


async def test_no_at_the_summary_creates_nothing(world, model):
    model["outputs"] = [complete()]
    await say("/newagent AI news every Thursday at 9")
    answer = await say("non")
    assert answer.sent[-1] == builder.CANCELLED
    assert await counts(world) == (1, 0)


async def test_cancel_ends_a_dialogue_and_says_when_there_is_none(world, model):
    assert (await say("/cancel")).sent == [builder.NOTHING_TO_CANCEL]
    model["outputs"] = [complete(question="Which topic?")]
    await say("/newagent a news agent")
    assert (await say("/cancel")).sent == [builder.CANCELLED]
    assert await counts(world) == (1, 0)
    assert await builder.waiting(builder.thread_id(Channel.TELEGRAM, "111")) is None


async def test_newagent_without_a_request_explains_and_calls_no_model(world, model):
    assert (await say("/newagent")).sent == [builder.USAGE]
    assert model["calls"] == []


async def test_at_most_five_questions_then_nothing_is_created(world, model):
    model["outputs"] = [complete(question=f"Question {i}?") for i in range(6)]
    sink = await say("/newagent something vague")
    for i in range(5):
        sink = await say(f"answer {i}")
    assert sink.sent[-1].startswith("I could not complete the agent after 5 questions")
    assert len(model["calls"]) == 6
    assert await counts(world) == (1, 0)
    assert await builder.waiting(builder.thread_id(Channel.TELEGRAM, "111")) is None


async def test_a_refused_spec_is_repaired_without_asking_the_user(world, model):
    # "default" is taken: the check's reason goes back to the model, which renames it.
    model["outputs"] = [complete(name="default"), complete(name="ai-news")]
    sink = await say("/newagent AI news every Thursday at 9")
    assert sink.sent[-1].startswith("Here is the agent I will create:")
    assert len(model["calls"]) == 2
    assert "already have an agent named 'default'" in model["calls"][1]["messages"][1]["content"]


async def test_a_spec_still_refused_after_the_repairs_is_asked_to_the_user(world, model):
    model["outputs"] = [complete(name="default")] * 3
    sink = await say("/newagent AI news every Thursday at 9")
    assert sink.sent[-1].startswith("I cannot create it yet: name: you already have")
    assert len(model["calls"]) == 1 + builder.REPAIRS


async def test_a_change_at_the_summary_goes_back_to_the_model(world, model):
    model["outputs"] = [complete(), complete(schedule_expr="0 8 * * 4")]
    await say("/newagent AI news every Thursday at 9")
    sink = await say("move it to 8:00")
    assert "cron 0 8 * * 4" in sink.sent[-1]
    assert "Change: move it to 8:00" in model["calls"][1]["messages"][1]["content"]


async def test_test_runs_once_then_yes_starts_the_schedule(world, model, monkeypatch):
    runs = []

    async def fake_execute(task_id, *, trigger):
        runs.append((task_id, trigger))
        return {"task_id": task_id, "status": "ok", "delivered": True}

    monkeypatch.setattr(builder.tasks, "execute", fake_execute)
    model["outputs"] = [complete()]
    await say("/newagent AI news every Thursday at 9")
    sink = await say("test")
    assert sink.sent[-1].startswith("Test run: ok")
    assert sink.choices[-1][0] == ["yes", "no"]
    from app.db.session import session_scope

    async with session_scope() as session:
        task = (await session.execute(select(ScheduledTask))).scalar_one()
        assert (task.enabled, task.next_run_at) == (False, None)
    await say("yes")
    # The node that ran the test is not run again when the dialogue resumes.
    assert runs == [(1, "builder-test")]
    async with session_scope() as session:
        task = (await session.execute(select(ScheduledTask))).scalar_one()
        assert task.enabled is True and task.next_run_at is not None
    assert await counts(world) == (2, 1)


async def test_test_then_no_deletes_the_task_and_disables_the_agent(world, model, monkeypatch):
    async def fake_execute(task_id, *, trigger):
        return {"task_id": task_id, "status": "ok", "delivered": True}

    monkeypatch.setattr(builder.tasks, "execute", fake_execute)
    model["outputs"] = [complete()]
    await say("/newagent AI news every Thursday at 9")
    await say("test")
    sink = await say("no")
    assert "disabled and its task deleted" in sink.sent[-1]
    from app.db.session import session_scope

    async with session_scope() as session:
        agent = (await session.execute(select(Agent).where(Agent.name == "ai-news"))).scalar_one()
        assert agent.is_active is False
    assert await counts(world) == (2, 0)


async def test_a_dialogue_survives_a_restart(world, model):
    from app import graph

    model["outputs"] = [complete(question="Which topic?"), complete()]
    await say("/newagent a news agent every Thursday at 9")
    await graph.close_graph()
    builder._compiled.clear()
    sink = await say("AI")
    assert sink.sent[-1].startswith("Here is the agent I will create:")


async def test_messages_go_to_the_agent_when_no_dialogue_is_open(world, model, turns):
    sink = await say("hello")
    assert sink.sent == ["agent reply"] and turns == ["hello"]
    assert model["calls"] == []


async def test_another_users_messages_never_reach_a_dialogue(world, model, turns):
    model["outputs"] = [complete(question="Which topic?")]
    await say("/newagent a news agent")
    sink = await say("yes", external="222")
    assert sink.sent == ["agent reply"]
    assert await builder.waiting(builder.thread_id(Channel.TELEGRAM, "111")) is not None


async def test_an_abandoned_dialogue_is_dropped_and_the_message_goes_to_the_agent(
    world, model, turns, monkeypatch
):
    model["outputs"] = [complete(question="Which topic?")]
    await say("/newagent a news agent")
    monkeypatch.setattr(builder, "IDLE_SECONDS", -1)
    sink = await say("what time is it?")
    assert sink.sent == ["agent reply"] and turns == ["what time is it?"]
    assert await builder.waiting(builder.thread_id(Channel.TELEGRAM, "111")) is None
    assert await counts(world) == (1, 0)


async def test_every_step_is_in_the_audit_trail_and_the_failure_apologises(world, model):
    from app.db.session import session_scope

    model["outputs"] = [complete(question="Which topic?")]
    await say("/newagent a news agent")
    sink = await say("AI")  # the script is empty: the model call raises
    assert sink.sent == ["Sorry, I cannot answer right now. Please try again in a few minutes."]
    async with session_scope() as session:
        rows = (await session.execute(select(ActionLog).order_by(ActionLog.id))).scalars().all()
    texts = [(r.direction.value, r.text, r.status.value) for r in rows]
    print(json.dumps(texts))
    assert texts[:3] == [
        ("inbound", "/newagent a news agent", "ok"),
        ("outbound", "Which topic?", "ok"),
        ("inbound", "AI", "ok"),
    ]
    assert texts[3][0] == "outbound" and texts[3][2] == "failed"


# --- Telegram answer buttons ---


async def test_telegram_buttons_carry_the_choices_and_answer_only_the_waiting_question(
    world, model, monkeypatch
):
    from types import SimpleNamespace

    from app.channels import telegram

    class Bot:
        def __init__(self):
            self.sent = []

        async def send_message(self, chat_id, text, reply_markup=None):
            self.sent.append((text, reply_markup))

    bot = Bot()
    update = SimpleNamespace(
        message=SimpleNamespace(text="/newagent AI news every Thursday at 9"),
        effective_user=SimpleNamespace(id=111),
        effective_chat=SimpleNamespace(id=111),
    )
    model["outputs"] = [complete()]
    await telegram._on_builder_command(update, SimpleNamespace(bot=bot))
    text, markup = bot.sent[-1]
    assert text.startswith("Here is the agent I will create:")
    data = [b.callback_data for row in markup.inline_keyboard for b in row]
    nonce = data[0].split(":")[1]
    assert data == [f"bld:{nonce}:yes", f"bld:{nonce}:test", f"bld:{nonce}:no"]

    dispatched, answers = [], []
    real_dispatch = telegram.dispatch_event

    async def spy(session, event, **kwargs):
        dispatched.append((event.user_id, event.text))
        return await real_dispatch(session, event, **kwargs)

    monkeypatch.setattr(telegram, "dispatch_event", spy)

    def press(user_id, data):
        async def answer(text):
            answers.append(text)

        async def edit(reply_markup=None):
            return None

        return SimpleNamespace(
            callback_query=SimpleNamespace(
                from_user=SimpleNamespace(id=user_id), data=data, answer=answer,
                edit_message_reply_markup=edit,
            ),
            effective_user=SimpleNamespace(id=user_id),
            effective_chat=SimpleNamespace(id=user_id),
        )  # fmt: skip

    context = SimpleNamespace(bot=bot)
    await telegram._on_choice(press(111, "bld:deadbeef:yes"), context)  # an old button
    await telegram._on_choice(press(222, f"bld:{nonce}:yes"), context)  # another user
    await telegram._on_choice(press(111, f"bld:{nonce}:rm"), context)  # not a choice
    assert dispatched == [] and answers == ["Expired", "Expired", "Expired"]
    assert await counts(world) == (1, 0)
    await telegram._on_choice(press(111, f"bld:{nonce}:yes"), context)
    assert dispatched == [("111", "yes")] and answers[-1] == "yes"
    assert await counts(world) == (2, 1)
    assert bot.sent[-1][0].startswith("Agent 'ai-news' created.")


async def test_purging_the_user_deletes_an_open_dialogue(world, model):
    from app.admin import service
    from app.db.session import session_scope

    model["outputs"] = [complete(question="Which topic?")]
    await say("/newagent a news agent")
    thread = builder.thread_id(Channel.TELEGRAM, "111")
    assert await builder.waiting(thread) is not None
    async with session_scope() as session:
        await service.delete_user(session, world["sam"], purge=True, actor="test")
        await session.commit()
    assert await builder.waiting(thread) is None


async def test_at_most_five_changes_at_the_summary(world, model):
    model["outputs"] = [complete()] * (1 + builder.MAX_CHANGES)
    await say("/newagent AI news every Thursday at 9")
    for i in range(builder.MAX_CHANGES):
        sink = await say(f"change number {i}")
        assert sink.sent[-1].startswith("Here is the agent I will create:")
    sink = await say("one change too many")
    assert sink.sent[-1] == builder.CANCELLED + " (too many changes)"
    assert await counts(world) == (1, 0)


async def test_the_builder_is_limited_like_a_turn(world, model, monkeypatch):
    from app.channels.limits import limiter
    from app.config import get_settings

    monkeypatch.setenv("RATE_LIMIT_MESSAGES_PER_MINUTE", "1")
    get_settings.cache_clear()
    limiter.clear()
    model["outputs"] = [complete(question="Which topic?")]
    await say("/newagent a news agent")
    sink = await say("AI")
    assert "Which topic?" not in sink.sent and len(model["calls"]) == 1
    assert sink.sent and "faster than this assistant" in sink.sent[-1]
    limiter.clear()


# --- the call to the engine ---


async def test_ask_model_sends_the_schema_without_thinking_and_parses_the_reply(
    fresh_db, monkeypatch
):
    import httpx

    from app.db.session import init_db

    await init_db()
    seen, replies = [], [json.dumps(complete()), "[1, 2]"]

    def handler(request):
        seen.append(json.loads(request.content))
        content = replies.pop(0)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    real_client = httpx.AsyncClient

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(builder.httpx, "AsyncClient", client)
    schema = builder.output_schema([], [])
    out = await builder.ask_model([{"role": "user", "content": "x"}], schema)
    assert out["name"] == "ai-news"
    body = seen[0]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "spec", "schema": schema},
    }
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] == 0 and body["stream"] is False and "model" not in body
    with pytest.raises(ValueError, match="not an object"):
        await builder.ask_model([{"role": "user", "content": "x"}], schema)


async def test_a_schedule_without_its_prompt_is_repaired_by_the_model(world, model):
    model["outputs"] = [complete(task_prompt=None), complete()]
    sink = await say("/newagent every hour from 9 to 17 on weekdays remind me to stretch")
    assert sink.sent[-1].startswith("Here is the agent I will create:")
    assert "cron 0 9 * * 4" in sink.sent[-1]
    assert "task_prompt missing" in model["calls"][1]["messages"][1]["content"]


async def test_known_feeds_are_offered_only_with_a_feeds_tool(world, model):
    from app.db.models import FeedSource
    from app.db.session import session_scope

    async with session_scope() as session:
        session.add(FeedSource(name="AI news", url="https://example.org/ai.rss", topics="ai"))
        await session.commit()
    model["outputs"] = [complete(), complete()]
    await say("/newagent AI news every Thursday at 9")
    assert "Known feeds" not in model["calls"][0]["messages"][1]["content"]
    await say("/cancel")

    async with session_scope() as session:
        session.add(
            McpServer(
                name="news", protocol="stdio", builtin_id="feeds",
                approved_definitions={"read_feed": {"sha256": "f", "definition": {}}},
            )
        )  # fmt: skip
        session.add(McpGrant(user_id=world["sam"], server_name="news"))
        await session.commit()
    await say("/newagent AI news every Thursday at 9")
    content = model["calls"][1]["messages"][1]["content"]
    assert "Known feeds: AI news [ai]: https://example.org/ai.rss" in content
    assert "mcp__news__read_feed" in content


async def test_from_the_terminal_the_summary_names_where_the_task_is_delivered(world, model):
    """Live: asked from the terminal, the summary said "Delivered on: terminal" for a task
    delivered on Telegram (the terminal cannot receive later)."""
    from app.admin import service
    from app.channels.dispatch import dispatch_event
    from app.db.session import session_scope

    async with session_scope() as session:
        identity = await service.add_channel_identity(session, world["sam"], Channel.TERMINAL,
                                                      "owner")  # fmt: skip
        await service.grant_identity_permission(session, world["sam"], identity.id,
                                                PermissionKind.CHAT)  # fmt: skip
        await session.commit()
    model["outputs"] = [complete()]
    sink = _Sink()
    async with session_scope() as session:
        await dispatch_event(session, NormalizedEvent("owner", Channel.TERMINAL,
                                                      "/newagent AI news", sink.reply))  # fmt: skip
    assert "Delivered on: telegram" in sink.sent[-1]
