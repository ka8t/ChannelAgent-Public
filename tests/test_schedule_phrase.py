"""Tests: a schedule written in words. The model is scripted (its real behaviour is
measured on a real engine): the task parser decides, a phrase that
cannot be scheduled ("le 31 février") gives the reason and creates no task, `/task add <phrase>`
and `POST /schedules/parse` share one implementation, a timezone is guessed from a city, and the
agent builder asks for one when the user has none.
"""

import secrets
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from app import builder, schedule_phrase, tasks
from app.admin.service import InvalidInputError
from app.channels.schema import NormalizedEvent
from app.db.models import Channel, PermissionKind, ScheduledTask, User

KEY = secrets.token_urlsafe(24)  # a throwaway key, generated at run time
MONDAY_NOON = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def reply(kind="cron", expr="0 9 * * 4", prompt="give me the AI news", reason=None) -> dict:
    return {"schedule_kind": kind, "schedule_expr": expr, "prompt": prompt, "reason": reason}


@pytest.fixture
def model(monkeypatch):
    state = {"outputs": [], "calls": []}

    async def fake(messages, schema):
        state["calls"].append({"messages": messages, "schema": schema})
        out = state["outputs"].pop(0)
        if isinstance(out, Exception):
            raise out
        return dict(out)

    monkeypatch.setattr(builder, "ask_model", fake)
    return state


# --- the phrase ---


async def test_a_phrase_becomes_a_checked_schedule_with_its_next_runs(model):
    model["outputs"] = [reply()]
    parsed = await schedule_phrase.parse(
        "every Thursday at 9 give me the AI news", "Europe/Paris", MONDAY_NOON
    )
    assert (parsed["kind"], parsed["expr"], parsed["prompt"]) == (
        "cron", "0 9 * * 4", "give me the AI news",
    )  # fmt: skip
    shown = schedule_phrase.describe_runs(parsed["next_runs"])
    assert shown == "Thu 2026-10-01 09:00, Thu 2026-10-08 09:00, Thu 2026-10-15 09:00"
    content = model["calls"][0]["messages"][1]["content"]
    assert "Monday 2026-09-28 14:00 (Europe/Paris)" in content


async def test_the_measured_mistakes_are_repaired(model):
    model["outputs"] = [reply(kind="daily", expr="0 18 * * 1-5"), reply(kind="cron", expr="7:30")]
    first = await schedule_phrase.parse("working days at 18", None, MONDAY_NOON)
    second = await schedule_phrase.parse("every day at 7:30", None, MONDAY_NOON)
    assert (first["kind"], first["expr"]) == ("cron", "0 18 * * 1-5")
    assert (second["kind"], second["expr"]) == ("daily", "7:30")


async def test_a_date_that_never_occurs_is_refused_with_the_reason(model):
    model["outputs"] = [reply(expr="0 9 31 2 *")]
    with pytest.raises(InvalidInputError, match="never falls due"):
        await schedule_phrase.parse("le 31 février à 9h", None, MONDAY_NOON)


async def test_words_without_a_schedule_are_refused_with_the_models_reason(model):
    model["outputs"] = [reply(kind=None, expr=None, prompt=None, reason="aucune date")]
    with pytest.raises(InvalidInputError, match="no schedule found in the words .aucune date."):
        await schedule_phrase.parse("bonjour", None, MONDAY_NOON)


async def test_a_form_the_parser_refuses_is_refused(model):
    model["outputs"] = [reply(kind="every", expr="0m")]
    with pytest.raises(InvalidInputError, match="1 minute to 30 days"):
        await schedule_phrase.parse("all the time", None, MONDAY_NOON)


async def test_a_kind_outside_the_three_forms_is_refused_by_the_parser(model):
    # An engine that does not enforce the schema could write another kind.
    model["outputs"] = [reply(kind="weekly", expr="thu 09:00")]
    with pytest.raises(InvalidInputError, match="kind is one of"):
        await schedule_phrase.parse("weekly on Thursday", None, MONDAY_NOON)


@pytest.mark.parametrize("text", ["", "   ", "x" * 2001, None])
async def test_an_empty_or_huge_phrase_calls_no_model(model, text):
    with pytest.raises(InvalidInputError, match="1 to 2000"):
        await schedule_phrase.parse(text, None)
    assert model["calls"] == []


@pytest.mark.parametrize(
    ("text", "zone"),
    [
        ("Paris", "Europe/Paris"),
        ("europe/paris", "Europe/Paris"),
        ("New York", "America/New_York"),
        ("sao paulo", "America/Sao_Paulo"),
        ("UTC.", "UTC"),
        ("Mars", None),
        ("Cordoba", None),  # America/Cordoba and America/Argentina/Cordoba: not guessed
        ("", None),
        ("../../etc/passwd", None),
        ("x" * 65, None),
        (None, None),
    ],
)
def test_guess_timezone_returns_only_names_of_the_system_list(text, zone):
    assert tasks.guess_timezone(text) == zone


# --- /task add <phrase> ---


@pytest.fixture
async def world(fresh_db):
    from app.admin import service
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as session:
        ids = {}
        for name, external in (("sam", "111"), ("alex", "222")):
            user = await service.create_user(session, name)
            await service.create_agent(session, user.id, "default")
            identity = await service.add_channel_identity(
                session, user.id, Channel.TELEGRAM, external
            )
            await service.grant_identity_permission(
                session, user.id, identity.id, PermissionKind.CHAT
            )
            ids[name] = user.id
        (await session.get(User, ids["sam"])).timezone = "Europe/Paris"
        await session.commit()
    return ids


class _Sink:
    def __init__(self):
        self.sent: list[str] = []

    async def reply(self, text):
        self.sent.append(text)


async def task_command(text: str, external="111", answer=True, asked=None) -> str:
    """`answer` is what the user says to the confirmation (True, False, None for no answer);
    "cannot" gives a channel that cannot ask. The questions asked are appended to `asked`."""
    from app.channels.dispatch import handle_task_command
    from app.db.session import session_scope

    async def confirm(question, timeout):
        if asked is not None:
            asked.append((question, timeout))
        return answer

    sink = _Sink()
    event = NormalizedEvent(
        external, Channel.TELEGRAM, f"/task {text}", sink.reply,
        confirm=None if answer == "cannot" else confirm,
    )  # fmt: skip
    async with session_scope() as session:
        await handle_task_command(session, event, text)
    return sink.sent[-1]


async def task_count() -> int:
    from app.db.session import session_scope

    async with session_scope() as session:
        return (await session.execute(select(func.count()).select_from(ScheduledTask))).scalar_one()


async def test_task_add_in_words_shows_the_next_runs_and_creates_on_yes(world, model):
    model["outputs"] = [reply()]
    asked = []
    answer = await task_command("add every Thursday at 9 give me the AI news", asked=asked)
    print(asked[0][0])
    question, timeout = asked[0]
    assert question.startswith("I read: cron 0 9 * * 4, next runs Thu")
    assert "(Europe/Paris)" in question and question.endswith("Create this task?")
    assert timeout == 120
    assert answer.startswith("Task created. #1 [on] cron 0 9 * * 4")
    assert "give me the AI news" in answer and "Next runs: Thu" in answer
    assert await task_count() == 1


@pytest.mark.parametrize(
    ("said", "reply_text"),
    [(False, "Task not created."), (None, "Task not created. (no answer in time)")],
)
async def test_task_add_in_words_creates_nothing_without_a_yes(world, model, said, reply_text):
    model["outputs"] = [reply()]
    answer = await task_command("add every Thursday at 9 give me the AI news", answer=said)
    assert answer == reply_text
    assert await task_count() == 0


async def test_a_channel_that_cannot_ask_gets_the_exact_form_to_send(world, model):
    model["outputs"] = [reply()]
    answer = await task_command("add every Thursday at 9 give me the AI news", answer="cannot")
    assert answer.endswith("To create it, send: /task add cron 0 9 * * 4 give me the AI news")
    assert await task_count() == 0


async def test_task_add_in_words_without_a_timezone_says_it_runs_in_utc(world, model):
    model["outputs"] = [reply()]
    asked = []
    await task_command("add every Thursday at 9 give me the AI news", external="222", asked=asked)
    assert "UTC: set your timezone with /task tz" in asked[0][0]


async def test_task_add_the_31st_of_february_gives_the_reason_and_creates_no_task(world, model):
    model["outputs"] = [reply(expr="0 9 31 2 *", prompt="pay the rent")]
    answer = await task_command("add le 31 février à 9h rappelle-moi de payer le loyer")
    assert "never falls due" in answer
    assert await task_count() == 0


async def test_task_add_in_words_without_a_prompt_creates_no_task(world, model):
    model["outputs"] = [reply(prompt=None)]
    answer = await task_command("add every Thursday at 9")
    assert "not what to do at each run" in answer
    assert await task_count() == 0


async def test_task_add_in_words_with_the_engine_down_creates_no_task(world, model):
    model["outputs"] = [ConnectionError("engine down")]
    answer = await task_command("add every Thursday at 9 give me the AI news")
    assert answer.startswith("The schedule could not be read right now")
    assert await task_count() == 0


async def test_task_add_with_the_three_forms_calls_no_model(world, model):
    answer = await task_command("add daily 07:45 Tell me the news")
    assert answer.startswith("Task created. #1 [on] daily 07:45")
    assert model["calls"] == []


# --- POST /schedules/parse ---


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
        yield c
    app.dependency_overrides.clear()


async def test_api_parses_a_phrase_and_writes_nothing(api, model):
    model["outputs"] = [reply()]
    response = await api.post(
        "/schedules/parse", json={"text": "every Thursday at 9", "timezone": "Europe/Paris"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["kind"], body["expr"], body["timezone"]) == ("cron", "0 9 * * 4", "Europe/Paris")
    assert len(body["next_runs"]) == 3 and body["next_runs"][0].endswith("09:00:00+02:00")
    assert await task_count() == 0


async def test_api_refuses_the_31st_of_february_and_an_unknown_timezone(api, model):
    model["outputs"] = [reply(expr="0 9 31 2 *")]
    never = await api.post("/schedules/parse", json={"text": "le 31 février à 9h"})
    zone = await api.post("/schedules/parse", json={"text": "at 9", "timezone": "Mars/Base"})
    assert never.status_code == 422 and "never falls due" in never.json()["detail"]
    assert zone.status_code == 422 and "Unknown timezone" in zone.json()["detail"]
    assert len(model["calls"]) == 1, "an unknown timezone is refused before the model"


async def test_api_route_needs_the_operate_scope(api, model):
    from app.api.app import app
    from app.api.scopes import Principal, Scope, get_principal

    app.dependency_overrides[get_principal] = lambda: Principal("reader", Scope.READ)
    response = await api.post("/schedules/parse", json={"text": "every Thursday at 9"})
    assert response.status_code == 403 and model["calls"] == []


# --- the agent builder asks for a timezone ---


def spec_output(**changes) -> dict:
    data = {
        "name": "ai-news", "purpose": "AI news", "system_prompt": "Digest.",
        "memory_mode": "off", "tools": [], "skills": [], "schedule_kind": "cron",
        "schedule_expr": "0 9 * * 4", "task_prompt": "The AI news of the week.", "question": None,
    }  # fmt: skip
    data.update(changes)
    return data


async def say(text: str, external="222") -> list[str]:
    from app.channels.dispatch import dispatch_event
    from app.db.session import session_scope

    sink = _Sink()
    event = NormalizedEvent(external, Channel.TELEGRAM, text, sink.reply)
    async with session_scope() as session:
        await dispatch_event(session, event)
    return sink.sent


async def test_the_builder_asks_for_a_timezone_then_sets_it(world, model):
    from app.db.session import session_scope

    model["outputs"] = [spec_output()]
    assert (await say("/newagent AI news every Thursday at 9")) == [builder.TIMEZONE_QUESTION]
    again = await say("Mars")
    assert again == ["I do not know the timezone 'Mars'. " + builder.TIMEZONE_QUESTION]
    summary = (await say("Paris"))[-1]
    assert "cron 0 9 * * 4 (Europe/Paris)" in summary and "09:00" in summary
    async with session_scope() as session:
        assert (await session.get(User, world["alex"])).timezone == "Europe/Paris"
    assert len(model["calls"]) == 1


async def test_no_timezone_question_for_an_agent_without_a_schedule(world, model):
    model["outputs"] = [spec_output(schedule_kind=None, schedule_expr=None, task_prompt=None)]
    sent = await say("/newagent a Spanish tutor")
    assert sent[-1].startswith("Here is the agent I will create:")


async def test_timezone_answers_count_as_questions(world, model):
    model["outputs"] = [spec_output()]
    await say("/newagent AI news every Thursday at 9")
    for _ in range(builder.MAX_QUESTIONS - 1):
        await say("Mars")
    last = await say("Mars")
    assert last[-1].startswith("I could not complete the agent after 5 questions (no timezone")
    assert await task_count() == 0
