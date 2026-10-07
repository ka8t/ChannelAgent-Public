"""Tests: scheduled tasks. Schedules (parsed or refused), next times in the user's
timezone, ownership (a user never reaches another user's task), the Admin API, the runner
(a due task runs once and is delivered; paused, none runs), purge and identity removal, and
the /task command.
"""

import json
import secrets
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app import tasks
from app.admin.service import InvalidInputError
from app.db.models import ActionLog, AdminEvent, Channel, PermissionKind, ScheduledTask

KEY = secrets.token_urlsafe(24)  # a throwaway key, generated at run time
PROMPT = "Summarise the weather for my commute"


def utc(*parts) -> datetime:
    return datetime(*parts, tzinfo=UTC)


# --- schedules ---


@pytest.mark.parametrize(
    ("kind", "expr"),
    [
        ("cron", "*/15 * * * *"),
        ("cron", "0 8 * * 1-5"),
        ("cron", "30 6 1,15 * *"),
        ("cron", "0 9-17/2 * * 7"),
        ("every", "30m"),
        ("every", "2h"),
        ("every", "1d"),
        ("daily", "08:30"),
        ("daily", "7:05"),
    ],
)
def test_valid_schedules_parse(kind, expr):
    assert tasks.check_schedule(kind, expr) == (kind, expr)


@pytest.mark.parametrize(
    ("kind", "expr", "reason"),
    [
        ("cron", "* * * *", "five fields"),
        ("cron", "60 * * * *", "minute"),
        ("cron", "5-1 * * * *", "backwards"),
        ("cron", "*/0 * * * *", "step"),
        ("cron", "a * * * *", "minute"),
        ("cron", "0 0 * 13 *", "month"),
        ("every", "0m", "1 minute to 30 days"),
        ("every", "31d", "1 minute to 30 days"),
        ("every", "5x", "number and a unit"),
        ("daily", "24:00", "HH:MM"),
        ("daily", "8h30", "HH:MM"),
        ("weekly", "mon", "kind is one of"),
        ("daily", "", "1 to 100"),
    ],
)
def test_invalid_schedules_are_refused_with_the_reason(kind, expr, reason):
    with pytest.raises(InvalidInputError, match=reason):
        tasks.check_schedule(kind, expr)


def test_daily_is_read_in_the_users_timezone_across_the_clock_change():
    # 07:00 in Paris (UTC+2): today's 08:30 is 06:30 UTC.
    assert tasks.next_time("daily", "08:30", "Europe/Paris", utc(2026, 9, 28, 5, 0)) == utc(
        2026, 9, 28, 6, 30
    )
    # Past it: tomorrow.
    assert tasks.next_time("daily", "08:30", "Europe/Paris", utc(2026, 9, 28, 7, 0)) == utc(
        2026, 9, 29, 6, 30
    )
    # Summer time ends on 25 October 2026: 08:30 in Paris is then 07:30 UTC.
    assert tasks.next_time("daily", "08:30", "Europe/Paris", utc(2026, 10, 24, 7, 0)) == utc(
        2026, 10, 25, 7, 30
    )
    assert tasks.next_time("daily", "08:30", None, utc(2026, 9, 28, 5, 0)) == utc(
        2026, 9, 28, 8, 30
    )


def test_cron_weekdays_and_the_either_day_rule():
    # Friday 2 October 2026, 09:00 UTC: "weekdays at 08:00" is next on Monday.
    assert tasks.next_time("cron", "0 8 * * 1-5", "UTC", utc(2026, 10, 2, 9, 0)) == utc(
        2026, 10, 5, 8, 0
    )
    # Both day fields restricted: the 13th OR a Friday, whichever comes first.
    assert tasks.next_time("cron", "0 0 13 * 5", "UTC", utc(2026, 10, 3, 0, 0)) == utc(
        2026, 10, 9, 0, 0
    )
    assert tasks.next_time("cron", "*/15 * * * *", "UTC", utc(2026, 10, 3, 0, 7, 30)) == utc(
        2026, 10, 3, 0, 15
    )


def test_every_counts_from_the_given_time_and_a_date_that_never_comes_is_refused():
    assert tasks.next_time("every", "2h", "UTC", utc(2026, 1, 1, 0, 0)) == utc(2026, 1, 1, 2, 0)
    with pytest.raises(InvalidInputError, match="never falls due"):
        tasks.next_time("cron", "0 0 31 2 *", "UTC", utc(2026, 1, 1))


@pytest.mark.parametrize("name", ["Europe/Paris", "UTC", "America/New_York"])
def test_known_timezones_are_accepted(name):
    assert tasks.check_timezone(name) == name


@pytest.mark.parametrize("name", ["Mars/Olympus", "../../etc/passwd", "", None, "/etc/localtime"])
def test_unknown_timezones_and_paths_are_refused(name):
    with pytest.raises(InvalidInputError, match="Unknown timezone"):
        tasks.check_timezone(name)


# --- the world ---


@pytest.fixture
async def world(fresh_db):
    """Sam (Telegram 111, chat) and Alex (Telegram 222, chat), one agent each."""
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
                user.id,
                agent.id,
                identity.id,
            )
        await session.commit()
    return ids


async def _create(world, who="sam", kind="daily", expr="08:30", now=None, **extra) -> int:
    from app.db.session import session_scope

    async with session_scope() as session:
        task = await tasks.create_task(
            session, user_id=world[who], prompt=PROMPT, kind=kind, expr=expr, actor="test",
            now=now, **extra,
        )  # fmt: skip
        await session.commit()
        return task.id


async def _task(task_id) -> ScheduledTask:
    from app.db.session import session_scope

    async with session_scope() as session:
        return await session.get(ScheduledTask, task_id)


async def _logs() -> list[tuple[str, str, str]]:
    from app.db.session import session_scope

    async with session_scope() as session:
        rows = (await session.execute(select(ActionLog).order_by(ActionLog.id))).scalars()
        return [(r.direction.value, r.status.value, r.text) for r in rows]


@pytest.fixture
def sent(monkeypatch):
    """A running Telegram adapter: what it would send, by chat id."""
    from app.channels import notify

    out: list[tuple[str, str]] = []

    async def sender(external_id, text):
        out.append((external_id, text))

    notify.register_sender(Channel.TELEGRAM, sender)
    yield out
    notify.unregister_sender(Channel.TELEGRAM)


@pytest.fixture
def turn(monkeypatch):
    """The model's reply, and the arguments of each turn."""
    import app.graph

    calls: list[dict] = []
    state = {"reply": "Dry, 14 degrees.", "raise": None}

    async def fake_run_turn(channel, user_id, agent_id, text, **kwargs):
        calls.append({"channel": channel, "user_id": user_id, "agent_id": agent_id,
                      "text": text, **kwargs})  # fmt: skip
        if state["raise"]:
            raise state["raise"]
        return state["reply"]

    monkeypatch.setattr(app.graph, "run_turn", fake_run_turn)
    state["calls"] = calls
    return state


# --- the runner ---


async def test_a_task_due_in_one_minute_runs_once_and_is_delivered(world, sent, turn):
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    due = now + timedelta(minutes=1)
    task_id = await _create(world, kind="daily", expr=due.strftime("%H:%M"), now=now)
    assert tasks._as_utc((await _task(task_id)).next_run_at) == due

    assert await tasks.run_due(now) == []  # not yet
    assert await tasks.run_due(due + timedelta(seconds=1)) == [task_id]
    assert await tasks.run_due(due + timedelta(seconds=2)) == []  # claimed: runs once

    assert sent == [("111", f"Scheduled task #{task_id}:\nDry, 14 degrees.")]
    assert turn["calls"][0]["thread_id"] == f"task_{task_id}"
    assert turn["calls"][0]["text"] == PROMPT
    assert await _logs() == [
        ("inbound", "ok", f"[task {task_id}] {PROMPT}"),
        ("outbound", "ok", "Dry, 14 degrees."),
    ]
    task = await _task(task_id)
    assert (task.run_count, task.last_status, task.last_error) == (1, "ok", None)
    assert tasks._as_utc(task.next_run_at) == due + timedelta(days=1)


async def test_paused_no_task_runs_and_due_tasks_move_on(world, sent, turn):
    from app.db.session import session_scope

    now = datetime.now(UTC).replace(second=0, microsecond=0)
    task_id = await _create(world, kind="every", expr="1m", now=now)
    async with session_scope() as session:
        await tasks.set_paused(session, True, actor="test")
        await session.commit()
    for minute in range(1, 4):
        assert await tasks.run_due(now + timedelta(minutes=minute, seconds=1)) == []
    task = await _task(task_id)
    assert (task.run_count, task.last_status) == (0, "skipped")
    assert sent == [] and turn["calls"] == [] and await _logs() == []
    assert tasks._as_utc(task.next_run_at) > now + timedelta(minutes=3)


async def test_a_failed_turn_is_recorded_by_class_name_and_nothing_is_sent(world, sent, turn):
    turn["raise"] = RuntimeError("engine down: secret detail")
    task_id = await _create(world)
    result = await tasks.execute(task_id)
    task = await _task(task_id)
    assert result["status"] == "failed" and (task.last_status, task.last_error) == (
        "failed",
        "RuntimeError",
    )
    assert sent == []
    assert [(d, s) for d, s, _ in await _logs()] == [("inbound", "ok"), ("outbound", "failed")]


async def test_an_empty_reply_is_a_failed_run(world, sent, turn):
    turn["reply"] = "  \n"
    task_id = await _create(world)
    assert (await tasks.execute(task_id))["status"] == "failed"
    assert (await _task(task_id)).last_error == "EmptyReply" and sent == []
    # Since 2026-10-04 the graph itself raises on an empty reply (not stored): same label.
    from app.replies import EmptyReplyError

    turn["raise"] = EmptyReplyError("the model returned an empty reply")
    assert (await tasks.execute(task_id))["status"] == "failed"
    assert (await _task(task_id)).last_error == "EmptyReply" and sent == []


async def test_without_a_running_adapter_the_run_is_undelivered(world, turn):
    task_id = await _create(world)
    assert (await tasks.execute(task_id))["status"] == "undelivered"
    assert (await _task(task_id)).last_error == "not delivered"
    assert [(d, s) for d, s, _ in await _logs()] == [("inbound", "ok"), ("outbound", "failed")]


async def test_an_inactive_user_s_task_is_refused_without_a_turn(world, sent, turn):
    from app.admin import service
    from app.db.session import session_scope

    task_id = await _create(world)
    async with session_scope() as session:
        await service.update_user(session, world["sam"], is_active=False)
        await session.commit()
    assert (await tasks.execute(task_id))["status"] == "refused"
    assert turn["calls"] == [] and sent == [] and await _logs() == []


async def test_the_scheduler_survives_a_failed_pass(monkeypatch, caplog):
    import asyncio

    passes = []

    async def failing(now=None):
        passes.append(1)
        raise RuntimeError("database is locked")

    monkeypatch.setattr(tasks, "run_due", failing)
    runner = asyncio.create_task(tasks.run_scheduler(poll_seconds=0.01))
    await asyncio.sleep(1.3)
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner
    assert len(passes) >= 2
    assert "one pass failed" in caplog.text


# --- ownership ---


async def test_a_user_cannot_list_or_edit_another_users_task(world):
    from app.db.session import session_scope

    task_id = await _create(world, who="sam")
    async with session_scope() as session:
        assert await tasks.list_tasks(session, world["alex"]) == []
        for attempt in (
            tasks.get_task(session, task_id, world["alex"]),
            tasks.update_task(
                session, task_id, {"enabled": False}, actor="a", user_id=world["alex"]
            ),
            tasks.delete_task(session, task_id, actor="a", user_id=world["alex"]),
        ):
            with pytest.raises(tasks.TaskNotFoundError, match=f"No task {task_id}"):
                await attempt
    task = await _task(task_id)
    assert task is not None and task.enabled


async def test_a_task_is_delivered_only_on_one_of_the_owners_identities(world):
    with pytest.raises(Exception, match="No channel identity"):
        await _create(world, who="sam", channel_identity_id=world["alex_identity"])
    with pytest.raises(Exception, match="No agent"):
        await _create(world, who="sam", agent_id=world["alex_agent"])


async def test_a_user_holds_at_most_the_limit_of_tasks(world, monkeypatch):
    from app.admin.service import ConflictError

    monkeypatch.setattr(tasks, "MAX_TASKS_PER_USER", 2)
    await _create(world)
    await _create(world)
    with pytest.raises(ConflictError, match="already has 2 tasks"):
        await _create(world)


# --- purge and identity removal ---


async def test_purging_a_user_deletes_their_tasks_and_task_threads(world, monkeypatch):
    import app.graph
    from app.admin import service
    from app.db.session import session_scope

    deleted: list[str] = []

    async def fake_delete(thread_ids):
        deleted.extend(thread_ids)
        return len(deleted)

    monkeypatch.setattr(app.graph, "delete_threads", fake_delete)
    task_id = await _create(world, who="sam")
    other = await _create(world, who="alex")
    async with session_scope() as session:
        await service.delete_user(session, world["sam"], purge=True)
        await session.commit()
    assert await _task(task_id) is None and await _task(other) is not None
    assert f"task_{task_id}" in deleted and f"task_{other}" not in deleted


async def test_removing_the_identity_removes_the_tasks_delivered_on_it(world, monkeypatch):
    import app.graph
    from app.admin import service
    from app.db.session import session_scope

    async def fake_delete(thread_ids):
        return 0

    monkeypatch.setattr(app.graph, "delete_threads", fake_delete)
    task_id = await _create(world)
    async with session_scope() as session:
        await service.remove_channel_identity(session, world["sam"], world["sam_identity"])
        await session.commit()
    assert await _task(task_id) is None


async def test_the_prompt_is_encrypted_at_rest_and_counted_by_the_rekey(world):
    import sqlite3

    from app.admin.rekey import APP_COLUMNS
    from app.config import get_settings

    await _create(world)
    con = sqlite3.connect(get_settings().database_url.split("///", 1)[1])
    try:
        (stored,) = con.execute("select prompt from scheduled_tasks").fetchone()
    finally:
        con.close()
    assert PROMPT not in stored and stored.startswith("gAAAA")
    assert ("scheduled_tasks", "id", "prompt") in APP_COLUMNS


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
        c.app, c.ids = app, world
        yield c
    app.dependency_overrides.clear()
    deps.reset_failure_state()


async def test_the_api_refuses_bad_schedules_with_422(api):
    body = {"user_id": api.ids["sam"], "prompt": PROMPT}
    statuses = {}
    for kind, expr in [
        ("cron", "61 * * * *"),
        ("cron", "0 0 31 2 *"),
        ("every", "0m"),
        ("daily", "25:00"),
        ("weekly", "mon"),
    ]:
        statuses[f"{kind} {expr}"] = (
            await api.post("/tasks", json={**body, "kind": kind, "expr": expr})
        ).status_code
    assert set(statuses.values()) == {422}, statuses
    good = await api.post("/tasks", json={**body, "kind": "cron", "expr": "0 8 * * 1-5"})
    assert good.status_code == 201, good.text


async def test_the_api_lists_edits_pauses_runs_and_deletes_with_admin_events(api, sent, turn):
    from app.db.session import session_scope

    created = await api.post(
        "/tasks", json={"user_id": api.ids["sam"], "prompt": PROMPT, "kind": "daily",
                        "expr": "08:30"},
    )  # fmt: skip
    assert created.status_code == 201, created.text
    task = created.json()
    assert (task["channel"], task["timezone"], task["enabled"]) == ("telegram", "UTC", True)
    assert task["agent_id"] == api.ids["sam_agent"] and task["next_run_at"]

    assert (await api.get("/tasks", params={"user_id": api.ids["alex"]})).json() == []
    assert len((await api.get("/tasks")).json()) == 1
    assert (await api.get("/tasks/999")).status_code == 404

    stopped = await api.patch(f"/tasks/{task['id']}", json={"enabled": False})
    assert stopped.json()["enabled"] is False and stopped.json()["next_run_at"] is None
    started = await api.patch(f"/tasks/{task['id']}", json={"enabled": True, "expr": "09:15"})
    assert (
        started.json()["next_run_at"].endswith("09:15:00Z")
        or "09:15:00" in started.json()["next_run_at"]
    )
    assert (await api.patch(f"/tasks/{task['id']}", json={"kind": "cron"})).status_code == 422

    assert (await api.get("/tasks/pause")).json() == {"paused": False}
    assert (await api.post("/tasks/pause", json={"paused": True})).json() == {"paused": True}
    assert (await api.get("/tasks/pause")).json() == {"paused": True}

    job = await api.post(f"/tasks/{task['id']}/run")
    assert job.status_code == 202
    from app.admin.jobs import registry

    await registry.get(job.json()["id"]).task
    assert registry.get(job.json()["id"]).result["status"] == "ok"
    assert sent == [("111", f"Scheduled task #{task['id']}:\nDry, 14 degrees.")]

    assert (await api.delete(f"/tasks/{task['id']}")).status_code == 204
    assert (await api.get(f"/tasks/{task['id']}")).status_code == 404

    async with session_scope() as session:
        events = list(
            (
                await session.execute(
                    select(AdminEvent)
                    .where(AdminEvent.action.like("task%"))
                    .order_by(AdminEvent.id)
                )
            ).scalars()
        )
    assert [e.action for e in events] == [
        "task.create",
        "task.update",
        "task.update",
        "tasks.pause",
        "task.run",
        "task.delete",
    ]
    assert all(PROMPT not in json.dumps(e.details) for e in events)


async def test_the_timezone_is_set_through_the_user_and_moves_the_next_run(api):
    created = await api.post(
        "/tasks", json={"user_id": api.ids["sam"], "prompt": PROMPT, "kind": "daily",
                        "expr": "08:30"},
    )  # fmt: skip
    before = created.json()["next_run_at"]
    assert (
        await api.patch(f"/users/{api.ids['sam']}", json={"timezone": "Mars/Olympus"})
    ).status_code == 422
    changed = await api.patch(f"/users/{api.ids['sam']}", json={"timezone": "Asia/Tokyo"})
    assert changed.status_code == 200 and changed.json()["timezone"] == "Asia/Tokyo"
    after = (await api.get(f"/tasks/{created.json()['id']}")).json()
    assert after["timezone"] == "Asia/Tokyo" and after["next_run_at"] != before
    assert "23:30:00" in after["next_run_at"]  # 08:30 in Tokyo is 23:30 UTC


async def test_task_routes_need_the_admin_scope(api):
    from app.api.scopes import Principal, Scope, get_principal

    api.app.dependency_overrides[get_principal] = lambda: Principal("op", Scope.OPERATE)
    assert (await api.get("/tasks")).status_code == 403
    assert (await api.get("/tasks/pause")).status_code == 403
    assert (await api.post("/tasks/pause", json={"paused": True})).status_code == 403
    assert (await api.post("/tasks/1/run")).status_code == 403


# --- the /task command ---


class _Sink:
    def __init__(self):
        self.sent: list[str] = []

    async def reply(self, text: str) -> None:
        self.sent.append(text)


async def _command(external_id: str, text: str) -> str:
    from app.channels.dispatch import handle_task_command
    from app.channels.schema import NormalizedEvent
    from app.db.session import session_scope

    sink = _Sink()
    event = NormalizedEvent(external_id, Channel.TELEGRAM, f"/task {text}".strip(), sink.reply)
    async with session_scope() as session:
        await handle_task_command(session, event, text)
    return sink.sent[-1]


async def test_the_task_command_adds_lists_pauses_and_sets_the_timezone(world, monkeypatch):
    from app import builder

    async def no_schedule(messages, schema):
        # "every often" is not the form "every 2h": read as words, no schedule found.
        return {"schedule_kind": None, "schedule_expr": None, "prompt": None, "reason": "vague"}

    monkeypatch.setattr(builder, "ask_model", no_schedule)
    answer = await _command("111", "add daily 07:45 Tell me   the news")
    assert answer.startswith("Task created. #1 [on] daily 07:45")
    assert "Tell me   the news" in answer, "the prompt is kept as typed"
    assert (await _command("111", "add cron 0 8 * * 1-5 Weekly plan")).startswith(
        "Task created. #2 [on] cron 0 8 * * 1-5"
    )
    assert "no schedule found in the words (vague)" in await _command("111", "add every often Nope")
    listed = await _command("111", "")
    assert "#1 [on]" in listed and "#2 [on]" in listed and "Timezone: UTC." in listed
    assert (await _command("111", "pause 1")).startswith("#1 [off]")
    assert (await _command("111", "tz Europe/Paris")).startswith("Your timezone is now")
    assert "Unknown timezone" in await _command("111", "tz Nowhere/Land")
    assert (await _command("111", "resume 1")).startswith("#1 [on]")
    assert await _command("111", "delete 2") == "Task #2 deleted."


async def test_the_task_command_never_reaches_another_users_task(world):
    await _command("111", "add daily 07:45 Sam's own")
    assert "You have no task." in await _command("222", "")
    for action in ("pause", "resume", "delete", "run"):
        assert await _command("222", f"{action} 1") == "No task 1"
    task = await _task(1)
    assert task is not None and task.enabled


async def test_an_email_identity_gets_the_reply_by_smtp_at_its_stored_address(
    fresh_db, monkeypatch
):
    from app.admin import service
    from app.channels import email
    from app.config import get_settings
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as session:
        user = await service.create_user(session, "Mail")
        identity = await service.add_channel_identity(
            session, user.id, Channel.EMAIL, "someone@example.org"
        )
        await session.commit()
        mailed: list[tuple] = []
        monkeypatch.setattr(email, "_send_reply_sync", lambda *a: mailed.append(a))
        monkeypatch.setenv("EMAIL_SMTP_HOST", "smtp.example.org")
        get_settings.cache_clear()
        assert await tasks.deliver(identity, "the reply", "[task 7] result") is True
        assert mailed == [("someone@example.org", "[task 7] result", "the reply")]
        monkeypatch.setenv("EMAIL_SMTP_HOST", "")
        get_settings.cache_clear()
        assert await tasks.deliver(identity, "the reply", "[task 7] result") is False
    assert len(mailed) == 1


# --- the shortest time between two runs of a task a user schedules themselves ---


@pytest.mark.parametrize(
    "kind, expr, minutes",
    [
        ("every", "1m", 1),
        ("every", "2h", 120),
        ("daily", "09:00", 1440),
        ("cron", "* * * * *", 1),
        ("cron", "*/10 * * * *", 10),
        ("cron", "0,1 9 * * *", 1),
        ("cron", "0 9 * * 4", 10080),
        # The first gap (00:59 to 00:00 the next day) is the long one: every gap is compared.
        ("cron", "0,59 0 * * *", 59),
    ],
)
def test_the_shortest_gap_of_a_schedule(kind, expr, minutes):
    assert tasks.shortest_gap_minutes(kind, expr) == minutes


async def _count_tasks() -> int:
    from sqlalchemy import func, select

    from app.db.session import session_scope

    async with session_scope() as session:
        return (await session.execute(select(func.count()).select_from(ScheduledTask))).scalar_one()


async def test_a_user_cannot_schedule_a_task_more_often_than_the_floor(world):
    for form in ("every 1m", "every 14m", "cron * * * * *", "cron 0,1 9 * * *"):
        answer = await _command("111", f"add {form} Ping")
        assert answer.startswith("a task you schedule yourself runs at most every 15 minutes"), (
            form, answer,
        )  # fmt: skip
    assert await _count_tasks() == 0
    assert (await _command("111", "add every 15m Ping")).startswith("Task created.")
    assert await _count_tasks() == 1


async def test_a_schedule_in_words_under_the_floor_is_refused_before_asking(world, monkeypatch):
    from app import builder

    async def every_minute(messages, schema):
        return {"schedule_kind": "every", "schedule_expr": "1m", "prompt": "Ping", "reason": ""}

    monkeypatch.setattr(builder, "ask_model", every_minute)
    asked: list[str] = []

    async def confirm(question, timeout):
        asked.append(question)
        return True

    from app.channels.dispatch import handle_task_command
    from app.channels.schema import NormalizedEvent
    from app.db.session import session_scope

    sink = _Sink()
    event = NormalizedEvent(
        "111", Channel.TELEGRAM, "/task add every minute ping me", sink.reply, confirm=confirm
    )
    async with session_scope() as session:
        await handle_task_command(session, event, "add every minute ping me")
    assert sink.sent[-1].startswith("a task you schedule yourself runs at most every 15 minutes")
    assert asked == [] and await _count_tasks() == 0


async def test_the_floor_is_a_setting_and_zero_turns_it_off(world, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("TASK_MIN_INTERVAL_MINUTES", "0")
    get_settings.cache_clear()
    try:
        assert (await _command("111", "add every 1m Ping")).startswith("Task created.")
        monkeypatch.setenv("TASK_MIN_INTERVAL_MINUTES", "120")
        get_settings.cache_clear()
        assert "at most every 120 minutes" in await _command("111", "add every 1h Ping")
    finally:
        get_settings.cache_clear()


async def test_an_administrator_is_not_held_to_the_floor(world):
    await _create(world, kind="every", expr="1m")
    assert await _count_tasks() == 1
