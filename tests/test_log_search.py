"""Tests: searching the audit trail.

One dataset (7 rows, 2 users, 3 agents, 2 channels, both directions),
queried through the service function, through GET /logs and through the
admin console, plus a spy test proving the API and the console call the
same service function.

Row ids follow insertion order, which is also chronological order:

 id user agent channel  dir       created (UTC)         text
  1   1    1    telegram inbound  2026-09-10 08:00:00   Please remind me to buy milk
  2   1    1    telegram outbound 2026-09-10 08:00:05   Sure, I will remind you about the milk
  3   1    2    email    inbound  2026-09-12 09:30:00   Quarterly report draft attached
  4   2    3    telegram inbound  2026-09-15 12:00:00   What is the weather in Paris?
  5   2    3    telegram outbound 2026-09-15 12:00:03   It is sunny in Paris.
  6   1    1    telegram inbound  2026-09-20 07:00:00   MILK again, do not forget
  7   2    3    email    inbound  2026-09-20 10:00:00   Invoice for the milk delivery
"""

import re
import sqlite3
from datetime import UTC, datetime, timedelta, timezone

import httpx
import pytest

from app.db.models import Channel, Direction
from tests._console import command, run_console

ROWS = [
    (1, 1, "telegram", "inbound", datetime(2026, 9, 10, 8, 0, 0, tzinfo=UTC),
     "Please remind me to buy milk"),
    (1, 1, "telegram", "outbound", datetime(2026, 9, 10, 8, 0, 5, tzinfo=UTC),
     "Sure, I will remind you about the milk"),
    (1, 2, "email", "inbound", datetime(2026, 9, 12, 9, 30, 0, tzinfo=UTC),
     "Quarterly report draft attached"),
    (2, 3, "telegram", "inbound", datetime(2026, 9, 15, 12, 0, 0, tzinfo=UTC),
     "What is the weather in Paris?"),
    (2, 3, "telegram", "outbound", datetime(2026, 9, 15, 12, 0, 3, tzinfo=UTC),
     "It is sunny in Paris."),
    (1, 1, "telegram", "inbound", datetime(2026, 9, 20, 7, 0, 0, tzinfo=UTC),
     "MILK again, do not forget"),
    (2, 3, "email", "inbound", datetime(2026, 9, 20, 10, 0, 0, tzinfo=UTC),
     "Invoice for the milk delivery"),
]


@pytest.fixture
async def populated(fresh_db):
    from app.db.models import ActionLog, Agent, User
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as session:
        alice, bob = User(display_name="Alice"), User(display_name="Bob")
        session.add_all([alice, bob])
        await session.flush()
        session.add_all([
            Agent(user_id=alice.id, name="default"),
            Agent(user_id=alice.id, name="work"),
            Agent(user_id=bob.id, name="default"),
        ])
        await session.flush()
        for user_id, agent_id, channel, direction, created_at, text in ROWS:
            session.add(ActionLog(
                user_id=user_id, agent_id=agent_id, channel=Channel(channel),
                direction=Direction(direction), text=text, created_at=created_at,
            ))
        await session.commit()


async def _search(**filters) -> list[int]:
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as session:
        return [e.id for e in await service.search_action_logs(session, **filters)]


# --- service function ---


async def test_no_filter_returns_everything_newest_first(populated):
    assert await _search() == [7, 6, 5, 4, 3, 2, 1]


async def test_each_single_filter(populated):
    assert await _search(user_id=2) == [7, 5, 4]
    assert await _search(agent_id=2) == [3]
    assert await _search(channel=Channel.EMAIL) == [7, 3]
    assert await _search(direction=Direction.OUTBOUND) == [5, 2]


async def test_user_plus_keyword_returns_only_matching_rows(populated):
    # The acceptance criterion: two filters combined. User 2 also has a
    # "milk" row (id 7) and user 1 has non-milk rows, neither may leak in.
    assert await _search(user_id=1, keyword="milk") == [6, 2, 1]


async def test_three_filters_combined(populated):
    ids = await _search(channel=Channel.TELEGRAM, direction=Direction.INBOUND, keyword="paris")
    assert ids == [4]


async def test_keyword_is_case_insensitive_and_matches_decrypted_text(populated):
    assert await _search(keyword="MILK") == await _search(keyword="milk") == [7, 6, 2, 1]
    assert await _search(keyword="  Milk  ") == [7, 6, 2, 1]


async def test_ciphertext_is_never_what_the_keyword_is_matched_against(populated):
    from app.config import get_settings

    db_file = get_settings().database_url.split("///", 1)[1]
    raw = [r[0] for r in sqlite3.connect(db_file).execute("select text from action_logs")]
    assert all(t.startswith("gAAAA") for t in raw), "the column really holds ciphertext"
    assert await _search(keyword="gAAAA") == []


async def test_blank_keyword_means_no_keyword(populated):
    assert await _search(keyword="   ") == [7, 6, 5, 4, 3, 2, 1]
    assert await _search(keyword="") == [7, 6, 5, 4, 3, 2, 1]


async def test_no_match_returns_empty_list(populated):
    assert await _search(user_id=1, keyword="paris") == []
    assert await _search(user_id=99) == []


async def test_date_window_is_since_inclusive_until_exclusive(populated):
    row3 = datetime(2026, 9, 12, 9, 30, 0, tzinfo=UTC)
    day = datetime(2026, 9, 12, tzinfo=UTC)
    end = datetime(2026, 9, 16, tzinfo=UTC)
    assert await _search(since=day, until=end) == [5, 4, 3]
    assert await _search(since=row3, until=end) == [5, 4, 3]
    assert await _search(since=row3 + timedelta(seconds=1), until=end) == [5, 4]
    assert await _search(since=day, until=row3) == [], "until is exclusive"
    assert await _search(since=day, until=row3 + timedelta(seconds=1)) == [3]


async def test_naive_datetimes_are_utc_and_aware_ones_are_converted(populated):
    assert await _search(since=datetime(2026, 9, 20, 7, 0, 0)) == [7, 6]
    paris = timezone(timedelta(hours=2))
    # 09:00 at UTC+2 is 07:00 UTC, so row 6 (07:00 UTC) is still included.
    assert await _search(since=datetime(2026, 9, 20, 9, 0, 0, tzinfo=paris)) == [7, 6]
    # Read as UTC (timezone ignored) it would wrongly drop row 6.
    assert await _search(since=datetime(2026, 9, 20, 9, 0, 0)) == [7]


async def test_limit_and_offset_page_through_matches(populated):
    assert await _search(keyword="milk", limit=2) == [7, 6]
    assert await _search(keyword="milk", limit=2, offset=2) == [2, 1]
    assert await _search(keyword="milk", limit=2, offset=4) == []
    assert await _search(limit=3, offset=1) == [6, 5, 4]
    pages: list[int] = []
    for offset in range(4):
        pages += await _search(keyword="milk", limit=1, offset=offset)
    assert pages == await _search(keyword="milk")


@pytest.mark.parametrize("limit", [0, -1, 501])
async def test_limit_out_of_range_is_rejected(populated, limit):
    with pytest.raises(ValueError, match="limit"):
        await _search(limit=limit)


async def test_negative_offset_is_rejected(populated):
    with pytest.raises(ValueError, match="offset"):
        await _search(offset=-1)


async def test_keyword_search_crosses_batch_boundaries_and_stops_early(populated, monkeypatch):
    from app.admin import service
    from app.db.session import session_scope

    monkeypatch.setattr(service, "_LOG_SEARCH_BATCH", 2)
    assert await _search(keyword="milk") == [7, 6, 2, 1]
    assert await _search(keyword="milk", offset=1, limit=2) == [6, 2]

    async with session_scope() as session:
        original = session.execute
        calls = 0

        async def counting(*args, **kwargs):
            nonlocal calls
            calls += 1
            return await original(*args, **kwargs)

        monkeypatch.setattr(session, "execute", counting)
        found = await service.search_action_logs(session, keyword="milk", limit=1)
    assert [e.id for e in found] == [7]
    assert calls == 1, "the newest row already matches: no further batch may be read"


# --- Admin API: GET /logs ---


@pytest.fixture
async def api(populated, monkeypatch):
    from app.api.app import app
    from app.config import get_settings

    monkeypatch.setenv("API_SERVER_KEY", "test-key")
    get_settings.cache_clear()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        yield client


AUTH = {"Authorization": "Bearer test-key"}


async def test_api_requires_the_key(api):
    assert (await api.get("/logs")).status_code == 401
    assert (await api.get("/logs", headers={"Authorization": "Bearer wrong"})).status_code == 401


async def test_api_two_filters_return_only_matching_rows_with_decrypted_text(api):
    r = await api.get("/logs", params={"user_id": 1, "keyword": "milk"}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert [e["id"] for e in body] == [6, 2, 1]
    assert body[2]["text"] == "Please remind me to buy milk"
    assert set(body[0]) == {
        "id", "user_id", "agent_id", "channel", "direction", "status", "text", "created_at",
    }
    assert body[0]["channel"] == "telegram" and body[0]["direction"] == "inbound"


async def test_api_channel_direction_and_date_filters(api):
    r = await api.get("/logs", params={"channel": "email", "direction": "inbound"}, headers=AUTH)
    assert [e["id"] for e in r.json()] == [7, 3]
    r = await api.get("/logs", params={"since": "2026-09-20T09:00:00+02:00"}, headers=AUTH)
    assert [e["id"] for e in r.json()] == [7, 6]
    r = await api.get(
        "/logs", params={"since": "2026-09-12", "until": "2026-09-16"}, headers=AUTH
    )
    assert [e["id"] for e in r.json()] == [5, 4, 3]


async def test_api_paging(api):
    r = await api.get("/logs", params={"keyword": "milk", "limit": 2, "offset": 2}, headers=AUTH)
    assert [e["id"] for e in r.json()] == [2, 1]


@pytest.mark.parametrize(
    "params",
    [{"channel": "fax"}, {"direction": "sideways"}, {"limit": 0}, {"limit": 501},
     {"offset": -1}, {"user_id": "abc"}, {"since": "not-a-date"}],
)
async def test_api_rejects_invalid_parameters(api, params):
    assert (await api.get("/logs", params=params, headers=AUTH)).status_code == 422


# --- Admin console (through the API) ---


def _ids(out: str) -> set[int]:
    """The ids of the rows of the table the console printed."""
    return {int(m.group(1)) for m in re.finditer(r"^(\d+)\s+\d+\s+\d+\s", out, re.M)}


async def test_console_search_combines_filters(populated, monkeypatch):
    out = await run_console(monkeypatch, *command("search-logs", user_id=1, keyword="milk"))
    assert _ids(out) == {1, 2, 6}
    assert "MILK again, do not forget" in out


async def test_console_date_filters_bound_the_search(populated, monkeypatch):
    out = await run_console(
        monkeypatch,
        *command("search-logs", since="2026-09-20T00:00:00Z", until="2026-09-21T00:00:00Z"),
    )
    assert _ids(out) == {6, 7}


async def test_console_reports_no_match_and_a_limit(populated, monkeypatch):
    out = await run_console(monkeypatch, *command("search-logs", user_id=1, keyword="paris"))
    assert "(none)" in out
    out = await run_console(monkeypatch, *command("search-logs", limit=2))
    assert len(_ids(out)) == 2


@pytest.mark.parametrize(
    "values",
    [
        {"user_id": "abc"},
        {"channel": "fax"},
        {"direction": "sideways"},
        {"since": "20/09/2026"},
        {"limit": "abc"},
        {"limit": "0"},
        {"limit": "501"},
    ],
)
async def test_console_invalid_input_is_reported_not_raised(populated, monkeypatch, values):
    out = await run_console(monkeypatch, *command("search-logs", **values), *command("whoami"))
    assert "Invalid input, nothing sent." in out or "Not done (HTTP 422)" in out
    assert '"actor"' in out, "the session went on to the next command"


# --- one service function behind both front ends ---


async def test_api_and_console_call_the_same_service_function(api, populated, monkeypatch):
    from app.admin import service

    calls: list[dict] = []
    real = service.search_action_logs

    async def spy(session, **kwargs):
        calls.append(kwargs)
        return await real(session, **kwargs)

    monkeypatch.setattr(service, "search_action_logs", spy)

    r = await api.get("/logs", params={"user_id": 1, "keyword": "milk"}, headers=AUTH)
    assert r.status_code == 200
    api_ids = [e["id"] for e in r.json()]

    await run_console(monkeypatch, *command("search-logs", user_id=1, keyword="milk"))

    assert len(calls) == 2, "exactly one service call per front end (the console via the API)"
    for kwargs in calls:
        assert kwargs["user_id"] == 1 and kwargs["keyword"] == "milk"
    assert api_ids == await _search(user_id=1, keyword="milk") == [6, 2, 1]


async def test_keyword_search_over_more_rows_than_one_real_batch(populated):
    """The batch boundary tests above shrink the batch to 2. This one keeps
    the real 500-row batch and 1,300 rows, so it crosses two real boundaries.
    """
    from app.admin import service
    from app.db.models import ActionLog
    from app.db.session import session_scope

    assert service._LOG_SEARCH_BATCH == 500
    base = datetime(2026, 9, 21, tzinfo=UTC)
    async with session_scope() as session:
        for i in range(1300):  # ids 8..1307, after the 7 seeded rows
            text = f"row {i} needle" if i % 7 == 0 else f"row {i}"
            if i == 3:
                text = "row 3 rare"
            session.add(ActionLog(
                user_id=1, agent_id=1, channel=Channel.TELEGRAM, direction=Direction.INBOUND,
                text=text, created_at=base + timedelta(seconds=i),
            ))
        await session.commit()

    needle_ids = [8 + i for i in range(1300) if i % 7 == 0]  # oldest first
    expected_newest_first = needle_ids[::-1]
    assert len(needle_ids) == 186

    assert await _search(keyword="needle", limit=500) == expected_newest_first
    assert await _search(keyword="needle", limit=50, offset=100) == expected_newest_first[100:150]
    assert await _search(keyword="needle", limit=500, offset=180) == expected_newest_first[180:]
    assert await _search(keyword="needle", limit=10, offset=186) == []

    # An old-only match forces a read of every batch: 1,307 rows = 500 + 500 + 307.
    async with session_scope() as session:
        original = session.execute
        calls = 0

        async def counting(*args, **kwargs):
            nonlocal calls
            calls += 1
            return await original(*args, **kwargs)

        monkey = pytest.MonkeyPatch()
        monkey.setattr(session, "execute", counting)
        try:
            found = await service.search_action_logs(session, keyword="rare")
        finally:
            monkey.undo()
    assert [e.text for e in found] == ["row 3 rare"]
    assert calls == 3, "1,307 rows must be read as 500 + 500 + 307"

    # The newest matching row is in the first batch: one query is enough.
    async with session_scope() as session:
        original = session.execute
        calls = 0
        monkey = pytest.MonkeyPatch()
        monkey.setattr(session, "execute", counting)
        try:
            found = await service.search_action_logs(session, keyword="needle", limit=1)
        finally:
            monkey.undo()
    assert [e.id for e in found] == [expected_newest_first[0]]
    assert calls == 1


# --- status filter and display ---


@pytest.fixture
async def with_statuses(populated):
    """Row 5 failed, row 7 denied, the others ok."""
    from sqlalchemy import update

    from app.db.models import ActionLog, ActionStatus
    from app.db.session import session_scope

    async with session_scope() as session:
        await session.execute(
            update(ActionLog).where(ActionLog.id == 5).values(status=ActionStatus.FAILED)
        )
        await session.execute(
            update(ActionLog).where(ActionLog.id == 7).values(status=ActionStatus.DENIED)
        )
        await session.commit()


async def test_status_filter(with_statuses):
    from app.db.models import ActionStatus

    assert await _search(status=ActionStatus.FAILED) == [5]
    assert await _search(status=ActionStatus.DENIED) == [7]
    assert await _search(status=ActionStatus.OK) == [6, 4, 3, 2, 1]
    assert await _search(status=ActionStatus.FAILED, user_id=1) == []
    assert await _search(status=ActionStatus.DENIED, keyword="milk") == [7]


async def test_new_rows_default_to_ok(populated):
    from sqlalchemy import select

    from app.db.models import ActionLog, ActionStatus
    from app.db.session import session_scope

    async with session_scope() as session:
        statuses = {e.status for e in (await session.execute(select(ActionLog))).scalars()}
    assert statuses == {ActionStatus.OK}


async def test_api_status_filter_and_field(api, with_statuses):
    r = await api.get("/logs", params={"status": "failed"}, headers=AUTH)
    assert [(e["id"], e["status"]) for e in r.json()] == [(5, "failed")]
    r = await api.get("/logs", params={"status": "denied", "keyword": "milk"}, headers=AUTH)
    assert [(e["id"], e["status"]) for e in r.json()] == [(7, "denied")]
    assert (await api.get("/logs", params={"status": "weird"}, headers=AUTH)).status_code == 422


async def test_console_status_filter(with_statuses, monkeypatch):
    out = await run_console(monkeypatch, *command("search-logs", status="failed"))
    assert _ids(out) == {5} and "failed" in out
    out = await run_console(monkeypatch, *command("search-logs"))
    assert "failed" in out and "denied" in out
    out = await run_console(monkeypatch, *command("search-logs", status="weird"))
    assert "Not done (HTTP 422)" in out
