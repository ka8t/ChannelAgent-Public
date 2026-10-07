"""Tests: the admin's own actions are recorded. One event per
mutating service call, written in the caller's transaction, identical
through the API and the console, details encrypted and free of secrets,
and reads of the decrypted logs recorded too.
"""

import json
import sqlite3

import httpx
import pytest

from app.db.models import Channel, Direction, PermissionKind
from tests._console import command, run_console

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
AUTH = {"Authorization": f"Bearer {KEY}"}


def _sql(query: str, *args):
    from app.config import get_settings

    con = sqlite3.connect(get_settings().database_url.split("///", 1)[1])
    try:
        return con.execute(query, args).fetchall()
    finally:
        con.close()


def _events():
    return _sql("select actor, action, target_type, target_id from admin_events order by id")


@pytest.fixture
async def world(fresh_db, monkeypatch):
    from app.admin import service
    from app.config import get_settings
    from app.db.session import init_db, session_scope

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    await init_db()
    async with session_scope() as s:
        alice = await service.create_user(s, "Alice", actor="test")
        identity = await service.add_channel_identity(
            s, alice.id, Channel.TELEGRAM, "1", actor="test"
        )
        await service.grant_identity_permission(
            s, alice.id, identity.id, PermissionKind.CHAT, actor="test"
        )
        agent = await service.create_agent(s, alice.id, "default", actor="test")
        await service.record_action(
            s,
            user_id=alice.id,
            agent_id=agent.id,
            channel=Channel.TELEGRAM,
            direction=Direction.INBOUND,
            text="secret question",
        )
        await service.request_access(s, Channel.TELEGRAM, "777", "let me in")
        await service.request_access(s, Channel.TELEGRAM, "888", "me too")
        await s.commit()
    _sql_exec("delete from admin_events")


def _sql_exec(query: str):
    from app.config import get_settings

    con = sqlite3.connect(get_settings().database_url.split("///", 1)[1])
    try:
        con.execute(query)
        con.commit()
    finally:
        con.close()


# --- exactly one event per mutating function ---

MUTATIONS = [
    ("create_user", lambda s, sv: sv.create_user(s, "Bob", actor="t"), "user.create", "user"),
    (
        "update_user",
        lambda s, sv: sv.update_user(s, 1, is_active=False, actor="t"),
        "user.update",
        "user",
    ),
    (
        "delete_user",
        lambda s, sv: sv.delete_user(s, 1, purge=True, actor="t"),
        "user.delete",
        "user",
    ),
    (
        "add_channel_identity",
        lambda s, sv: sv.add_channel_identity(s, 1, Channel.MATRIX, "@a:b", actor="t"),
        "identity.add",
        "user",
    ),
    (
        "remove_channel_identity",
        lambda s, sv: sv.remove_channel_identity(s, 1, 1, actor="t"),
        "identity.remove",
        "user",
    ),
    (
        "set_identity_agent",
        lambda s, sv: sv.set_identity_agent(s, 1, 1, 1, actor="t"),
        "identity.set_agent",
        "user",
    ),
    (
        "grant_identity_permission",
        lambda s, sv: sv.grant_identity_permission(s, 1, 1, PermissionKind.ADMIN, actor="t"),
        "permission.grant",
        "user",
    ),
    (
        "revoke_identity_permission",
        lambda s, sv: sv.revoke_identity_permission(s, 1, 1, PermissionKind.CHAT, actor="t"),
        "permission.revoke",
        "user",
    ),
    (
        "approve_request",
        lambda s, sv: sv.approve_request(s, 1, resolved_by="t"),
        "request.approve",
        "access_request",
    ),
    (
        "deny_request",
        lambda s, sv: sv.deny_request(s, 2, resolved_by="t"),
        "request.deny",
        "access_request",
    ),
    (
        "create_agent",
        lambda s, sv: sv.create_agent(s, 1, "second", actor="t"),
        "agent.create",
        "agent",
    ),
    (
        "rename_agent",
        lambda s, sv: sv.rename_agent(s, 1, "renamed", actor="t"),
        "agent.rename",
        "agent",
    ),
    (
        "set_agent_active",
        lambda s, sv: sv.set_agent_active(s, 1, False, actor="t"),
        "agent.set_active",
        "agent",
    ),
]


@pytest.mark.parametrize(
    ("name", "call", "action", "target_type"), MUTATIONS, ids=[m[0] for m in MUTATIONS]
)
async def test_each_mutating_function_writes_exactly_one_event(
    world, name, call, action, target_type
):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        await call(s, service)
        await s.commit()
    rows = _events()
    assert len(rows) == 1, rows
    actor, got_action, got_type, target_id = rows[0]
    assert (actor, got_action, got_type) == ("t", action, target_type)
    assert target_id is not None


async def test_the_mutations_table_covers_every_public_mutating_function():
    """A new mutating service function must be added to MUTATIONS."""
    import inspect

    from app.admin import service

    covered = {m[0] for m in MUTATIONS}
    exempt = {
        # System paths, not admin actions: lazy default agent, access requests
        # raised by a message, conversation log lines.
        "get_or_create_default_agent",
        "request_access",
        "ensure_access_request",
        "record_action",
        "record_admin_event",
    }
    mutating = {
        name
        for name, fn in inspect.getmembers(service, inspect.iscoroutinefunction)
        if fn.__module__ == service.__name__
        and not name.startswith("_")
        and name.split("_")[0]
        in {
            "create",
            "update",
            "delete",
            "add",
            "remove",
            "grant",
            "revoke",
            "approve",
            "deny",
            "rename",
            "set",
        }
    }
    assert mutating - exempt == covered, (mutating - exempt) ^ covered


async def test_read_functions_write_no_event(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        await service.list_users(s)
        await service.get_user(s, 1)
        await service.get_user_detail(s, 1)
        await service.list_agents(s, 1)
        await service.list_requests(s, None)
        await service.list_channel_identities(s, 1)
        await service.list_identity_permissions(s, 1, 1)
        await service.storage_overview(s)
        await s.commit()
    assert _events() == []


async def test_a_refused_operation_writes_no_event(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        with pytest.raises(service.UserNotFoundError):
            await service.update_user(s, 999, is_active=False, actor="t")
        with pytest.raises(service.UserHasHistoryError):
            await service.delete_user(s, 1, actor="t")
        await s.commit()
    assert _events() == []


async def test_the_event_is_in_the_same_transaction_as_the_change(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        await service.create_user(s, "Ghost", actor="t")
        await s.rollback()
    assert _sql("select count(*) from users where display_name = 'Ghost'") == [(0,)]
    assert _events() == []


# --- what is recorded ---


async def test_details_are_encrypted_at_rest_and_readable_through_the_service(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        await service.create_user(s, "Zebulon Quux", actor="t")
        await s.commit()
    raw = _sql("select details from admin_events")[0][0]
    assert "Zebulon" not in raw
    async with session_scope() as s:
        [event] = await service.search_admin_events(s, actor="t")
    assert json.loads(event.details) == {"display_name": "Zebulon Quux"}


async def test_an_email_address_is_never_written_to_the_event(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        await service.add_channel_identity(
            s, 1, Channel.EMAIL, "private.person@example.org", actor="t"
        )
        await s.commit()
    async with session_scope() as s:
        [event] = await service.search_admin_events(s, action="identity.add")
    assert "private.person" not in event.details
    assert "example.org" not in event.details
    assert json.loads(event.details)["channel"] == "email"


async def test_deleting_a_user_keeps_its_event_after_the_purge(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        await service.delete_user(s, 1, purge=True, actor="t")
        await s.commit()
    assert _sql("select count(*) from users where id = 1") == [(0,)]
    async with session_scope() as s:
        [event] = await service.search_admin_events(s, action="user.delete")
    assert event.target_id == 1
    details = json.loads(event.details)
    assert details["purge"] is True and details["logs_deleted"] == 1


async def test_the_actor_defaults_to_unspecified_and_must_not_be_blank(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        await service.create_user(s, "NoActor")
        with pytest.raises(service.InvalidInputError):
            await service.create_user(s, "Blank", actor="  ")
        await s.commit()
    assert [e[0] for e in _events()] == ["unspecified"]


# --- reading the decrypted logs is recorded ---


async def test_searching_the_logs_records_the_filters_and_not_the_results(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        found = await service.search_action_logs(
            s, keyword="secret", channel=Channel.TELEGRAM, user_id=1, limit=5, actor="t"
        )
        await s.commit()
    assert len(found) == 1
    [event] = _events()
    assert event == ("t", "logs.search", "action_logs", None)
    async with session_scope() as s:
        [recorded] = await service.search_admin_events(s, action="logs.search")
    details = json.loads(recorded.details)
    assert details["keyword"] == "secret"
    assert details["channel"] == "telegram" and details["user_id"] == 1
    assert details["limit"] == 5 and details["results"] == 1
    assert "secret question" not in recorded.details


# --- searching the events ---


async def _three_events():
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        await service.create_user(s, "A", actor="api")
        await service.create_user(s, "B", actor="console")
        await service.create_agent(s, 1, "x", actor="api")
        await s.commit()


async def test_search_events_filters_and_paging(world):
    from app.admin import service
    from app.db.session import session_scope

    await _three_events()
    async with session_scope() as s:
        by_actor = await service.search_admin_events(s, actor="api")
        by_action = await service.search_admin_events(s, action="user.create")
        by_target = await service.search_admin_events(s, target_type="agent")
        newest_first = await service.search_admin_events(s)
        page = await service.search_admin_events(s, limit=1, offset=1)
    assert [e.action for e in by_actor] == ["agent.create", "user.create"]
    assert len(by_action) == 2 and len(by_target) == 1
    assert [e.id for e in newest_first] == sorted((e.id for e in newest_first), reverse=True)
    assert [e.id for e in page] == [newest_first[1].id]


async def test_search_events_date_window_and_limits(world):
    from datetime import UTC, datetime, timedelta

    from app.admin import service
    from app.db.session import session_scope

    await _three_events()
    now = datetime.now(UTC)
    async with session_scope() as s:
        assert len(await service.search_admin_events(s, since=now - timedelta(hours=1))) == 3
        assert await service.search_admin_events(s, since=now + timedelta(hours=1)) == []
        assert await service.search_admin_events(s, until=now - timedelta(hours=1)) == []
        with pytest.raises(ValueError):
            await service.search_admin_events(s, limit=0)
        with pytest.raises(ValueError):
            await service.search_admin_events(s, offset=-1)


# --- API and console: same events, and no actor left unspecified ---


@pytest.fixture
async def api(world):
    from app.api.app import app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def _recorded(actor):
    """(action, target_type, decrypted details) of one actor's events. Ids
    inside the details differ between two different rows, so they are left
    out of the comparison.
    """
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        events = await service.search_admin_events(s, actor=actor, limit=500)
    return [
        (
            e.action,
            e.target_type,
            {k: v for k, v in json.loads(e.details).items() if not k.endswith("_id")},
        )
        for e in reversed(events)
    ]


async def test_api_and_console_write_the_same_events_for_the_same_operations(
    world, api, monkeypatch
):
    from app.admin import service
    from app.admin.client import client_label
    from app.db.session import session_scope

    async with session_scope() as s:  # two more users to act on, one per front end
        await service.create_user(s, "ForApi", actor="t")
        await service.create_user(s, "ForConsole", actor="t")
        await s.commit()
    _sql_exec("delete from admin_events")

    # The API side.
    ok = {200, 201, 204}
    calls = [
        api.post("/users", json={"display_name": "Same"}, headers=AUTH),
        api.patch("/users/2", json={"is_active": False}, headers=AUTH),
        api.post("/users/2/agents", json={"name": "bot"}, headers=AUTH),
        api.patch("/agents/2", json={"name": "bot2"}, headers=AUTH),
        api.post(
            "/users/2/channels", json={"channel": "matrix", "identifier": "@a:b"}, headers=AUTH
        ),
        api.get("/logs", params={"keyword": "secret", "limit": 20}, headers=AUTH),
    ]
    for call in calls:
        assert (await call).status_code in ok

    # The console side (through the API), the same operations on the other user.
    await run_console(
        monkeypatch,
        *command("create-user", display_name="Same"),
        *command("update-user", user_id=3, is_active="false"),
        *command("create-agent", user_id=3, name="bot"),
        *command("update-agent", agent_id=3, name="bot2"),
        *command("add-channel-identity", user_id=3, channel="matrix", identifier="@c:d"),
        *command("search-logs", keyword="secret", limit=20),
    )

    from_api = await _recorded("api")
    from_console = await _recorded(client_label())
    assert len(from_api) == 6
    assert from_api == from_console  # same actions, target types and details
    assert _sql("select count(*) from admin_events where actor = 'unspecified'") == [(0,)]


async def test_console_operations_carry_the_cli_actor(world, monkeypatch):
    from app.admin.client import client_label

    await run_console(
        monkeypatch,
        *command("create-user", display_name="Dan"),
        *command("grant", user_id=1, channel_identity_id=1, kind="admin"),
        *command("update-agent", agent_id=1, name="renamed"),
        *command("approve-request", request_id=1),
        *command("search-logs", keyword="secret"),
    )
    actors = {r[0] for r in _events()}
    actions = [r[1] for r in _events()]
    assert actors == {client_label()} and client_label().startswith("cli:")
    assert actions == [
        "user.create",
        "permission.grant",
        "agent.rename",
        "request.approve",
        "logs.search",
    ]


async def test_the_console_records_the_log_search_even_when_nothing_matches(world, monkeypatch):
    await run_console(monkeypatch, *command("search-logs", keyword="no-such-word"))
    assert [r[1] for r in _events()] == ["logs.search"]


async def test_the_api_records_a_log_search_that_returns_nothing(world, api):
    r = await api.get("/logs", params={"keyword": "no-such-word"}, headers=AUTH)
    assert r.json() == []
    assert [r[:2] for r in _events()] == [("api", "logs.search")]


async def test_events_endpoint_requires_the_key_and_filters(world, api):
    assert (await api.get("/admin-events")).status_code == 401
    await api.post("/users", json={"display_name": "Cara"}, headers=AUTH)
    await api.patch("/users/2", json={"is_active": False}, headers=AUTH)
    r = await api.get("/admin-events", params={"action": "user.update"}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert [e["action"] for e in body] == ["user.update"]
    assert body[0]["actor"] == "api" and body[0]["target_id"] == 2
    assert json.loads(body[0]["details"]) == {"is_active": False}
    bad = await api.get("/admin-events", params={"limit": 0}, headers=AUTH)
    assert bad.status_code == 422


async def test_reading_the_events_is_itself_recorded(world, api):
    await api.get("/admin-events", headers=AUTH)
    await api.get("/admin-events", headers=AUTH)
    assert [r[:2] for r in _events()] == [("api", "admin_events.search")] * 2


def test_every_encrypted_column_is_covered_by_rekey_and_the_unreadable_count():
    """A new encrypted column must be re-encrypted by a key rotation and
    counted by the storage overview, otherwise it silently breaks.
    """
    from app.admin import rekey, service
    from app.db.models import Base
    from app.db.types import EncryptedString

    in_schema = {
        (table.name, column.name)
        for table in Base.metadata.tables.values()
        for column in table.columns
        if isinstance(column.type, EncryptedString)
    }
    assert in_schema, "the schema has encrypted columns"
    assert in_schema == {(t, c) for t, _key, c in rekey.APP_COLUMNS}
    assert in_schema == set(service._ENCRYPTED_COLUMNS)
