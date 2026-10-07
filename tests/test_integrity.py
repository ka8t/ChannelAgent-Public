"""Tests: referential integrity and what deleting a user means.

Policy decided in the issue (recommendation applied): deleting a user who
has audit-trail entries is refused (deactivate instead); an explicit
purge deletes the user's agents and logs in one transaction. A user with
no history is deleted together with their agents.
"""

import sqlite3

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.db.models import Channel, Direction


def _db_file() -> str:
    from app.config import get_settings

    return get_settings().database_url.split("///", 1)[1]


def _count(table: str, where: str = "1=1") -> int:
    con = sqlite3.connect(_db_file())
    try:
        return con.execute(f"select count(*) from {table} where {where}").fetchone()[0]
    finally:
        con.close()


@pytest.fixture
async def world(fresh_db):
    """User 1 (Alice): identity, permission, 2 agents, 3 logs. User 2 (Bob):
    identity, 1 agent, 1 log. User 3 (Carol): identity only, no history.
    """
    from app.admin import service
    from app.db.models import ChannelIdentity, PermissionKind, User
    from app.db.session import init_db, session_scope
    from app.security.auth import grant_permission

    await init_db()
    async with session_scope() as s:
        users = [User(display_name=n) for n in ("Alice", "Bob", "Carol")]
        s.add_all(users)
        await s.flush()
        for i, u in enumerate(users, start=1):
            ident = ChannelIdentity(user_id=u.id, channel=Channel.TELEGRAM, external_id=str(i))
            s.add(ident)
            await s.flush()
            await grant_permission(s, ident, PermissionKind.CHAT)
        a1 = await service.create_agent(s, 1, "default")
        await service.create_agent(s, 1, "work")
        b1 = await service.create_agent(s, 2, "default")
        for text_, agent in (("a1", a1), ("a2", a1), ("a3", a1)):
            await service.record_action(
                s, user_id=1, agent_id=agent.id, channel=Channel.TELEGRAM,
                direction=Direction.INBOUND, text=text_,
            )
        await service.record_action(
            s, user_id=2, agent_id=b1.id, channel=Channel.TELEGRAM,
            direction=Direction.INBOUND, text="b1",
        )
        await s.commit()


# --- foreign keys are really enforced ---


async def test_foreign_keys_pragma_is_on_for_application_connections(world):
    from app.db.session import get_engine

    async with get_engine().connect() as conn:
        assert (await conn.execute(text("PRAGMA foreign_keys"))).scalar() == 1


async def test_the_database_rejects_an_agent_for_an_unknown_user(world):
    from app.db.models import Agent
    from app.db.session import session_scope

    with pytest.raises(IntegrityError):
        async with session_scope() as s:
            s.add(Agent(user_id=999, name="ghost"))
            await s.commit()
    assert _count("agents", "user_id = 999") == 0


async def test_the_database_rejects_a_log_for_an_unknown_agent(world):
    from app.admin import service
    from app.db.session import session_scope

    with pytest.raises(IntegrityError):
        async with session_scope() as s:
            await service.record_action(
                s, user_id=1, agent_id=999, channel=Channel.EMAIL,
                direction=Direction.INBOUND, text="x",
            )
            await s.commit()
    assert _count("action_logs") == 4


# --- service layer validates before touching the database ---


async def test_create_agent_for_an_unknown_user_raises_a_clear_error(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        with pytest.raises(service.UserNotFoundError, match="999"):
            await service.create_agent(s, 999, "ghost")
    assert _count("agents") == 3


async def test_default_agent_for_an_unknown_user_raises(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        with pytest.raises(service.UserNotFoundError):
            await service.get_or_create_default_agent(s, 999)
    assert _count("agents") == 3


# --- deletion policy ---


async def test_user_with_history_cannot_be_deleted_without_purge(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        with pytest.raises(service.UserHasHistoryError) as exc:
            await service.delete_user(s, 1)
    assert exc.value.agents == 2 and exc.value.logs == 3
    # Nothing was touched (direct SQL, not the ORM).
    assert _count("users") == 3 and _count("agents", "user_id = 1") == 2
    assert _count("action_logs", "user_id = 1") == 3
    assert _count("channel_identities", "user_id = 1") == 1


async def test_user_without_history_is_deleted_with_identities_and_permissions(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        report = await service.delete_user(s, 3)
        await s.commit()
    assert (report.agents_deleted, report.logs_deleted) == (0, 0)
    assert _count("users", "id = 3") == 0
    assert _count("channel_identities", "user_id = 3") == 0
    assert _count("permissions") == 2, "only Carol's permission is gone"


async def test_user_with_only_agents_and_no_logs_is_deleted_with_them(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        await service.create_agent(s, 3, "empty")
        await s.commit()
    assert _count("agents", "user_id = 3") == 1
    async with session_scope() as s:
        report = await service.delete_user(s, 3)
        await s.commit()
    assert report.agents_deleted == 1
    assert _count("agents", "user_id = 3") == 0 and _count("users", "id = 3") == 0


async def test_purge_deletes_everything_of_that_user_and_nothing_else(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        report = await service.delete_user(s, 1, purge=True)
        await s.commit()
    assert (report.agents_deleted, report.logs_deleted) == (2, 3)
    assert _count("users", "id = 1") == 0
    assert _count("agents", "user_id = 1") == 0
    assert _count("action_logs", "user_id = 1") == 0
    assert _count("channel_identities", "user_id = 1") == 0
    # Bob and Carol are untouched.
    assert _count("users") == 2 and _count("agents") == 1 and _count("action_logs") == 1
    assert _count("permissions") == 2


async def test_purge_is_atomic(world, monkeypatch):
    """If the last step fails, nothing that came before it may persist."""
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        original = s.delete

        async def failing(obj, *a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(s, "delete", failing)
        with pytest.raises(RuntimeError):
            await service.delete_user(s, 1, purge=True)
        await s.rollback()
        monkeypatch.setattr(s, "delete", original)
    assert _count("users") == 3 and _count("agents", "user_id = 1") == 2
    assert _count("action_logs", "user_id = 1") == 3


async def test_deleting_an_unknown_user_raises(world):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        with pytest.raises(service.UserNotFoundError):
            await service.delete_user(s, 999)


# --- Admin API ---


@pytest.fixture
async def api(world, monkeypatch):
    from app.api.app import app
    from app.config import get_settings

    monkeypatch.setenv("API_SERVER_KEY", "k" * 32)
    get_settings.cache_clear()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


AUTH = {"Authorization": "Bearer " + "k" * 32}


async def test_api_delete_user_with_history_is_409_and_touches_nothing(api):
    r = await api.delete("/users/1", headers=AUTH)
    assert r.status_code == 409
    assert "2 agent" in r.text and "3 log" in r.text and "purge" in r.text
    assert _count("users") == 3 and _count("action_logs", "user_id = 1") == 3


async def test_api_delete_user_without_history_is_204(api):
    assert (await api.delete("/users/3", headers=AUTH)).status_code == 204
    assert _count("users", "id = 3") == 0


async def test_api_purge_is_204_and_removes_the_history(api):
    assert (await api.delete("/users/1", params={"purge": "true"}, headers=AUTH)).status_code == 204
    assert _count("users", "id = 1") == 0 and _count("action_logs", "user_id = 1") == 0


async def test_api_delete_unknown_user_is_404(api):
    assert (await api.delete("/users/999", headers=AUTH)).status_code == 404


async def test_api_delete_requires_the_key(api):
    assert (await api.delete("/users/3")).status_code == 401
    assert _count("users") == 3
