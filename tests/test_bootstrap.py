"""Tests: the first admin is seeded from TELEGRAM_ALLOWED_USERS on an
empty database only, once, and the environment is never consulted again.
"""

import logging
import sqlite3

import pytest

from app.db.models import Channel


def _sql(query: str, *args):
    from app.config import get_settings

    con = sqlite3.connect(get_settings().database_url.split("///", 1)[1])
    try:
        return con.execute(query, args).fetchall()
    finally:
        con.close()


async def _bootstrap(monkeypatch, allowed: str | None):
    from app.config import get_settings
    from app.db.bootstrap import bootstrap_admin_from_env
    from app.db.session import session_scope

    if allowed is None:
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
    else:
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", allowed)
    get_settings.cache_clear()
    async with session_scope() as session:
        await bootstrap_admin_from_env(session)


@pytest.fixture
async def empty(fresh_db):
    from app.db.session import init_db

    await init_db()


async def test_seeds_one_admin_with_a_telegram_identity_and_the_admin_permission(
    empty, monkeypatch
):
    from app.db.session import session_scope
    from app.security.auth import authorize

    await _bootstrap(monkeypatch, "111")
    assert _sql("select count(*) from users") == [(1,)]
    assert _sql("select channel, external_id from channel_identities") == [("telegram", "111")]
    assert _sql("select kind from permissions") == [("admin",)]
    async with session_scope() as s:
        decision = await authorize(s, Channel.TELEGRAM, "111")
    assert decision.allowed and decision.is_admin


async def test_running_twice_does_not_duplicate_the_admin(empty, monkeypatch):
    await _bootstrap(monkeypatch, "111")
    await _bootstrap(monkeypatch, "111")
    assert _sql("select count(*) from users") == [(1,)]
    assert _sql("select count(*) from channel_identities") == [(1,)]
    assert _sql("select count(*) from permissions") == [(1,)]


async def test_a_comma_separated_list_seeds_one_admin_each(empty, monkeypatch):
    await _bootstrap(monkeypatch, "111, 222 ,,333")
    assert sorted(_sql("select external_id from channel_identities")) == [
        ("111",),
        ("222",),
        ("333",),
    ]
    assert _sql("select count(*) from permissions") == [(3,)]


async def test_the_environment_is_never_consulted_once_the_database_has_users(empty, monkeypatch):
    from app.db.session import session_scope
    from app.security.auth import authorize

    await _bootstrap(monkeypatch, "111")
    await _bootstrap(monkeypatch, "222")  # the environment changed later
    assert _sql("select external_id from channel_identities") == [("111",)]
    async with session_scope() as s:
        assert (await authorize(s, Channel.TELEGRAM, "222")).allowed is False


async def test_a_database_that_already_has_a_user_is_left_alone(empty, monkeypatch):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        await service.create_user(s, "Someone else")
        await s.commit()
    await _bootstrap(monkeypatch, "111")
    assert _sql("select count(*) from users") == [(1,)]
    assert _sql("select count(*) from channel_identities") == [(0,)]


# "" rather than an unset variable: an unset one would fall back to the real .env file.
@pytest.mark.parametrize("value", ["", "  ,  "])
async def test_no_allowed_users_seeds_nobody_and_warns(empty, monkeypatch, caplog, value):
    caplog.set_level(logging.WARNING, logger="channelagent")
    await _bootstrap(monkeypatch, value)
    assert _sql("select count(*) from users") == [(0,)]
    if value != "  ,  ":
        assert any("no admin user" in r.message for r in caplog.records)
