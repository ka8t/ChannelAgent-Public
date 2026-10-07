"""The tamper-evident chain of the admin events: each event carries the hash of the one
before it and of its own content; an edited event, a removed one or a rewritten hash is
reported by `GET /audit/verify` (`./start.sh --admin verify-audit`), and a key rotation, which
re-encrypts the details, leaves the chain intact.
"""

import sqlite3
from pathlib import Path

import httpx
import pytest
from cryptography.fernet import Fernet

from app.admin import audit_chain, service

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
AUTH = {"Authorization": f"Bearer {KEY}"}


def _db() -> Path:
    from app.config import get_settings

    return Path(get_settings().database_url.split("///", 1)[1])


def _sql(query: str, *args):
    con = sqlite3.connect(_db())
    try:
        rows = con.execute(query, args).fetchall()
        con.commit()
        return rows
    finally:
        con.close()


async def _events(n: int) -> None:
    from app.db.session import session_scope

    async with session_scope() as s:
        for i in range(n):
            await service.record_admin_event(
                s, actor="cli:test", action=f"test.{i}", target_type="test", target_id=i,
                details={"i": i, "note": "private"},
            )  # fmt: skip
        await s.commit()


async def _verify() -> dict:
    from app.db.session import session_scope

    async with session_scope() as s:
        return await audit_chain.verify(s)


@pytest.fixture
async def db(fresh_db, monkeypatch):
    from app.config import get_settings
    from app.db.session import init_db

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    await init_db()


async def test_each_event_is_chained_to_the_one_before(db):
    await _events(3)
    rows = _sql("select prev_hash, hash from admin_events order by id")
    assert rows[0][0] == audit_chain.GENESIS
    assert [r[0] for r in rows[1:]] == [r[1] for r in rows[:-1]]
    assert len({r[1] for r in rows}) == 3


async def test_an_untouched_table_verifies(db):
    await _events(5)
    result = await _verify()
    assert (result["events"], result["problems"], result["intact"]) == (5, [], True)
    assert result["last_hash"] == _sql("select hash from admin_events order by id desc")[0][0]


@pytest.mark.parametrize(
    "column, value",
    [("actor", "intruder"), ("action", "test.x"), ("target_id", 99), ("created_at", "2020-01-01")],
)
async def test_an_edited_event_is_reported_alone(db, column, value):
    await _events(5)
    _sql(f"update admin_events set {column} = ? where id = 3", value)
    assert (await _verify())["problems"] == [{"id": 3, "problem": "its content was changed"}]


async def test_edited_details_are_reported(db):
    from app.security.encryption import encrypt_value

    await _events(4)
    _sql("update admin_events set details = ? where id = 2", encrypt_value('{"i": 7}'))
    assert [p["id"] for p in (await _verify())["problems"]] == [2]


async def test_a_removed_event_is_reported_on_the_next(db):
    await _events(5)
    _sql("delete from admin_events where id = 3")
    problems = (await _verify())["problems"]
    assert [p["id"] for p in problems] == [4] and "link" in problems[0]["problem"]


async def test_a_recomputed_hash_breaks_the_next_link(db):
    """An edit whose own hash is recomputed still breaks the link of the event after it."""
    from app.db.models import AdminEvent
    from app.db.session import session_scope

    await _events(4)
    _sql("update admin_events set actor = 'intruder' where id = 2")
    async with session_scope() as s:
        event = await s.get(AdminEvent, 2)
        forged = audit_chain.link(event.prev_hash, audit_chain._event_content(event))
    _sql("update admin_events set hash = ? where id = 2", forged)
    assert [p["id"] for p in (await _verify())["problems"]] == [3]


async def test_an_event_without_a_hash_is_reported(db):
    await _events(2)
    _sql("update admin_events set hash = null where id = 2")
    assert (await _verify())["problems"] == [{"id": 2, "problem": "not chained (no hash)"}]


async def test_a_key_rotation_keeps_the_chain(fresh_db, monkeypatch):
    from app.admin.rekey import run_rekey
    from app.checkpoints import checkpoint_db_path
    from app.config import get_settings
    from app.db.session import init_db
    from app.security import encryption

    old, new = Fernet.generate_key().decode(), Fernet.generate_key().decode()

    def use(key):
        monkeypatch.setenv("ENCRYPTION_KEY", key)
        get_settings.cache_clear()
        encryption._fernet.cache_clear()

    use(old)
    await init_db()
    await _events(4)
    before = _sql("select details from admin_events order by id")
    report = run_rekey(_db(), checkpoint_db_path(), old, new)
    assert report.applied
    assert _sql("select details from admin_events order by id") != before  # re-encrypted
    use(new)
    try:
        assert (await _verify())["intact"] is True
    finally:
        monkeypatch.undo()  # the key cached by encryption must not outlive this test
        get_settings.cache_clear()
        encryption._fernet.cache_clear()


async def test_the_migration_chains_the_events_already_stored(fresh_db, monkeypatch):
    from alembic.command import downgrade, upgrade
    from alembic.config import Config

    from app.config import get_settings
    from app.db.session import _REPO_ROOT, init_db

    await init_db()
    await _events(6)
    cfg = Config(str(_REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_REPO_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url)
    cfg.attributes["configure_logger"] = False
    import asyncio

    await asyncio.to_thread(downgrade, cfg, "4770d34ea658")
    assert "hash" not in [r[1] for r in _sql("pragma table_info(admin_events)")]
    await asyncio.to_thread(upgrade, cfg, "head")
    result = await _verify()
    assert (result["events"], result["intact"]) == (6, True)


async def test_the_api_route_reports_the_problems(db):
    from app.api.app import app

    await _events(3)
    _sql("update admin_events set actor = 'intruder' where id = 1")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        response = await c.get("/audit/verify", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["problems"] == [{"id": 1, "problem": "its content was changed"}]
    assert response.json()["intact"] is False
