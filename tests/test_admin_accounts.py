"""Named administrators: accounts and scrypt passwords, the TOTP second
factor, per-client tokens and the static key turned off. Through the real API in
process, on a fresh database; scrypt runs with a lower cost to keep the tests fast.
"""

import base64
import sqlite3
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app.security import totp

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
STATIC = {"Authorization": f"Bearer {KEY}"}
PASSWORD = "a long enough passphrase"


@pytest.fixture
async def api(fresh_db, monkeypatch, tmp_path):
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db
    from app.security import passwords

    monkeypatch.setattr(passwords, "LOG2_N", 10)
    passwords.dummy_hash.cache_clear()
    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 5000))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        yield c
    deps.reset_failure_state()
    passwords.dummy_hash.cache_clear()


def _basic(name: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{name}:{password}".encode()).decode()


async def _create(api, name="alice", scope="owner", headers=STATIC, **extra):
    body = {"name": name, "scope": scope, "password": PASSWORD, **extra}
    return await api.post("/admins", json=body, headers=headers)


def _code(secret: str, offset: int = 0) -> str:
    return totp.code_at(secret, totp.current_step() + offset)


async def _sign_in(api, name, secret=None, code=None, params=None, password=PASSWORD):
    headers = {"Authorization": _basic(name, password)}
    if code is not None or secret is not None:
        headers["X-TOTP"] = code if code is not None else _code(secret)
    return await api.post("/auth/token", headers=headers, params=params or {})


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --- the first owner, and the static key turned off ---


async def test_the_static_key_works_until_a_named_owner_exists(api):
    assert (await api.get("/users", headers=STATIC)).status_code == 200
    created = await _create(api)
    assert created.status_code == 201
    refused = await api.get("/users", headers=STATIC)
    assert refused.status_code == 401
    assert "named owner" in refused.json()["detail"]


async def test_an_admin_below_owner_does_not_turn_the_static_key_off(api):
    assert (await _create(api, name="bob", scope="admin")).status_code == 201
    assert (await api.get("/users", headers=STATIC)).status_code == 200


async def test_the_in_process_script_keeps_the_static_key(api):
    from app.api import deps

    await _create(api)
    marker = deps.INPROCESS_CLI.set(True)
    try:
        assert (await api.get("/users", headers=STATIC)).status_code == 200
    finally:
        deps.INPROCESS_CLI.reset(marker)


async def test_a_disabled_owner_turns_the_static_key_back_on_only_when_none_is_left(api):
    secret = (await _create(api)).json()["totp_secret"]
    token = (await _sign_in(api, "alice", secret)).json()["token"]
    await _create(api, name="carol", headers=_bearer(token))
    assert (await api.post("/admins/carol/disable", headers=_bearer(token))).status_code == 200
    assert (await api.get("/users", headers=STATIC)).status_code == 401  # alice is left


# --- accounts and passwords ---


async def test_an_owner_always_gets_a_totp_secret_shown_once(api):
    created = (await _create(api)).json()
    assert created["totp"] is True and len(created["totp_secret"]) >= 32
    assert created["totp_uri"].startswith("otpauth://totp/ChannelAgent:alice?")
    token = (await _sign_in(api, "alice", created["totp_secret"])).json()["token"]
    listed = (await api.get("/admins", headers=_bearer(token))).json()
    assert listed[0]["name"] == "alice" and "totp_secret" not in listed[0]


async def test_below_owner_the_second_factor_is_optional(api):
    assert (await _create(api, name="bob", scope="admin")).json()["totp_secret"] is None
    with_code = (await _create(api, name="dan", scope="read", with_totp=True)).json()
    assert with_code["totp_secret"]


@pytest.mark.parametrize(
    "body, status",
    [
        ({"name": "Alice!", "scope": "owner", "password": PASSWORD}, 422),
        ({"name": "alice", "scope": "root", "password": PASSWORD}, 422),
        ({"name": "alice", "scope": "owner", "password": "short"}, 422),
    ],
)
async def test_a_bad_account_is_refused(api, body, status):
    assert (await api.post("/admins", json=body, headers=STATIC)).status_code == status


async def test_a_name_is_taken_once(api):
    await _create(api, name="bob", scope="admin")
    assert (await _create(api, name="bob", scope="read")).status_code == 409


async def test_the_password_is_stored_as_a_scrypt_hash(api, fresh_db):
    from app.config import get_settings

    await _create(api, name="bob", scope="admin")
    path = get_settings().database_url.split("///", 1)[1]
    stored = sqlite3.connect(path).execute("select password_hash from admin_accounts").fetchone()
    assert stored[0].startswith("scrypt$10$8$1$") and PASSWORD not in stored[0]


async def test_a_sign_in_runs_scrypt_once_whether_the_name_exists_or_not(api, monkeypatch):
    from app.security import passwords

    await _create(api, name="bob", scope="admin")
    checked = []
    real = passwords.verify_password
    monkeypatch.setattr(passwords, "verify_password", lambda p, h: checked.append(h) or real(p, h))
    assert (await _sign_in(api, "bob", password="wrong password here")).status_code == 401
    assert (await _sign_in(api, "nobody", password=PASSWORD)).status_code == 401
    assert len(checked) == 2 and checked[1] == passwords.dummy_hash()


async def test_one_answer_for_every_failed_sign_in(api):
    secret = (await _create(api)).json()["totp_secret"]
    answers = [
        await _sign_in(api, "alice", password="wrong password here", code=_code(secret)),
        await _sign_in(api, "nobody", code="123456"),
        await _sign_in(api, "alice"),
    ]
    assert {r.status_code for r in answers} == {401}
    assert len({r.text for r in answers}) == 1


async def test_a_password_change_is_ones_own_or_an_owners(api):
    secret = (await _create(api)).json()["totp_secret"]
    owner = (await _sign_in(api, "alice", secret)).json()["token"]
    await _create(api, name="bob", scope="admin", headers=_bearer(owner))
    await _create(api, name="dan", scope="admin", headers=_bearer(owner))
    bob = (await _sign_in(api, "bob")).json()["token"]
    new = {"password": "another long passphrase"}
    assert (
        await api.post("/admins/bob/password", json=new, headers=_bearer(bob))
    ).status_code == 200
    assert (
        await api.post("/admins/dan/password", json=new, headers=_bearer(bob))
    ).status_code == 403
    assert (
        await api.post("/admins/dan/password", json=new, headers=_bearer(owner))
    ).status_code == 200
    assert (await _sign_in(api, "bob")).status_code == 401
    assert (await _sign_in(api, "bob", password="another long passphrase")).status_code == 201


async def test_the_last_owner_cannot_be_disabled(api):
    secret = (await _create(api)).json()["totp_secret"]
    token = (await _sign_in(api, "alice", secret)).json()["token"]
    assert (await api.post("/admins/alice/disable", headers=_bearer(token))).status_code == 409


# --- the second factor ---


async def test_an_owner_without_a_valid_code_gets_no_token(api):
    secret = (await _create(api)).json()["totp_secret"]
    assert (await _sign_in(api, "alice")).status_code == 401
    assert (
        await _sign_in(api, "alice", code="000000" if _code(secret) != "000000" else "111111")
    ).status_code == 401
    assert (await _sign_in(api, "alice", code="12345")).status_code == 401
    assert (await _sign_in(api, "alice", secret)).status_code == 201


async def test_a_code_works_once(api):
    secret = (await _create(api)).json()["totp_secret"]
    code = _code(secret)
    assert (await _sign_in(api, "alice", code=code)).status_code == 201
    assert (await _sign_in(api, "alice", code=code)).status_code == 401


async def test_one_step_of_clock_drift_is_accepted_two_are_not(api):
    secret = (await _create(api)).json()["totp_secret"]
    assert (await _sign_in(api, "alice", code=_code(secret, -2))).status_code == 401
    assert (await _sign_in(api, "alice", code=_code(secret, -1))).status_code == 201


async def test_a_reset_gives_a_new_secret_and_the_old_one_stops(api):
    old = (await _create(api)).json()["totp_secret"]
    token = (await _sign_in(api, "alice", old)).json()["token"]
    new = (await api.post("/admins/alice/totp", headers=_bearer(token))).json()["totp_secret"]
    assert new and new != old
    assert (await _sign_in(api, "alice", code=_code(old, 1))).status_code == 401
    assert (await _sign_in(api, "alice", code=_code(new, 1))).status_code == 201


# --- tokens ---


async def test_a_token_acts_as_its_account_with_its_scope(api):
    secret = (await _create(api)).json()["totp_secret"]
    signed = await _sign_in(api, "alice", secret, params={"label": "laptop", "scope": "read"})
    assert signed.status_code == 201
    data = signed.json()
    assert data["token"].startswith("ca_") and data["scope"] == "read"
    who = (await api.get("/whoami", headers=_bearer(data["token"]))).json()
    assert who["actor"] == "adm:alice" and who["scope"] == "read"
    assert (await api.get("/users", headers=_bearer(data["token"]))).status_code == 200
    assert (await api.post("/users", json={}, headers=_bearer(data["token"]))).status_code == 403


async def test_a_token_cannot_go_above_its_account(api):
    await _create(api, name="bob", scope="operate")
    assert (await _sign_in(api, "bob", params={"scope": "admin"})).status_code == 422


async def test_only_the_sign_in_route_takes_a_password(api):
    await _create(api, name="bob", scope="admin")
    basic = {"Authorization": _basic("bob", PASSWORD)}
    assert (await api.get("/users", headers=basic)).status_code == 401


async def test_a_token_cannot_make_another_token(api):
    await _create(api, name="bob", scope="admin")
    token = (await _sign_in(api, "bob")).json()["token"]
    assert (await api.post("/auth/token", headers=_bearer(token))).status_code == 409


async def test_an_expired_token_is_refused(api, monkeypatch):
    from app.admin import accounts

    await _create(api, name="bob", scope="admin")
    token = (await _sign_in(api, "bob", params={"hours": 1})).json()["token"]
    later = datetime.now(UTC) + timedelta(hours=1, seconds=1)
    monkeypatch.setattr(accounts, "_now", lambda: later)
    assert (await api.get("/users", headers=_bearer(token))).status_code == 401


async def test_a_revoked_token_is_refused(api):
    await _create(api, name="bob", scope="admin")
    signed = (await _sign_in(api, "bob")).json()
    assert (
        await api.delete(f"/auth/tokens/{signed['id']}", headers=_bearer(signed["token"]))
    ).status_code == 200
    assert (await api.get("/users", headers=_bearer(signed["token"]))).status_code == 401


async def test_a_disabled_account_loses_its_tokens(api):
    secret = (await _create(api)).json()["totp_secret"]
    owner = (await _sign_in(api, "alice", secret)).json()["token"]
    await _create(api, name="bob", scope="admin", headers=_bearer(owner))
    bob = (await _sign_in(api, "bob")).json()["token"]
    assert (await api.post("/admins/bob/disable", headers=_bearer(owner))).status_code == 200
    assert (await api.get("/users", headers=_bearer(bob))).status_code == 401
    assert (await _sign_in(api, "bob")).status_code == 401


async def test_someone_elses_token_is_not_found_and_not_listed(api):
    await _create(api, name="bob", scope="admin")
    await _create(api, name="dan", scope="admin")
    bob = (await _sign_in(api, "bob")).json()
    dan = (await _sign_in(api, "dan")).json()
    assert (
        await api.delete(f"/auth/tokens/{bob['id']}", headers=_bearer(dan["token"]))
    ).status_code == 404
    listed = (await api.get("/auth/tokens", headers=_bearer(dan["token"]))).json()
    assert [t["id"] for t in listed] == [dan["id"]] and "token" not in listed[0]
    every = await api.get(
        "/auth/tokens", params={"every_account": True}, headers=_bearer(dan["token"])
    )
    assert every.status_code == 403


async def test_only_the_hash_of_a_token_is_stored(api):
    from app.config import get_settings

    await _create(api, name="bob", scope="admin")
    token = (await _sign_in(api, "bob")).json()["token"]
    path = get_settings().database_url.split("///", 1)[1]
    raw = open(path, "rb").read()
    assert token.encode() not in raw
    assert PASSWORD.encode() not in raw


async def test_the_changes_are_admin_events_without_any_secret(api):
    from app.config import get_settings

    secret = (await _create(api)).json()["totp_secret"]
    token = (await _sign_in(api, "alice", secret)).json()["token"]
    await api.post("/admins/alice/totp", headers=_bearer(token))
    path = get_settings().database_url.split("///", 1)[1]
    rows = sqlite3.connect(path).execute("select actor, action from admin_events").fetchall()
    assert ("api", "admin.create") in rows  # the static key, no X-Client label
    assert ("adm:alice", "token.create") in rows and ("adm:alice", "admin.totp_reset") in rows
    from sqlalchemy import select

    from app.db.models import AdminEvent
    from app.db.session import session_scope

    async with session_scope() as s:
        details = [e.details or "" for e in (await s.scalars(select(AdminEvent))).all()]
    assert not any(secret in d or token in d or PASSWORD in d for d in details)


async def test_failed_sign_ins_count_for_the_limiter(api):
    await _create(api, name="bob", scope="admin")
    codes = [
        (await _sign_in(api, "bob", password="wrong password here")).status_code for _ in range(11)
    ]
    assert codes == [401] * 10 + [429]


async def test_disabling_an_account_revokes_its_tokens_in_the_table(api):
    from app.config import get_settings

    secret = (await _create(api)).json()["totp_secret"]
    owner = (await _sign_in(api, "alice", secret)).json()["token"]
    await _create(api, name="bob", scope="admin", headers=_bearer(owner))
    await _sign_in(api, "bob")
    await api.post("/admins/bob/disable", headers=_bearer(owner))
    path = get_settings().database_url.split("///", 1)[1]
    rows = sqlite3.connect(path).execute(
        "select t.revoked_at is not null from api_tokens t join admin_accounts a"
        " on a.id = t.account_id where a.name = 'bob'"
    )
    assert rows.fetchall() == [(1,)]


async def test_a_token_of_a_disabled_account_is_refused_even_unrevoked(api):
    from sqlalchemy import update

    from app.db.models import AdminAccount
    from app.db.session import session_scope

    await _create(api, name="bob", scope="admin")
    token = (await _sign_in(api, "bob")).json()["token"]
    async with session_scope() as s:
        await s.execute(update(AdminAccount).values(disabled_at=datetime.now(UTC)))
        await s.commit()
    assert (await api.get("/users", headers=_bearer(token))).status_code == 401


@pytest.mark.parametrize("code", ["12345", "1234567", "12345é", "abcdef", ""])
def test_a_malformed_code_is_refused_without_an_error(code):
    secret = totp.new_secret()
    assert totp.matching_step(secret, code, None) is None
