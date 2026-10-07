"""Tests for the Admin API over the service layer: users, channel
identities, permissions (-), access requests, agents,
and the protection of every route including FastAPI's own /docs.
"""

import sqlite3

import httpx
import pytest

from app.db.models import Channel

KEY = "k" * 32
AUTH = {"Authorization": f"Bearer {KEY}"}


def _db() -> str:
    from app.config import get_settings

    return get_settings().database_url.split("///", 1)[1]


def _sql(query: str, *args):
    con = sqlite3.connect(_db())
    try:
        return con.execute(query, args).fetchall()
    finally:
        con.close()


@pytest.fixture
async def api(fresh_db, monkeypatch):
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    await init_db()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def _user(api, name="Alice") -> int:
    return (await api.post("/users", json={"display_name": name}, headers=AUTH)).json()["id"]


async def _identity(api, uid, channel="telegram", identifier="123") -> int:
    r = await api.post(
        f"/users/{uid}/channels", json={"channel": channel, "identifier": identifier}, headers=AUTH
    )
    return r.json()["id"]


# --- every route is protected, including the automatic documentation ---


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/users"),
        ("POST", "/users"),
        ("GET", "/users/1"),
        ("PATCH", "/users/1"),
        ("DELETE", "/users/1"),
        ("GET", "/users/1/channels"),
        ("POST", "/users/1/channels"),
        ("DELETE", "/users/1/channels/1"),
        ("GET", "/users/1/channels/1/permissions"),
        ("POST", "/users/1/channels/1/permissions"),
        ("DELETE", "/users/1/channels/1/permissions/chat"),
        ("GET", "/requests"),
        ("POST", "/requests/1/approve"),
        ("POST", "/requests/1/deny"),
        ("GET", "/users/1/agents"),
        ("POST", "/users/1/agents"),
        ("GET", "/agents/1"),
        ("PATCH", "/agents/1"),
        ("GET", "/logs"),
        ("GET", "/storage"),
        ("GET", "/docs"),
        ("GET", "/openapi.json"),
    ],
)
async def test_every_route_needs_the_key(api, method, path):
    assert (await api.request(method, path)).status_code == 401
    assert (
        await api.request(method, path, headers={"Authorization": "Bearer wrong"})
    ).status_code == 401


async def test_docs_answer_with_the_key(api):
    assert (await api.get("/docs", headers=AUTH)).status_code == 200
    assert (await api.get("/openapi.json", headers=AUTH)).status_code == 200


async def test_the_automatic_redoc_is_not_exposed(api):
    assert (await api.get("/redoc", headers=AUTH)).status_code == 404


# --- users ---


async def test_user_crud(api):
    r = await api.post("/users", json={"display_name": "Alice"}, headers=AUTH)
    assert r.status_code == 201 and r.json()["is_active"] is True
    uid = r.json()["id"]
    assert [u["id"] for u in (await api.get("/users", headers=AUTH)).json()] == [uid]
    assert (await api.get(f"/users/{uid}", headers=AUTH)).json()["display_name"] == "Alice"
    r = await api.patch(f"/users/{uid}", json={"is_active": False}, headers=AUTH)
    assert r.status_code == 200 and r.json() == {
        "id": uid,
        "display_name": "Alice",
        "is_active": False,
        "timezone": None,
        "tools_suspended": False,
    }
    r = await api.patch(f"/users/{uid}", json={"display_name": "Bob"}, headers=AUTH)
    assert r.json()["display_name"] == "Bob" and r.json()["is_active"] is False
    assert (await api.delete(f"/users/{uid}", headers=AUTH)).status_code == 204
    assert _sql("select count(*) from users")[0][0] == 0


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/users/999", None),
        ("PATCH", "/users/999", {"is_active": True}),
        ("DELETE", "/users/999", None),
        ("GET", "/users/999/channels", None),
        ("POST", "/users/999/channels", {"channel": "telegram", "identifier": "1"}),
        ("GET", "/users/999/agents", None),
        ("POST", "/users/999/agents", {"name": "x"}),
    ],
)
async def test_unknown_user_is_404(api, method, path, body):
    r = await api.request(method, path, json=body, headers=AUTH)
    assert r.status_code == 404 and "999" in r.json()["detail"]


# --- channel identities and permissions ---


async def test_identity_lifecycle_and_immediate_effect_on_authorization(api):
    from app.db.session import session_scope
    from app.security.auth import authorize

    uid = await _user(api)
    cid = await _identity(api, uid, identifier="123")
    async with session_scope() as s:
        assert (await authorize(s, Channel.TELEGRAM, "123")).allowed is False
    r = await api.post(
        f"/users/{uid}/channels/{cid}/permissions", json={"kind": "chat"}, headers=AUTH
    )
    assert r.status_code == 201 and r.json()["kind"] == "chat"
    async with session_scope() as s:
        assert (await authorize(s, Channel.TELEGRAM, "123")).allowed is True
    assert [
        p["kind"]
        for p in (await api.get(f"/users/{uid}/channels/{cid}/permissions", headers=AUTH)).json()
    ] == ["chat"]
    r = await api.delete(f"/users/{uid}/channels/{cid}/permissions/chat", headers=AUTH)
    assert r.status_code == 204
    async with session_scope() as s:
        assert (await authorize(s, Channel.TELEGRAM, "123")).allowed is False
    assert (
        await api.delete(f"/users/{uid}/channels/{cid}/permissions/chat", headers=AUTH)
    ).status_code == 404
    assert (await api.delete(f"/users/{uid}/channels/{cid}", headers=AUTH)).status_code == 204
    assert _sql("select count(*) from channel_identities")[0][0] == 0


async def test_granting_twice_is_harmless(api):
    uid = await _user(api)
    cid = await _identity(api, uid)
    for _ in range(2):
        r = await api.post(
            f"/users/{uid}/channels/{cid}/permissions", json={"kind": "admin"}, headers=AUTH
        )
        assert r.status_code == 201
    assert _sql("select count(*) from permissions")[0][0] == 1


async def test_email_identity_is_hashed_and_the_address_is_encrypted_and_never_returned(api):
    from app.security.hashing import channel_identifier_key

    uid = await _user(api)
    addr = "Secret.Person@example.com"
    r = await api.post(
        f"/users/{uid}/channels", json={"channel": "email", "identifier": addr}, headers=AUTH
    )
    assert r.status_code == 201
    assert addr.lower() not in r.text.lower() and "raw_address" not in r.text
    ((external_id, raw),) = _sql("select external_id, raw_address from channel_identities")
    assert external_id == channel_identifier_key(Channel.EMAIL, addr)
    assert raw.startswith("gAAAA") and "example.com" not in raw


async def test_duplicate_identity_is_409_even_for_another_user(api):
    a, b = await _user(api, "A"), await _user(api, "B")
    await _identity(api, a, identifier="123")
    r = await api.post(
        f"/users/{b}/channels", json={"channel": "telegram", "identifier": "123"}, headers=AUTH
    )
    assert r.status_code == 409 and "already linked" in r.json()["detail"]


async def test_bad_identity_input_is_422(api):
    uid = await _user(api)
    for body in (
        {"channel": "telegram", "identifier": "   "},
        {"channel": "fax", "identifier": "1"},
    ):
        r = await api.post(f"/users/{uid}/channels", json=body, headers=AUTH)
        assert r.status_code == 422


async def test_an_identity_of_another_user_is_404(api):
    a, b = await _user(api, "A"), await _user(api, "B")
    cid = await _identity(api, a)
    assert (await api.delete(f"/users/{b}/channels/{cid}", headers=AUTH)).status_code == 404
    r = await api.post(
        f"/users/{b}/channels/{cid}/permissions", json={"kind": "chat"}, headers=AUTH
    )
    assert r.status_code == 404
    assert _sql("select count(*) from channel_identities")[0][0] == 1


async def test_unknown_permission_kind_is_422(api):
    uid = await _user(api)
    cid = await _identity(api, uid)
    r = await api.post(
        f"/users/{uid}/channels/{cid}/permissions", json={"kind": "root"}, headers=AUTH
    )
    assert r.status_code == 422


# --- access requests ---


@pytest.fixture
async def pending(api):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        await service.request_access(s, Channel.TELEGRAM, "777", "let me in")
        await service.request_access(s, Channel.EMAIL, "hash-abc", "hello there")
        await s.commit()


async def test_list_requests_pending_by_default_with_decrypted_first_message(api, pending):
    body = (await api.get("/requests", headers=AUTH)).json()
    assert [(r["id"], r["channel"], r["status"]) for r in body] == [
        (1, "telegram", "pending"),
        (2, "email", "pending"),
    ]
    assert body[0]["first_message_text"] == "let me in"
    assert body[0]["resolved_by"] is None and body[0]["resolved_at"] is None


async def test_request_status_filter(api, pending):
    await api.post("/requests/1/deny", headers=AUTH)
    assert [r["id"] for r in (await api.get("/requests", headers=AUTH)).json()] == [2]
    assert [r["id"] for r in (await api.get("/requests?status=denied", headers=AUTH)).json()] == [1]
    assert [r["id"] for r in (await api.get("/requests?status=all", headers=AUTH)).json()] == [1, 2]
    assert (await api.get("/requests?status=weird", headers=AUTH)).status_code == 422


async def test_approving_creates_user_identity_and_chat_permission_and_records_the_actor(
    api, pending
):
    from app.db.session import session_scope
    from app.security.auth import authorize

    r = await api.post("/requests/1/approve", headers=AUTH)
    assert r.status_code == 200 and r.json()["is_active"] is True
    assert _sql("select count(*) from users")[0][0] == 1
    assert _sql("select channel, external_id from channel_identities") == [("telegram", "777")]
    assert _sql("select kind from permissions") == [("chat",)]
    ((status, resolved_at, by),) = _sql(
        "select status, resolved_at, resolved_by from access_requests where id = 1"
    )
    assert (status, by) == ("approved", "api") and resolved_at is not None
    async with session_scope() as s:
        assert (await authorize(s, Channel.TELEGRAM, "777")).allowed is True


async def test_denying_resolves_without_creating_a_user(api, pending):
    assert (await api.post("/requests/1/deny", headers=AUTH)).status_code == 204
    assert _sql("select count(*) from users")[0][0] == 0
    assert _sql("select status, resolved_by from access_requests where id = 1") == [
        ("denied", "api")
    ]


async def test_resolving_twice_is_409_and_unknown_is_404(api, pending):
    await api.post("/requests/1/approve", headers=AUTH)
    for action in ("approve", "deny"):
        r = await api.post(f"/requests/1/{action}", headers=AUTH)
        assert r.status_code == 409 and "already approved" in r.json()["detail"]
        assert (await api.post(f"/requests/999/{action}", headers=AUTH)).status_code == 404
    assert _sql("select count(*) from users")[0][0] == 1, "no second user was created"


# --- agents ---


async def test_agent_lifecycle_and_admin_edits_any_users_agent(api):
    alice, bob = await _user(api, "Alice"), await _user(api, "Bob")
    r = await api.post(f"/users/{alice}/agents", json={"name": "work"}, headers=AUTH)
    assert r.status_code == 201 and r.json() == {
        "id": 1,
        "user_id": alice,
        "name": "work",
        "is_active": True,
        "system_prompt": None,
        "has_system_prompt": False,
        "model": None,
        "memory_mode": "off",
        "tools": [],
        "skills": [],
        "purpose": None,
    }
    await api.post(
        f"/users/{bob}/agents", json={"name": "work"}, headers=AUTH
    )  # same name, other user
    assert [a["name"] for a in (await api.get(f"/users/{alice}/agents", headers=AUTH)).json()] == [
        "work"
    ]
    # the caller is the admin: both agents belong to other people
    r = await api.patch("/agents/2", json={"name": "renamed"}, headers=AUTH)
    assert r.status_code == 200 and r.json()["name"] == "renamed" and r.json()["user_id"] == bob
    r = await api.patch("/agents/2", json={"is_active": False}, headers=AUTH)
    assert r.json()["is_active"] is False and r.json()["name"] == "renamed"
    r = await api.patch("/agents/2", json={"name": "again", "is_active": True}, headers=AUTH)
    assert r.json() == {
        "id": 2,
        "user_id": bob,
        "name": "again",
        "is_active": True,
        "system_prompt": None,
        "has_system_prompt": False,
        "model": None,
        "memory_mode": "off",
        "tools": [],
        "skills": [],
        "purpose": None,
    }
    assert (await api.get("/agents/2", headers=AUTH)).json()["name"] == "again"
    assert _sql("select name, is_active from agents order by id") == [("work", 1), ("again", 1)]


async def test_agent_errors(api):
    uid = await _user(api)
    await api.post(f"/users/{uid}/agents", json={"name": "work"}, headers=AUTH)
    await api.post(f"/users/{uid}/agents", json={"name": "other"}, headers=AUTH)
    assert (
        await api.post(f"/users/{uid}/agents", json={"name": "work"}, headers=AUTH)
    ).status_code == 409
    assert (await api.patch("/agents/2", json={"name": "work"}, headers=AUTH)).status_code == 409
    assert (await api.patch("/agents/1", json={"name": "work"}, headers=AUTH)).status_code == 200
    for name in ("", "   ", "x" * 101):
        r = await api.post(f"/users/{uid}/agents", json={"name": name}, headers=AUTH)
        assert r.status_code == 422, name
    assert (await api.get("/agents/999", headers=AUTH)).status_code == 404
    assert (await api.patch("/agents/999", json={"name": "x"}, headers=AUTH)).status_code == 404
    assert _sql("select count(*) from agents")[0][0] == 2


# --- a deactivated agent does not answer ---


async def test_a_deactivated_agent_gets_a_refusal_and_no_llm_call(fresh_db, monkeypatch):
    from app.admin import service
    from app.channels import dispatch
    from app.channels.schema import NormalizedEvent
    from app.db.models import ChannelIdentity, PermissionKind, User
    from app.db.session import init_db, session_scope
    from app.security.auth import grant_permission

    await init_db()
    async with session_scope() as s:
        u = User(display_name="Alice")
        s.add(u)
        await s.flush()
        ident = ChannelIdentity(user_id=u.id, channel=Channel.TELEGRAM, external_id="55")
        s.add(ident)
        await s.flush()
        await grant_permission(s, ident, PermissionKind.CHAT)
        agent = await service.get_or_create_default_agent(s, u.id)
        await s.commit()
        agent_id = agent.id

    calls = []

    async def fake_turn(channel, user_id, agent_id_, text, **_kwargs):
        calls.append(text)
        return "answer"

    monkeypatch.setattr(dispatch, "run_turn", fake_turn)
    sent: list[str] = []

    async def reply(text):
        sent.append(text)

    async def send(text):
        async with session_scope() as session:
            return await dispatch.dispatch_event(
                session, NormalizedEvent("55", Channel.TELEGRAM, text, reply)
            )

    async with session_scope() as s:
        await service.set_agent_active(s, agent_id, False)
        await s.commit()
    outcome = await send("hello")
    assert outcome is dispatch.DispatchOutcome.DENIED
    assert sent == [dispatch.AGENT_DISABLED_MESSAGE]
    assert calls == [], "the LLM must not be called for a deactivated agent"
    assert _sql("select direction, status from action_logs") == [("inbound", "denied")]

    async with session_scope() as s:
        await service.set_agent_active(s, agent_id, True)
        await s.commit()
    assert await send("hello again") is dispatch.DispatchOutcome.OK
    assert calls == ["hello again"] and sent[-1] == "answer"


async def test_approving_a_request_whose_identity_already_exists_grants_chat_without_a_duplicate(
    api, pending
):
    """Found on the real database: the request was made, then an admin added
    the same identity by hand. Approving must not crash (it did: HTTP 500).
    Since adding the identity through the service resolves the request, so
    this state is built by writing the row directly, as the old data was.
    """
    from app.db.models import ChannelIdentity, User
    from app.db.session import session_scope
    from app.security.hashing import channel_identifier_key

    async with session_scope() as session:
        user = User(display_name="Already there")
        session.add(user)
        await session.flush()
        session.add(
            ChannelIdentity(
                user_id=user.id,
                channel=Channel.TELEGRAM,
                external_id=channel_identifier_key(Channel.TELEGRAM, "777"),
            )
        )
        await session.commit()
        uid = user.id
    r = await api.post("/requests/1/approve", headers=AUTH)
    assert r.status_code == 200 and r.json()["id"] == uid
    assert _sql("select count(*) from users")[0][0] == 1, "no second user"
    assert _sql("select count(*) from channel_identities")[0][0] == 1, "no second identity"
    assert _sql("select kind from permissions") == [("chat",)]
    assert _sql("select status, resolved_by from access_requests where id = 1") == [
        ("approved", "api")
    ]
