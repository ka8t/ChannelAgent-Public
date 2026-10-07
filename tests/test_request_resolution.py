"""Tests: the pending list only holds people who still need a
decision. An identity that an admin creates or grants a permission resolves
its pending request; automated mail is not a request and gets no reply.
"""

import email
import sqlite3
from email.mime.text import MIMEText

import httpx
import pytest

from app.admin import service
from app.channels import email as email_adapter
from app.db.models import Channel, PermissionKind
from app.db.session import init_db, session_scope
from app.security.hashing import channel_identifier_key

KEY = "test-key-" + "a" * 20
AUTH = {"Authorization": f"Bearer {KEY}"}


def _sql(query: str):
    from app.config import get_settings

    con = sqlite3.connect(get_settings().database_url.split("///", 1)[1])
    try:
        return con.execute(query).fetchall()
    finally:
        con.close()


@pytest.fixture
async def db(fresh_db):
    await init_db()


# --- an identity that is created or granted resolves its request ---


async def test_adding_the_identity_approves_its_pending_request(db):
    async with session_scope() as s:
        await service.request_access(s, Channel.TELEGRAM, "777", "let me in")
        user = await service.create_user(s, "Sam")
        await service.add_channel_identity(s, user.id, Channel.TELEGRAM, "777", actor="console")
        await s.commit()
    assert _sql("select status, resolved_by from access_requests") == [("approved", "console")]
    assert _sql("select resolved_at is not null from access_requests") == [(1,)]
    async with session_scope() as s:
        assert await service.list_pending_requests(s) == []


async def test_granting_a_permission_approves_a_pending_request(db):
    """The identity existed already (as on the real database)."""
    from app.db.models import ChannelIdentity

    async with session_scope() as s:
        user = await service.create_user(s, "Sam")
        s.add(ChannelIdentity(user_id=user.id, channel=Channel.TELEGRAM, external_id="777"))
        await s.flush()
        await service.request_access(s, Channel.TELEGRAM, "777", "let me in")
        identity_id = (await service.list_channel_identities(s, user.id))[0].id
        await service.grant_identity_permission(
            s, user.id, identity_id, PermissionKind.CHAT, actor="api"
        )
        await s.commit()
    assert _sql("select status, resolved_by from access_requests") == [("approved", "api")]


async def test_an_email_request_is_resolved_by_adding_the_address(db):
    address = "montezuma@outlook.fr"
    async with session_scope() as s:
        await service.request_access(
            s, Channel.EMAIL, channel_identifier_key(Channel.EMAIL, address), "hello"
        )
        user = await service.create_user(s, "Monty")
        await service.add_channel_identity(s, user.id, Channel.EMAIL, address, actor="api")
        await s.commit()
    assert _sql("select status from access_requests") == [("approved",)]


async def test_other_requests_are_left_alone(db):
    async with session_scope() as s:
        await service.request_access(s, Channel.TELEGRAM, "777", "mine")
        await service.request_access(s, Channel.TELEGRAM, "888", "someone else")
        await service.request_access(s, Channel.EMAIL, "777", "same id, other channel")
        user = await service.create_user(s, "Sam")
        await service.add_channel_identity(s, user.id, Channel.TELEGRAM, "777", actor="api")
        await s.commit()
    assert _sql("select channel, external_id, status from access_requests order by id") == [
        ("telegram", "777", "approved"),
        ("telegram", "888", "pending"),
        ("email", "777", "pending"),
    ]


async def test_an_already_denied_request_is_not_reopened_or_changed(db):
    async with session_scope() as s:
        request = await service.request_access(s, Channel.TELEGRAM, "777", "hi")
        await service.deny_request(s, request.id, resolved_by="console")
        user = await service.create_user(s, "Sam")
        await service.add_channel_identity(s, user.id, Channel.TELEGRAM, "777", actor="api")
        await s.commit()
    assert _sql("select status, resolved_by from access_requests") == [("denied", "console")]


async def test_the_resolution_is_recorded_in_the_admin_events(db):
    async with session_scope() as s:
        await service.request_access(s, Channel.TELEGRAM, "777", "hi")
        user = await service.create_user(s, "Sam")
        await service.add_channel_identity(s, user.id, Channel.TELEGRAM, "777", actor="api")
        await s.commit()
    assert _sql(
        "select actor, action, target_type from admin_events where action = 'request.auto_approve'"
    ) == [("api", "request.auto_approve", "access_request")]


async def test_through_the_api_the_request_is_approved_by_api(fresh_db, monkeypatch):
    from app.api.app import app
    from app.config import get_settings

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    await init_db()
    async with session_scope() as s:
        await service.request_access(s, Channel.TELEGRAM, "777", "let me in")
        await s.commit()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        uid = (await c.post("/users", json={"display_name": "Sam"}, headers=AUTH)).json()["id"]
        r = await c.post(
            f"/users/{uid}/channels",
            json={"channel": "telegram", "identifier": "777"},
            headers=AUTH,
        )
        assert r.status_code == 201
        pending = (await c.get("/requests", headers=AUTH)).json()
        approved = (await c.get("/requests?status=approved", headers=AUTH)).json()
    assert pending == []
    assert [(x["id"], x["resolved_by"]) for x in approved] == [(1, "api")]


# --- automated mail is not a request and gets no reply ---


def _raw(headers: dict[str, str], sender="robot@example.com") -> bytes:
    msg = MIMEText("[Django] collector failed, see the log", "plain", "utf-8")
    msg["From"] = sender
    msg["Subject"] = "[agent] alert"
    for name, value in headers.items():
        msg[name] = value
    return msg.as_bytes()


def _parsed(headers: dict[str, str]):
    return email.message_from_bytes(_raw(headers))


@pytest.mark.parametrize(
    "headers",
    [
        {"Auto-Submitted": "auto-generated"},
        {"Auto-Submitted": "auto-replied"},
        {"Auto-Submitted": "Auto-Generated; type=notification"},
        {"Precedence": "bulk"},
        {"Precedence": "list"},
        {"Precedence": "JUNK"},
    ],
)
def test_automated_headers_are_recognised(headers):
    assert email_adapter._is_automated(_parsed(headers)) is True


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Auto-Submitted": "no"},
        {"Auto-Submitted": "No"},
        {"Auto-Submitted": "no; comment=written-by-a-person"},
        {"Precedence": "first-class"},
    ],
)
def test_ordinary_mail_is_not_automated(headers):
    assert email_adapter._is_automated(_parsed(headers)) is False


@pytest.fixture
def fake_imap(monkeypatch):
    from tests.test_email_adapter import _FakeIMAP

    def install(messages):
        fake = _FakeIMAP(messages)
        monkeypatch.setattr(email_adapter.imaplib, "IMAP4_SSL", lambda *a, **k: fake)
        return fake

    return install


@pytest.fixture
def mailbox(db, monkeypatch):
    """Settings for _poll_once and a recorder for every outgoing reply."""
    sent: list[tuple] = []
    monkeypatch.setattr(
        email_adapter, "_send_reply_sync", lambda to, subject, body: sent.append((to, body))
    )
    monkeypatch.setattr(
        email_adapter,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "email_imap_host": "h", "email_imap_port": 993, "email_username": "bot@x.org",
                "email_password": "p", "email_trigger_tag": "[agent]",
                "email_agent_folder": "INBOX.Agent", "llm_task_timeout_seconds": 600,
            },
        )(),
    )
    email_adapter._attempts.clear()
    return sent


@pytest.mark.parametrize(
    "headers", [{"Auto-Submitted": "auto-generated"}, {"Precedence": "bulk"}]
)
async def test_a_tagged_automated_mail_creates_no_request_and_no_reply(
    mailbox, fake_imap, headers
):
    fake = fake_imap({b"21": _raw(headers)})
    await email_adapter._poll_once()
    assert _sql("select count(*) from access_requests") == [(0,)]
    assert mailbox == [], "no reply sent"
    assert fake.moves == [(b"21", "INBOX.Agent")], "filed, not left to be fetched again"


async def test_the_same_mail_without_the_header_does_create_a_request(mailbox, fake_imap):
    """The control: the header is what prevents the request."""
    fake_imap({b"21": _raw({})})
    await email_adapter._poll_once()
    assert _sql("select count(*) from access_requests") == [(1,)]
    assert mailbox == [], "an unknown sender is never answered"


async def test_an_authorized_sender_automatic_reply_never_reaches_the_model(
    mailbox, fake_imap, monkeypatch
):
    """An out-of-office answer to the bot's own reply must not start a loop."""
    async with session_scope() as s:
        user = await service.create_user(s, "Robot owner")
        identity = await service.add_channel_identity(
            s, user.id, Channel.EMAIL, "robot@example.com"
        )
        await service.grant_identity_permission(s, user.id, identity.id, PermissionKind.CHAT)
        await s.commit()
    calls = []

    async def no_model(*a, **k):
        calls.append(a)
        return "should not happen"

    monkeypatch.setattr("app.channels.dispatch.run_turn", no_model)
    fake_imap({b"22": _raw({"Auto-Submitted": "auto-replied"})})
    await email_adapter._poll_once()
    assert calls == [] and mailbox == []
    assert _sql("select count(*) from action_logs") == [(0,)]


def test_the_bots_own_replies_are_marked_auto_replied(monkeypatch):
    captured = {}

    class _Smtp:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, *a):
            pass

        def send_message(self, msg):
            captured["msg"] = msg

    monkeypatch.setattr(email_adapter.smtplib, "SMTP_SSL", _Smtp)
    monkeypatch.setattr(
        email_adapter,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "email_username": "bot@x.org", "email_password": "p",
                "email_smtp_host": "h", "email_smtp_port": 465,
            },
        )(),
    )
    email_adapter._send_reply_sync("someone@example.com", "Re: [agent] hi", "answer")
    assert captured["msg"]["Auto-Submitted"] == "auto-replied"
    assert email_adapter._is_automated(captured["msg"]) is True, "so two bots cannot ping-pong"
