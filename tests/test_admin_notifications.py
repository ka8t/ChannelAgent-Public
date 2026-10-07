"""Tests: a new access request really notifies the administrators
(the denial message used to claim it while nothing did), once per request,
and telling the admins never breaks the message that triggered it.
"""

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from app.channels import notify
from app.channels.dispatch import DENIED_MESSAGE, DispatchOutcome, dispatch_event
from app.channels.schema import NormalizedEvent
from app.db.models import Channel, PermissionKind


def _sql(query, *args):
    from app.config import get_settings

    con = sqlite3.connect(get_settings().database_url.split("///", 1)[1])
    try:
        return con.execute(query, args).fetchall()
    finally:
        con.close()


@pytest.fixture
def sent():
    """A fake Telegram sender registered for the test, removed afterwards."""
    messages: list[tuple[str, str]] = []

    async def sender(external_id, text):
        messages.append((external_id, text))

    notify.register_sender(Channel.TELEGRAM, sender)
    yield messages
    notify.unregister_sender(Channel.TELEGRAM)


@pytest.fixture
async def admins(fresh_db):
    """1 Alice admin, 2 Bob admin, 3 Carol chat only, 4 Dave admin but inactive,
    5 Eve admin on email (no sender exists for email), 6 Fay known but no
    permission at all.
    """
    from app.admin import service
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as s:
        for name, ext, kind, active, channel in [
            ("Alice", "1", PermissionKind.ADMIN, True, Channel.TELEGRAM),
            ("Bob", "2", PermissionKind.ADMIN, True, Channel.TELEGRAM),
            ("Carol", "3", PermissionKind.CHAT, True, Channel.TELEGRAM),
            ("Dave", "4", PermissionKind.ADMIN, False, Channel.TELEGRAM),
            ("Eve", "eve@example.com", PermissionKind.ADMIN, True, Channel.EMAIL),
        ]:
            user = await service.create_user(s, name)
            identity = await service.add_channel_identity(s, user.id, channel, ext)
            await service.grant_identity_permission(s, user.id, identity.id, kind)
            await service.update_user(s, user.id, is_active=active)
        fay = await service.create_user(s, "Fay")
        await service.add_channel_identity(s, fay.id, Channel.TELEGRAM, "6")
        await s.commit()


async def _write(user, text="let me in", channel=Channel.TELEGRAM):
    from app.db.session import session_scope

    replies: list[str] = []

    async def reply(t):
        replies.append(t)

    async with session_scope() as s:
        outcome = await dispatch_event(s, NormalizedEvent(user, channel, text, reply))
    return outcome, replies


async def test_a_new_request_notifies_each_active_admin_once_and_only_them(admins, sent):
    outcome, replies = await _write("999", "let me in please")
    assert outcome is DispatchOutcome.DENIED and replies == [DENIED_MESSAGE]
    assert [m[0] for m in sent] == ["1", "2"], "Alice and Bob only"
    text = sent[0][1]
    assert "#1" in text and "telegram/999" in text and "let me in please" in text
    assert sent[0][1] == sent[1][1]


async def test_repeated_messages_from_the_same_pending_identity_do_not_notify_again(admins, sent):
    await _write("999", "first")
    await _write("999", "second")
    await _write("999", "third")
    assert len(sent) == 2, "2 admins x 1 request"
    assert _sql("select count(*) from access_requests") == [(1,)]


async def test_another_identity_is_another_request_and_notifies_again(admins, sent):
    await _write("999")
    await _write("888")
    assert len(sent) == 4
    assert {"#1", "#2"} <= {w for m in sent for w in m[1].split() if w.startswith("#")}


async def test_a_known_identity_without_permission_also_creates_a_notified_request(admins, sent):
    await _write("6", "I am Fay")
    assert len(sent) == 2 and "telegram/6" in sent[0][1]


async def test_a_request_from_email_still_notifies_the_telegram_admins_silently(admins, sent):
    outcome, replies = await _write("stranger@example.com", "hello", Channel.EMAIL)
    assert outcome is DispatchOutcome.DENIED and replies == [], "no reply on email"
    assert [m[0] for m in sent] == ["1", "2"]


async def test_an_admin_on_a_channel_without_a_sender_is_skipped(admins, sent):
    from app.db.session import session_scope

    async with session_scope() as s:
        reached = await notify.notify_admins(s, "hello")
    assert reached == 2 and [m[0] for m in sent] == ["1", "2"], "Eve on email is skipped"


async def test_a_failing_send_neither_stops_the_others_nor_the_user_reply(admins, monkeypatch):
    got: list[str] = []

    async def flaky(external_id, text):
        if external_id == "1":
            raise RuntimeError("telegram down")
        got.append(external_id)

    notify.register_sender(Channel.TELEGRAM, flaky)
    try:
        outcome, replies = await _write("999")
    finally:
        notify.unregister_sender(Channel.TELEGRAM)
    assert got == ["2"], "Bob is still notified"
    assert outcome is DispatchOutcome.DENIED and replies == [DENIED_MESSAGE]
    assert _sql("select count(*) from access_requests") == [(1,)]


async def test_with_no_sender_registered_nothing_breaks(admins):
    assert notify.get_sender(Channel.TELEGRAM) is None
    outcome, replies = await _write("999")
    assert outcome is DispatchOutcome.DENIED and replies == [DENIED_MESSAGE]
    assert _sql("select count(*) from access_requests") == [(1,)]


async def test_a_long_first_message_is_truncated_and_flattened(admins, sent):
    await _write("999", "x" * 500 + "\nsecond line")
    text = sent[0][1]
    assert "x" * notify.MAX_QUOTED_MESSAGE + "..." in text
    assert "x" * (notify.MAX_QUOTED_MESSAGE + 1) not in text and "\n" not in text


def test_the_denial_message_no_longer_claims_a_notification():
    assert "notified" not in DENIED_MESSAGE
    assert "recorded" in DENIED_MESSAGE


# --- the Telegram adapter is the sender while it runs ---


class _FakeApplication:
    def __init__(self):
        self.sent = []
        self.bot = SimpleNamespace(send_message=self._send)
        self.updater = SimpleNamespace(start_polling=self._noop, stop=self._noop)

    async def _send(self, chat_id, text):
        self.sent.append((chat_id, text))

    async def _noop(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def start(self):
        pass

    async def stop(self):
        pass


async def test_the_telegram_adapter_registers_a_sender_while_running_and_removes_it(monkeypatch):
    from app.channels import telegram

    fake = _FakeApplication()
    monkeypatch.setattr(telegram, "build_application", lambda: fake)
    assert notify.get_sender(Channel.TELEGRAM) is None
    task = asyncio.create_task(telegram.run_telegram_adapter())
    for _ in range(50):
        await asyncio.sleep(0.01)
        if notify.get_sender(Channel.TELEGRAM) is not None:
            break
    sender = notify.get_sender(Channel.TELEGRAM)
    assert sender is not None
    await sender("100200300", "hello admin")
    assert fake.sent == [(100200300, "hello admin")], "the id is sent as an integer chat id"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert notify.get_sender(Channel.TELEGRAM) is None
