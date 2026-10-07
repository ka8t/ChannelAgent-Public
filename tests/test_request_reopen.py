"""Tests: a person whose earlier access request was resolved is
handled deliberately. Before the fix, writing again after a denial raised an
IntegrityError (unique constraint on channel + external_id).
"""

import sqlite3

import pytest

from app.admin import service
from app.channels import notify
from app.channels.dispatch import DENIED_MESSAGE, DispatchOutcome, dispatch_event
from app.channels.schema import NormalizedEvent
from app.db.models import Channel, PermissionKind
from app.db.session import init_db, session_scope


def _sql(query):
    from app.config import get_settings

    con = sqlite3.connect(get_settings().database_url.split("///", 1)[1])
    try:
        return con.execute(query).fetchall()
    finally:
        con.close()


@pytest.fixture
def told():
    """Admin notifications: (external_id, text) sent through a fake sender."""
    messages: list[tuple[str, str]] = []

    async def sender(external_id, text):
        messages.append((external_id, text))

    notify.register_sender(Channel.TELEGRAM, sender)
    yield messages
    notify.unregister_sender(Channel.TELEGRAM)


@pytest.fixture
async def admin(fresh_db):
    await init_db()
    async with session_scope() as s:
        user = await service.create_user(s, "Admin")
        identity = await service.add_channel_identity(s, user.id, Channel.TELEGRAM, "1")
        await service.grant_identity_permission(s, user.id, identity.id, PermissionKind.ADMIN)
        await s.commit()


async def _write(user, text, channel=Channel.TELEGRAM):
    replies: list[str] = []

    async def reply(t):
        replies.append(t)

    async with session_scope() as s:
        outcome = await dispatch_event(s, NormalizedEvent(user, channel, text, reply))
    return outcome, replies


# --- (a) denied: the denial stands ---


async def test_a_denied_sender_writing_again_gets_no_exception_and_no_new_row(admin, told):
    assert (await _write("900", "let me in"))[0] is DispatchOutcome.DENIED
    async with session_scope() as s:
        request_id = (await service.list_pending_requests(s))[0].id
        await service.deny_request(s, request_id, resolved_by="console")
        await s.commit()
    told.clear()

    outcome, replies = await _write("900", "please, again")

    assert outcome is DispatchOutcome.DENIED
    assert replies == [DENIED_MESSAGE], "the denial reply is sent"
    assert told == [], "no second notification"
    assert _sql("select count(*), min(status) from access_requests") == [(1, "denied")]


async def test_a_denied_email_sender_gets_no_reply_and_no_row(admin, told):
    from app.security.hashing import channel_identifier_key

    key = channel_identifier_key(Channel.EMAIL, "denied@example.com")
    async with session_scope() as s:
        request = await service.request_access(s, Channel.EMAIL, key, "hello")
        await service.deny_request(s, request.id, resolved_by="api")
        await s.commit()
    told.clear()
    outcome, replies = await _write("denied@example.com", "again", channel=Channel.EMAIL)
    assert outcome is DispatchOutcome.DENIED and replies == []
    assert _sql("select count(*), min(status) from access_requests") == [(1, "denied")]


# --- (b) approved, identity gone: reopened ---


async def test_an_approved_person_whose_identity_was_removed_reopens_the_request(admin, told):
    await _write("900", "let me in")
    async with session_scope() as s:
        request_id = (await service.list_pending_requests(s))[0].id
        user = await service.approve_request(s, request_id, resolved_by="api")
        identity = (await service.list_channel_identities(s, user.id))[0]
        await service.remove_channel_identity(s, user.id, identity.id)
        await s.commit()
    assert _sql("select status from access_requests") == [("approved",)]
    told.clear()

    outcome, replies = await _write("900", "I am back")

    assert outcome is DispatchOutcome.DENIED and replies == [DENIED_MESSAGE]
    assert _sql("select count(*), min(status) from access_requests") == [(1, "pending")]
    assert _sql("select resolved_at is null, resolved_by is null from access_requests") == [(1, 1)]
    assert len(told) == 1, "the admins are told once"
    async with session_scope() as s:
        assert (await service.list_pending_requests(s))[0].first_message_text == "I am back"


async def test_the_reopened_request_does_not_notify_again_on_the_next_message(admin, told):
    await _write("900", "first")
    async with session_scope() as s:
        request_id = (await service.list_pending_requests(s))[0].id
        user = await service.approve_request(s, request_id, resolved_by="api")
        identity = (await service.list_channel_identities(s, user.id))[0]
        await service.remove_channel_identity(s, user.id, identity.id)
        await s.commit()
    told.clear()
    await _write("900", "second")
    await _write("900", "third")
    assert len(told) == 1


async def test_a_first_request_and_a_pending_one_behave_as_before(admin, told):
    await _write("900", "first")
    await _write("900", "second")
    assert _sql("select count(*), min(status) from access_requests") == [(1, "pending")]
    assert len(told) == 1
    async with session_scope() as s:
        assert (await service.list_pending_requests(s))[0].first_message_text == "first"


async def test_the_service_flag_says_when_to_notify(fresh_db):
    await init_db()
    async with session_scope() as s:
        first, new_first = await service.ensure_access_request(s, Channel.TELEGRAM, "5", "a")
        again, new_again = await service.ensure_access_request(s, Channel.TELEGRAM, "5", "b")
        await service.deny_request(s, first.id)
        denied, new_denied = await service.ensure_access_request(s, Channel.TELEGRAM, "5", "c")
        await s.commit()
    assert (new_first, new_again, new_denied) == (True, False, False)
    assert first.id == again.id == denied.id
