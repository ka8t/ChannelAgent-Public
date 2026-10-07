"""Integration tests for the Auth Node: authorize() against a
real (temp file) SQLite DB, not mocks — the property that matters is
that it fails closed and reacts to DB changes with no caching to
invalidate, which a mock of the DB layer wouldn't actually exercise.
"""

import pytest


@pytest.mark.asyncio
async def test_unknown_identity_is_denied(fresh_db):
    from app.db.models import Channel
    from app.db.session import init_db, session_scope
    from app.security.auth import authorize

    await init_db()
    async with session_scope() as session:
        decision = await authorize(session, Channel.TELEGRAM, "does-not-exist")
    assert decision.allowed is False
    assert decision.user is None


@pytest.mark.asyncio
async def test_known_identity_with_chat_permission_is_allowed(fresh_db):
    from app.db.models import Channel, ChannelIdentity, PermissionKind, User
    from app.db.session import init_db, session_scope
    from app.security.auth import authorize, grant_permission

    await init_db()
    async with session_scope() as session:
        user = User(display_name="Test")
        session.add(user)
        await session.flush()
        identity = ChannelIdentity(user_id=user.id, channel=Channel.TELEGRAM, external_id="1")
        session.add(identity)
        await session.flush()
        await grant_permission(session, identity, PermissionKind.CHAT)
        await session.commit()

    async with session_scope() as session:
        decision = await authorize(session, Channel.TELEGRAM, "1")
    assert decision.allowed is True
    assert decision.is_admin is False


@pytest.mark.asyncio
async def test_known_identity_without_any_permission_is_denied(fresh_db):
    from app.db.models import Channel, ChannelIdentity, User
    from app.db.session import init_db, session_scope
    from app.security.auth import authorize

    await init_db()
    async with session_scope() as session:
        user = User(display_name="Test")
        session.add(user)
        await session.flush()
        session.add(ChannelIdentity(user_id=user.id, channel=Channel.TELEGRAM, external_id="2"))
        await session.commit()

    async with session_scope() as session:
        decision = await authorize(session, Channel.TELEGRAM, "2")
    assert decision.allowed is False


@pytest.mark.asyncio
async def test_revoked_permission_is_denied_on_the_next_call(fresh_db):
    from app.db.models import Channel, ChannelIdentity, PermissionKind, User
    from app.db.session import init_db, session_scope
    from app.security.auth import authorize, grant_permission, revoke_permission

    await init_db()
    async with session_scope() as session:
        user = User(display_name="Test")
        session.add(user)
        await session.flush()
        identity = ChannelIdentity(user_id=user.id, channel=Channel.TELEGRAM, external_id="3")
        session.add(identity)
        await session.flush()
        await grant_permission(session, identity, PermissionKind.CHAT)
        await session.commit()
        identity_id = identity.id

    async with session_scope() as session:
        decision = await authorize(session, Channel.TELEGRAM, "3")
        assert decision.allowed is True

    async with session_scope() as session:
        identity = await session.get(ChannelIdentity, identity_id)
        revoked = await revoke_permission(session, identity, PermissionKind.CHAT)
        await session.commit()
    assert revoked is True

    async with session_scope() as session:
        decision = await authorize(session, Channel.TELEGRAM, "3")
    assert decision.allowed is False, "revocation must take effect on the very next check"


@pytest.mark.asyncio
async def test_inactive_user_is_denied_even_with_a_permission(fresh_db):
    from app.db.models import Channel, ChannelIdentity, PermissionKind, User
    from app.db.session import init_db, session_scope
    from app.security.auth import authorize, grant_permission

    await init_db()
    async with session_scope() as session:
        user = User(display_name="Test", is_active=False)
        session.add(user)
        await session.flush()
        identity = ChannelIdentity(user_id=user.id, channel=Channel.TELEGRAM, external_id="4")
        session.add(identity)
        await session.flush()
        await grant_permission(session, identity, PermissionKind.ADMIN)
        await session.commit()

    async with session_scope() as session:
        decision = await authorize(session, Channel.TELEGRAM, "4")
    assert decision.allowed is False
