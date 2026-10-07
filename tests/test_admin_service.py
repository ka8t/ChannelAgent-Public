"""Tests (AccessRequest) and (ActionLog) via app.admin.service
and their wiring into app.channels.dispatch.dispatch_event.
"""

import pytest
from sqlalchemy import select


@pytest.mark.asyncio
async def test_request_access_upserts_one_pending_row(fresh_db):
    from app.admin.service import request_access
    from app.db.models import AccessRequest, Channel
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as session:
        await request_access(session, Channel.TELEGRAM, "111", "hi")
        await request_access(session, Channel.TELEGRAM, "111", "hi again")
        await session.commit()

    async with session_scope() as session:
        rows = (await session.execute(select(AccessRequest))).scalars().all()
    assert len(rows) == 1, "a second message from the same pending identity must not duplicate"


@pytest.mark.asyncio
async def test_approve_request_creates_user_and_grants_chat(fresh_db):
    from app.admin.service import approve_request, request_access
    from app.db.models import Channel
    from app.db.session import init_db, session_scope
    from app.security.auth import authorize

    await init_db()
    async with session_scope() as session:
        req = await request_access(session, Channel.TELEGRAM, "222", "let me in")
        await session.commit()
        req_id = req.id

    async with session_scope() as session:
        await approve_request(session, req_id)
        await session.commit()

    async with session_scope() as session:
        decision = await authorize(session, Channel.TELEGRAM, "222")
    assert decision.allowed is True, "approving must grant CHAT immediately, no separate step"


@pytest.mark.asyncio
async def test_deny_request_does_not_create_a_user(fresh_db):
    from sqlalchemy import func

    from app.admin.service import deny_request, request_access
    from app.db.models import Channel, User
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as session:
        req = await request_access(session, Channel.TELEGRAM, "333", "let me in")
        await session.commit()
        req_id = req.id

    async with session_scope() as session:
        await deny_request(session, req_id)
        await session.commit()
        count = (await session.execute(select(func.count()).select_from(User))).scalar_one()
    assert count == 0


@pytest.mark.asyncio
async def test_dispatch_denied_unknown_identity_creates_request_no_action_log(
    fresh_db, monkeypatch
):
    from sqlalchemy import func

    from app.channels.dispatch import DENIED_MESSAGE, dispatch_event
    from app.channels.schema import NormalizedEvent
    from app.db.models import ActionLog, Channel
    from app.db.session import init_db, session_scope

    await init_db()
    sent = []

    async def reply(text):
        sent.append(text)

    async with session_scope() as session:
        event = NormalizedEvent(user_id="444", channel=Channel.TELEGRAM, text="hi", reply=reply)
        await dispatch_event(session, event)

    assert sent == [DENIED_MESSAGE]
    async with session_scope() as session:
        from app.db.models import AccessRequest

        requests = (await session.execute(select(AccessRequest))).scalars().all()
        logs = (await session.execute(select(func.count()).select_from(ActionLog))).scalar_one()
    assert len(requests) == 1
    assert logs == 0, "no User exists yet for an unknown identity, so no ActionLog row is possible"


@pytest.mark.asyncio
async def test_dispatch_allowed_writes_inbound_and_outbound_logs(fresh_db, monkeypatch):
    import app.graph as graph_module
    from app.channels.dispatch import dispatch_event
    from app.channels.schema import NormalizedEvent
    from app.db.models import ActionLog, Channel, ChannelIdentity, Direction, PermissionKind, User
    from app.db.session import init_db, session_scope
    from app.security.auth import grant_permission

    async def fake_run_turn(channel, user_id, agent_id, text, **_kwargs):
        return "canned reply"

    monkeypatch.setattr(graph_module, "run_turn", fake_run_turn)
    import app.channels.dispatch as dispatch_module

    monkeypatch.setattr(dispatch_module, "run_turn", fake_run_turn)

    await init_db()
    async with session_scope() as session:
        user = User(display_name="Log test")
        session.add(user)
        await session.flush()
        identity = ChannelIdentity(user_id=user.id, channel=Channel.TELEGRAM, external_id="555")
        session.add(identity)
        await session.flush()
        await grant_permission(session, identity, PermissionKind.CHAT)
        await session.commit()

    sent = []

    async def reply(text):
        sent.append(text)

    async with session_scope() as session:
        event = NormalizedEvent(user_id="555", channel=Channel.TELEGRAM, text="hi", reply=reply)
        await dispatch_event(session, event)

    assert sent == ["canned reply"]
    async with session_scope() as session:
        logs = (await session.execute(select(ActionLog).order_by(ActionLog.id))).scalars().all()
    assert len(logs) == 2
    assert logs[0].direction == Direction.INBOUND and logs[0].text == "hi"
    assert logs[1].direction == Direction.OUTBOUND and logs[1].text == "canned reply"
