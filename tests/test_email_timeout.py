"""Tests: an email turn waits for the engine as long as a scheduled task does
(LLM_TASK_TIMEOUT_SECONDS), a chat turn keeps the chat limit, and the setting never leaks
into the next turn."""

import pytest

from app.channels.schema import NormalizedEvent
from app.db.models import Channel


@pytest.fixture
async def senders(fresh_db, monkeypatch):
    from app.admin import service
    from app.channels import dispatch
    from app.config import get_settings
    from app.db.models import PermissionKind
    from app.db.session import init_db, session_scope
    from app.graph import llm_timeout

    monkeypatch.setenv("LLM_TASK_TIMEOUT_SECONDS", "600")
    get_settings.cache_clear()
    await init_db()
    async with session_scope() as s:
        user = await service.create_user(s, "Eve")
        for channel, external in ((Channel.TELEGRAM, "55"), (Channel.EMAIL, "eve@example.org")):
            identity = await service.add_channel_identity(s, user.id, channel, external)
            await service.grant_identity_permission(s, user.id, identity.id, PermissionKind.CHAT)
        await s.commit()
    seen: list = []

    async def fake_turn(*args, **kwargs):
        seen.append(llm_timeout.get())
        return "ok"

    monkeypatch.setattr(dispatch, "run_turn", fake_turn)
    yield seen
    get_settings.cache_clear()


async def _send(channel: Channel, user_id: str) -> None:
    from app.channels.dispatch import dispatch_event
    from app.db.session import session_scope

    async def reply(_text):
        pass

    async with session_scope() as s:
        await dispatch_event(s, NormalizedEvent(user_id, channel, "hello", reply))


async def test_an_email_turn_gets_the_task_limit_and_a_chat_turn_keeps_its_own(senders):
    from app.graph import llm_timeout

    await _send(Channel.EMAIL, "eve@example.org")
    assert llm_timeout.get() is None, "reset after the email turn"
    await _send(Channel.TELEGRAM, "55")
    assert senders == [600, None]


async def test_a_terminal_turn_also_gets_the_task_limit(senders):
    """A terminal turn is not streamed: the chat limit cut a whole answer at 120 s."""
    from app.admin import service
    from app.db.models import PermissionKind
    from app.db.session import session_scope

    async with session_scope() as s:
        user = await service.create_user(s, "Owner")
        identity = await service.add_channel_identity(s, user.id, Channel.TERMINAL, "owner")
        await service.grant_identity_permission(s, user.id, identity.id, PermissionKind.CHAT)
        await s.commit()
    await _send(Channel.TERMINAL, "owner")
    assert senders == [600]
