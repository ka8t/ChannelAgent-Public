"""Tests: rate limit and turn limit per user, the same for everyone.

- With a limit of L per minute, message L+1 gets the refusal text and 1 log row (status
  limited), and is not run; a second user is unaffected; after 60 seconds the user may write.
- One turn at a time: a second message while the first is being answered gets the notice.
- 0 turns a limit off; an email over the limit gets no reply and is taken again later.
- An invalid value is refused by `./start.sh --set` (exit code).
"""

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from app.admin import service
from app.channels import dispatch, limits
from app.channels.dispatch import DispatchOutcome, dispatch_event
from app.channels.schema import NormalizedEvent
from app.db.models import Channel, PermissionKind

REPO = Path(__file__).resolve().parents[1]


def _rows(user_id: int):
    import sqlite3

    from app.config import get_settings

    con = sqlite3.connect(get_settings().database_url.split("///", 1)[1])
    try:
        return con.execute(
            "select direction, status from action_logs where user_id = ? order by id", (user_id,)
        ).fetchall()
    finally:
        con.close()


@pytest.fixture
async def world(fresh_db, monkeypatch):
    """Alice (Telegram "1") and Bob (Telegram "2"), both allowed to chat; turns are fake."""
    from app.config import get_settings
    from app.db.session import init_db, session_scope

    monkeypatch.setenv("RATE_LIMIT_MESSAGES_PER_MINUTE", "3")
    monkeypatch.setenv("RATE_LIMIT_CONCURRENT_TURNS", "1")
    get_settings.cache_clear()
    await init_db()
    async with session_scope() as s:
        for name, ext in (("Alice", "1"), ("Bob", "2")):
            user = await service.create_user(s, name)
            identity = await service.add_channel_identity(s, user.id, Channel.TELEGRAM, ext)
            await service.grant_identity_permission(s, user.id, identity.id, PermissionKind.CHAT)
            await service.create_agent(s, user.id, "default")
        await s.commit()
    turns = []

    async def fake_turn(channel, user_id, agent_id, text, **_kw):
        turns.append((user_id, text))
        return f"answer to {text}"

    monkeypatch.setattr(dispatch, "run_turn", fake_turn)
    yield turns
    get_settings.cache_clear()


async def _send(user: str, text: str, channel=Channel.TELEGRAM):
    from app.db.session import session_scope

    replies = []

    async def reply(t):
        replies.append(t)

    event = NormalizedEvent(user_id=user, channel=channel, text=text, reply=reply)
    async with session_scope() as s:
        outcome = await dispatch_event(s, event)
    return outcome, replies


async def test_message_l_plus_one_is_refused_and_logged_once_and_others_are_unaffected(world):
    outcomes = [await _send("1", f"m{i}") for i in range(4)]
    assert [o for o, _ in outcomes] == [DispatchOutcome.OK] * 3 + [DispatchOutcome.LIMITED]
    assert outcomes[3][1] == [limits.MESSAGES["rate"]]
    assert world == [("1", "m0"), ("1", "m1"), ("1", "m2")], "m3 was not run"
    alice = _rows(1)
    assert alice.count(("inbound", "limited")) == 1 and len(alice) == 7  # 3 x 2 + 1
    bob = await _send("2", "hello")
    assert bob == (DispatchOutcome.OK, ["answer to hello"])


async def test_the_window_is_sixty_seconds(world, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(limits, "_now", lambda: clock[0])
    for i in range(3):
        await _send("1", f"m{i}")
    assert (await _send("1", "too soon"))[0] == DispatchOutcome.LIMITED
    clock[0] += 59.9
    assert (await _send("1", "still too soon"))[0] == DispatchOutcome.LIMITED
    clock[0] += 0.1
    assert (await _send("1", "now"))[0] == DispatchOutcome.OK


async def test_one_turn_at_a_time(world, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_turn(channel, user_id, agent_id, text, **_kw):
        started.set()
        await release.wait()
        return "done"

    monkeypatch.setattr(dispatch, "run_turn", slow_turn)
    first = asyncio.create_task(_send("1", "long question"))
    await started.wait()
    try:  # without the turn limit the second message would wait for the first: bounded
        second = await asyncio.wait_for(_send("1", "and another"), timeout=5)
    finally:
        release.set()
    assert second == (DispatchOutcome.LIMITED, [limits.MESSAGES["busy"]])
    assert (await first)[0] == DispatchOutcome.OK
    assert (await _send("1", "after it"))[0] == DispatchOutcome.OK


async def test_zero_turns_the_limits_off(world, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("RATE_LIMIT_MESSAGES_PER_MINUTE", "0")
    get_settings.cache_clear()
    outcomes = [(await _send("1", f"m{i}"))[0] for i in range(10)]
    assert outcomes == [DispatchOutcome.OK] * 10


async def test_an_email_over_the_limit_gets_no_reply(fresh_db, monkeypatch):
    from app.config import get_settings
    from app.db.session import init_db, session_scope

    monkeypatch.setenv("RATE_LIMIT_MESSAGES_PER_MINUTE", "1")
    get_settings.cache_clear()
    await init_db()
    async with session_scope() as s:
        user = await service.create_user(s, "Mail")
        identity = await service.add_channel_identity(s, user.id, Channel.EMAIL, "a@example.org")
        await service.grant_identity_permission(s, user.id, identity.id, PermissionKind.CHAT)
        await s.commit()

    async def fake_turn(*_a, **_kw):
        return "ok"

    monkeypatch.setattr(dispatch, "run_turn", fake_turn)
    assert (await _send("a@example.org", "one", Channel.EMAIL))[0] == DispatchOutcome.OK
    outcome, replies = await _send("a@example.org", "two", Channel.EMAIL)
    assert outcome == DispatchOutcome.LIMITED and replies == []


def test_an_invalid_value_is_refused_by_set(tmp_path):
    shutil.copy(REPO / "start.sh", tmp_path / "start.sh")
    shutil.copy(REPO / ".env.example", tmp_path / ".env.example")
    (tmp_path / "app").mkdir()
    shutil.copy(REPO / "app" / "settings_rules.py", tmp_path / "app" / "settings_rules.py")
    (tmp_path / ".env").write_text("RATE_LIMIT_MESSAGES_PER_MINUTE=20\n")
    (tmp_path / ".env").chmod(0o600)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    bad = subprocess.run(
        ["bash", "start.sh", "--config", "RATE_LIMIT_MESSAGES_PER_MINUTE=-5"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    good = subprocess.run(
        ["bash", "start.sh", "--config", "RATE_LIMIT_MESSAGES_PER_MINUTE=30"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert bad.returncode == 1 and "RATE_LIMIT_MESSAGES_PER_MINUTE" in bad.stderr
    assert good.returncode == 0
    assert "RATE_LIMIT_MESSAGES_PER_MINUTE=30" in (tmp_path / ".env").read_text()
