"""Tests: the container healthcheck.

The application touches a heartbeat file while it is doing its job;
`python -m app.health` succeeds only when that file is recent.
"""

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from app import health

REPO = Path(__file__).resolve().parent.parent


# --- check() ---


def test_no_heartbeat_file_is_unhealthy(tmp_path):
    ok, message = health.check(tmp_path / "missing")
    assert not ok and "no heartbeat file" in message


def test_a_fresh_heartbeat_is_healthy(tmp_path):
    path = tmp_path / "beat"
    health.beat(path)
    ok, _ = health.check(path, max_age=60)
    assert ok


def test_a_stale_heartbeat_is_unhealthy(tmp_path):
    path = tmp_path / "beat"
    health.beat(path)
    old = time.time() - 120
    os.utime(path, (old, old))
    ok, message = health.check(path, max_age=60)
    assert not ok and "(limit 60 s)" in message


def test_the_age_limit_is_the_boundary(tmp_path):
    path = tmp_path / "beat"
    health.beat(path)
    at_50 = time.time() - 50
    os.utime(path, (at_50, at_50))
    assert health.check(path, max_age=60)[0]
    assert not health.check(path, max_age=40)[0]


def test_the_check_interval_is_well_inside_the_age_limit():
    assert health.INTERVAL_SECONDS * 3 < health.MAX_AGE_SECONDS


# --- heartbeat() ---


async def _running():
    await asyncio.sleep(3600)


async def test_the_heartbeat_is_written_while_a_component_runs():
    component = asyncio.create_task(_running())
    beating = asyncio.create_task(health.heartbeat([component], interval=0.05))
    await asyncio.sleep(0.2)
    assert health.check()[0]
    assert not beating.done()
    beating.cancel()
    component.cancel()
    await asyncio.gather(beating, component, return_exceptions=True)


async def test_the_heartbeat_stops_when_every_component_has_ended():
    async def ends():
        return None

    component = asyncio.create_task(ends())
    await component
    await asyncio.wait_for(health.heartbeat([component], interval=0.05), timeout=2)
    assert not health.HEARTBEAT_PATH.exists()


async def test_the_heartbeat_continues_while_one_of_several_components_runs():
    async def ends():
        return None

    dead, alive = asyncio.create_task(ends()), asyncio.create_task(_running())
    await dead
    beating = asyncio.create_task(health.heartbeat([dead, alive], interval=0.05))
    await asyncio.sleep(0.2)
    assert not beating.done() and health.check()[0]
    beating.cancel()
    alive.cancel()
    await asyncio.gather(beating, alive, return_exceptions=True)


async def test_with_no_component_the_application_is_idling_and_alive():
    beating = asyncio.create_task(health.heartbeat([], interval=0.05))
    await asyncio.sleep(0.2)
    assert not beating.done() and health.check()[0]
    beating.cancel()
    await asyncio.gather(beating, return_exceptions=True)


# --- the real process and the real command ---


def _env(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABASE_", "CHECKPOINT_"))}
    env.update(
        PYTHONPATH=str(REPO),
        TMPDIR=str(tmp_path),
        DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path}/data/channelagent.db",
        TELEGRAM_BOT_TOKEN="",
        EMAIL_IMAP_HOST="",
        API_SERVER_KEY="",
        MIGRATION_BACKUPS_KEEP="0",
    )
    return env


def _run_health(env):
    return subprocess.run(
        [sys.executable, "-m", "app.health"], env=env, capture_output=True, text=True, timeout=30
    )


def test_the_command_exits_1_without_a_heartbeat_and_0_with_one(tmp_path):
    env = _env(tmp_path)
    missing = _run_health(env)
    assert missing.returncode == 1 and missing.stdout.startswith("unhealthy")
    (tmp_path / "channelagent.heartbeat").touch()
    present = _run_health(env)
    assert present.returncode == 0 and present.stdout.startswith("healthy")


def test_a_running_application_is_healthy_and_a_stuck_one_is_not(tmp_path):
    env = _env(tmp_path)
    beat_file = tmp_path / "channelagent.heartbeat"
    app = subprocess.Popen(
        [sys.executable, "-m", "app.main"],
        env=env,
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 30
        while not beat_file.exists() and time.time() < deadline:
            time.sleep(0.2)
        assert beat_file.exists(), "the application never wrote a heartbeat"
        assert _run_health(env).returncode == 0

        # Freeze the process: its event loop stops, so the file stops being
        # touched. Age the file as the passing of MAX_AGE_SECONDS would.
        os.kill(app.pid, signal.SIGSTOP)
        old = time.time() - 2 * health.MAX_AGE_SECONDS
        os.utime(beat_file, (old, old))
        time.sleep(1)
        assert os.stat(beat_file).st_mtime == old
        assert _run_health(env).returncode == 1
    finally:
        os.kill(app.pid, signal.SIGCONT)
        app.terminate()
        app.wait(timeout=30)


# --- a component that runs but has stopped succeeding ---


def test_a_registered_component_is_stale_only_after_its_max_age(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(health.time, "monotonic", lambda: now[0])
    health.register("email", 300)
    assert health.stale_components() == []
    now[0] += 299
    assert health.stale_components() == []
    now[0] += 2
    assert health.stale_components() == ["email"]
    health.mark("email")
    assert health.stale_components() == []
    health.unregister("email")
    now[0] += 10_000
    assert health.stale_components() == []


def test_marking_an_unknown_component_does_nothing():
    health.mark("nobody")
    assert health.stale_components() == []


async def test_the_heartbeat_stops_while_a_component_is_stale_and_resumes_after_a_success(caplog):
    health.register("telegram", 0.15)
    component = asyncio.create_task(_running())
    beating = asyncio.create_task(health.heartbeat([component], interval=0.03))
    await asyncio.sleep(0.05)
    assert health.check(max_age=0.2)[0], "fresh at first"
    await asyncio.sleep(0.5)  # no success for longer than the component's max age
    assert not health.check(max_age=0.2)[0], "the file went stale"
    assert "no recent success from telegram" in caplog.text
    health.mark("telegram")
    await asyncio.sleep(0.1)
    assert health.check(max_age=0.2)[0], "a success brings the heartbeat back"
    beating.cancel()
    component.cancel()
    await asyncio.gather(beating, component, return_exceptions=True)


async def test_the_email_adapter_marks_only_after_a_completed_poll(monkeypatch):
    from app.channels import email as email_adapter

    calls = []

    async def poll():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("IMAP down")
        if len(calls) == 4:
            raise asyncio.CancelledError

    marks = []
    monkeypatch.setattr(email_adapter, "_poll_once", poll)
    monkeypatch.setattr(email_adapter, "POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(email_adapter.health, "mark", lambda name: marks.append(name))
    monkeypatch.setattr(
        email_adapter, "get_settings",
        lambda: type("S", (), {"email_trigger_tag": "[agent]", "email_agent_folder": ""})(),
    )
    with pytest.raises(asyncio.CancelledError):
        await email_adapter.run_email_adapter()
    assert calls == [1, 1, 1, 1]
    assert marks == ["email", "email"], "marked after polls 1 and 3, not after the failed 2"
    assert health.stale_components() == [] and "email" not in health._components


async def test_the_telegram_liveness_loop_marks_on_get_me_success_only(monkeypatch):
    from app.channels import telegram

    outcomes = [True, False, True]
    marks = []

    class Bot:
        async def get_me(self):
            ok = outcomes.pop(0)
            if not outcomes:
                raise asyncio.CancelledError
            if not ok:
                raise RuntimeError("network down")

    monkeypatch.setattr(telegram.health, "mark", lambda name: marks.append(name))
    with pytest.raises(asyncio.CancelledError):
        await telegram._liveness_loop(Bot(), interval=0)
    assert marks == ["telegram"], "one success marked, the failure did not"


def test_long_work_extends_the_limit_only_while_it_runs(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(health.time, "monotonic", lambda: now[0])
    health.register("email", 300)
    try:
        with health.working("email", 600):
            now[0] += 900
            assert health.stale_components() == [], "300 + 600 s while the turn runs"
            now[0] += 2
            assert health.stale_components() == ["email"], "and not one second more"
        health.mark("email")
        now[0] += 301
        assert health.stale_components() == ["email"], "back to 300 s after the turn"
        with health.working("nobody", 600):  # an unregistered name: nothing changes
            assert "nobody" not in health._components
    finally:
        health.unregister("email")


async def test_an_email_turn_runs_with_the_task_limit_after_the_mailbox_answered(monkeypatch):
    """The poll marks a success once the mailbox answered, and the turn that follows may
    wait for the engine up to LLM_TASK_TIMEOUT_SECONDS without turning the container
    unhealthy."""
    from app.channels import email as email_adapter
    from app.channels.dispatch import DispatchOutcome

    settings = type(
        "S",
        (),
        {"email_trigger_tag": "[agent]", "email_agent_folder": "", "llm_task_timeout_seconds": 600},
    )()
    message = email_adapter.TaggedMessage(b"7", "eve@example.org", "[agent] hi", "hello")
    log = []
    monkeypatch.setattr(email_adapter, "get_settings", lambda: settings)
    monkeypatch.setattr(email_adapter, "_fetch_tagged_unseen", lambda tag, folder: [message])
    monkeypatch.setattr(email_adapter, "_finalize_message", lambda uid, target: None)
    real_mark = health.mark
    monkeypatch.setattr(
        email_adapter.health, "mark", lambda name: (log.append("mark"), real_mark(name))
    )

    async def dispatch(session, event, **kwargs):
        log.append(("limit during the turn", health._components["email"][0]))
        return DispatchOutcome.OK

    monkeypatch.setattr(email_adapter, "dispatch_event", dispatch)
    health.register("email", 300)
    try:
        await email_adapter._poll_once()
        assert log == ["mark", ("limit during the turn", 300 + 3 * 600)]
        assert health._components["email"][0] == 300
    finally:
        health.unregister("email")
