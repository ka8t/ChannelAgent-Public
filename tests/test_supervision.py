"""Tests: one component that cannot run (a Telegram token that is
rejected, an API port already in use, an unusable email setting) is logged and
stopped, and the others keep running. The process ends, non-zero, only when
every component has stopped.
"""

import asyncio
import contextlib
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parent.parent
TOKEN = "123456789:AAHfakeFAKEfake-fake_fakeFAKEfakeFAKEfa"
API_KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"


class Running:
    """A fake component that records that it started and stays up until cancelled."""

    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = False

    async def __call__(self, *_a, **_k):
        self.started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.fixture
def boot(fresh_db, monkeypatch, caplog):
    """Configure the environment for app.main.main() and return a helper."""
    from app.config import get_settings

    def configure(*, telegram=False, email=False, api=False):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN if telegram else "")
        monkeypatch.setenv("EMAIL_IMAP_HOST", "imap.example.org" if email else "")
        monkeypatch.setenv("EMAIL_USERNAME", "u@example.org" if email else "")
        monkeypatch.setenv("EMAIL_PASSWORD", "Test-Only-Passw0rd-Value" if email else "")
        monkeypatch.setenv("API_SERVER_KEY", API_KEY if api else "")
        get_settings.cache_clear()

    caplog.set_level(logging.INFO, logger="channelagent")
    return configure


async def _stop(task):
    """Cancel a main() task and wait for it, so nothing leaks into the next test."""
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _wait(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        await asyncio.sleep(0.05)
    return False


def _errors(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


def _fake_api(monkeypatch, behaviour):
    """Replace uvicorn's Config and Server: `behaviour` is the coroutine `serve` runs."""
    import uvicorn

    class Config:
        def __init__(self, app, **kwargs):
            pass

    class Server:
        def __init__(self, config):
            pass

        async def serve(self):
            await behaviour()

    monkeypatch.setattr(uvicorn, "Config", Config)
    monkeypatch.setattr(uvicorn, "Server", Server)


# --- one failing component does not stop the others ---


async def test_a_failing_telegram_adapter_does_not_stop_the_email_adapter(
    boot, monkeypatch, caplog
):
    from app import main as app_main

    boot(telegram=True, email=True)
    email = Running()

    async def telegram_fails():
        raise RuntimeError(f"The token `{TOKEN}` was rejected by the server.")

    monkeypatch.setattr("app.channels.telegram.run_telegram_adapter", telegram_fails)
    monkeypatch.setattr("app.channels.email.run_email_adapter", email)
    task = asyncio.create_task(app_main.main())
    assert await _wait(email.started.is_set), "the email adapter started"
    assert await _wait(lambda: _errors(caplog)), "the failure was logged"
    await asyncio.sleep(0.3)
    assert not task.done(), "the process is still up"
    assert not email.cancelled
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert email.cancelled, "a shutdown still stops the surviving component"


async def test_the_error_names_the_adapter_and_the_fix_and_never_the_token(
    boot, monkeypatch, caplog
):
    from app import main as app_main

    boot(telegram=True, email=True)

    async def telegram_fails():
        raise RuntimeError(f"The token `{TOKEN}` was rejected by the server.")

    monkeypatch.setattr("app.channels.telegram.run_telegram_adapter", telegram_fails)
    monkeypatch.setattr("app.channels.email.run_email_adapter", Running())
    task = asyncio.create_task(app_main.main())
    assert await _wait(lambda: _errors(caplog))
    await _stop(task)
    text = "\n".join(r.getMessage() + (r.exc_text or "") for r in _errors(caplog))
    assert "Telegram adapter" in text and "TELEGRAM_BOT_TOKEN" in text
    assert "RuntimeError" in text, "the traceback is kept for whoever investigates"
    assert TOKEN not in text and "AAHfake" not in text
    assert "keep running" in text or "other components" in text


async def test_a_shutdown_is_not_logged_as_a_failure(boot, monkeypatch, caplog):
    from app import main as app_main

    boot(telegram=True, email=True)
    telegram, email = Running(), Running()
    monkeypatch.setattr("app.channels.telegram.run_telegram_adapter", telegram)
    monkeypatch.setattr("app.channels.email.run_email_adapter", email)
    task = asyncio.create_task(app_main.main())
    assert await _wait(lambda: telegram.started.is_set() and email.started.is_set())
    await _stop(task)
    assert telegram.cancelled and email.cancelled
    assert _errors(caplog) == []


async def test_a_shutdown_waits_for_every_component_to_finish_its_cleanup(boot, monkeypatch):
    """Ctrl+C: gather gave up as soon as the API ended cancelled, and asyncio.run then
    cancelled the Telegram adapter in the middle of stopping its poller."""
    from app import main as app_main

    boot(telegram=True, api=True)
    api = Running()
    cleanup = {"started": False, "finished": False}

    async def telegram_with_slow_cleanup():
        try:
            await asyncio.sleep(3600)
        finally:
            cleanup["started"] = True
            await asyncio.sleep(0.3)  # stopping the poller, one last getUpdates
            cleanup["finished"] = True

    _fake_api(monkeypatch, api)
    monkeypatch.setattr("app.channels.telegram.run_telegram_adapter", telegram_with_slow_cleanup)
    task = asyncio.create_task(app_main.main())
    assert await _wait(api.started.is_set)
    await _stop(task)
    assert cleanup["started"] and cleanup["finished"], "main() ended before Telegram stopped"


async def test_a_shutdown_closes_the_open_mcp_connections(boot, monkeypatch):
    from app import main as app_main
    from app.mcp.manager import manager

    boot(telegram=True)
    closed = []

    async def reset():
        closed.append(True)

    monkeypatch.setattr(manager, "reset", reset)
    telegram = Running()
    monkeypatch.setattr("app.channels.telegram.run_telegram_adapter", telegram)
    task = asyncio.create_task(app_main.main())
    assert await _wait(telegram.started.is_set)
    await _stop(task)
    assert closed == [True]


async def test_the_backup_scheduler_runs_beside_the_components_and_stops_with_them(
    boot, monkeypatch
):
    """Started with the application, stopped at shutdown, not counted as a component."""
    from app import main as app_main

    boot(telegram=True)
    scheduler, telegram = Running(), Running()
    monkeypatch.setattr("app.admin.backup_schedule.run_scheduler", scheduler)
    monkeypatch.setattr("app.channels.telegram.run_telegram_adapter", telegram)
    task = asyncio.create_task(app_main.main())
    assert await _wait(lambda: scheduler.started.is_set() and telegram.started.is_set())
    await _stop(task)
    assert scheduler.cancelled


async def test_the_failure_is_logged_once(boot, monkeypatch, caplog):
    from app import main as app_main

    boot(telegram=True, email=True)

    async def telegram_fails():
        raise RuntimeError("boom")

    monkeypatch.setattr("app.channels.telegram.run_telegram_adapter", telegram_fails)
    monkeypatch.setattr("app.channels.email.run_email_adapter", Running())
    task = asyncio.create_task(app_main.main())
    assert await _wait(lambda: _errors(caplog))
    await asyncio.sleep(0.5)
    await _stop(task)
    assert len([r for r in _errors(caplog) if "Telegram adapter" in r.getMessage()]) == 1


async def test_a_failing_telegram_adapter_does_not_stop_the_admin_api(boot, monkeypatch, caplog):
    from app import main as app_main

    boot(telegram=True, api=True)
    api = Running()

    async def telegram_fails():
        raise RuntimeError("boom")

    _fake_api(monkeypatch, api)
    monkeypatch.setattr("app.channels.telegram.run_telegram_adapter", telegram_fails)
    task = asyncio.create_task(app_main.main())
    assert await _wait(api.started.is_set)
    assert await _wait(lambda: _errors(caplog))
    await asyncio.sleep(0.3)
    assert not task.done() and not api.cancelled
    await _stop(task)


async def test_a_failing_admin_api_does_not_stop_the_adapters(boot, monkeypatch, caplog):
    """uvicorn ends with SystemExit when its port is taken."""
    from app import main as app_main

    boot(telegram=True, api=True)
    telegram = Running()

    async def api_cannot_bind():
        raise SystemExit(1)

    _fake_api(monkeypatch, api_cannot_bind)
    monkeypatch.setattr("app.channels.telegram.run_telegram_adapter", telegram)
    task = asyncio.create_task(app_main.main())
    assert await _wait(telegram.started.is_set)
    assert await _wait(lambda: _errors(caplog))
    await asyncio.sleep(0.3)
    assert not task.done() and not telegram.cancelled
    text = " ".join(r.getMessage() for r in _errors(caplog))
    assert "Admin API" in text and "API_SERVER_PORT" in text
    await _stop(task)


# --- the process ends, non-zero, only when everything has stopped ---


async def test_when_every_component_has_failed_the_process_exits_non_zero(
    boot, monkeypatch, caplog
):
    from app import main as app_main

    boot(telegram=True, email=True)

    async def fails():
        raise RuntimeError("boom")

    monkeypatch.setattr("app.channels.telegram.run_telegram_adapter", fails)
    monkeypatch.setattr("app.channels.email.run_email_adapter", fails)
    code = await asyncio.wait_for(app_main.main(), timeout=10)
    assert code == 1
    assert any("Every component has stopped" in r.getMessage() for r in _errors(caplog))
    assert len([r for r in _errors(caplog) if "adapter" in r.getMessage().lower()]) >= 2


async def test_a_component_that_simply_returns_counts_as_stopped(boot, monkeypatch, caplog):
    """The email adapter returns, logging an error, when its tag is unusable."""
    from app import main as app_main

    boot(email=True)

    async def returns():
        return None

    monkeypatch.setattr("app.channels.email.run_email_adapter", returns)
    code = await asyncio.wait_for(app_main.main(), timeout=10)
    assert code == 1


async def test_with_no_component_the_process_keeps_running(boot, caplog):
    from app import main as app_main

    boot()
    task = asyncio.create_task(app_main.main())
    await asyncio.sleep(0.5)
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_healthy_component_alone_keeps_the_process_running(boot, monkeypatch):
    from app import main as app_main

    boot(telegram=True)
    telegram = Running()
    monkeypatch.setattr("app.channels.telegram.run_telegram_adapter", telegram)
    task = asyncio.create_task(app_main.main())
    assert await _wait(telegram.started.is_set)
    await asyncio.sleep(0.3)
    assert not task.done()
    await _stop(task)


# --- the real process ---


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _env(tmp_path, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABASE_", "CHECKPOINT_"))}
    env.update(
        PYTHONPATH=str(REPO),
        DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path}/data/channelagent.db",
        TELEGRAM_BOT_TOKEN=TOKEN,
        EMAIL_IMAP_HOST="",
        API_SERVER_KEY=API_KEY,
        MIGRATION_BACKUPS_KEEP="0",
    )
    env.update(extra)
    return env


def test_the_real_process_keeps_the_api_up_when_telegram_rejects_its_token(tmp_path):
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.main"],
        cwd=tmp_path,
        env=_env(tmp_path, API_SERVER_PORT=str(port)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        url = f"http://127.0.0.1:{port}/users"
        deadline = time.time() + 40
        answered = None
        while time.time() < deadline and answered is None:
            try:
                answered = httpx.get(url, timeout=2).status_code
            except httpx.HTTPError:
                time.sleep(0.5)
        assert answered == 401, "the API answers"
        time.sleep(12)  # long past the moment Telegram refuses (or cannot be reached)
        assert proc.poll() is None, "the process is still running"
        assert httpx.get(url, timeout=2).status_code == 401
        good = httpx.get(url, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=2)
        assert good.status_code == 200
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            output, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            output, _ = proc.communicate()
    assert "Telegram adapter" in output and "ERROR" in output, output[-800:]
    assert TOKEN not in output and "AAHfake" not in output


def test_ctrl_c_stops_the_real_process_without_a_traceback(tmp_path):
    """asyncio.run re-raises KeyboardInterrupt after main() has stopped everything:
    it printed a traceback after a clean stop (2026-09-28)."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.main"],
        cwd=tmp_path,
        env=_env(tmp_path, TELEGRAM_BOT_TOKEN="", API_SERVER_PORT=str(port)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 40
        while time.time() < deadline:
            try:
                httpx.get(f"http://127.0.0.1:{port}/users", timeout=2)
                break
            except httpx.HTTPError:
                time.sleep(0.5)
        proc.send_signal(signal.SIGINT)
        output, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert "Application shutdown complete" in output, output[-800:]
    assert "Traceback" not in output and "KeyboardInterrupt" not in output, output[-1500:]
    assert proc.returncode == 130


def test_the_real_process_exits_non_zero_when_its_only_component_fails(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "app.main"],
        cwd=tmp_path,
        env=_env(tmp_path, API_SERVER_KEY="", API_SERVER_PORT=str(_free_port())),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 1, proc.stderr[-800:]
    output = proc.stdout + proc.stderr
    assert "Every component has stopped" in output
    assert TOKEN not in output and "AAHfake" not in output
