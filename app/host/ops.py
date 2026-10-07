"""What the host helper does: stop, start and restart the application, restore a
backup and rotate the key with the application stopped. Each one is a job.

There is one implementation of each operation, and it is `start.sh`: the helper runs it
with an argument list, never a shell string, in the project directory and with a minimal
environment (the script reads `.env` itself). START_SH_LOCAL=1 tells the script to act on
this machine even when API_URL names an API elsewhere, so a call can never loop back.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
from pathlib import Path

from app import settings_rules as rules
from app.admin.jobs import Job, JobError
from app.admin.service import ConflictError, InvalidInputError
from app.logging_setup import scrub

PROJECT_DIR = Path(__file__).resolve().parents[2]
RUN_MODE_FILE = ".run-mode"
# The API answers its caller before the helper stops the application under it.
STOP_GRACE_SECONDS = 1.0
SCRIPT_TIMEOUT_SECONDS = 900
START_TIMEOUT_SECONDS = 300
POLL_SECONDS = 1.0
OUTPUT_LINES_KEPT = 15


def env_values() -> dict:
    return rules.read_env(str(PROJECT_DIR / ".env"))


def database_path() -> Path | None:
    from app.db.session import sqlite_file_path

    url = env_values().get("DATABASE_URL") or "sqlite+aiosqlite:///./data/channelagent.db"
    path = sqlite_file_path(url)
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_DIR / path


def run_mode() -> str:
    """The mode start.sh last started the application in: native or docker."""
    try:
        mode = (PROJECT_DIR / RUN_MODE_FILE).read_text().strip()
    except OSError:
        return "native"
    return mode if mode in ("native", "docker") else "native"


def _api_answers() -> bool:
    values = env_values()
    host = values.get("API_SERVER_HOST") or "127.0.0.1"
    host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    port = int(values.get("API_SERVER_PORT") or 8700)
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _native_pid() -> int | None:
    try:
        pid = int((PROJECT_DIR / ".app.pid").read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        command = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat=,command="],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not command or command.startswith("Z") or "app.main" not in command:
        return None
    return pid


def _container_running() -> bool:
    try:
        out = subprocess.run(
            ["docker", "compose", "ps", "--status", "running", "-q", "channelagent"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(out)


def app_state() -> dict:
    """Whether the application runs, and what shows it."""
    reasons = []
    pid = _native_pid()
    if pid is not None:
        reasons.append(f"native process {pid}")
    if _container_running():
        reasons.append("container running")
    if _api_answers():
        reasons.append("Admin API answering")
    return {"mode": run_mode(), "running": bool(reasons), "reasons": reasons}


async def run_script(*args: str, timeout: float = SCRIPT_TIMEOUT_SECONDS) -> tuple[int, str]:
    """Run `start.sh ARGS` and return (exit code, the end of its output, secrets removed)."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", str(PROJECT_DIR)),
        "LANG": os.environ.get("LANG", "C"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "START_SH_LOCAL": "1",
    }
    process = await asyncio.create_subprocess_exec(
        "bash",
        str(PROJECT_DIR / "start.sh"),
        *args,
        cwd=PROJECT_DIR,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,  # a detached application must not share the helper's group
    )
    try:
        out, _ = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise JobError(f"start.sh {args[0]} did not finish within {int(timeout)} s") from None
    lines = scrub(out.decode(errors="replace")).strip().splitlines()
    return process.returncode, "\n".join(lines[-OUTPUT_LINES_KEPT:])


async def _state() -> dict:
    return await asyncio.to_thread(app_state)


async def _stop(job: Job, progress: float) -> None:
    job.update(progress, "Stopping the application")
    await asyncio.sleep(STOP_GRACE_SECONDS)
    code, out = await run_script("--stop")
    if (await _state())["running"]:
        raise JobError(f"The application is still running after start.sh --stop (exit {code})")


async def _start(job: Job, progress: float) -> None:
    mode = run_mode()
    job.update(progress, f"Starting the application ({mode})")
    code, out = await run_script(f"--{mode}", "--detach")
    if code != 0:
        raise JobError(f"start.sh --{mode} --detach failed (exit {code}): {_last_line(out)}")
    waited = 0.0
    while waited < START_TIMEOUT_SECONDS:
        no_api = not env_values().get("API_SERVER_KEY")
        if _api_answers() or (no_api and (await _state())["running"]):
            return
        await asyncio.sleep(POLL_SECONDS)
        waited += POLL_SECONDS
    raise JobError(f"The application did not answer within {START_TIMEOUT_SECONDS} s")


def _last_line(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1][:300] if lines else "no output"


async def stop(job: Job) -> dict:
    if not (await _state())["running"]:
        raise JobError("The application is not running")
    await _stop(job, 0.2)
    return await _state()


async def start(job: Job) -> dict:
    await _start(job, 0.2)
    return await _state()


async def restart(job: Job) -> dict:
    if (await _state())["running"]:
        await _stop(job, 0.1)
    await _start(job, 0.5)
    return await _state()


def check_not_running() -> None:
    """Refuse `start` when the application already runs (before any job is created)."""
    if app_state()["running"]:
        raise ConflictError("The application is already running")


def backup_file(name: str) -> Path:
    """The backup called `name`, which must be one of the listed backups: a name, never a path."""
    from app.admin.restore import list_backups

    database = database_path()
    if database is None:
        raise ConflictError("Backups are only available for a SQLite database")
    for backup in list_backups(database):
        if backup.path.name == name:
            return backup.path
    raise InvalidInputError(f"No backup named {name!r} (GET /backups lists them)")


def _rows(path: Path) -> dict:
    from app.admin.restore import row_counts

    return row_counts(path) if path.exists() else {}


async def _around_stopped(job: Job, work) -> dict:
    """Stop the application if it runs, do `work`, start it again if it was running. When
    `work` says the data may be unsafe (it returns restart=False), it stays stopped."""
    was_running = (await _state())["running"]
    if was_running:
        await _stop(job, 0.1)
    result, restart_after = await work()
    if was_running and restart_after:
        await _start(job, 0.8)
    result["was_running"] = was_running
    result["restarted"] = was_running and restart_after
    return result


async def restore(job: Job, name: str, allow_unreadable: bool) -> dict:
    backup = backup_file(name)
    database = database_path()

    async def work():
        job.update(0.4, f"Restoring {name}")
        args = ["--restore", name, "--yes"] + (["--allow-unreadable"] if allow_unreadable else [])
        code, out = await run_script(*args)
        backup_rows = await asyncio.to_thread(_rows, backup)
        after = await asyncio.to_thread(_rows, database)
        result = {
            "backup": name,
            "exit_code": code,
            "backup_rows": backup_rows,
            "database_rows_after": after,
            "rows_equal": code == 0 and backup_rows == after,
            "output": out,
        }
        if code != 0:
            result["error"] = _last_line(out)
        return result, True

    result = await _around_stopped(job, work)
    if result["exit_code"] != 0:
        raise JobError(f"Restore refused: {result['error']}")
    return result


async def rekey(job: Job, allow_unreadable: bool, dry_run: bool) -> dict:
    database = database_path()

    async def work():
        job.update(0.4, "Dry run of the key rotation" if dry_run else "Rotating the key")
        before = await asyncio.to_thread(_rows, database)
        args = ["--rekey", "--yes"]
        args += ["--allow-unreadable"] if allow_unreadable else []
        args += ["--dry-run"] if dry_run else []
        code, out = await run_script(*args)
        after = await asyncio.to_thread(_rows, database)
        result = {
            "dry_run": dry_run,
            "exit_code": code,
            "database_rows_before": before,
            "database_rows_after": after,
            "rows_equal": before == after,
            "output": out,
        }
        if code != 0:
            result["error"] = _last_line(out)
        # Exit 3: the rotation failed half-way or the read-back found unreadable values; the
        # guided command says not to start the application, so it stays stopped.
        return result, code != 3

    result = await _around_stopped(job, work)
    if result["exit_code"] != 0:
        still = "" if result["restarted"] or not result["was_running"] else " (left stopped)"
        raise JobError(f"Rekey failed{still}: {result['error']}")
    return result
