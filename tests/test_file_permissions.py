"""Tests: secrets and data are readable by their owner only.

Everything the application creates is private (umask 077 at every entry
point, explicit modes for backups), files the owner already placed are only
reported, and start.sh creates .env as 600 without ever loosening it.
"""

import logging
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _shared(path: Path) -> bool:
    return bool(_mode(path) & 0o077)


# --- helpers in app.security.permissions ---


def test_harden_process_sets_umask_077():
    from app.security.permissions import harden_process

    before = os.umask(0o022)
    try:
        harden_process()
        current = os.umask(0o022)
        assert current == 0o077
    finally:
        os.umask(before)


def test_loose_files_reports_only_group_or_world_accessible_existing_files(tmp_path):
    from app.security.permissions import loose_files

    private = tmp_path / "private"
    private.write_text("x")
    private.chmod(0o600)
    group = tmp_path / "group"
    group.write_text("x")
    group.chmod(0o640)
    world = tmp_path / "world"
    world.write_text("x")
    world.chmod(0o644)
    result = loose_files([private, group, world, tmp_path / "missing"])
    assert result == [(group, 0o640), (world, 0o644)]


def test_one_warning_lists_every_loose_file_and_none_when_all_are_private(tmp_path, caplog):
    from app.security.permissions import warn_about_loose_files

    a = tmp_path / ".env"
    a.write_text("x")
    a.chmod(0o644)
    b = tmp_path / "app.db"
    b.write_text("x")
    b.chmod(0o664)
    with caplog.at_level(logging.WARNING, logger="channelagent"):
        assert warn_about_loose_files([a, b, tmp_path / "missing"]) == 2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    text = warnings[0].getMessage()
    assert ".env (644)" in text and "app.db (664)" in text and "chmod 600" in text
    assert "missing" not in text

    a.chmod(0o600)
    b.chmod(0o600)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="channelagent"):
        assert warn_about_loose_files([a, b]) == 0
    assert caplog.records == []
    assert _mode(a) == 0o600, "the check never changes a mode"


def test_the_warning_does_not_change_the_files_it_reports(tmp_path):
    from app.security.permissions import warn_about_loose_files

    f = tmp_path / ".env"
    f.write_text("x")
    f.chmod(0o644)
    warn_about_loose_files([f])
    assert _mode(f) == 0o644


# --- backups ---


def test_a_backup_and_its_directory_are_private_even_under_a_permissive_umask(tmp_path):
    import sqlite3

    from app.db.backup import make_backup

    db = tmp_path / "data" / "app.db"
    db.parent.mkdir()
    con = sqlite3.connect(db)
    con.execute("create table t (x)")
    con.commit()
    con.close()
    before = os.umask(0o022)
    try:
        backup = make_backup(db, "test")
    finally:
        os.umask(before)
    assert _mode(backup) == 0o600
    assert _mode(backup.parent) == 0o700


def test_an_existing_backup_directory_keeps_the_mode_its_owner_gave_it(tmp_path):
    import sqlite3

    from app.db.backup import make_backup

    db = tmp_path / "app.db"
    con = sqlite3.connect(db)
    con.execute("create table t (x)")
    con.commit()
    con.close()
    (tmp_path / "backups").mkdir()
    (tmp_path / "backups").chmod(0o755)
    backup = make_backup(db, "test")
    assert _mode(backup.parent) == 0o755
    assert _mode(backup) == 0o600


# --- the real entry points ---


# The console reaches the database through the API in process: it needs the key.
CONSOLE_KEY = "console-test-key-" + "x" * 20


def _env(tmp_path: Path, **extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABASE_", "CHECKPOINT_"))}
    env.update(
        PYTHONPATH=str(REPO),
        DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path}/newdata/channelagent.db",
        TELEGRAM_BOT_TOKEN="",
        EMAIL_IMAP_HOST="",
        API_SERVER_KEY="",
        MIGRATION_BACKUPS_KEEP="0",
    )
    env.update(extra)
    return env


def _files_under(root: Path):
    return [p for p in root.rglob("*") if p.is_file()]


def test_the_admin_console_creates_a_private_data_directory_and_database(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "app.admin.cli", "--transport", "inprocess"],
        input="q\n",
        cwd=tmp_path,
        env=_env(tmp_path, API_SERVER_KEY=CONSOLE_KEY),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    data = tmp_path / "newdata"
    assert _mode(data) == 0o700
    files = _files_under(data)
    assert any(p.name == "channelagent.db" for p in files)
    assert [p.name for p in files if _shared(p)] == []


def test_the_application_creates_private_database_and_checkpoint_files(tmp_path):
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.main"],
        cwd=tmp_path,
        env=_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 60
        seen = ""
        while time.time() < deadline and "No channel adapters" not in seen:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            seen += line
        assert "No channel adapters" in seen, seen
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    data = tmp_path / "newdata"
    names = {p.name for p in _files_under(data)}
    assert {"channelagent.db", "checkpoints.db"} <= names
    assert _mode(data) == 0o700
    assert [p.name for p in _files_under(data) if _shared(p)] == []


def test_the_key_rotation_command_hardens_the_process(monkeypatch):
    from app.admin import rekey

    calls = []
    monkeypatch.setattr(rekey, "harden_process", lambda: calls.append(1))
    monkeypatch.setenv("OLD_ENCRYPTION_KEY", "")
    monkeypatch.setenv("DATABASE_URL", "postgresql://nope")
    assert rekey.main(["--dry-run"]) == 1  # not SQLite: refused right after
    assert calls == [1]


def test_startup_warns_once_about_a_loose_env_file_and_database(tmp_path):
    (tmp_path / ".env").write_text("X=1\n")
    (tmp_path / ".env").chmod(0o644)
    data = tmp_path / "newdata"
    data.mkdir()
    db = data / "channelagent.db"
    import sqlite3

    sqlite3.connect(db).close()
    db.chmod(0o644)
    checkpoints = data / "checkpoints.db"
    checkpoints.write_text("")
    checkpoints.chmod(0o644)
    proc = subprocess.run(
        [sys.executable, "-m", "app.admin.cli", "--transport", "inprocess"],
        input="q\n",
        cwd=tmp_path,
        env=_env(tmp_path, ENCRYPTION_KEY=os.environ["ENCRYPTION_KEY"], API_SERVER_KEY=CONSOLE_KEY),
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = proc.stdout + proc.stderr
    lines = [ln for ln in output.splitlines() if "readable by other users" in ln.lower()]
    assert len(lines) == 1, output
    assert ".env (644)" in lines[0] and "channelagent.db (644)" in lines[0]
    assert "checkpoints.db (644)" in lines[0]
    assert _mode(tmp_path / ".env") == 0o644, "existing files are reported, not changed"


# --- start.sh ---


@pytest.fixture
def sandbox(tmp_path):
    shutil.copy(REPO / "start.sh", tmp_path / "start.sh")
    shutil.copy(REPO / ".env.example", tmp_path / ".env.example")
    (tmp_path / "app").mkdir()
    shutil.copy(REPO / "app" / "settings_rules.py", tmp_path / "app" / "settings_rules.py")
    return tmp_path


def _run(sandbox: Path, *args: str):
    return subprocess.run(
        ["bash", "start.sh", *args], cwd=sandbox, capture_output=True, text=True, timeout=60
    )


def test_start_script_creates_env_as_600_whatever_the_umask(sandbox):
    # Since only a start creates .env (without a terminal it copies .env.example);
    # the empty key then stops it, right after.
    before = os.umask(0o022)
    try:
        result = _run(sandbox, "--native")
    finally:
        os.umask(before)
    assert "ENCRYPTION_KEY is missing" in result.stderr, result.stderr
    assert _mode(sandbox / ".env") == 0o600


def test_set_keeps_the_mode_it_finds(sandbox):
    (sandbox / ".env").write_text((sandbox / ".env.example").read_text())
    (sandbox / ".env").chmod(0o600)
    result = _run(sandbox, "--config", "LLAMA_PORT=9999")
    assert result.returncode == 0, result.stderr
    assert _mode(sandbox / ".env") == 0o600
    (sandbox / ".env").chmod(0o640)
    _run(sandbox, "--config", "LLAMA_PORT=9998")
    assert _mode(sandbox / ".env") == 0o640, "--set does not loosen or tighten what it finds"
    assert "LLAMA_PORT=9998" in (sandbox / ".env").read_text()


def test_a_shared_env_is_reported_exactly_once_and_not_changed(sandbox):
    (sandbox / ".env").write_text((sandbox / ".env.example").read_text())
    (sandbox / ".env").chmod(0o644)
    for args in (("--config",), ("--config", "LLAMA_PORT=9997")):
        result = _run(sandbox, *args)
        assert result.returncode == 0, result.stderr
        lines = [ln for ln in result.stderr.splitlines() if "readable by other users" in ln]
        assert len(lines) == 1, result.stderr
        assert "chmod 600 .env" in lines[0]
        assert _mode(sandbox / ".env") == 0o644


def test_a_private_env_produces_no_warning(sandbox):
    (sandbox / ".env").write_text((sandbox / ".env.example").read_text())
    (sandbox / ".env").chmod(0o600)
    result = _run(sandbox, "--config")
    assert "readable by other users" not in result.stderr + result.stdout


def test_start_script_never_copies_the_example_without_a_private_umask():
    text = (REPO / "start.sh").read_text()
    copies = [ln for ln in text.splitlines() if "cp .env.example .env" in ln]
    assert len(copies) == 1, copies  # one helper, no scattered copies
    assert "umask 077" in copies[0]
