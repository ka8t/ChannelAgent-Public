"""Tests: restoring a database backup is one guided, checked command.

Real SQLite files, real encryption, a real listening socket for the "the
application is running" refusal, and the real command line in a subprocess.
"""

import hashlib
import os
import socket
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from app.admin import restore as rs
from app.db.backup import make_backup

REPO = Path(__file__).resolve().parent.parent
KEY = os.environ["ENCRYPTION_KEY"]


@pytest.fixture(autouse=True)
def no_docker_on_this_machine(monkeypatch):
    """The refusal looks at running `channelagent` containers: a real one on the machine
    running the tests (the owner's live application) must not change their result.
    """
    monkeypatch.setattr(rs.shutil, "which", lambda _name: None)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sql(path: Path, query: str, *args):
    con = sqlite3.connect(path)
    try:
        rows = con.execute(query, args).fetchall()
        con.commit()
        return rows
    finally:
        con.close()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _users(path: Path) -> int:
    return _sql(path, "select count(*) from users")[0][0]


@pytest.fixture
async def world(fresh_db):
    """A migrated database with 2 users, 1 identity with an encrypted address, 1
    action log; and a backup of it taken with the real backup function.
    """
    from app.admin import service
    from app.config import get_settings
    from app.db.models import Channel, Direction
    from app.db.session import init_db, session_scope, sqlite_file_path

    await init_db()
    async with session_scope() as s:
        a = await service.create_user(s, "Alice", actor="t")
        await service.create_user(s, "Bob", actor="t")
        await service.add_channel_identity(s, a.id, Channel.EMAIL, "alice@example.org", actor="t")
        agent = await service.create_agent(s, a.id, "default", actor="t")
        await service.record_action(
            s,
            user_id=a.id,
            agent_id=agent.id,
            channel=Channel.EMAIL,
            direction=Direction.INBOUND,
            text="hello",
        )
        await s.commit()
    db = sqlite_file_path(get_settings().database_url)
    revision = _sql(db, "select version_num from alembic_version")[0][0]
    backup = make_backup(db, revision)
    return db, backup


def _restore(db, backup, **kw):
    lines: list[str] = []
    kw.setdefault("encryption_key", KEY)
    kw.setdefault("api_port", _free_port())
    kw.setdefault("yes", True)
    report = rs.run_restore(db, backup, out=lines.append, **kw)
    return report, lines


# --- the round trip ---


def test_restore_puts_back_the_backups_data_and_keeps_the_current_database(world):
    db, backup = world
    _sql(
        db, "insert into users (display_name, is_active, created_at) values ('C', 1, '2026-01-01')"
    )
    _sql(
        db, "insert into users (display_name, is_active, created_at) values ('D', 1, '2026-01-01')"
    )
    _sql(
        db, "insert into users (display_name, is_active, created_at) values ('E', 1, '2026-01-01')"
    )
    assert _users(db) == 5 and _users(backup) == 2

    report, lines = _restore(db, backup)

    assert _users(db) == 2, "the live database now holds the backup's rows"
    assert report.before_restore_copy is not None
    assert report.before_restore_copy.name.startswith(db.stem + "-before-restore-")
    assert _users(report.before_restore_copy) == 5, "the changed data is kept in the copy"
    assert report.counts_before["users"] == 5 and report.counts_after["users"] == 2
    assert _sql(db, "pragma integrity_check") == [("ok",)]
    assert _sql(db, "select version_num from alembic_version") == _sql(
        backup, "select version_num from alembic_version"
    )
    assert any("integrity_check ok" in ln for ln in lines)


def test_restored_and_saved_files_are_private(world):
    db, backup = world
    os.chmod(db, 0o644)
    report, _ = _restore(db, backup)
    assert _mode(db) == 0o600
    assert _mode(report.before_restore_copy) == 0o600


async def test_the_restored_data_is_readable_with_the_current_key(world):
    from app.admin import service
    from app.db.session import get_engine, get_sessionmaker, session_scope

    db, backup = world
    _sql(db, "delete from action_logs")
    _restore(db, backup)
    # The application is stopped during a restore; this process still holds a
    # connection to the replaced file, so start from a fresh engine.
    await get_engine().dispose()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    async with session_scope() as s:
        logs = await service.search_action_logs(s, keyword="hello", actor="t")
    assert [x.text for x in logs] == ["hello"]


def test_the_backup_file_itself_is_untouched(world):
    db, backup = world
    before = _sha(backup)
    _restore(db, backup)
    assert _sha(backup) == before


def test_the_write_ahead_log_of_the_old_database_does_not_survive(world):
    """A real WAL: journal_mode=wal, a write that is not checkpointed, and a
    connection still open (a crashed process leaves the same two files).
    """
    db, backup = world
    holder = sqlite3.connect(db, isolation_level=None)
    try:
        holder.execute("pragma journal_mode=wal")
        holder.execute("pragma wal_autocheckpoint=0")
        holder.execute(
            "insert into users (display_name, is_active, created_at) values ('C', 1, '2026-01-01')"
        )
        wal = db.with_name(db.name + "-wal")
        shm = db.with_name(db.name + "-shm")
        assert wal.exists() and wal.stat().st_size > 0 and shm.exists(), "the leftovers exist"
        report, _ = _restore(db, backup)
        assert not wal.exists() and not shm.exists()
        assert _users(report.before_restore_copy) == 3, "the WAL data went into the copy"
        assert _sql(db, "pragma integrity_check") == [("ok",)] and _users(db) == 2
    finally:
        holder.close()
    assert _users(db) == 2


def test_a_failure_after_staging_leaves_the_database_alone_and_no_staging_file(world, monkeypatch):
    db, backup = world
    sha = _sha(db)
    real = rs.row_counts

    def lying(path):
        counts = real(path)
        return {**counts, "users": 999} if str(path).endswith(".restoring") else counts

    monkeypatch.setattr(rs, "row_counts", lying)
    with pytest.raises(rs.RestoreError, match="differs from the backup"):
        _restore(db, backup)
    assert _sha(db) == sha, "the live database was not replaced"
    assert not db.with_name(db.name + ".restoring").exists()
    assert len(list(db.parent.glob("backups/*before-restore*"))) == 1, "the copy was made first"


def test_no_staging_file_is_left_behind(world):
    db, backup = world
    _restore(db, backup)
    assert not db.with_name(db.name + ".restoring").exists()


def test_the_conversation_checkpoints_are_not_touched_and_the_output_says_so(world, tmp_path):
    db, backup = world
    checkpoints = db.parent / "checkpoints.db"
    checkpoints.write_bytes(b"conversation history")
    before = _sha(checkpoints)
    _, lines = _restore(db, backup)
    assert _sha(checkpoints) == before
    assert any("checkpoints.db" in ln and "NOT restored" in ln for ln in lines)


def test_restore_into_a_directory_with_no_database_yet(world, tmp_path):
    _, backup = world
    fresh = tmp_path / "elsewhere" / backup.name.split("-")[0]
    target = tmp_path / "elsewhere" / "channelagent.db"
    report, _ = _restore(target, backup)
    assert report.before_restore_copy is None
    assert _users(target) == 2 and _mode(target) == 0o600
    assert not fresh.exists()


# --- refusals: nothing changes ---


def _assert_untouched(db: Path, sha: str):
    assert _sha(db) == sha
    assert not list(db.parent.glob("backups/*before-restore*"))
    assert not db.with_name(db.name + ".restoring").exists()


def test_refused_while_the_admin_api_answers(world):
    db, backup = world
    _sql(
        db, "insert into users (display_name, is_active, created_at) values ('C', 1, '2026-01-01')"
    )
    sha = _sha(db)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        with pytest.raises(rs.RestoreError, match="running.*Admin API"):
            _restore(db, backup, api_port=port)
    _assert_untouched(db, sha)


def test_refused_while_the_database_is_locked_by_another_process(world):
    db, backup = world
    sha = _sha(db)
    holder = sqlite3.connect(db, isolation_level=None)
    holder.execute("begin exclusive")
    try:
        with pytest.raises(rs.RestoreError, match="locked"):
            _restore(db, backup)
    finally:
        holder.execute("rollback")
        holder.close()
    _assert_untouched(db, sha)


def test_refused_for_a_file_that_is_not_sqlite(world, tmp_path):
    db, _ = world
    sha = _sha(db)
    fake = tmp_path / "fake.db"
    fake.write_text("this is not a database at all, just text " * 20)
    with pytest.raises(rs.RestoreError, match="not a readable SQLite"):
        _restore(db, fake)
    _assert_untouched(db, sha)


def test_refused_for_a_corrupt_backup(world, tmp_path):
    db, backup = world
    sha = _sha(db)
    broken = tmp_path / "broken.db"
    data = backup.read_bytes()
    broken.write_bytes(data[: len(data) // 2] + b"\x00" * (len(data) // 2))
    with pytest.raises(rs.RestoreError):
        _restore(db, broken)
    _assert_untouched(db, sha)


def test_refused_for_a_backup_that_opens_but_fails_the_integrity_check(world, tmp_path):
    """An index that disagrees with its table: every query works, integrity_check does not."""
    db, backup = world
    sha = _sha(db)
    ext = _sql(backup, "select external_id from channel_identities")[0][0].encode()
    data = bytearray(backup.read_bytes())
    at = data.find(ext)
    assert at > 0
    data[at + 5] = ord("0") if data[at + 5] != ord("0") else ord("1")  # one occurrence only
    damaged = tmp_path / "damaged.db"
    damaged.write_bytes(bytes(data))
    con = sqlite3.connect(damaged)
    assert con.execute("pragma integrity_check").fetchall() != [("ok",)]
    assert con.execute("select count(*) from users").fetchone()[0] == 2, "it still opens"
    con.close()
    with pytest.raises(rs.RestoreError, match="integrity"):
        _restore(db, damaged)
    _assert_untouched(db, sha)


def test_refused_for_a_sqlite_file_that_is_not_a_channelagent_database(world, tmp_path):
    db, _ = world
    sha = _sha(db)
    other = tmp_path / "other.db"
    _sql(other, "create table things (x)")
    with pytest.raises(rs.RestoreError, match="not a ChannelAgent database"):
        _restore(db, other)
    _assert_untouched(db, sha)


def test_refused_for_a_schema_revision_this_code_does_not_know(world, tmp_path):
    db, backup = world
    sha = _sha(db)
    newer = tmp_path / "newer.db"
    newer.write_bytes(backup.read_bytes())
    _sql(newer, "update alembic_version set version_num = 'ffffffffffff'")
    with pytest.raises(rs.RestoreError, match="does not know"):
        _restore(db, newer)
    _assert_untouched(db, sha)


def test_refused_when_the_backup_is_the_live_database(world):
    db, _ = world
    with pytest.raises(rs.RestoreError, match="live database itself"):
        _restore(db, db)


def test_refused_for_a_missing_file(world, tmp_path):
    db, _ = world
    with pytest.raises(rs.RestoreError, match="not a file"):
        _restore(db, tmp_path / "nope.db")


class _Docker:
    def __init__(self, names: list[str]):
        self.stdout = "\n".join(names) + "\n"


def _fake_docker(monkeypatch, names: list[str]):
    monkeypatch.setattr(rs.shutil, "which", lambda _name: "/usr/local/bin/docker")
    calls = []

    def run(cmd, **_kw):
        calls.append(cmd)
        return _Docker(names)

    monkeypatch.setattr(rs.subprocess, "run", run)
    return calls


def test_refused_while_a_channelagent_container_is_running(world, monkeypatch):
    db, backup = world
    sha = _sha(db)
    calls = _fake_docker(monkeypatch, ["channelagent-channelagent-1", "channelagent-llama-1"])
    with pytest.raises(rs.RestoreError, match="container is running.*channelagent-channelagent-1"):
        _restore(db, backup)
    assert calls and calls[0][:2] == ["docker", "ps"]
    _assert_untouched(db, sha)


def test_a_running_llama_server_container_alone_does_not_block_a_restore(world, monkeypatch):
    db, backup = world
    _fake_docker(monkeypatch, ["channelagent-llama-server-1"])
    report, _ = _restore(db, backup)
    assert _users(db) == 2 and report.before_restore_copy is not None


def test_no_container_and_a_failing_docker_command_do_not_block_a_restore(world, monkeypatch):
    db, backup = world
    _fake_docker(monkeypatch, [])
    _restore(db, backup)

    def broken(cmd, **_kw):
        raise OSError("docker is not reachable")

    monkeypatch.setattr(rs.subprocess, "run", broken)
    _restore(db, backup)


# --- an older schema is accepted, with a note ---


def test_a_backup_behind_the_current_schema_is_restored_with_a_note(world, tmp_path):
    db, backup = world
    old = tmp_path / "old.db"
    old.write_bytes(backup.read_bytes())
    # Same file, but it says it is one revision behind head (a known revision).
    from app.admin.restore import _known_revisions

    head, known = _known_revisions()
    behind = sorted(known - {head})[0]
    _sql(old, "update alembic_version set version_num = ?", behind)
    report, lines = _restore(db, old)
    assert report.needs_migration is True
    assert any("behind the current schema" in ln for ln in lines)


# --- the key ---


def test_a_backup_made_under_another_key_reports_its_unreadable_count_and_is_refused(
    world, monkeypatch
):
    db, backup = world
    encrypted = (
        _sql(backup, "select count(*) from action_logs")[0][0]
        + _sql(backup, "select count(*) from channel_identities where raw_address is not null")[0][
            0
        ]
        + _sql(backup, "select count(*) from admin_events where details is not null")[0][0]
    )
    assert encrypted >= 3
    other_key = Fernet.generate_key().decode()
    sha = _sha(db)
    with pytest.raises(rs.RestoreError, match="another key|OLD key") as exc:
        _restore(db, backup, encryption_key=other_key)
    assert "--allow-unreadable" in str(exc.value)
    assert (
        rs.unreadable_values(backup, other_key)
        and sum(rs.unreadable_values(backup, other_key).values()) == encrypted
    )
    _assert_untouched(db, sha)


def test_allow_unreadable_restores_anyway_and_still_reports_the_count(world):
    db, backup = world
    _sql(
        db, "insert into users (display_name, is_active, created_at) values ('C', 1, '2026-01-01')"
    )
    other_key = Fernet.generate_key().decode()
    report, lines = _restore(db, backup, encryption_key=other_key, allow_unreadable=True)
    assert report.unreadable_total >= 3
    assert any("WARNING" in ln and "cannot be decrypted" in ln for ln in lines)
    assert _users(db) == 2


def test_a_backup_readable_with_the_current_key_reports_nothing_unreadable(world):
    db, backup = world
    report, lines = _restore(db, backup)
    assert report.unreadable == {} and not any("WARNING" in ln for ln in lines)


# --- confirmation ---


def test_without_yes_the_word_restore_is_required(world):
    db, backup = world
    _sql(
        db, "insert into users (display_name, is_active, created_at) values ('C', 1, '2026-01-01')"
    )
    sha = _sha(db)
    with pytest.raises(rs.RestoreError, match="cancelled"):
        _restore(db, backup, yes=False, confirm=lambda _prompt: "yes")
    _assert_untouched(db, sha)
    _restore(db, backup, yes=False, confirm=lambda _prompt: "RESTORE")
    assert _users(db) == 2


# --- listing ---


def test_listing_shows_kind_revision_and_stamp_newest_first_and_skips_other_databases(world):
    db, backup = world
    d = db.parent / "backups"
    (d / f"{db.stem}-prerekey-20260101T000000000000Z.db").write_bytes(b"x")
    (d / f"{db.stem}-before-restore-20260102T000000000000Z.db").write_bytes(b"x")
    (d / "checkpoints-prerekey-20260103T000000000000Z.db").write_bytes(b"x")
    (d / f"{db.stem}-notes.db").write_bytes(b"x")
    infos = rs.list_backups(db)
    kinds = {i.path.name: i.kind for i in infos}
    assert kinds[backup.name] == "migration"
    assert kinds[f"{db.stem}-prerekey-20260101T000000000000Z.db"] == "prerekey"
    assert kinds[f"{db.stem}-before-restore-20260102T000000000000Z.db"] == "before-restore"
    assert kinds[f"{db.stem}-notes.db"] == "other"
    assert not any(i.path.name.startswith("checkpoints-") for i in infos)
    assert infos[0].path == backup, "the backup made today is the newest"
    assert (
        next(i for i in infos if i.path == backup).revision
        == _sql(db, "select version_num from alembic_version")[0][0]
    )


def test_listing_when_there_is_no_backup_directory(tmp_path):
    assert rs.list_backups(tmp_path / "channelagent.db") == []


# --- the command line, for real ---


def _cli(world_db: Path, *args: str, stdin: str = "", key: str = KEY, port: int | None = None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABASE_", "CHECKPOINT_"))}
    env.update(
        PATH=f"{Path(sys.executable).parent}:/usr/bin:/bin",  # no docker: see the fixture above
        PYTHONPATH=str(REPO),
        DATABASE_URL=f"sqlite+aiosqlite:///{world_db}",
        ENCRYPTION_KEY=key,
        API_SERVER_PORT=str(port or _free_port()),
        API_SERVER_HOST="127.0.0.1",
    )
    return subprocess.run(
        [sys.executable, "-m", "app.admin.restore", *args],
        input=stdin,
        text=True,
        capture_output=True,
        env=env,
        cwd=REPO,
        timeout=120,
    )


def test_cli_list(world):
    db, backup = world
    result = _cli(db, "--list")
    assert result.returncode == 0, result.stderr
    assert backup.name in result.stdout and "before a migration" in result.stdout


def test_cli_restores_a_named_backup_and_prints_the_counts(world):
    db, backup = world
    _sql(
        db, "insert into users (display_name, is_active, created_at) values ('C', 1, '2026-01-01')"
    )
    result = _cli(db, backup.name, "--yes")
    assert result.returncode == 0, result.stderr
    assert "users 3 -> 2" in result.stdout
    assert "integrity_check ok" in result.stdout
    assert "Kept" in result.stdout and "before-restore" in result.stdout
    assert _users(db) == 2


def test_cli_interactive_choice_and_confirmation(world):
    db, backup = world
    _sql(
        db, "insert into users (display_name, is_active, created_at) values ('C', 1, '2026-01-01')"
    )
    result = _cli(db, stdin="1\nRESTORE\n")
    assert result.returncode == 0, result.stderr
    assert _users(db) == 2


def test_cli_wrong_confirmation_or_number_changes_nothing(world):
    db, _ = world
    _sql(
        db, "insert into users (display_name, is_active, created_at) values ('C', 1, '2026-01-01')"
    )
    sha = _sha(db)
    assert _cli(db, stdin="1\nno\n").returncode == 1
    assert _cli(db, stdin="99\n").returncode == 1
    assert _cli(db, stdin="\n").returncode == 1
    assert _cli(db, stdin="").returncode == 1  # input closed
    _assert_untouched(db, sha)


def test_cli_refuses_while_the_api_answers_and_explains(world):
    db, backup = world
    sha = _sha(db)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        result = _cli(db, backup.name, "--yes", port=server.getsockname()[1])
    assert result.returncode == 1
    assert "Restore refused" in result.stderr and "Stop it first" in result.stderr
    _assert_untouched(db, sha)


def test_cli_unknown_backup_name(world):
    db, _ = world
    result = _cli(db, "nope.db", "--yes")
    assert result.returncode == 1 and "no backup named nope.db" in result.stderr


def test_cli_a_wrong_key_is_refused_without_the_flag(world):
    db, backup = world
    other = Fernet.generate_key().decode()
    result = _cli(db, backup.name, "--yes", key=other)
    assert result.returncode == 1 and "cannot be decrypted" in result.stdout
    assert _cli(db, backup.name, "--yes", "--allow-unreadable", key=other).returncode == 0


# --- start.sh wiring ---


def test_start_script_routes_restore_through_the_virtualenv_module():
    text = (REPO / "start.sh").read_text()
    assert '--restore) MODE="restore"' in text
    assert 'python3 -m app.admin.restore "$@"' in text
    assert "--restore" in text.split("set -euo pipefail")[0], "the header documents it"


def test_only_this_projects_container_is_asked_for(world, monkeypatch, tmp_path):
    """A copy of the project elsewhere (another compose project) is not blocked by the
    owner's running container, and the owner's is still found."""
    db, backup = world
    calls = _fake_docker(monkeypatch, [])
    monkeypatch.delenv("COMPOSE_PROJECT_NAME", raising=False)
    sandbox = tmp_path / "Sand.Box-1"
    sandbox.mkdir()
    monkeypatch.chdir(sandbox)
    _restore(db, backup)
    assert "label=com.docker.compose.project=sandbox-1" in calls[0]
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "channelagent")
    assert rs.compose_project() == "channelagent"
