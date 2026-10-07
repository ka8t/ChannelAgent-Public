"""Tests: rotating ENCRYPTION_KEY. The real key had been committed, so
it has to be replaceable without losing the audit trail, the stored addresses
or the conversations.
"""

import hashlib
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from cryptography.fernet import Fernet, InvalidToken

from app.admin import service
from app.admin.rekey import RekeyError, main, run_rekey
from app.db.models import Channel, Direction

OLD = Fernet.generate_key().decode()
NEW = Fernet.generate_key().decode()


def _use_key(monkeypatch, key: str):
    from app.config import get_settings
    from app.security import encryption

    monkeypatch.setenv("ENCRYPTION_KEY", key)
    get_settings.cache_clear()
    encryption._fernet.cache_clear()


def _paths():
    from app.checkpoints import checkpoint_db_path
    from app.config import get_settings

    return Path(get_settings().database_url.split("///", 1)[1]), checkpoint_db_path()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _values(path: Path, table: str, column: str) -> list[str]:
    con = sqlite3.connect(path)
    try:
        return [
            r[0]
            for r in con.execute(f'select "{column}" from "{table}" where "{column}" is not null')
        ]
    finally:
        con.close()


def _readable_with(key: str, values) -> int:
    f = Fernet(key.encode())
    n = 0
    for v in values:
        try:
            f.decrypt(v.encode() if isinstance(v, str) else bytes(v))
            n += 1
        except InvalidToken:
            pass
    return n


@pytest.fixture
async def world(fresh_db, monkeypatch):
    """Data written with the OLD key: 3 log entries, 2 access requests,
    1 email address."""
    from app.db.session import init_db, session_scope

    _use_key(monkeypatch, OLD)
    await init_db()
    async with session_scope() as s:
        user = await service.create_user(s, "Alice")
        await service.add_channel_identity(s, user.id, Channel.EMAIL, "alice@example.com")
        agent = await service.create_agent(s, user.id, "default")
        for text in ("first secret", "second secret", "third"):
            await service.record_action(
                s,
                user_id=user.id,
                agent_id=agent.id,
                channel=Channel.TELEGRAM,
                direction=Direction.INBOUND,
                text=text,
            )
        await service.request_access(s, Channel.TELEGRAM, "777", "let me in")
        await service.request_access(s, Channel.TELEGRAM, "888", "me too")
        await s.commit()


# --- the rotation itself ---


async def test_every_value_moves_to_the_new_key_and_the_app_reads_it_back(world, monkeypatch):
    db, cp = _paths()
    before = _values(db, "action_logs", "text")
    assert _readable_with(OLD, before) == 3 and _readable_with(NEW, before) == 0

    report = run_rekey(db, cp, OLD, NEW)
    assert report.applied and report.to_rewrite == 9 and report.unreadable == 0
    assert (report.verified_new, report.still_old) == (9, 0)

    for table, column, count in (
        ("action_logs", "text", 3),
        ("access_requests", "first_message_text", 2),
        ("channel_identities", "raw_address", 1),
    ):
        values = _values(db, table, column)
        assert _readable_with(NEW, values) == count, table
        assert _readable_with(OLD, values) == 0, f"{table} still readable with the old key"

    _use_key(monkeypatch, NEW)
    from app.db.session import session_scope

    async with session_scope() as s:
        logs = await service.search_action_logs(s)
        requests = await service.list_requests(s)
    assert [e.text for e in logs] == ["third", "second secret", "first secret"]
    assert sorted(r.first_message_text for r in requests) == ["let me in", "me too"]


async def test_the_old_key_can_no_longer_decrypt_anything_afterwards(world, monkeypatch):
    from app.security.encryption import decrypt_value

    db, cp = _paths()
    run_rekey(db, cp, OLD, NEW)
    _use_key(monkeypatch, OLD)
    for table, column in (
        ("action_logs", "text"),
        ("access_requests", "first_message_text"),
        ("channel_identities", "raw_address"),
    ):
        for value in _values(db, table, column):
            with pytest.raises(ValueError, match="Cannot decrypt"):
                decrypt_value(value)


async def test_a_dry_run_changes_nothing_and_says_what_it_would_do(world):
    db, cp = _paths()
    digest = _sha(db)
    report = run_rekey(db, cp, OLD, NEW, dry_run=True)
    assert not report.applied and report.to_rewrite == 9 and report.unreadable == 0
    assert _sha(db) == digest
    assert not (db.parent / "backups").exists()


async def test_running_it_twice_leaves_everything_alone_the_second_time(world):
    db, cp = _paths()
    run_rekey(db, cp, OLD, NEW)
    digest = _sha(db)
    backups_after_first = sorted(p.name for p in (db.parent / "backups").glob("*.db"))
    second = run_rekey(db, cp, OLD, NEW)
    assert second.backups == [] and not second.applied, "nothing to do, no useless backup"
    assert sorted(p.name for p in (db.parent / "backups").glob("*.db")) == backups_after_first
    assert second.to_rewrite == 0 and sum(t.new for t in second.targets.values()) == 9
    assert _sha(db) == digest, "nothing was rewritten"


async def test_a_run_interrupted_half_way_can_be_finished(world):
    """Two values are already on the new key (a partial earlier run)."""
    db, cp = _paths()
    con = sqlite3.connect(db)
    new = Fernet(NEW.encode())
    old = Fernet(OLD.encode())
    for row_id, value in con.execute("select id, text from action_logs limit 2").fetchall():
        con.execute(
            "update action_logs set text = ? where id = ?",
            (new.encrypt(old.decrypt(value.encode())).decode(), row_id),
        )
    con.commit()
    con.close()
    report = run_rekey(db, cp, OLD, NEW)
    assert (
        report.targets["action_logs.text"].new == 2 and report.targets["action_logs.text"].old == 1
    )
    assert _readable_with(NEW, _values(db, "action_logs", "text")) == 3


# --- refusals: nothing is written ---


async def test_an_unreadable_value_aborts_before_anything_is_written(world):
    db, cp = _paths()
    con = sqlite3.connect(db)
    con.execute("update action_logs set text = 'garbage' where id = 2")
    con.commit()
    con.close()
    digest = _sha(db)
    with pytest.raises(RekeyError, match="neither key"):
        run_rekey(db, cp, OLD, NEW)
    assert _sha(db) == digest, "the database is byte for byte unchanged"
    assert not (db.parent / "backups").exists()


async def test_allow_unreadable_rotates_the_rest_and_leaves_the_bad_value_as_it_is(world):
    db, cp = _paths()
    con = sqlite3.connect(db)
    con.execute("update action_logs set text = 'garbage' where id = 2")
    con.commit()
    con.close()
    report = run_rekey(db, cp, OLD, NEW, allow_unreadable=True)
    assert report.unreadable == 1 and report.to_rewrite == 8
    assert "garbage" in _values(db, "action_logs", "text")
    assert _readable_with(NEW, _values(db, "action_logs", "text")) == 2


async def test_the_wrong_old_key_is_refused_without_writing(world):
    db, cp = _paths()
    digest = _sha(db)
    with pytest.raises(RekeyError, match="neither key"):
        run_rekey(db, cp, Fernet.generate_key().decode(), NEW)
    assert _sha(db) == digest


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("", NEW, "required"),
        (OLD, "", "required"),
        (OLD, OLD, "same"),
        ("not-a-key", NEW, "not a valid"),
        (OLD, "short", "not a valid"),
    ],
)
async def test_bad_keys_are_refused(world, old, new, message):
    db, cp = _paths()
    digest = _sha(db)
    with pytest.raises(RekeyError, match=message):
        run_rekey(db, cp, old, new)
    assert _sha(db) == digest


def test_a_missing_database_is_refused(tmp_path):
    with pytest.raises(RekeyError, match="does not exist"):
        run_rekey(tmp_path / "nope.db", None, OLD, NEW)


# --- backups ---


async def test_a_prerekey_backup_with_the_old_data_is_made_first(world):
    db, cp = _paths()
    report = run_rekey(db, cp, OLD, NEW)
    (backup,) = report.backups
    assert "-prerekey-" in backup.name
    assert _readable_with(OLD, _values(backup, "action_logs", "text")) == 3, "the way back"


async def test_prerekey_backups_survive_the_rotation_of_migration_backups(world):
    """With three migration backups and one prerekey copy in the folder, keeping
    one migration backup removes the two oldest and never the prerekey copy.
    """
    from app.db.backup import _prune

    db, cp = _paths()
    (prerekey,) = run_rekey(db, cp, OLD, NEW).backups
    folder = prerekey.parent
    migration_backups = [
        folder / f"{db.stem}-rev{i}-2020010{i}T000000000000Z.db" for i in (1, 2, 3)
    ]
    for f in migration_backups:
        f.write_bytes(b"x")
    _prune(folder, db.stem, 1)
    assert prerekey.exists(), "the way back from a rotation is never rotated away"
    assert [f.exists() for f in migration_backups] == [False, False, True]


async def test_no_backup_flag(world):
    db, cp = _paths()
    assert run_rekey(db, cp, OLD, NEW, backup=False).backups == []


# --- the conversations ---


class _Counting(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        data = json.dumps(
            {"choices": [{"message": {"content": f"N={len(body['messages'])} REPLY-MARKER"}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def llm(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _Counting)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    server.shutdown()


async def test_a_conversation_continues_after_the_rotation(world, llm, monkeypatch):
    from app import graph
    from app.graph import run_turn

    for text in ("one", "two", "three"):
        reply = await run_turn(Channel.TELEGRAM, "42", 1, text)
    assert reply.split()[0] == "N=5"
    await graph.close_graph()

    db, cp = _paths()
    checkpoint_values = _values(cp, "checkpoints", "checkpoint")
    assert _readable_with(OLD, checkpoint_values) > 0
    report = run_rekey(db, cp, OLD, NEW)
    assert (
        report.targets["checkpoints.checkpoint"].old > 0 and report.targets["writes.value"].old > 0
    )
    assert _readable_with(OLD, _values(cp, "checkpoints", "checkpoint")) == 0
    assert _readable_with(OLD, _values(cp, "writes", "value")) == 0

    _use_key(monkeypatch, NEW)
    reply = await run_turn(Channel.TELEGRAM, "42", 1, "four")
    assert reply.split()[0] == "N=7", "the history survived the key change"


async def test_the_old_key_cannot_read_the_conversations_afterwards(world, llm, monkeypatch):
    from app import graph
    from app.graph import run_turn

    await run_turn(Channel.TELEGRAM, "42", 1, "one")
    await graph.close_graph()
    db, cp = _paths()
    run_rekey(db, cp, OLD, NEW)
    _use_key(monkeypatch, OLD)
    with pytest.raises(ValueError, match="Cannot decrypt"):
        await run_turn(Channel.TELEGRAM, "42", 1, "two")


# --- the command line ---


async def test_the_command_reports_and_returns_zero(world, monkeypatch, capsys):
    _use_key(monkeypatch, NEW)
    monkeypatch.setenv("OLD_ENCRYPTION_KEY", OLD)
    assert main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "action_logs.text" in out and "old key:     3" in out and "Nothing written" in out
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Done: 9 value(s) re-encrypted; verified 9 readable with the new key, 0 still" in out
    assert "Backup:" in out


async def test_the_command_refuses_without_the_old_key_and_returns_two(world, monkeypatch, capsys):
    _use_key(monkeypatch, NEW)
    monkeypatch.delenv("OLD_ENCRYPTION_KEY", raising=False)
    assert main([]) == 2
    assert "Rekey refused" in capsys.readouterr().err
