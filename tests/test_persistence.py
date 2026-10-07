"""Tests: conversations survive a restart, encrypted at rest.

The restart is real: each phase runs in its own Python process against the
same files. A mock LLM replies with the number of messages it received, so
"the thread survived" is a number (N=7 after 3 turns and 1 more), not a claim.
"""

import json
import os
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from app.db.models import Channel

REPO_ROOT = Path(__file__).resolve().parent.parent
USER_MARKER = "SECRET-USER-MARKER-XYZ"
REPLY_MARKER = "SECRET-REPLY-MARKER-QRS"


class _CountingLLM(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        if self.path != "/v1/chat/completions":  # no tokenizer on this mock
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = f"N={len(body['messages'])} {REPLY_MARKER}"
        data = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(autouse=True)
async def _database(fresh_db):
    """A turn reads its agent's settings from the database: a throwaway one, which the
    subprocesses that simulate a restart inherit through DATABASE_URL."""
    from app.db.session import init_db

    await init_db()


@pytest.fixture
def llm(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _CountingLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("LLAMA_SERVER_URL", url)
    from app.config import get_settings

    get_settings.cache_clear()
    yield url
    server.shutdown()


_TURNS = """
import asyncio, sys
from app.db.models import Channel
from app.graph import close_graph, run_turn

async def main():
    user, agent, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    last = None
    for i in range(n):
        last = await run_turn(Channel.TELEGRAM, user, agent, f"msg {i} @@MARKER@@")
    await close_graph()
    print(last.split()[0])

asyncio.run(main())
""".replace("@@MARKER@@", USER_MARKER)


def _process(user: str, agent: int, turns: int) -> str:
    """One separate Python process = one 'run' of the application."""
    result = subprocess.run(
        [sys.executable, "-c", _TURNS, user, str(agent), str(turns)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr[-800:]
    return result.stdout.strip()


def _checkpoint_file() -> Path:
    return Path(os.environ["CHECKPOINT_DB_PATH"])


# --- the acceptance criterion: a restart resumes the thread ---


def test_history_survives_a_process_restart(fresh_db, llm):
    assert _process("42", 1, 3) == "N=5"  # 3 turns: 3 human + 2 earlier AI + the reply
    assert _process("42", 1, 1) == "N=7", "after the restart the thread must continue"


def test_threads_stay_independent_after_a_restart(fresh_db, llm):
    assert _process("1", 1, 2) == "N=3"
    assert _process("2", 1, 1) == "N=1"
    assert _process("1", 2, 1) == "N=1"  # same user, other agent
    # a second life of the application
    assert _process("1", 1, 1) == "N=5"
    assert _process("2", 1, 1) == "N=3"
    assert _process("1", 2, 1) == "N=3"


# --- encrypted at rest ---


def _all_checkpoint_bytes() -> bytes:
    """The database file plus its WAL and shared-memory files. SQLite runs in
    WAL mode here, so while the application is running recent writes sit in
    the -wal file, not in the main one.
    """
    base = _checkpoint_file()
    parts = [base, Path(f"{base}-wal"), Path(f"{base}-shm")]
    return b"".join(p.read_bytes() for p in parts if p.exists())


def test_checkpoint_file_never_contains_the_conversation_in_clear(fresh_db, llm):
    _process("42", 1, 2)
    raw = _all_checkpoint_bytes()
    assert raw, "the checkpoint file exists and is not empty"
    assert USER_MARKER.encode() not in raw, "the user's text is in clear in the file"
    assert REPLY_MARKER.encode() not in raw, "the model's reply is in clear in the file"


async def test_nothing_in_clear_while_the_connection_is_still_open(fresh_db, llm):
    """The running application never closes the connection between turns:
    the WAL file holds the latest writes and must not hold clear text either.
    """
    from app.graph import run_turn

    await run_turn(Channel.TELEGRAM, "42", 1, f"hello {USER_MARKER}")
    wal = Path(f"{_checkpoint_file()}-wal")
    assert wal.exists() and wal.stat().st_size > 0, "the write is really sitting in the WAL"
    raw = _all_checkpoint_bytes()
    assert USER_MARKER.encode() not in raw and REPLY_MARKER.encode() not in raw


def test_checkpoint_payloads_are_marked_as_fernet_and_thread_ids_stay_readable(fresh_db, llm):
    _process("42", 1, 1)
    con = sqlite3.connect(_checkpoint_file())
    try:
        types = {r[0] for r in con.execute("select type from checkpoints")}
        write_types = {r[0] for r in con.execute("select type from writes")}
        threads = {r[0] for r in con.execute("select distinct thread_id from checkpoints")}
    finally:
        con.close()
    assert types and all(t.endswith("+fernet") for t in types), types
    assert write_types and all(t.endswith("+fernet") for t in write_types), write_types
    assert threads == {"telegram_42_1"}


async def test_a_wrong_key_cannot_read_the_history(fresh_db, llm, monkeypatch):
    from cryptography.fernet import Fernet

    from app import graph
    from app.config import get_settings
    from app.graph import run_turn
    from app.security import encryption

    await run_turn(Channel.TELEGRAM, "42", 1, "first")
    await graph.close_graph()
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    get_settings.cache_clear()
    encryption._fernet.cache_clear()
    try:
        with pytest.raises(ValueError, match="Cannot decrypt"):
            await run_turn(Channel.TELEGRAM, "42", 1, "second")
    finally:
        encryption._fernet.cache_clear()


# --- one bad thread does not stop the others ---


async def test_one_corrupted_thread_does_not_stop_other_threads(fresh_db, llm):
    from app.graph import run_turn

    await run_turn(Channel.TELEGRAM, "1", 1, "a")
    await run_turn(Channel.TELEGRAM, "2", 1, "b")
    con = sqlite3.connect(_checkpoint_file())
    con.execute(
        "update checkpoints set checkpoint = ? where thread_id = 'telegram_1_1'", (b"garbage",)
    )
    con.commit()
    con.close()

    with pytest.raises(ValueError):
        await run_turn(Channel.TELEGRAM, "1", 1, "again")
    assert (await run_turn(Channel.TELEGRAM, "2", 1, "still fine")).split()[0] == "N=3"


# --- close and reopen inside one process ---


async def test_closing_and_reopening_the_graph_keeps_the_history(fresh_db, llm):
    from app import graph
    from app.graph import run_turn

    assert (await run_turn(Channel.TELEGRAM, "5", 1, "one")).split()[0] == "N=1"
    await graph.close_graph()
    assert (await run_turn(Channel.TELEGRAM, "5", 1, "two")).split()[0] == "N=3"


# --- where the file lives ---


def test_checkpoint_path_defaults_next_to_the_main_database(monkeypatch, tmp_path):
    from app import checkpoints
    from app.config import get_settings

    monkeypatch.delenv("CHECKPOINT_DB_PATH", raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/sub/app.db")
    get_settings.cache_clear()
    assert checkpoints.checkpoint_db_path() == tmp_path / "sub" / "checkpoints.db"


def test_checkpoint_path_can_be_set_explicitly(monkeypatch, tmp_path):
    from app import checkpoints
    from app.config import get_settings

    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "elsewhere.db"))
    get_settings.cache_clear()
    assert checkpoints.checkpoint_db_path() == tmp_path / "elsewhere.db"


def test_checkpoint_path_for_a_non_sqlite_database(monkeypatch):
    from app import checkpoints
    from app.config import get_settings

    monkeypatch.delenv("CHECKPOINT_DB_PATH", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/db")
    get_settings.cache_clear()
    assert checkpoints.checkpoint_db_path() == Path("data") / "checkpoints.db"


# --- deleting a user purges their conversations ---


def _thread_rows(thread_id: str) -> int:
    con = sqlite3.connect(_checkpoint_file())
    try:
        return sum(
            con.execute(f"select count(*) from {t} where thread_id = ?", (thread_id,)).fetchone()[0]
            for t in ("checkpoints", "writes")
        )
    finally:
        con.close()


@pytest.fixture
async def two_users(fresh_db, llm):
    from app.admin import service
    from app.db.models import ChannelIdentity, Direction, PermissionKind, User
    from app.db.session import init_db, session_scope
    from app.graph import run_turn
    from app.security.auth import grant_permission

    await init_db()
    async with session_scope() as s:
        for i in (1, 2):
            u = User(display_name=f"user{i}")
            s.add(u)
            await s.flush()
            ident = ChannelIdentity(user_id=u.id, channel=Channel.TELEGRAM, external_id=str(i))
            s.add(ident)
            await s.flush()
            await grant_permission(s, ident, PermissionKind.CHAT)
            agent = await service.create_agent(s, u.id, "default")
            await service.record_action(
                s, user_id=u.id, agent_id=agent.id, channel=Channel.TELEGRAM,
                direction=Direction.INBOUND, text="hello",
            )
        await s.commit()
    await run_turn(Channel.TELEGRAM, "1", 1, "hi")
    await run_turn(Channel.TELEGRAM, "2", 2, "hi")


async def test_purge_deletes_that_users_conversations_and_no_one_elses(two_users):
    from app.admin import service
    from app.db.session import session_scope
    from app.graph import run_turn

    assert _thread_rows("telegram_1_1") > 0 and _thread_rows("telegram_2_2") > 0
    async with session_scope() as s:
        report = await service.delete_user(s, 1, purge=True)
        await s.commit()
    assert report.threads_deleted == 1
    assert _thread_rows("telegram_1_1") == 0
    assert _thread_rows("telegram_2_2") > 0, "the other user's conversation is untouched"
    assert (await run_turn(Channel.TELEGRAM, "2", 2, "more")).split()[0] == "N=3"


async def test_a_refused_deletion_keeps_the_conversations(two_users):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        with pytest.raises(service.UserHasHistoryError):
            await service.delete_user(s, 1)
    assert _thread_rows("telegram_1_1") > 0
