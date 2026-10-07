"""Tests: the storage overview cannot drift from the schema, and it
reports the conversation checkpoints (their own SQLite file, in WAL mode).

Every number is compared with an independent source: a direct SQL count and
the sizes read from the filesystem.
"""

import json
import os
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
import pytest

from app.db.models import Base, Channel
from tests._console import command, run_console


def _main_db_file() -> Path:
    from app.config import get_settings

    return Path(get_settings().database_url.split("///", 1)[1])


def _checkpoint_files() -> list[Path]:
    from app.checkpoints import checkpoint_db_path

    path = checkpoint_db_path()
    return [p for p in (path, Path(f"{path}-wal"), Path(f"{path}-shm")) if p.exists()]


def _direct_checkpoint_counts() -> dict[str, int]:
    from app.checkpoints import checkpoint_db_path

    con = sqlite3.connect(checkpoint_db_path())
    try:
        return {
            t: con.execute(f"select count(*) from {t}").fetchone()[0]
            for t in ("checkpoints", "writes")
        }
    finally:
        con.close()


async def _overview():
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as session:
        return await service.storage_overview(session)


@pytest.fixture
async def empty_db(fresh_db):
    from app.db.session import init_db

    await init_db()


class _LLM(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        if self.path != "/v1/chat/completions":  # no tokenizer on this mock
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.rfile.read(int(self.headers["Content-Length"]))
        data = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def llm(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _LLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    server.shutdown()
    get_settings.cache_clear()


# --- the guard: no table can be missing from the overview unnoticed ---


def test_every_table_of_the_schema_is_counted_or_explicitly_excluded():
    from app.admin import service

    counted = {model.__tablename__ for model in service._COUNTED_MODELS}
    schema = set(Base.metadata.tables)
    assert schema - counted - service.STORAGE_EXCLUDED_TABLES == set(), (
        "a table was added to the schema: count it in service._COUNTED_MODELS or "
        "add it, reviewed, to service.STORAGE_EXCLUDED_TABLES"
    )
    assert counted <= schema
    assert not (counted & service.STORAGE_EXCLUDED_TABLES)


async def test_every_table_of_the_migrated_database_is_counted_or_excluded(empty_db):
    """The same guard against the real migrated database, not only the models:
    a migration that creates a table without a model would be caught here.
    """
    from app.admin import service

    con = sqlite3.connect(_main_db_file())
    try:
        tables = {
            row[0]
            for row in con.execute(
                "select name from sqlite_master where type = 'table' and name not like 'sqlite_%'"
            )
        }
    finally:
        con.close()
    counted = {model.__tablename__ for model in service._COUNTED_MODELS}
    assert tables - counted - service.STORAGE_EXCLUDED_TABLES == set()
    assert tables == counted | service.STORAGE_EXCLUDED_TABLES


def test_the_excluded_list_is_only_the_migration_bookkeeping_table():
    from app.admin import service

    assert service.STORAGE_EXCLUDED_TABLES == {"alembic_version"}


# --- checkpoints in the overview ---


async def test_no_checkpoint_file_yet_means_no_numbers_and_no_file_created(empty_db):
    from app.checkpoints import checkpoint_db_path

    overview = await _overview()
    assert overview.checkpoint_size_bytes is None
    assert overview.checkpoint_row_counts == {}
    assert not checkpoint_db_path().exists(), "reading the overview must not create the file"


async def test_checkpoint_rows_and_size_match_sql_and_the_filesystem(empty_db, llm):
    from app import graph

    for i in range(3):
        await graph.run_turn(Channel.TELEGRAM, "42", 1, f"message {i}")

    overview = await _overview()
    assert overview.checkpoint_row_counts == _direct_checkpoint_counts()
    assert overview.checkpoint_row_counts["checkpoints"] > 0
    assert overview.checkpoint_row_counts["writes"] > 0
    files = _checkpoint_files()
    assert overview.checkpoint_size_bytes == sum(os.path.getsize(p) for p in files)


async def test_the_size_includes_the_wal_file(empty_db, llm):
    from app import graph

    await graph.run_turn(Channel.TELEGRAM, "42", 1, "hello")
    wal = [p for p in _checkpoint_files() if p.name.endswith("-wal")]
    assert wal, "the checkpoint database is expected to run in WAL mode"
    overview = await _overview()
    main_only = _checkpoint_files()[0].stat().st_size
    assert overview.checkpoint_size_bytes >= main_only + wal[0].stat().st_size


async def test_the_counts_follow_the_conversations(empty_db, llm):
    from app import graph

    await graph.run_turn(Channel.TELEGRAM, "42", 1, "one")
    first = (await _overview()).checkpoint_row_counts
    await graph.run_turn(Channel.TELEGRAM, "43", 1, "another user")
    second = (await _overview()).checkpoint_row_counts
    assert second["checkpoints"] > first["checkpoints"]
    assert second == _direct_checkpoint_counts()


async def test_an_unreadable_checkpoint_file_does_not_break_the_overview(empty_db, monkeypatch):
    from app.checkpoints import checkpoint_db_path

    path = checkpoint_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not a sqlite database" * 20)
    overview = await _overview()
    assert overview.checkpoint_row_counts == {}
    assert overview.checkpoint_size_bytes == path.stat().st_size
    assert overview.row_counts["users"] == 0


# --- API and console ---


async def test_the_api_returns_the_checkpoint_numbers(empty_db, llm, monkeypatch):
    from app import graph
    from app.api.app import app
    from app.config import get_settings

    api_key = "test-key-" + "a" * 20
    monkeypatch.setenv("API_SERVER_KEY", api_key)
    get_settings.cache_clear()
    await graph.run_turn(Channel.TELEGRAM, "42", 1, "hello")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        auth = {"Authorization": f"Bearer {api_key}"}
        body = (await c.get("/storage", headers=auth)).json()
    assert body["checkpoint_row_counts"] == _direct_checkpoint_counts()
    assert body["checkpoint_size_bytes"] == sum(os.path.getsize(p) for p in _checkpoint_files())
    from app.checkpoints import checkpoint_db_path

    assert str(checkpoint_db_path().parent) not in json.dumps(body)


async def test_the_console_lists_the_checkpoint_tables(empty_db, llm, monkeypatch):
    from app import graph

    out = await run_console(monkeypatch, *command("storage"))
    assert '"checkpoint_size_bytes": null' in out

    await graph.run_turn(Channel.TELEGRAM, "42", 1, "hello")
    out = await run_console(monkeypatch, *command("storage"))
    shown = json.loads(out[out.index("{") : out.rindex("}") + 1])
    assert shown["checkpoint_size_bytes"] > 0
    assert shown["checkpoint_row_counts"] == _direct_checkpoint_counts()
