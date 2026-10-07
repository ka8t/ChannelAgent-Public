"""Tests for the interactive admin console, rebuilt on the Admin API (owner option 2).

- It never opens the database: it imports neither the service layer nor a session.
- Its menus come from the manifest: every API command is reachable (contract N of N).
- The six menus of the old console (requests, users, agents, logs, admin events, storage)
  work through the API, recorded with the actor `cli:<user>`; a route's scope applies.
- Deletions and owner operations ask for a confirmation; no answer ends the session except
  `q`, Ctrl+D or Ctrl+C; without API_SERVER_KEY it says so and exits 2.
"""

import re
import sqlite3
from pathlib import Path

import pytest

from app.db.models import Channel, Direction, PermissionKind
from tests._console import command, run_console

KEY = "k" * 32
REPO = Path(__file__).resolve().parents[1]


def _db() -> str:
    from app.config import get_settings

    return get_settings().database_url.split("///", 1)[1]


def _sql(query: str, *args):
    con = sqlite3.connect(_db())
    try:
        return con.execute(query, args).fetchall()
    finally:
        con.close()


async def _seed():
    """User 1 Alice: telegram identity 1 (chat), agent 1 default, 1 log entry "hello".
    User 2 Bob: no history. Pending request 1: telegram 777 "let me in"."""
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        alice = await service.create_user(s, "Alice")
        await service.create_user(s, "Bob")
        identity = await service.add_channel_identity(s, alice.id, Channel.TELEGRAM, "1")
        await service.grant_identity_permission(s, alice.id, identity.id, PermissionKind.CHAT)
        agent = await service.create_agent(s, alice.id, "default")
        await service.record_action(
            s,
            user_id=alice.id,
            agent_id=agent.id,
            channel=Channel.TELEGRAM,
            direction=Direction.INBOUND,
            text="hello",
        )
        await service.request_access(s, Channel.TELEGRAM, "777", "let me in")
        await s.commit()


@pytest.fixture
async def world(fresh_db, monkeypatch):
    from app.api import deps
    from app.config import get_settings
    from app.db.session import init_db

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    await _seed()
    yield
    get_settings.cache_clear()


def _cli_actor() -> str:
    from app.admin.client import client_label

    return client_label()


# --- it is a client of the API ---


def test_the_console_never_opens_the_database():
    source = (REPO / "app" / "admin" / "cli.py").read_text()
    imports = [ln for ln in source.splitlines() if re.match(r"\s*(from|import)\s", ln)]
    assert not [ln for ln in imports if "service" in ln or "app.db" in ln or "session" in ln]
    assert "session_scope" not in source


async def test_every_api_command_is_reachable_from_the_console(world, monkeypatch):
    """Contract: each command of the manifest is listed under its tag and runs when chosen;
    the call itself is recorded instead of sent, so no command changes anything here."""
    from app.admin import cli
    from app.admin import client as api

    called = []

    async def record(_http, op, _args):
        called.append(op["command"])
        return 200, {"ok": True}

    monkeypatch.setattr(api, "execute", record)
    ops = cli._manifest()
    listed = sum(len(v) for v in cli.groups(ops).values())
    values = {"integer": "1", "number": "1", "boolean": "true", "array": "[]", "object": "{}"}
    for op in ops:
        answers = [op["command"]]
        for f in op["fields"]:
            answers.append(str(f["enum"][0]) if f["enum"] else values.get(f["type"], "x"))
        if cli._needs_confirmation(op):
            answers.append(cli.CONFIRM_WORD)
        await run_console(monkeypatch, *answers)
    print(f"console reach: {len(set(called))} of {len(ops)} commands, {listed} listed")
    assert called == [op["command"] for op in ops] and listed == len(ops) >= 66


async def test_a_command_is_also_picked_by_its_menu_and_number(world, monkeypatch):
    from app.admin import cli

    menus = cli.groups(cli._manifest())
    number = str([op["command"] for op in menus["users"]].index("list-users") + 1)
    out = await run_console(monkeypatch, str(list(menus).index("users") + 1), number, "", "", "b")
    assert "Alice" in out and "Bob" in out


# --- the six menus of the old console, through the API ---


async def test_an_empty_list_says_so(fresh_db, monkeypatch):
    from app.db.session import init_db

    await init_db()
    out = await run_console(monkeypatch, *command("list-requests"))
    assert "(none)" in out


async def test_requests_are_listed_and_approved(world, monkeypatch):
    out = await run_console(
        monkeypatch, *command("list-requests"), *command("approve-request", request_id=1)
    )
    assert "let me in" in out
    assert _sql("select status, resolved_by from access_requests") == [("approved", _cli_actor())]


async def test_users_are_created_changed_and_deleted_with_a_confirmation(world, monkeypatch):
    out = await run_console(
        monkeypatch,
        *command("create-user", display_name="Carol"),
        *command("update-user", user_id=2, is_active="false"),
        *command("delete-user", user_id=3, confirm=False),
    )
    assert "Not run." in out
    assert _sql("select id, display_name, is_active from users order by id")[1:] == [
        (2, "Bob", 0),
        (3, "Carol", 1),
    ]
    await run_console(monkeypatch, *command("delete-user", user_id=3))
    assert _sql("select count(*) from users where id = 3") == [(0,)]


async def test_agents_are_created_renamed_and_deactivated(world, monkeypatch):
    await run_console(
        monkeypatch,
        *command("create-agent", user_id=2, name="work"),
        *command("update-agent", agent_id=2, name="job", is_active="false"),
    )
    assert _sql("select user_id, name, is_active from agents where id = 2") == [(2, "job", 0)]


async def test_logs_are_searched(world, monkeypatch):
    out = await run_console(monkeypatch, *command("search-logs", keyword="hello", limit=20))
    assert "hello" in out and "telegram" in out


async def test_admin_events_are_searched_and_show_the_cli_actor(world, monkeypatch):
    await run_console(monkeypatch, *command("create-user", display_name="Dan"))
    out = await run_console(monkeypatch, *command("search-admin-events", action="user.create"))
    assert _cli_actor() in out
    rows = _sql("select actor from admin_events where action = 'user.create' order by id")
    assert rows[-1] == (_cli_actor(),)


async def test_storage_is_shown(world, monkeypatch):
    out = await run_console(monkeypatch, *command("storage"))
    assert "row_counts" in out and "undecryptable_rows" in out


# --- scope, confirmation, bad input, ending ---


async def test_a_refusal_of_the_api_is_shown_not_bypassed(world, monkeypatch):
    from app.api.app import app
    from app.api.scopes import Principal, Scope, get_principal

    app.dependency_overrides[get_principal] = lambda: Principal("cli:reader", Scope.READ)
    try:
        out = await run_console(monkeypatch, *command("create-user", display_name="Nope"))
    finally:
        app.dependency_overrides.clear()
    assert "Not done (HTTP 403)" in out
    assert _sql("select count(*) from users where display_name = 'Nope'") == [(0,)]


async def test_no_answer_ends_the_session_or_raises(world, monkeypatch):
    out = await run_console(
        monkeypatch,
        "nonsense",
        "users",
        "99",
        "b",
        *command("create-agent", user_id="abc", name="x"),
        "update-user",
        "",  # user_id is required: the other fields are not asked
        *command("get-user", user_id=999),
        *command("set-config", key="KEY", value="v", confirm=False),
    )
    assert out.count("Unknown option.") == 2
    assert "Invalid input, nothing sent. --user-id: 'abc' is not a valid integer" in out
    assert "Invalid input, nothing sent. user_id is required" in out
    assert "Not done (HTTP 404)" in out and "Not run." in out


async def test_a_job_command_waits_for_its_job(world, monkeypatch):
    out = await run_console(monkeypatch, *command("create-database-backup"))
    assert '"status": "done"' in out and "-manual-" in out


def test_the_console_program_says_what_is_missing_and_exits_2(tmp_path):
    import os
    import subprocess
    import sys

    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(REPO),
        "HOME": str(tmp_path),
        "ENCRYPTION_KEY": os.environ["ENCRYPTION_KEY"],  # the conftest's throwaway key
    }
    done = subprocess.run(
        [sys.executable, "-m", "app.admin.cli", "--transport", "inprocess"],
        cwd=tmp_path,
        env=env,
        input="q\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 2 and "API_SERVER_KEY is not set" in done.stderr


async def test_the_console_process_exits_after_a_purge_and_on_closed_input(world, monkeypatch):
    """A purge opens the conversation checkpoint file, whose connection runs a thread that
    would keep the process alive; closed input (Ctrl+D) ends the session cleanly."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONPATH": str(REPO), "API_SERVER_KEY": KEY}
    purge = "\n".join(command("delete-user", user_id=1, purge="true")) + "\n"
    result = subprocess.run(
        [sys.executable, "-m", "app.admin.cli", "--transport", "inprocess"],
        cwd=REPO,
        env=env,
        input=purge,  # no q: the input ends
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert "in process" in result.stdout and "Input closed, leaving the console." in result.stdout
    assert "Type YES to run delete-user: ok" in result.stdout  # 204: the API answers no body
    assert _sql("select count(*) from users where id = 1") == [(0,)]


# --- models ("the admin API/console" entry point) ---


@pytest.fixture
async def models_env(world, monkeypatch, tmp_path):
    from app.config import get_settings

    models = tmp_path / "models"
    models.mkdir()
    (models / "tiny.gguf").write_bytes(b"GGUF" + b"\0" * 60)
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:9")
    get_settings.cache_clear()
    return models


async def test_models_are_listed_and_deleted_after_confirmation(models_env, monkeypatch):
    out = await run_console(
        monkeypatch,
        *command("list-models"),
        *command("delete-model", name="tiny.gguf", confirm=False),
    )
    assert "tiny.gguf" in out and "Not run." in out and (models_env / "tiny.gguf").exists()
    await run_console(monkeypatch, *command("delete-model", name="tiny.gguf"))
    assert not (models_env / "tiny.gguf").exists()
    events = _sql("select actor from admin_events where action = 'model.delete'")
    assert events == [(_cli_actor(),)]


async def test_a_refused_model_name_is_reported(models_env, monkeypatch):
    out = await run_console(monkeypatch, *command("delete-model", name="a;b.gguf"))
    assert "Not done (HTTP 422)" in out and "A model name is" in out


async def test_a_failed_job_is_reported_as_not_done(world, monkeypatch):
    from app.api import operations
    from app.db.backup import BackupError

    def fail(_path, _label):
        raise BackupError("the disk is full")

    monkeypatch.setattr(operations, "make_backup", fail)
    out = await run_console(monkeypatch, *command("create-database-backup"))
    assert "Not done (HTTP 202): the disk is full" in out
