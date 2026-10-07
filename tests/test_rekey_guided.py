"""Tests: `./start.sh --rekey`, the guided rotation of ENCRYPTION_KEY.

The rotation itself is tested in test_rekey.py. Here: the order of the steps,
the refusals, what is written where, and that no key ever reaches the output.
Data is written with the OLD key through the real service layer, `.env` holds
that key like a real one, and the new key is read back from `.env`.
"""

import hashlib
import os
import socket
import sqlite3
import stat
from pathlib import Path

import pytest
from cryptography.fernet import Fernet, InvalidToken

from app.admin import rekey_guided, service
from app.db.models import Channel, Direction

OLD = Fernet.generate_key().decode()
THIRD = Fernet.generate_key().decode()
REPO = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _paths():
    from app.checkpoints import checkpoint_db_path
    from app.config import get_settings

    return Path(get_settings().database_url.split("///", 1)[1]), checkpoint_db_path()


def _values(db: Path) -> list[bytes]:
    con = sqlite3.connect(db)
    try:
        out = []
        for table, column in (
            ("action_logs", "text"),
            ("access_requests", "first_message_text"),
            ("channel_identities", "raw_address"),
            ("admin_events", "details"),
        ):
            query = f'select "{column}" from "{table}" where "{column}" is not null'
            out += [r[0].encode() for r in con.execute(query)]
        return out
    finally:
        con.close()


def _readable(key: str, values) -> int:
    f, n = Fernet(key.encode()), 0
    for v in values:
        try:
            f.decrypt(v)
            n += 1
        except InvalidToken:
            pass
    return n


def _env_key(root: Path) -> str:
    for line in (root / ".env").read_text().splitlines():
        if line.startswith("ENCRYPTION_KEY="):
            return line.split("=", 1)[1]
    return ""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
async def world(fresh_db, tmp_path, monkeypatch):
    """Data under the OLD key; the current directory holds `.env` with that key;
    the process environment does not hold it (start.sh removes it too), so the
    key comes from the file.
    """
    from app.config import get_settings
    from app.db.session import init_db, session_scope
    from app.security import encryption

    monkeypatch.setenv("ENCRYPTION_KEY", OLD)
    get_settings.cache_clear()
    encryption._fernet.cache_clear()
    await init_db()
    async with session_scope() as s:
        user = await service.create_user(s, "Alice")
        await service.add_channel_identity(s, user.id, Channel.EMAIL, "alice@example.com")
        agent = await service.create_agent(s, user.id, "default")
        for text in ("first secret", "second secret", "third"):
            await service.record_action(
                s, user_id=user.id, agent_id=agent.id, channel=Channel.TELEGRAM,
                direction=Direction.INBOUND, text=text,
            )
        await service.request_access(s, Channel.TELEGRAM, "777", "let me in")
        await service.request_access(s, Channel.TELEGRAM, "888", "me too")
        await s.commit()

    monkeypatch.delenv("ENCRYPTION_KEY")
    monkeypatch.setenv("API_SERVER_PORT", str(_free_port()))
    monkeypatch.setattr("app.admin.restore.shutil.which", lambda name: None)  # no host Docker
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f"TELEGRAM_BOT_TOKEN=\nENCRYPTION_KEY={OLD}\nLLAMA_PORT=8080\n")
    (tmp_path / ".env").chmod(0o600)
    get_settings.cache_clear()
    encryption._fernet.cache_clear()
    return tmp_path


def _run(args=("--yes",)) -> int:
    from app.config import get_settings

    get_settings.cache_clear()
    return rekey_guided.main(list(args))


# --- the whole sequence ---


async def test_a_full_rotation_moves_every_value_and_puts_the_new_key_in_env(world, capsys):
    db, _ = _paths()
    before = _values(db)
    assert len(before) == 9 and _readable(OLD, before) == 9

    code = _run()
    out = capsys.readouterr()

    assert code == 0, out.err
    new_key = _env_key(world)
    assert new_key and new_key != OLD
    Fernet(new_key.encode())  # a valid Fernet key
    after = _values(db)
    assert _readable(OLD, after) == 0, "the old key reads nothing any more"
    assert _readable(new_key, after) == 9, "the new key reads everything"
    assert "Values the key in .env cannot decrypt: 0" in out.out
    assert "9 value(s) re-encrypted; 9 read with the new key, 0 still readable" in out.out


async def test_no_key_appears_in_the_output_and_the_old_one_only_in_the_safety_net(world, capsys):
    assert _run() == 0
    out = capsys.readouterr()
    new_key = _env_key(world)
    for key in (OLD, new_key):
        assert key not in out.out and key not in out.err
    holders = {
        p.name for p in world.rglob("*") if p.is_file() and OLD.encode() in p.read_bytes()
    }
    assert holders == {".env.pre-rekey"}, holders
    new_holders = {
        p.name for p in world.rglob("*") if p.is_file() and new_key.encode() in p.read_bytes()
    }
    assert new_holders == {".env"}, "the new key is only in .env"


async def test_the_files_holding_keys_are_private(world):
    assert _run() == 0
    for name in (".env", ".env.pre-rekey"):
        assert stat.S_IMODE(os.stat(world / name).st_mode) == 0o600


async def test_the_prerekey_backups_are_made_and_the_end_message_says_what_to_delete(
    world, capsys
):
    assert _run() == 0
    text = capsys.readouterr().out
    backups = list(_paths()[0].parent.glob("backups/*prerekey*"))
    # One copy: this fixture has no checkpoint file (test_rekey.py covers that one).
    assert len(backups) == 1 and backups[0].name.startswith("test-prerekey-")
    assert ".env.pre-rekey" in text and "password manager" in text and "prerekey copies" in text
    assert (world / ".env.pre-rekey").exists(), "nothing is deleted by the tool"


async def test_the_other_lines_of_env_are_kept(world):
    assert _run() == 0
    lines = (world / ".env").read_text().splitlines()
    assert "TELEGRAM_BOT_TOKEN=" in lines and "LLAMA_PORT=8080" in lines
    assert len([ln for ln in lines if ln.startswith("ENCRYPTION_KEY=")]) == 1


# --- dry run and confirmation ---


async def test_dry_run_shows_the_counts_and_changes_nothing(world, capsys):
    db, cp = _paths()
    env_before = (world / ".env").read_text()
    sha = _sha(db)
    assert _run(("--dry-run",)) == 0
    out = capsys.readouterr().out
    assert "9 value(s) to re-encrypt, 0 unreadable" in out and "Dry run only" in out
    assert (world / ".env").read_text() == env_before
    assert not (world / ".env.pre-rekey").exists()
    assert _sha(db) == sha
    assert not list(db.parent.glob("backups/*prerekey*"))


async def test_the_dry_run_comes_before_the_confirmation_prompt(world, monkeypatch, capsys):
    prompts = []

    def fake_input(prompt):
        prompts.append((prompt, capsys.readouterr().out))
        return "no"

    monkeypatch.setattr("builtins.input", fake_input)
    assert _run(()) == 1
    (prompt, seen), = prompts
    assert "ROTATE" in prompt
    assert "9 value(s) to re-encrypt" in seen, "the counts were shown before asking"


async def test_a_wrong_confirmation_changes_nothing(world, monkeypatch, capsys):
    db, _ = _paths()
    env_before, sha = (world / ".env").read_text(), _sha(db)
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert _run(()) == 1
    assert "Cancelled" in capsys.readouterr().out
    assert (world / ".env").read_text() == env_before and _sha(db) == sha
    assert not (world / ".env.pre-rekey").exists()


async def test_the_confirmation_word_starts_the_rotation(world, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "ROTATE")
    assert _run(()) == 0
    assert _env_key(world) != OLD


# --- refusals ---


async def test_refused_while_the_application_runs_nothing_changed(world, capsys):
    from app.config import get_settings

    db, _ = _paths()
    env_before, sha = (world / ".env").read_text(), _sha(db)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", int(os.environ["API_SERVER_PORT"])))
        listener.listen()
        code = _run()
    get_settings.cache_clear()
    err = capsys.readouterr().err
    assert code == 2 and "looks like it is running" in err and "./start.sh --stop" in err
    assert (world / ".env").read_text() == env_before and _sha(db) == sha


async def test_refused_when_a_container_is_running(world, monkeypatch, capsys):
    monkeypatch.setattr(
        "app.admin.restore.running_reasons",
        lambda db_path, host, port: ["a container is running (channelagent-channelagent-1)"],
    )
    assert _run() == 2
    assert "a container is running" in capsys.readouterr().err
    assert _env_key(world) == OLD


async def test_unreadable_values_stop_the_run_unless_allowed(world, capsys):
    db, _ = _paths()
    con = sqlite3.connect(db)
    stray = Fernet(THIRD.encode()).encrypt(b"written with another key").decode()
    con.execute("update action_logs set text = ? where id = 1", (stray,))
    con.commit()
    con.close()
    env_before, sha = (world / ".env").read_text(), _sha(db)

    assert _run() == 2
    err = capsys.readouterr()
    assert "8 value(s) to re-encrypt, 1 unreadable" in err.out
    assert "Refused" in err.err and "--allow-unreadable" in err.err
    assert (world / ".env").read_text() == env_before and _sha(db) == sha

    assert _run(("--yes", "--allow-unreadable")) == 0
    after = _values(db)
    assert _readable(_env_key(world), after) == 8, "the readable ones moved"
    assert _readable(OLD, after) == 0 and _readable(THIRD, after) == 1, "the stray one is untouched"


async def test_an_existing_safety_net_is_never_overwritten(world, capsys):
    (world / ".env.pre-rekey").write_text("ENCRYPTION_KEY=an-earlier-key\n")
    assert _run() == 2
    assert "already exists" in capsys.readouterr().err
    assert (world / ".env.pre-rekey").read_text() == "ENCRYPTION_KEY=an-earlier-key\n"
    assert _env_key(world) == OLD


async def test_an_empty_key_has_nothing_to_rotate(world, capsys):
    (world / ".env").write_text("ENCRYPTION_KEY=\n")
    assert _run() == 2
    assert "nothing to rotate" in capsys.readouterr().err


# --- failures ---


async def test_a_failure_before_anything_is_rewritten_restores_env(world, monkeypatch, capsys):
    original = rekey_guided.run_rekey

    def fail_on_the_real_run(*args, **kwargs):
        if kwargs.get("dry_run"):
            return original(*args, **kwargs)
        raise rekey_guided.RekeyError("disk full")

    monkeypatch.setattr(rekey_guided, "run_rekey", fail_on_the_real_run)
    env_before = (world / ".env").read_text()
    assert _run() == 3
    err = capsys.readouterr().err
    assert "disk full" in err and ".env was restored" in err
    assert (world / ".env").read_text() == env_before
    assert not (world / ".env.pre-rekey").exists()
    assert _readable(OLD, _values(_paths()[0])) == 9


async def test_a_failure_after_the_data_moved_keeps_both_keys_and_says_how_to_finish(
    world, monkeypatch, capsys
):
    original = rekey_guided.run_rekey

    def apply_then_fail(*args, **kwargs):
        if kwargs.get("dry_run"):
            return original(*args, **kwargs)
        original(*args, **kwargs)
        raise rekey_guided.RekeyError("verification failed")

    monkeypatch.setattr(rekey_guided, "run_rekey", apply_then_fail)
    assert _run() == 3
    err = capsys.readouterr().err
    new_key = _env_key(world)
    assert new_key != OLD, ".env keeps the new key"
    assert (world / ".env.pre-rekey").exists() and OLD in (world / ".env.pre-rekey").read_text()
    assert "OLD_ENCRYPTION_KEY" in err and ".env.pre-rekey" in err
    assert OLD not in err and new_key not in err


# --- start.sh wiring ---


def test_start_sh_removes_the_exported_old_key_before_the_guided_tool():
    text = (REPO / "start.sh").read_text()
    assert (
        "exec env -u ENCRYPTION_KEY -u OLD_ENCRYPTION_KEY python3 -m app.admin.rekey_guided"
        in text
    )
    assert '--rekey) MODE="rekey" ;;' in text


def test_the_rekey_mode_never_reaches_the_model_server_start():
    text = (REPO / "start.sh").read_text()
    assert text.index('if [ "$MODE" = "rekey" ]') < text.index("# --- 2. Native llama-server")
