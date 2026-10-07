"""Tests: every start.sh command refuses an argument it does not take, before the
.env step, the virtualenv and the engine; `--describe`, `--host-helper` and `--models` are
gone and `--show-config`, `--set`, `--configure` became `--config` (each removed name says what
replaced it); `--config` and `--config KEY=VALUE` never create `.env`.

Every run is in a temporary directory holding a copy of start.sh and the two modules the
configuration commands need. Nothing there can install or start anything: a command that got
past its argument check would show it (a `.venv`, "Installing dependencies", a `.env`).
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def box(tmp_path):
    (tmp_path / "app").mkdir()
    shutil.copy(REPO / "start.sh", tmp_path / "start.sh")
    shutil.copy(REPO / ".env.example", tmp_path / ".env.example")
    for name in ("settings_rules.py", "env_wizard.py"):
        shutil.copy(REPO / "app" / name, tmp_path / "app" / name)
    return tmp_path


def _run(box: Path, *args: str):
    return subprocess.run(
        ["bash", "start.sh", *args], cwd=box, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=60,
    )


def _nothing_happened(box: Path, result) -> None:
    assert not (box / ".env").exists(), "the .env step was reached"
    assert not (box / ".venv").exists(), "the virtualenv step was reached"
    assert "Installing dependencies" not in result.stdout + result.stderr
    assert "ChannelAgent start.sh (mode:" not in result.stdout, "the start was reached"


REFUSED = [
    (("--native", "--bogus"), 2),
    (("--native", "--detach", "--bogus"), 2),
    (("--status", "--bogus"), 2),
    (("--stop", "--all", "--bogus"), 2),
    (("--config", "--bogus"), 2),
    (("--config", "LLAMA_PORT=8181", "EXTRA=1"), 2),
    (("--config", "--edit", "--bogus"), 2),
    (("--admin", "--bogus"), 2),
    (("--rekey", "--bogus"), 2),
    (("--restore", "--bogus"), 2),
    (("--restore", "one.db", "two.db"), 2),
    (("--chat", "--bogus"), 2),
    (("--chat", "--agent"), 2),
    (("--config", "--edit", "x"), 2),
    (("--docker", "--bogus"), 2),
]


@pytest.mark.parametrize(("args", "code"), REFUSED, ids=[" ".join(a) for a, _ in REFUSED])
def test_an_argument_a_command_does_not_take_is_refused_before_anything(box, args, code):
    result = _run(box, *args)
    assert result.returncode == code, (result.stdout, result.stderr)
    assert "Usage" in result.stderr or "Unknown" in result.stderr
    _nothing_happened(box, result)


REMOVED = {
    "--describe": "./start.sh --admin describe",
    "--show-config": "./start.sh --config",
    "--set": "./start.sh --config KEY=VALUE",
    "--configure": "./start.sh --config --edit",
    "--models": "./start.sh --admin list-models",
    "--host-helper": "starts with the application",
    "--api": "./start.sh --admin COMMAND",  # One command for the menus and the calls
}


@pytest.mark.parametrize("name", sorted(REMOVED))
def test_a_removed_command_is_unknown_and_says_what_replaced_it(box, name):
    result = _run(box, name, "anything")
    assert result.returncode == 2
    assert f"Unknown command: {name}" in result.stderr
    assert REMOVED[name] in result.stderr
    _nothing_happened(box, result)


def test_the_help_names_only_the_current_commands(box):
    text = _run(box, "--help").stdout
    for name in REMOVED:
        assert f"  {name} " not in text and f"  {name}\n" not in text, name
    for line in ("  --config  ", "  --config KEY=VALUE", "  --config --edit", "  --admin  ",
                 "  --admin COMMAND"):
        assert line in text


def test_describe_is_unknown_and_the_help_points_to_admin_describe(box):
    result = _run(box, "--describe")
    assert "Unknown command: --describe" in result.stderr
    assert "--admin describe lists every command" in _run(box, "--help").stdout


def test_the_help_has_one_docker_line_and_no_removed_form(box):
    text = _run(box, "--help").stdout
    assert text.count("--docker [--detach]") == 1
    assert "--docker --detach " not in text
    assert "--describe" not in text and "--host-helper" not in text and "--models" not in text


def test_show_config_without_env_shows_the_defaults_and_creates_nothing(box):
    result = _run(box, "--config")
    assert result.returncode == 0, result.stderr
    assert "No .env yet: the defaults of .env.example are shown" in result.stderr
    assert "LLAMA_PORT" in result.stdout and "8080" in result.stdout
    assert not (box / ".env").exists()
    assert not list(box.glob(".env.bak*"))


def test_config_set_without_env_points_to_edit_and_creates_nothing(box):
    result = _run(box, "--config", "LLAMA_PORT=8181")
    assert result.returncode == 1
    assert "LLAMA_PORT was NOT set" in result.stderr and "--config --edit" in result.stderr
    assert not (box / ".env").exists()


def test_set_and_show_config_still_work_on_an_existing_env(box):
    shutil.copy(box / ".env.example", box / ".env")
    (box / ".env").chmod(0o600)
    assert _run(box, "--config", "LLAMA_PORT=8181").returncode == 0
    shown = _run(box, "--config")
    assert shown.returncode == 0 and "8181" in shown.stdout
