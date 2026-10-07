"""Tests: ./start.sh --show-config and --set. Every run is in a
temporary directory holding a *copy* of start.sh, because the script resolves
its own directory from its path: running the real one from elsewhere would
still modify the real .env (this happened once).
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


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


def _example_keys() -> list[str]:
    keys = []
    for line in (REPO / ".env.example").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.append(line.partition("=")[0])
    return keys


def _make_env(sandbox: Path) -> None:
    """The .env a configuration leaves: --show-config and --set no longer create one."""
    (sandbox / ".env").write_text((sandbox / ".env.example").read_text())
    (sandbox / ".env").chmod(0o600)


def _env(sandbox: Path) -> dict[str, str]:
    values = {}
    for line in (sandbox / ".env").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key] = value
    return values


def test_show_config_without_env_lists_every_variable_and_creates_nothing(sandbox):
    result = _run(sandbox, "--config")
    assert result.returncode == 0, result.stderr
    assert not (sandbox / ".env").exists(), "Reading creates nothing"
    shown = [line.split()[0] for line in result.stdout.splitlines() if line.startswith("  ")]
    assert shown == _example_keys()
    assert len(shown) >= 25


@pytest.mark.parametrize(
    "key",
    [
        "ENCRYPTION_KEY",
        "API_SERVER_KEY",
        "TELEGRAM_BOT_TOKEN",
        "EMAIL_PASSWORD",
        "MATRIX_BOT_ACCESS_TOKEN",
    ],
)
def test_show_config_never_prints_a_secret_in_full(sandbox, key):
    # Values that --set accepts for each variable.
    secrets = {
        "ENCRYPTION_KEY": "SECR" + "A" * 39 + "=",
        "API_SERVER_KEY": "SECRk3Jf9Zq2Lm8Xw5Tn7Vb4Hd6Gs1Yc",
        "TELEGRAM_BOT_TOKEN": "1234567890:ABCdefGhiJklMnoPqrStuVwxYz012345",
    }
    secret = secrets.get(key, "SECRETVALUE-0123456789-abcdef")
    (sandbox / ".env").write_text((sandbox / ".env.example").read_text())
    assert _run(sandbox, "--config", f"{key}={secret}").returncode == 0
    out = _run(sandbox, "--config").stdout
    assert secret not in out, f"{key} is printed in clear"
    assert f"{secret[:4]}...(hidden)" in [
        w for line in out.splitlines() if key in line for w in line.split()
    ]


def test_show_config_shows_ordinary_values_and_marks_missing_ones(sandbox):
    _make_env(sandbox)
    _run(sandbox, "--config", "LLAMA_PORT=9999")
    out = _run(sandbox, "--config").stdout
    port_line = next(line for line in out.splitlines() if "LLAMA_PORT" in line)
    assert port_line.split() == ["LLAMA_PORT", "9999"]
    assert "(not set)" in next(line for line in out.splitlines() if "MODEL_FILE" in line)


def test_set_updates_an_existing_key_and_keeps_everything_else(sandbox):
    _make_env(sandbox)
    before = (sandbox / ".env").read_text().splitlines()
    result = _run(sandbox, "--config", "LLAMA_PORT=8123")
    assert result.returncode == 0
    after = (sandbox / ".env").read_text().splitlines()
    assert len(after) == len(before)
    changed = [(a, b) for a, b in zip(before, after, strict=True) if a != b]
    assert changed == [(next(a for a in before if a.startswith("LLAMA_PORT=")), "LLAMA_PORT=8123")]


def test_set_keeps_brackets_and_equals_signs_in_the_value(sandbox):
    _make_env(sandbox)
    assert _run(sandbox, "--config", "EMAIL_TRIGGER_TAG=[agent]a=b").returncode == 0
    assert _env(sandbox)["EMAIL_TRIGGER_TAG"] == "[agent]a=b"


def test_set_without_env_creates_nothing_and_points_to_configure(sandbox):
    result = _run(sandbox, "--config", "LLAMA_PORT=8123")
    assert result.returncode == 1 and "--config --edit" in result.stderr
    assert not (sandbox / ".env").exists(), "Creating .env is the configuration's job"


def test_set_works_on_a_file_without_a_trailing_newline(sandbox):
    (sandbox / ".env").write_text("A=1")
    assert _run(sandbox, "--config", "LLAMA_PORT=7").returncode == 0
    assert (sandbox / ".env").read_text() == "A=1\nLLAMA_PORT=7\n"


def test_set_without_key_equals_value_prints_usage_and_changes_nothing(sandbox):
    # `--config` alone shows the configuration: only a word without "=" is a usage error.
    _make_env(sandbox)
    before = (sandbox / ".env").read_text()
    result = _run(sandbox, "--config", "NOEQUALSIGN")
    assert result.returncode == 1 and "Usage: ./start.sh --config KEY=VALUE" in result.stderr
    assert (sandbox / ".env").read_text() == before


@pytest.mark.parametrize("argument", ["=value", "lower_case=1", "BAD KEY=1", "LLAMA-PORT=1"])
def test_set_refuses_an_invalid_variable_name_and_changes_nothing(sandbox, argument):
    _make_env(sandbox)
    before = (sandbox / ".env").read_text()
    result = _run(sandbox, "--config", argument)
    assert result.returncode == 1 and "not a valid variable name" in result.stderr
    assert (sandbox / ".env").read_text() == before


def test_set_refuses_a_key_the_application_does_not_know(sandbox):
    _make_env(sandbox)
    before = (sandbox / ".env").read_text()
    result = _run(sandbox, "--config", "LLAMA_PROT=9999")  # a typo of LLAMA_PORT
    assert result.returncode == 1 and "unknown variable" in result.stderr
    assert "LLAMA_PORT" in result.stderr, "it suggests the known names"
    assert (sandbox / ".env").read_text() == before


# --- a machine that is only a client of another machine's API ---


def _fake_venv(sandbox: Path) -> Path:
    """A .venv whose pip does nothing and whose python3 records the call: the client
    command start.sh hands over to, without installing anything."""
    bin_dir = sandbox / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    calls = sandbox / "client_calls"
    (bin_dir / "activate").write_text(f'export PATH="{bin_dir}:$PATH"\n')
    (bin_dir / "pip").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "python3").write_text(f'#!/bin/sh\necho "$@" >> "{calls}"\n')
    for name in ("pip", "python3"):
        (bin_dir / name).chmod(0o755)
    return calls


def _client_env(sandbox: Path, api_url: str) -> None:
    (sandbox / ".env").write_text(f"API_URL='{api_url}'\nAPI_SERVER_KEY='{'k' * 32}'\n")
    (sandbox / ".env").chmod(0o600)


@pytest.mark.parametrize("args", [("--admin", "whoami"), ("--admin", "describe")])
def test_a_client_of_a_remote_api_needs_no_encryption_key(sandbox, args):
    calls = _fake_venv(sandbox)
    _client_env(sandbox, "https://admin.example.org")
    result = _run(sandbox, *args)
    assert result.returncode == 0, result.stderr
    assert "ENCRYPTION_KEY" not in result.stderr
    assert calls.read_text().startswith("-m app.admin.client")


def test_without_api_url_the_encryption_key_is_still_required(sandbox):
    calls = _fake_venv(sandbox)
    (sandbox / ".env").write_text(f"API_SERVER_KEY='{'k' * 32}'\n")
    (sandbox / ".env").chmod(0o600)
    result = _run(sandbox, "--admin", "whoami")
    assert result.returncode == 1
    assert "ENCRYPTION_KEY is missing" in result.stderr
    assert not calls.exists()
