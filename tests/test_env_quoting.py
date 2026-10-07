"""Tests: a value written by `start.sh --set` means the same thing to
the shell (`source .env`), to docker compose, to python-dotenv and to the
application, and is never executed. Values that cannot be represented the same
way by all of them are refused.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from app import settings_rules as rules

REPO = Path(__file__).resolve().parent.parent
SPECIALS = list('&;|<>()$!#*?{} `"~')
COMBINED = 'a b&c;d|e<f>g(h)i$j!k#l*m?n{o}p`q"r~s'


@pytest.fixture
def sandbox(tmp_path):
    shutil.copy(REPO / "start.sh", tmp_path / "start.sh")
    shutil.copy(REPO / ".env.example", tmp_path / ".env.example")
    (tmp_path / "app").mkdir()
    shutil.copy(REPO / "app" / "settings_rules.py", tmp_path / "app" / "settings_rules.py")
    (tmp_path / ".env").write_text((tmp_path / ".env.example").read_text())
    (tmp_path / ".env").chmod(0o600)
    return tmp_path


def _run(sandbox: Path, *args: str):
    return subprocess.run(
        ["bash", "start.sh", *args], cwd=sandbox, capture_output=True, text=True, timeout=60
    )


def _line(sandbox: Path, key: str) -> str:
    return next(
        ln for ln in (sandbox / ".env").read_text().splitlines() if ln.startswith(f"{key}=")
    )


def _bash_reads(sandbox: Path, key: str) -> str:
    out = subprocess.run(
        ["bash", "-c", f'set -a; source ./.env; set +a; printf %s "${key}"'],
        cwd=sandbox,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def _dotenv_reads(sandbox: Path, key: str) -> str:
    from dotenv import dotenv_values

    return dotenv_values(sandbox / ".env")[key]


def _settings_reads(sandbox: Path) -> str:
    from app.config import Settings

    return Settings(_env_file=str(sandbox / ".env"), ENCRYPTION_KEY="x" * 44).email_password


def _compose_reads(sandbox: Path, key: str) -> tuple[str, str]:
    """The value through env_file and through ${...} interpolation."""
    (sandbox / "compose.yml").write_text(
        "services:\n  x:\n    image: alpine\n    env_file: .env\n    environment:\n"
        f'      VIA_INTERPOLATION: "${{{key}}}"\n'
    )
    out = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            ".env",
            "-f",
            "compose.yml",
            "config",
            "--format",
            "json",
        ],
        cwd=sandbox,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    env = json.loads(out.stdout)["services"]["x"]["environment"]
    # `config` prints a literal "$" as "$$" (escaping of its output only).
    return env[key].replace("$$", "$"), env["VIA_INTERPOLATION"].replace("$$", "$")


needs_compose = pytest.mark.skipif(
    subprocess.run(["docker", "compose", "version"], capture_output=True).returncode != 0
    if shutil.which("docker")
    else True,
    reason="docker compose is not available",
)


# --- the rules module ---


def test_a_value_of_safe_characters_is_written_as_it_is():
    for value in ("8081", "https://host:8080/x", "a=b", "[agent]", "INBOX.Agent", "u@example.org"):
        assert rules.format_value(value) == value


@pytest.mark.parametrize("char", SPECIALS)
def test_a_value_with_a_special_character_is_accepted_and_quoted(char):
    value = f"ab{char}cd"
    assert rules.validate("EMAIL_PASSWORD", value) is None
    assert rules.format_value(value) == f"'{value}'"


def test_an_empty_value_stays_empty():
    assert rules.format_value("") == ""


@pytest.mark.parametrize("value", ["it's", "back\\slash", "a${HOME}b", "${X}", "new\nline", "a\tb"])
def test_a_value_the_readers_would_disagree_on_is_refused(value):
    reason = rules.validate("EMAIL_PASSWORD", value)
    assert reason and value not in reason


def test_a_dollar_without_a_brace_is_fine():
    assert rules.validate("EMAIL_PASSWORD", "a$b$$c$") is None


def test_the_tilde_is_quoted_so_the_shell_does_not_expand_it():
    assert rules.format_value("~/models") == "'~/models'"


# --- start.sh --set writes it quoted, and the readers agree ---


@pytest.mark.parametrize("char", SPECIALS)
def test_each_special_character_reads_back_identically_in_the_shell_and_dotenv(sandbox, char):
    value = f"ab{char}cd"
    assert _run(sandbox, "--config", f"EMAIL_PASSWORD={value}").returncode == 0
    assert _line(sandbox, "EMAIL_PASSWORD") == f"EMAIL_PASSWORD='{value}'"
    assert _bash_reads(sandbox, "EMAIL_PASSWORD") == value
    assert _dotenv_reads(sandbox, "EMAIL_PASSWORD") == value


def test_every_reader_gets_the_same_string_for_a_value_with_all_of_them(sandbox):
    assert _run(sandbox, "--config", f"EMAIL_PASSWORD={COMBINED}").returncode == 0
    assert _bash_reads(sandbox, "EMAIL_PASSWORD") == COMBINED
    assert _dotenv_reads(sandbox, "EMAIL_PASSWORD") == COMBINED
    assert _settings_reads(sandbox) == COMBINED


@needs_compose
def test_docker_compose_reads_the_same_string_through_env_file_and_interpolation(sandbox):
    assert _run(sandbox, "--config", f"EMAIL_PASSWORD={COMBINED}").returncode == 0
    via_env_file, via_interpolation = _compose_reads(sandbox, "EMAIL_PASSWORD")
    assert via_env_file == COMBINED
    assert via_interpolation == COMBINED


def test_a_command_in_the_value_is_data_and_is_never_executed(sandbox):
    marker = sandbox / "pwned"
    value = f"ab&touch {marker}"
    assert _run(sandbox, "--config", f"EMAIL_PASSWORD={value}").returncode == 0
    assert _bash_reads(sandbox, "EMAIL_PASSWORD") == value
    assert not marker.exists()
    for payload in (f"$(touch {marker})", f"`touch {marker}`", f"x;touch {marker}"):
        assert _run(sandbox, "--config", f"EMAIL_PASSWORD={payload}").returncode == 0
        assert _bash_reads(sandbox, "EMAIL_PASSWORD") == payload
    assert not marker.exists()


def test_a_value_the_readers_would_disagree_on_leaves_env_unchanged(sandbox):
    before = (sandbox / ".env").read_bytes()
    for value in ("it's", "back\\slash", "a${HOME}b", "a\nb"):
        result = _run(sandbox, "--config", f"EMAIL_PASSWORD={value}")
        assert result.returncode != 0, value
        assert (sandbox / ".env").read_bytes() == before
        assert not list(sandbox.glob(".env.bak*"))
        assert value not in result.stdout + result.stderr


def test_plain_values_are_still_written_unquoted(sandbox):
    assert _run(sandbox, "--config", "LLAMA_PORT=8081").returncode == 0
    assert _line(sandbox, "LLAMA_PORT") == "LLAMA_PORT=8081"


def test_a_variable_that_is_not_in_env_yet_is_appended_quoted(sandbox):
    text = "\n".join(
        ln
        for ln in (sandbox / ".env").read_text().splitlines()
        if not ln.startswith("EMAIL_PASSWORD")
    )
    (sandbox / ".env").write_text(text + "\n")
    assert _run(sandbox, "--config", "EMAIL_PASSWORD=a b&c").returncode == 0
    assert (sandbox / ".env").read_text().splitlines()[-1] == "EMAIL_PASSWORD='a b&c'"
    assert _bash_reads(sandbox, "EMAIL_PASSWORD") == "a b&c"


def test_when_the_rules_cannot_be_loaded_the_message_says_so_and_nothing_is_written(sandbox):
    (sandbox / "app" / "settings_rules.py").write_text("raise ImportError('broken')\n")
    before = (sandbox / ".env").read_bytes()
    result = _run(sandbox, "--config", "LLAMA_PORT=8123")
    assert result.returncode != 0
    assert "could not be loaded" in result.stderr and "Traceback" not in result.stderr
    assert (sandbox / ".env").read_bytes() == before


def test_setting_a_variable_again_replaces_a_quoted_line(sandbox):
    _run(sandbox, "--config", "EMAIL_PASSWORD=a b")
    _run(sandbox, "--config", "EMAIL_PASSWORD=c d")
    lines = [
        ln for ln in (sandbox / ".env").read_text().splitlines() if ln.startswith("EMAIL_PASSWORD")
    ]
    assert lines == ["EMAIL_PASSWORD='c d'"]
    assert _bash_reads(sandbox, "EMAIL_PASSWORD") == "c d"


# --- --show-config unquotes ---


def test_show_config_shows_the_value_without_the_quotes(sandbox):
    _run(sandbox, "--config", "EMAIL_USERNAME=first last")
    out = _run(sandbox, "--config").stdout
    line = next(ln for ln in out.splitlines() if "EMAIL_USERNAME" in ln)
    assert "first last" in line and "'" not in line


def test_show_config_masks_the_unquoted_secret_and_never_shows_the_quotes(sandbox):
    _run(sandbox, "--config", "EMAIL_PASSWORD=Zq9 rest of it")
    out = _run(sandbox, "--config").stdout
    line = next(ln for ln in out.splitlines() if "EMAIL_PASSWORD" in ln)
    assert "Zq9 ...(hidden)" in line
    assert "rest of it" not in out and "'" not in line


def test_show_config_also_unquotes_a_line_the_owner_quoted_by_hand(sandbox):
    text = (sandbox / ".env").read_text().replace("LLAMA_PORT=8080", "LLAMA_PORT='8080'")
    (sandbox / ".env").write_text(text)
    out = _run(sandbox, "--config").stdout
    line = next(ln for ln in out.splitlines() if "LLAMA_PORT" in ln)
    assert line.split() == ["LLAMA_PORT", "8080"]


# --- the encryption key guard sees through quotes ---


def test_an_empty_quoted_key_counts_as_empty_and_a_quoted_key_as_set(sandbox):
    from cryptography.fernet import Fernet

    key = Fernet.generate_key().decode()
    text = (sandbox / ".env").read_text().replace("ENCRYPTION_KEY=", "ENCRYPTION_KEY=''", 1)
    (sandbox / ".env").write_text(text)
    assert _run(sandbox, "--config", f"ENCRYPTION_KEY={key}").returncode == 0
    (sandbox / ".env").write_text(
        (sandbox / ".env").read_text().replace(f"ENCRYPTION_KEY={key}", f"ENCRYPTION_KEY='{key}'")
    )
    before = (sandbox / ".env").read_bytes()
    other = Fernet.generate_key().decode()
    assert _run(sandbox, "--config", f"ENCRYPTION_KEY={other}").returncode != 0
    assert (sandbox / ".env").read_bytes() == before
