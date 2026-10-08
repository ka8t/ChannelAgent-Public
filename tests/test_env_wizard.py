"""Tests: the interactive configuration of .env (`app/env_wizard.py`) and when
`start.sh` runs it: only with --configure, or on a native start with no .env next to the
script, and never without a terminal.

The script runs are in a temporary directory holding a copy of start.sh and of the two
modules it needs, under a pseudo-terminal when a terminal is part of the case.
"""

import base64
import hashlib
import os
import pty
import select
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

from app import env_wizard, settings_rules

REPO = Path(__file__).resolve().parent.parent
TOKEN = "555666777:AAExistingTokenValueForTestsOnly_12345"
TYPED = "987654321:AATypedTokenValueForTestsOnly_67890"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _backups(directory: Path) -> list:
    return sorted(directory.glob(".env.bak*"))


def _questions(example: Path) -> int:
    return len(env_wizard.example_layout(str(example)))


class Answers:
    """Scripted answers: `ask` and `ask_secret` take the next one; prompts are recorded."""

    def __init__(self, *answers, default=""):
        self.answers = list(answers)
        self.default = default
        self.prompts = []
        self.secret_prompts = []
        self.out = []

    def _next(self):
        if not self.answers:
            return self.default
        answer = self.answers.pop(0)
        if answer is EOFError:
            raise EOFError
        return answer

    def ask(self, prompt):
        self.prompts.append(prompt)
        return self._next()

    def ask_secret(self, prompt):
        self.secret_prompts.append(prompt)
        return self._next()

    def run(self, env, example):
        return env_wizard.run(
            str(env), str(example), ask=self.ask, ask_secret=self.ask_secret,
            out=self.out.append,
        )


@pytest.fixture
def files(tmp_path):
    example = tmp_path / ".env.example"
    shutil.copy(REPO / ".env.example", example)
    return tmp_path / ".env", example


def _existing(env: Path, example: Path) -> None:
    key = base64.urlsafe_b64encode(os.urandom(32)).decode()
    lines = []
    for line in example.read_text().splitlines():
        if line.startswith("ENCRYPTION_KEY="):
            line = f"ENCRYPTION_KEY={key}"
        elif line.startswith("TELEGRAM_BOT_TOKEN="):
            line = f"TELEGRAM_BOT_TOKEN={TOKEN}"
        lines.append(line)
    env.write_text("\n".join(lines) + "\n")
    env.chmod(0o600)


def _answers_for(example: Path, by_key: dict, confirm="y") -> list:
    """One answer per question in the order of .env.example (Enter by default)."""
    keys = [key for _, _, key in env_wizard.example_layout(str(example))]
    return [by_key.get(key, "") for key in keys if key != "ENCRYPTION_KEY"] + [confirm]


# --- the questions and the write ---


def test_without_env_every_default_is_offered_and_a_private_env_is_created(files):
    env, example = files
    answers = Answers(default="")
    answers.answers = [""] * _questions(example) + ["y"]
    assert answers.run(env, example) == 0
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    values = settings_rules.read_env(str(env))
    defaults = settings_rules.read_env(str(example))
    key = values.pop("ENCRYPTION_KEY")
    assert len(base64.urlsafe_b64decode(key)) == 32, "a valid Fernet key was generated"
    assert settings_rules.validate("ENCRYPTION_KEY", key) is None
    defaults.pop("ENCRYPTION_KEY")
    assert values == defaults
    assert not _backups(env.parent), "nothing to back up when there was no .env"
    assert key not in "\n".join(answers.out), "the generated key is never printed"


def test_one_question_per_variable_in_the_order_of_the_example(files):
    env, example = files
    answers = Answers()
    answers.answers = [""] * _questions(example) + ["n"]
    answers.run(env, example)
    asked = [p.split(" [")[0] for p in answers.prompts[:-1] + answers.secret_prompts]
    assert sorted(asked) == sorted(settings_rules.example_keys(str(example)))
    assert len(asked) == 74 == len(settings_rules.example_keys(str(example)))


def test_changes_are_checked_written_once_and_the_previous_env_kept(files):
    env, example = files
    _existing(env, example)
    previous = env.read_bytes()
    answers = Answers(*_answers_for(example, {"LLAMA_PORT": "70000", "EMAIL_PASSWORD": "pw-1"}))
    # 70000 is refused: the question comes back and takes the next answer.
    i = answers.answers.index("70000")
    answers.answers.insert(i + 1, "9191")
    assert answers.run(env, example) == 0
    values = settings_rules.read_env(str(env))
    assert values["LLAMA_PORT"] == "9191" and values["EMAIL_PASSWORD"] == "pw-1"
    assert values["TELEGRAM_BOT_TOKEN"] == TOKEN, "Enter keeps a secret"
    (backup,) = _backups(env.parent)
    assert backup.read_bytes() == previous
    assert any("LLAMA_PORT was NOT changed" in line for line in answers.out)
    assert not any("70000" in line for line in answers.out), "a refused value is not echoed"


def test_a_dash_empties_a_value(files):
    env, example = files
    _existing(env, example)
    answers = Answers(*_answers_for(example, {"EMAIL_AGENT_FOLDER": "-"}))
    assert answers.run(env, example) == 0
    assert settings_rules.read_env(str(env))["EMAIL_AGENT_FOLDER"] == ""


def test_secrets_are_read_without_echo_and_never_printed(files):
    env, example = files
    _existing(env, example)
    answers = Answers(*_answers_for(example, {"TELEGRAM_BOT_TOKEN": TYPED}))
    assert answers.run(env, example) == 0
    secret_keys = {p.split(" [")[0] for p in answers.secret_prompts}
    known = set(settings_rules.example_keys(str(example)))
    assert secret_keys == (settings_rules.SENSITIVE & known) - {"ENCRYPTION_KEY"}
    printed = "\n".join(answers.out + answers.prompts + answers.secret_prompts)
    assert TOKEN not in printed and TYPED not in printed and TOKEN[:4] not in printed
    assert "TELEGRAM_BOT_TOKEN (secret, not shown)" in printed
    assert settings_rules.read_env(str(env))["TELEGRAM_BOT_TOKEN"] == TYPED


def test_an_existing_encryption_key_is_never_asked_nor_changed(files):
    env, example = files
    _existing(env, example)
    key = settings_rules.read_env(str(env))["ENCRYPTION_KEY"]
    answers = Answers(*_answers_for(example, {"LLAMA_PORT": "9191"}))
    assert answers.run(env, example) == 0
    assert not any(p.startswith("ENCRYPTION_KEY") for p in answers.secret_prompts)
    assert settings_rules.read_env(str(env))["ENCRYPTION_KEY"] == key


def test_nothing_changed_writes_nothing(files):
    env, example = files
    _existing(env, example)
    before = _sha(env)
    answers = Answers(default="")
    assert answers.run(env, example) == 0
    assert _sha(env) == before and not _backups(env.parent)
    assert not any("[y/N]" in p for p in answers.prompts), "no confirmation without a change"


@pytest.mark.parametrize("stop", ["n", EOFError])
def test_declining_or_stopping_writes_nothing(files, stop):
    env, example = files
    _existing(env, example)
    before = _sha(env)
    answers = _answers_for(example, {"LLAMA_PORT": "9191"}, confirm=stop)
    if stop is EOFError:
        answers = answers[:3] + [EOFError]
    assert Answers(*answers).run(env, example) == 1
    assert _sha(env) == before and not _backups(env.parent)


def test_stopping_without_env_creates_nothing(files):
    env, example = files
    assert Answers("", "", EOFError).run(env, example) == 1
    assert not env.exists()


# --- when start.sh runs it ---


@pytest.fixture
def sandbox(tmp_path):
    box = tmp_path / "box"
    (box / "app").mkdir(parents=True)
    shutil.copy(REPO / "start.sh", box / "start.sh")
    shutil.copy(REPO / ".env.example", box / ".env.example")
    for name in ("settings_rules.py", "env_wizard.py"):
        shutil.copy(REPO / "app" / name, box / "app" / name)
    return box


def _no_key_env(box: Path) -> None:
    """A .env whose empty key makes a native start stop right after the .env step."""
    shutil.copy(box / ".env.example", box / ".env")
    (box / ".env").chmod(0o600)


def _run_tty(box: Path, *args: str, feed: str = "", cwd: Path | None = None):
    """start.sh under a pseudo-terminal, `feed` typed in advance. Returns (exit code, output)."""
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        ["bash", str(box / "start.sh"), *args],
        cwd=cwd or box, stdin=slave, stdout=slave, stderr=slave, start_new_session=True,
    )
    os.close(slave)
    os.write(master, feed.encode())
    chunks, end = [], time.time() + 30
    while time.time() < end:
        ready, _, _ = select.select([master], [], [], 1)
        if not ready:
            if proc.poll() is not None:
                break
            continue
        try:
            data = os.read(master, 65536)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    else:  # still waiting for an answer: a question nobody expected
        proc.kill()
    os.close(master)
    return proc.wait(timeout=30), b"".join(chunks).decode(errors="replace")


def _run_plain(box: Path, *args: str):
    return subprocess.run(
        ["bash", "start.sh", *args], cwd=box, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=60,
    )


ASKED = "variables, the"  # the wizard's first line, with or without a .env


def test_configure_in_a_terminal_without_env_creates_it(sandbox):
    feed = "\n" * _questions(sandbox / ".env.example") + "y\n"
    code, out = _run_tty(sandbox, "--config", "--edit", feed=feed)
    assert code == 0, out
    assert ASKED in out and "Created .env (mode 600)" in out
    assert stat.S_IMODE((sandbox / ".env").stat().st_mode) == 0o600
    assert settings_rules.read_env(str(sandbox / ".env"))["ENCRYPTION_KEY"]


def test_configure_with_env_asks_and_keeps_everything_on_enter(sandbox):
    _no_key_env(sandbox)
    before = _sha(sandbox / ".env")
    # The empty key is asked too: '-' keeps it empty.
    feed = "-\n" + "\n" * _questions(sandbox / ".env.example")
    code, out = _run_tty(sandbox, "--config", "--edit", feed=feed)
    assert code == 0, out
    assert ASKED in out and "Nothing changed" in out
    assert _sha(sandbox / ".env") == before and not _backups(sandbox)


def test_configure_without_a_terminal_asks_nothing(sandbox):
    result = _run_plain(sandbox, "--config", "--edit")
    assert result.returncode == 2
    assert "needs a terminal" in result.stderr and ASKED not in result.stdout
    assert not (sandbox / ".env").exists()


def test_native_with_env_asks_nothing(sandbox):
    _no_key_env(sandbox)
    before = _sha(sandbox / ".env")
    code, out = _run_tty(sandbox, "--native")
    assert code == 1 and "ENCRYPTION_KEY is missing" in out, out
    assert ASKED not in out
    assert _sha(sandbox / ".env") == before


def test_native_without_env_asks_and_stopping_starts_nothing(sandbox):
    code, out = _run_tty(sandbox, "--native", feed="\x04")
    assert code == 1, out
    assert ASKED in out and "Stopped: nothing was written" in out
    assert "the application was not started" in out
    assert not (sandbox / ".env").exists()


def test_native_without_a_terminal_copies_the_example_as_before(sandbox):
    result = _run_plain(sandbox, "--native")
    assert ASKED not in result.stdout
    assert "--config --edit" in result.stderr
    assert (sandbox / ".env").read_bytes() == (sandbox / ".env.example").read_bytes()


def test_other_commands_without_env_ask_nothing(sandbox):
    code, out = _run_tty(sandbox, "--config")
    assert code == 0 and ASKED not in out, out


def test_launched_from_another_directory_the_env_next_to_the_script_counts(sandbox, tmp_path):
    """start.sh and .env in the sandbox, run from a directory without .env: 0 question,
    0 file written in that directory."""
    _no_key_env(sandbox)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    code, out = _run_tty(sandbox, "--native", cwd=elsewhere)
    assert code == 1 and "ENCRYPTION_KEY is missing" in out, out
    assert ASKED not in out
    assert list(elsewhere.iterdir()) == []
