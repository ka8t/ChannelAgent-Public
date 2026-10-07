"""The repository must not depend on, or mention, any other repository.

Run time: the defaults for the llama-server bundle and the model directory point
inside this repository. Text: no tracked file names the earlier project.
"""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# Built from two parts so this file does not contain the name it forbids.
EARLIER_PROJECT = "her" + "mes"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture(scope="module")
def tracked_files() -> list[str]:
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    return _git("ls-files", "-z").split("\0")[:-1]


def test_no_tracked_file_names_the_earlier_project(tracked_files):
    offenders = []
    for name in tracked_files:
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        if EARLIER_PROJECT in text.lower() or EARLIER_PROJECT in name.lower():
            offenders.append(name)
    assert offenders == []


def _example_value(key: str) -> str:
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"{key} missing from .env.example")


@pytest.mark.parametrize("key", ["LLAMA_SERVER_BIN", "MODELS_DIR"])
def test_example_paths_stay_inside_the_repository(key):
    value = _example_value(key)
    assert value.startswith("./") and ".." not in Path(value).parts, value


@pytest.mark.parametrize("ignored", ["models/x.gguf", "vendor/llama.cpp/llama-server"])
def test_local_assets_are_git_ignored(ignored):
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    result = subprocess.run(["git", "check-ignore", "-q", ignored], cwd=ROOT)
    assert result.returncode == 0


@pytest.mark.parametrize("line", ["models/", "vendor/"])
def test_local_assets_stay_out_of_the_docker_build_context(line):
    assert line in (ROOT / ".dockerignore").read_text().splitlines()
