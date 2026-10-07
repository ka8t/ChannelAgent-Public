"""Tests: the application does not run as root in the container.

File and code checks only (no Docker). The real runs (`id`, fresh volume,
existing data, root-owned directory) are recorded in the issue.
"""

import os
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text()
LINES = [line.strip() for line in DOCKERFILE.splitlines() if not line.strip().startswith("#")]


def test_the_image_switches_to_the_fixed_unprivileged_user():
    users = [line for line in LINES if line.startswith("USER ")]
    assert users == ["USER 10001:10001"]


def test_the_user_is_created_with_the_same_fixed_uid_and_gid():
    text = " ".join(LINES)
    assert re.search(r"groupadd --system --gid 10001 channelagent", text)
    assert re.search(r"useradd --system --uid 10001 --gid 10001", text)


def test_the_data_directory_is_given_to_that_user():
    assert "RUN mkdir -p /app/data && chown 10001:10001 /app/data" in LINES


def test_the_user_directive_comes_before_the_command_and_after_the_setup():
    index = {line: i for i, line in enumerate(LINES)}
    user = index["USER 10001:10001"]
    assert user > index["RUN mkdir -p /app/data && chown 10001:10001 /app/data"]
    assert user > index["RUN pip install --no-cache-dir -r requirements.txt"]
    assert user < index['CMD ["python", "-m", "app.main"]']


def test_the_code_is_not_given_to_the_application_user():
    assert not [line for line in LINES if line.startswith("COPY --chown")]
    assert not [line for line in LINES if "chown -R" in line], "only the data directory is chowned"


def test_compose_does_not_override_the_user_back_to_root():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    assert "user" not in compose["services"]["channelagent"]


def test_ci_asserts_the_uid_is_not_zero():
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert 'docker run --rm --entrypoint id channelagent:ci -u' in ci
    assert 'test "$uid" != 0' in ci


# --- a data directory the application cannot write gets a clear message ---


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_an_unwritable_data_directory_names_the_fix(tmp_path):
    from app.db.session import _ensure_sqlite_dir_exists

    data = tmp_path / "data"
    data.mkdir()
    data.chmod(0o500)
    try:
        with pytest.raises(PermissionError) as error:
            _ensure_sqlite_dir_exists(f"sqlite+aiosqlite:///{data}/channelagent.db")
    finally:
        data.chmod(0o700)
    message = str(error.value)
    assert str(data) in message and f"uid {os.getuid()}" in message
    assert "chown -R 10001:10001" in message


def test_a_writable_or_missing_data_directory_is_fine(tmp_path):
    from app.db.session import _ensure_sqlite_dir_exists

    _ensure_sqlite_dir_exists(f"sqlite+aiosqlite:///{tmp_path}/new/channelagent.db")
    assert (tmp_path / "new").is_dir()
    _ensure_sqlite_dir_exists(f"sqlite+aiosqlite:///{tmp_path}/new/channelagent.db")
    _ensure_sqlite_dir_exists("postgresql+asyncpg://u:p@h/db")


def test_start_sh_creates_data_before_compose_and_warns_on_linux():
    """A missing ./data would be created by Docker owned by root, and the
    uid 10001 application could not write to it (found by the TLS rehearsal).
    """
    lines = [line.strip() for line in (REPO_ROOT / "start.sh").read_text().splitlines()]
    mkdir_at = lines.index("mkdir -p data")
    starts = [i for i, line in enumerate(lines) if line.startswith("exec docker compose up")]
    assert len(starts) == 2 and all(mkdir_at < i for i in starts), starts  # detached or not
    text = "\n".join(lines)
    assert "sudo chown -R 10001:10001 data" in text and '"$(uname -s)" = "Linux"' in text
