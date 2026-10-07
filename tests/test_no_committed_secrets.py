"""A Fernet key must never be committed (found on 2026-09-20: the real
ENCRYPTION_KEY had been used as the test default in tests/conftest.py since
its first commit).

The only key allowed in a tracked file is the throwaway one in
tests/conftest.py. Anything else shaped like a Fernet key (43 url-safe base64
characters and a "=") fails this test.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FERNET_SHAPE = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{43}=(?![A-Za-z0-9_-])")
ALLOWED = {"tests/conftest.py", ".gitleaks.toml"}  # the throwaway key, and its allow-list


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout")
    return result.stdout.splitlines()


def test_no_fernet_shaped_key_in_any_tracked_file_except_the_test_conftest():
    offenders = []
    for name in _tracked_files():
        if name in ALLOWED:
            continue
        try:
            text = (REPO / name).read_text(errors="ignore")
        except (IsADirectoryError, FileNotFoundError):
            continue
        if FERNET_SHAPE.search(text):
            offenders.append(name)
    assert offenders == [], f"a key-shaped value is committed in: {offenders}"


def test_the_test_key_is_the_only_one_and_is_a_valid_fernet_key():
    from cryptography.fernet import Fernet

    found = FERNET_SHAPE.findall((REPO / "tests/conftest.py").read_text())
    assert len(found) == 1
    Fernet(found[0].encode())  # raises if it is not a valid key


def test_the_env_example_and_config_hold_no_key():
    for name in (".env.example", "app/config.py", "docker-compose.yml", "Dockerfile"):
        assert not FERNET_SHAPE.search((REPO / name).read_text()), name


# --- the CI job that starts the image ---


def _ci_boot_step() -> str:
    text = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    match = re.search(r"API_KEY=(.*)", text)
    assert match, "the boot step defines API_KEY"
    return match.group(1).strip()


def test_ci_starts_the_image_with_a_generated_api_key_that_the_app_accepts():
    """A literal key in the workflow is a secret-shaped string in the repository
    and is refused if it looks like a placeholder. The job generates
    one per run instead.
    """
    assert _ci_boot_step() == "$(openssl rand -hex 32)"
    import secrets

    from app.api.deps import api_key_is_acceptable

    assert api_key_is_acceptable(secrets.token_hex(32))


def test_the_old_literal_ci_key_is_what_the_app_now_refuses():
    from app.api.deps import api_key_is_acceptable

    assert api_key_is_acceptable("ci-key-0123456789abcdef") is False


def test_gitleaks_only_allows_exact_values_never_patterns_or_paths():
    import re
    import tomllib

    config = tomllib.loads((REPO / ".gitleaks.toml").read_text())
    allow = config["allowlist"]
    assert "paths" not in allow and "commits" not in allow
    for pattern in allow["regexes"]:
        assert re.fullmatch(r"[A-Za-z0-9_=\-]{16,}", pattern), (
            f"{pattern!r} must be one exact literal value, not a pattern"
        )


def test_the_throwaway_api_key_of_the_tests_is_an_allowed_exact_value():
    import tomllib

    allowed = tomllib.loads((REPO / ".gitleaks.toml").read_text())["allowlist"]["regexes"]
    assert "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA" in allowed
