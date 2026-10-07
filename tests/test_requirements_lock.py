"""Tests: the dependencies are locked, so the same commit builds the
same image, and the lock is what the image, CI and start.sh install.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PIN = re.compile(r"^([A-Za-z0-9_.\-]+)(\[[^\]]+\])?==([^\s;#]+)")


def _lines(path: str) -> list[str]:
    return [
        ln.strip()
        for ln in (REPO / path).read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def _name(requirement: str) -> str:
    base = re.split(r"[\[<>=!~ ;]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", base).lower()


def _pins(path: str) -> dict[str, str]:
    pins = {}
    for ln in _lines(path):
        if ln.startswith("-"):
            continue
        m = PIN.match(ln)
        assert m, f"{path}: not an exact pin: {ln!r}"
        pins[_name(ln)] = m.group(3)
    return pins


def test_every_runtime_dependency_is_pinned_to_an_exact_version():
    pins = _pins("requirements.txt")
    assert len(pins) >= 15


def test_every_direct_dependency_is_in_the_lock():
    direct = {_name(ln) for ln in _lines("requirements.in")}
    assert direct, "requirements.in lists the direct dependencies"
    assert direct <= set(_pins("requirements.txt"))


def test_the_direct_dependencies_are_the_ones_the_application_uses():
    direct = {_name(ln) for ln in _lines("requirements.in")}
    assert {"fastapi", "sqlalchemy", "alembic", "cryptography", "langgraph"} <= direct
    assert not any(re.search(r"[=<>~!]", ln) for ln in _lines("requirements.in")), (
        "requirements.in is loose on purpose: the versions live in requirements.txt"
    )


def test_the_dev_lock_is_pinned_and_contains_the_runtime_lock():
    dev = _pins("requirements-dev.txt")
    runtime = _pins("requirements.txt")
    assert {"pytest", "pytest-asyncio", "ruff"} <= set(dev)
    assert all(dev[name] == version for name, version in runtime.items())


def test_the_image_ci_and_start_script_install_the_lock():
    assert "pip install --no-cache-dir -r requirements.txt" in (REPO / "Dockerfile").read_text()
    assert "pip install -r requirements-dev.txt" in (
        REPO / ".github" / "workflows" / "ci.yml"
    ).read_text()
    assert "pip install --quiet -r requirements.txt" in (REPO / "start.sh").read_text()


def test_the_lock_is_generated_for_the_python_of_the_image():
    dockerfile = (REPO / "Dockerfile").read_text()
    image_python = re.search(r"FROM python:(\d+\.\d+)", dockerfile).group(1)
    header = (REPO / "requirements.txt").read_text()
    assert f"python{image_python}" in header.replace(" ", "").lower() or (
        f"Python {image_python}" in header
    )
    script = (REPO / "scripts" / "update_requirements.sh").read_text()
    runs = [ln for ln in script.splitlines() if ln.startswith("docker run")]
    assert runs and all(f"python:{image_python}" in ln for ln in runs)


def test_dependabot_proposes_weekly_grouped_updates():
    config = (REPO / ".github" / "dependabot.yml").read_text()
    assert "package-ecosystem: pip" in config
    ecosystems = config.count("package-ecosystem:")
    assert ecosystems == config.count("interval: weekly") >= 1, "every ecosystem is weekly"
    assert "groups:" in config


def test_ci_audits_the_lock_and_fails_on_a_known_vulnerability():
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert re.search(r"^\s+run: pip-audit -r requirements\.txt\b", ci, re.MULTILINE)
    assert "--ignore-vuln" not in ci.split("run: pip-audit", 1)[1].splitlines()[0], (
        "no exception is accepted silently: add it with a comment, then update this test"
    )
