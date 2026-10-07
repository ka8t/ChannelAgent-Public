"""Remote exposure of the Admin API refused by default (S5): a non-loopback
API_SERVER_HOST (native) or API_BIND_ADDRESS (Docker) stops `start.sh` before anything starts,
and the application itself outside its container, unless API_REMOTE=tls-proxy declares a TLS
proxy in front. One rule, `app/settings_rules.py::api_exposure_problem`.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import settings_rules
from tests.test_start_status_stop import Box

REPO = Path(__file__).resolve().parents[1]
KEY = "k" * 32


@pytest.mark.parametrize("address", ["127.0.0.1", "127.0.0.2", "::1", "[::1]", "localhost"])
def test_loopback_is_always_allowed(address):
    assert settings_rules.api_exposure_problem("API_SERVER_HOST", address, "") is None


@pytest.mark.parametrize("address", ["0.0.0.0", "::", "192.168.1.20", "10.0.0.5", "my-host"])
def test_any_other_address_needs_a_tls_proxy(address):
    reason = settings_rules.api_exposure_problem("API_BIND_ADDRESS", address, "")
    assert reason and reason.startswith(f"API_BIND_ADDRESS={address} ") and "API_REMOTE" in reason
    assert settings_rules.api_exposure_problem("API_BIND_ADDRESS", address, "tls-proxy") is None


@pytest.mark.parametrize("value, ok", [("", True), ("tls-proxy", True), ("yes", False)])
def test_api_remote_takes_empty_or_tls_proxy(value, ok):
    assert (settings_rules.validate("API_REMOTE", value) is None) is ok


def _exposure(variable: str, **env) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "app/settings_rules.py"), "--exposure", variable],
        env={"PATH": os.environ["PATH"], **env},
        capture_output=True,
        text=True,
    )


def test_the_script_check_exits_1_with_the_reason():
    refused = _exposure("API_BIND_ADDRESS", API_BIND_ADDRESS="0.0.0.0")
    assert refused.returncode == 1 and "API_REMOTE=tls-proxy" in refused.stderr
    allowed = _exposure("API_BIND_ADDRESS", API_BIND_ADDRESS="0.0.0.0", API_REMOTE="tls-proxy")
    assert allowed.returncode == 0
    assert _exposure("API_SERVER_HOST").returncode == 0  # unset: the loopback default


def _box(tmp_path: Path, **values) -> Box:
    box = Box(tmp_path)
    extra = "".join(f"{k}={v}\n" for k, v in values.items())
    env = box.path / ".env"
    env.write_text(env.read_text() + extra)
    return box


def test_docker_start_refused_on_a_public_bind_address(tmp_path):
    box = _box(tmp_path, API_BIND_ADDRESS="0.0.0.0")
    result = box.run("--docker")
    assert result.returncode == 1, result.stderr
    assert "API_BIND_ADDRESS=0.0.0.0" in result.stderr
    log = box.docker_log.read_text() if box.docker_log.exists() else ""
    assert "compose" not in log  # nothing was started


def test_native_start_refused_on_a_public_host(tmp_path):
    box = _box(tmp_path, API_SERVER_HOST="0.0.0.0")
    result = box.run("--native", docker=False)
    assert result.returncode == 1 and "API_SERVER_HOST=0.0.0.0" in result.stderr
    assert not (box.path / ".venv").exists()  # stopped before the virtualenv


def test_docker_start_passes_the_check_with_a_tls_proxy(tmp_path):
    box = _box(tmp_path, API_BIND_ADDRESS="0.0.0.0", API_REMOTE="tls-proxy")
    box.start_llama()
    try:
        result = box.run("--docker")
    finally:
        for proc in box.procs:
            proc.kill()
    assert "API_BIND_ADDRESS=0.0.0.0" not in result.stderr
    assert "compose" in box.docker_log.read_text()


# --- the application ---


def _settings(host: str, remote: str = "", key: str = KEY):
    from types import SimpleNamespace

    return SimpleNamespace(api_server_key=key, api_server_host=host, api_remote=remote)


def test_the_application_refuses_a_public_host_outside_a_container(monkeypatch, tmp_path):
    from app import main

    monkeypatch.setattr(main, "CONTAINER_MARKERS", (tmp_path / "none",))
    assert main.exposure_refusal(_settings("127.0.0.1")) is None
    assert "API_SERVER_HOST=0.0.0.0" in main.exposure_refusal(_settings("0.0.0.0"))
    assert main.exposure_refusal(_settings("0.0.0.0", "tls-proxy")) is None
    assert main.exposure_refusal(_settings("0.0.0.0", key="")) is None  # no API at all


def test_inside_its_container_the_application_binds_every_interface(monkeypatch, tmp_path):
    from app import main

    marker = tmp_path / ".dockerenv"
    marker.touch()
    monkeypatch.setattr(main, "CONTAINER_MARKERS", (marker,))
    assert main.exposure_refusal(_settings("0.0.0.0")) is None


async def test_main_exits_2_before_anything_starts(monkeypatch, tmp_path):
    from app import main

    monkeypatch.setattr(main, "CONTAINER_MARKERS", (tmp_path / "none",))
    monkeypatch.setattr(main, "get_settings", lambda: _settings("0.0.0.0"))

    async def must_not_run():
        raise AssertionError("the database was opened")

    monkeypatch.setattr(main, "init_db", must_not_run)
    assert await main.main() == 2
