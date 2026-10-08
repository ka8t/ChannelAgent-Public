"""Tests: the SearXNG that start.sh runs when SEARXNG_MANAGED=true (the docker-compose.yml
service, its start, its settings file, its --status line and its stop).

The start functions are read out of start.sh and run with a fake `docker` and a stub HTTP server
for /healthz; --status and --stop run the whole script in the sandbox of
tests/test_start_status_stop.py. Nothing touches the real Docker or data/.
"""

import re
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from tests.test_start_status_stop import Box, _free_port, _wait, httpx_ok

REPO = Path(__file__).resolve().parent.parent
SERVICE = yaml.safe_load((REPO / "docker-compose.yml").read_text())["services"]["searxng"]

HEALTHZ_STUB = textwrap.dedent(
    """
    import sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_GET(self):
            code = 200 if self.path == "/healthz" else 404
            self.send_response(code); self.send_header("Content-Length", "2"); self.end_headers()
            self.wfile.write(b"OK")

    HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
    """
)

# docker: `info` answers unless FAKE_DOCKER_DOWN is set, `compose ... up` fails when
# FAKE_DOCKER_UP_FAILS is set; every call is recorded.
FAKE_DOCKER = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_DOCKER_LOG"
if [ "$1" = "info" ]; then [ -z "${FAKE_DOCKER_DOWN:-}" ]; exit $?; fi
case "$*" in *" up -d searxng") [ -z "${FAKE_DOCKER_UP_FAILS:-}" ]; exit $?;; esac
exit 0
"""


def _functions() -> str:
    """The SearXNG functions of start.sh and the one helper they use, as bash source."""
    text = (REPO / "start.sh").read_text()
    wanted = ("http_code", "searxng_settings", "start_searxng", "stop_searxng")
    parts = []
    for name in wanted:
        match = re.search(rf"^{name}\(\) {{.*?^}}$", text, flags=re.S | re.M)
        assert match, name
        parts.append(match.group(0))
    return "\n".join(parts)


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "docker").write_text(FAKE_DOCKER)
    (tmp_path / "bin" / "docker").chmod(0o755)
    (tmp_path / "functions.sh").write_text(_functions())
    procs = []
    yield tmp_path, procs
    for proc in procs:
        proc.kill()
        proc.wait(timeout=10)


def _healthz(procs, port: int) -> None:
    proc = subprocess.Popen([sys.executable, "-c", HEALTHZ_STUB, str(port)])
    procs.append(proc)
    assert _wait(lambda: httpx_ok(f"http://127.0.0.1:{port}/healthz"))


def _start(path: Path, mode: str, **env) -> subprocess.CompletedProcess:
    script = f'. ./functions.sh; start_searxng {mode}; echo "URL=${{SEARXNG_URL:-}}"'
    return subprocess.run(
        ["bash", "-c", script],
        cwd=path,
        env={
            "PATH": f"{path / 'bin'}:/usr/bin:/bin",
            "FAKE_DOCKER_LOG": str(path / "docker.log"),
            **env,
        },
        capture_output=True,
        text=True,
        timeout=60,
    )


def _calls(path: Path) -> list[str]:
    log = path / "docker.log"
    return log.read_text().splitlines() if log.exists() else []


def test_the_service_is_pinned_hardened_on_loopback_and_never_started_by_default():
    assert re.fullmatch(r"searxng/searxng@sha256:[0-9a-f]{64}", SERVICE["image"])
    assert SERVICE["profiles"] == ["searxng"]
    assert SERVICE["user"] == "977:977"
    assert SERVICE["read_only"] is True
    assert SERVICE["cap_drop"] == ["ALL"]
    assert SERVICE["security_opt"] == ["no-new-privileges:true"]
    assert SERVICE["ports"] == ["127.0.0.1:${SEARXNG_PORT:-8888}:8080"]
    assert SERVICE["volumes"] == ["./data/searxng/settings.yml:/etc/searxng/settings.yml:ro"]


def test_not_managed_starts_nothing(sandbox):
    path, _ = sandbox
    result = _start(path, "native", SEARXNG_MANAGED="false")
    assert result.returncode == 0
    assert "URL=\n" in result.stdout
    assert _calls(path) == []
    assert not (path / "data").exists()


def test_native_start_writes_the_settings_once_and_exports_the_loopback_url(sandbox):
    path, procs = sandbox
    port = _free_port()
    _healthz(procs, port)
    result = _start(path, "native", SEARXNG_MANAGED="true", SEARXNG_PORT=str(port))
    assert result.returncode == 0, result.stderr
    assert f"URL=http://127.0.0.1:{port}\n" in result.stdout
    assert "compose --profile searxng up -d searxng" in _calls(path)

    settings = path / "data" / "searxng" / "settings.yml"
    text = settings.read_text()
    secret = yaml.safe_load(text)["server"]["secret_key"]
    assert re.fullmatch(r"[0-9a-f]{64}", secret)
    assert secret not in result.stdout + result.stderr, "the secret is never printed"
    assert yaml.safe_load(text)["search"]["formats"] == ["html", "json"]
    assert yaml.safe_load(text)["server"]["limiter"] is False
    assert stat.S_IMODE(settings.stat().st_mode) == 0o644

    again = _start(path, "native", SEARXNG_MANAGED="true", SEARXNG_PORT=str(port))
    assert again.returncode == 0
    assert settings.read_text() == text, "an existing settings file is kept"


def test_docker_mode_points_the_container_at_the_compose_service(sandbox):
    path, procs = sandbox
    port = _free_port()
    _healthz(procs, port)
    result = _start(path, "docker", SEARXNG_MANAGED="true", SEARXNG_PORT=str(port))
    assert "URL=http://searxng:8080\n" in result.stdout


def test_settings_without_json_are_kept_with_a_warning(sandbox):
    path, procs = sandbox
    port = _free_port()
    _healthz(procs, port)
    (path / "data" / "searxng").mkdir(parents=True)
    (path / "data" / "searxng" / "settings.yml").write_text("search:\n  formats:\n    - html\n")
    result = _start(path, "native", SEARXNG_MANAGED="true", SEARXNG_PORT=str(port))
    assert "does not list json" in result.stderr
    assert (path / "data" / "searxng" / "settings.yml").read_text().endswith("- html\n")


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"FAKE_DOCKER_DOWN": "1"}, "Docker is not running"),
        ({"FAKE_DOCKER_UP_FAILS": "1"}, "SearXNG did not start"),
    ],
)
def test_a_failure_leaves_search_off_and_the_run_going(sandbox, env, message):
    path, _ = sandbox
    result = _start(path, "native", SEARXNG_MANAGED="true", SEARXNG_PORT=str(_free_port()), **env)
    assert result.returncode == 0
    assert message in result.stderr and "web search is off for this run" in result.stderr
    assert "URL=\n" in result.stdout


def test_no_answer_on_healthz_leaves_search_off(sandbox):
    path, _ = sandbox
    # `seq 1 30` with `sleep 1`: a fake sleep keeps the test fast.
    (path / "bin" / "sleep").write_text("#!/bin/sh\nexit 0\n")
    (path / "bin" / "sleep").chmod(0o755)
    result = _start(path, "native", SEARXNG_MANAGED="true", SEARXNG_PORT=str(_free_port()))
    assert "does not answer" in result.stderr
    assert "URL=\n" in result.stdout


@pytest.fixture
def box(tmp_path):
    b = Box(tmp_path)
    yield b
    b.cleanup()


def _env(box: Box, extra: str) -> None:
    env = box.path / ".env"
    env.write_text(env.read_text() + extra)


def test_status_says_not_managed_by_default(box):
    result = box.run("--status")
    assert "  SearXNG          : not managed (SEARXNG_MANAGED is not true)" in result.stdout


def test_status_says_up_or_down_when_managed(box):
    port = _free_port()
    _env(box, f"SEARXNG_MANAGED=true\nSEARXNG_PORT={port}\n")
    down = box.run("--status")
    assert f"  SearXNG          : down on port {port}" in down.stdout
    proc = subprocess.Popen([sys.executable, "-c", HEALTHZ_STUB, str(port)])
    box.procs.append(proc)
    assert _wait(lambda: httpx_ok(f"http://127.0.0.1:{port}/healthz"))
    up = box.run("--status")
    assert f"  SearXNG          : up at http://127.0.0.1:{port}" in up.stdout


def test_stop_all_stops_a_managed_searxng_and_only_then(box):
    box.run("--stop", "--all")
    assert not any("searxng" in call for call in box.docker_calls())
    _env(box, "SEARXNG_MANAGED=true\n")
    box.run("--stop", "--all")
    assert "compose --profile searxng stop searxng" in box.docker_calls()
    box.docker_log.unlink()
    box.run("--stop")
    assert not any("searxng" in call for call in box.docker_calls()), "--stop alone keeps it"

