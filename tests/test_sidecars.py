"""Tests: sidecar containers for third-party MCP servers, run by the host helper.

- Only an image pinned in MCP_SIDECAR_IMAGES runs; a bad name or egress is refused.
- The server container: its own internal network, read-only, user nobody, no capability, no
  new privileges, limits, no volume; out to the default bridge only for egress lan/internet.
- The forwarder publishes on 127.0.0.1 only; a failed deploy removes what it created.
- Through the Admin API: owner scope, an admin event; the application never mounts the Docker
  socket (every compose file).
"""

import json
import shutil
from pathlib import Path

import httpx
import pytest
import yaml

from app.admin.jobs import Job, JobError
from app.admin.service import ConflictError, InvalidInputError
from app.host import sidecars

REPO = Path(__file__).resolve().parents[1]
IMAGE = "ghcr.io/acme/mcp-x@sha256:" + "a" * 64
FWD = "python@sha256:" + "f" * 64
SECRET = "Sq7vT3mK9xW2pL7nR4bY6cH1dF5gJ0sAZ"


class FakeDocker:
    def __init__(self, fail_on: str | None = None):
        self.calls: list[list[str]] = []
        self.fail_on = fail_on
        self.containers: list[str] = []

    async def __call__(self, *args: str, timeout=None):
        self.calls.append(list(args))
        if self.fail_on and self.fail_on in args:
            return 1, "Error response from daemon: boom"
        if args[:2] == ("image", "inspect"):
            return 0, json.dumps([FWD])
        if args[0] == "ps":
            return 0, "\n".join(f"{c}\tUp 1 minute\t\tx" for c in self.containers)
        return 0, "ok"


@pytest.fixture
def docker(monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(sidecars, "docker", fake)
    monkeypatch.setattr(sidecars, "free_loopback_port", lambda: 41234)
    fake.waited = []

    async def ready(port, timeout=None):
        fake.waited.append(port)

    monkeypatch.setattr(sidecars, "wait_ready", ready)
    return fake


def _job():
    return Job(id="host-000000000001", kind="sidecar-deploy")


async def _deploy(egress="local"):
    return await sidecars.deploy(
        _job(), name="notes", image=IMAGE, port=8000, egress=egress, memory_mb=256, cpus=0.5,
        forwarder_image="python:3.14-slim",
    )  # fmt: skip


def test_only_a_pinned_listed_image_a_good_name_and_egress_pass():
    allowed = sidecars.allowed_images(f"{IMAGE}, ghcr.io/x:latest ,sha256:" + "b" * 64)
    assert allowed == [IMAGE, "sha256:" + "b" * 64], "a tag without a digest is never allowed"
    sidecars.check_request("notes", IMAGE, allowed, "local")
    for name, image, egress in (
        ("Notes", IMAGE, "local"),
        ("../x", IMAGE, "local"),
        ("notes", "ghcr.io/acme/mcp-x:latest", "local"),
        ("notes", "ghcr.io/acme/other@sha256:" + "c" * 64, "local"),
        ("notes", IMAGE, "everywhere"),
    ):
        with pytest.raises(InvalidInputError):
            sidecars.check_request(name, image, allowed, egress)


def test_the_server_container_is_confined():
    argv = sidecars.server_argv("notes", IMAGE, 8000, 256, 0.5)
    joined = " ".join(argv)
    for flag in ("--read-only", "--user 65534:65534", "--cap-drop ALL",
                 "--security-opt no-new-privileges", "--memory 256m", "--cpus 0.5",
                 "--pids-limit 128", "--network ca-mcp-notes"):  # fmt: skip
        assert flag in joined, flag
    assert not {"-v", "--volume", "--mount", "--privileged", "-p", "--publish"} & set(argv)
    assert "docker.sock" not in joined and argv[-1] == IMAGE


def test_the_forwarder_publishes_on_loopback_only():
    argv = sidecars.forwarder_argv("notes", FWD, 8000, 41234)
    assert argv[argv.index("--publish") + 1] == "127.0.0.1:41234:8000"
    assert "--read-only" in argv and "--user" in argv and FWD in argv


async def test_a_local_server_gets_no_way_out(docker):
    result = await _deploy("local")
    assert ["network", "create", "--internal", "--label", "channelagent.sidecar=notes",
            "ca-mcp-notes"] in docker.calls  # fmt: skip
    assert not any(c[:2] == ["network", "connect"] and c[2] == "bridge" for c in docker.calls)
    assert ["network", "connect", "ca-mcp-notes", "ca-mcp-notes-fwd"] in docker.calls
    assert result["url_native"] == "http://127.0.0.1:41234/mcp"
    assert result["url_container"] == "http://host.docker.internal:41234/mcp"
    assert result["forwarder"] == FWD, "the forwarder runs pinned by digest"
    assert docker.waited == [41234], "done only once the server answers"


@pytest.mark.parametrize("egress", ["lan", "internet"])
async def test_a_server_that_may_go_out_joins_the_bridge(docker, egress):
    await _deploy(egress)
    assert ["network", "connect", "bridge", "ca-mcp-notes"] in docker.calls


async def test_waiting_for_the_server_ends_on_an_answer_or_a_timeout():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Answer(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(406)  # an MCP server refusing a plain GET is still up
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = HTTPServer(("127.0.0.1", 0), Answer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        await sidecars.wait_ready(server.server_port, timeout=5)
    finally:
        server.shutdown()
    with pytest.raises(JobError, match="did not answer"):
        await sidecars.wait_ready(sidecars.free_loopback_port(), timeout=1)


async def test_a_failed_deploy_removes_what_it_created(monkeypatch):
    fake = FakeDocker(fail_on="--publish")
    monkeypatch.setattr(sidecars, "docker", fake)
    monkeypatch.setattr(sidecars, "free_loopback_port", lambda: 41234)
    with pytest.raises(JobError):
        await _deploy()
    assert ["rm", "-f", "ca-mcp-notes"] in fake.calls
    assert ["network", "rm", "ca-mcp-notes"] in fake.calls


async def test_a_name_in_use_is_refused(docker):
    docker.containers = ["ca-mcp-notes"]
    with pytest.raises(ConflictError):
        await _deploy()
    assert not any(c[0] == "run" for c in docker.calls)


def test_the_application_never_mounts_the_docker_socket():
    """The application's service, in every compose file, has no Docker socket. The optional
    autoheal overlay mounts it for its own container, which restarts an unhealthy
    application and runs no application code."""
    files = sorted(REPO.glob("docker-compose*.yml"))
    assert files
    holders = []
    for path in files:
        for name, service in (
            (yaml.safe_load(path.read_text()) or {}).get("services") or {}
        ).items():
            if any("docker.sock" in str(v) for v in (service.get("volumes") or [])):
                holders.append((path.name, name))
    assert holders == [("docker-compose.autoheal.yml", "autoheal")]
    dockerfile = (REPO / "Dockerfile").read_text()
    assert "docker.sock" not in dockerfile
    assert not any(pkg in dockerfile for pkg in ("docker.io", "docker-ce", "docker-cli"))


# --- through the helper and the Admin API ---


@pytest.fixture
def project(tmp_path, monkeypatch):
    from app.admin.jobs import JobRegistry
    from app.host import helper, ops

    (tmp_path / ".env").write_text(
        f"HOST_HELPER_ENABLED=true\nHOST_HELPER_SECRET={SECRET}\nMCP_SIDECAR_IMAGES={IMAGE}\n"
    )
    shutil.copy(REPO / ".env.example", tmp_path / ".env.example")
    monkeypatch.setattr(ops, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(helper, "jobs", JobRegistry(prefix="host-"))
    return tmp_path


@pytest.fixture
async def api(fresh_db, project, docker, monkeypatch):
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db
    from app.host import client as host_client

    monkeypatch.setenv("API_SERVER_KEY", "k" * 32)
    monkeypatch.setenv("HOST_HELPER_ENABLED", "true")
    monkeypatch.setenv("HOST_HELPER_SECRET", SECRET)
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()

    def helper_client():
        from app.host.helper import build_app

        transport = httpx.ASGITransport(app=build_app(SECRET), raise_app_exceptions=False)
        return httpx.AsyncClient(transport=transport, base_url="http://helper")

    monkeypatch.setattr(host_client, "_client", helper_client)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    headers = {"Authorization": "Bearer " + "k" * 32}
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as c:
        yield c
    get_settings.cache_clear()


async def test_a_listed_image_is_deployed_and_an_unlisted_one_refused(api, docker):
    from app.host import helper

    refused = await api.post("/mcp/sidecars", json={"name": "x", "image": "sha256:" + "d" * 64})
    assert refused.status_code == 422 and "MCP_SIDECAR_IMAGES" in refused.json()["detail"]
    assert not any(c[0] == "run" for c in docker.calls)
    accepted = await api.post("/mcp/sidecars", json={"name": "notes", "image": IMAGE})
    assert accepted.status_code == 202
    job = accepted.json()
    await helper.jobs.get(job["id"]).task
    done = (await api.get(f"/jobs/{job['id']}")).json()
    assert done["status"] == "done" and done["result"]["url_native"].startswith("http://127.0.0.1:")
    events = (await api.get("/admin-events", params={"action": "mcp_sidecar.deploy"})).json()
    assert len(events) == 1


async def test_the_owner_scope_is_needed(api):
    from app.api.app import app
    from app.api.scopes import Principal, Scope, get_principal

    app.dependency_overrides[get_principal] = lambda: Principal("api", Scope.ADMIN)
    try:
        assert (
            await api.post("/mcp/sidecars", json={"name": "a", "image": IMAGE})
        ).status_code == 403
        assert (await api.delete("/mcp/sidecars/a")).status_code == 403
        assert (await api.get("/mcp/sidecars")).status_code == 200
    finally:
        app.dependency_overrides.clear()


async def test_removing_an_unknown_sidecar_is_404(api):
    assert (await api.delete("/mcp/sidecars/ghost")).status_code == 404
    assert (await api.delete("/mcp/sidecars/Bad_Name")).status_code == 422
