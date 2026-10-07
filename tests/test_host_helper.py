"""Tests: the host helper.

- Signed calls: an unsigned, badly signed, tampered, expired or replayed call is refused.
- The helper: a fixed table (anything else 403), the scope claim re-checked, typed bodies,
  two audit lines per call (before, after; a job's second line when it ends), a secret value
  never written to the audit, a body limit, the audit file readable by its owner only.
- The Admin API: host routes need the owner scope, forward a signed call, record an admin
  event, follow the helper's `host-...` jobs; the configuration and the models go through the
  helper where this process has no `.env` or `models/`; a helper that is off or has another
  secret is a 409 with the reason.
- The operations: start.sh run with an argument list, a minimal environment and
  START_SH_LOCAL=1; restore and rekey stop the application and start it again (not after a
  rekey that says the data may be unsafe).
- The transport over a real Unix socket, mode 660.

No real start.sh run, no Docker, no real data: a fake start.sh in a temporary project.
"""

import asyncio
import json
import os
import shutil
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import pytest

from app.host import signing

SECRET = "Hq7vT3mK9xW2pL7nR4bY6cH1dF5gJ0sAZ"
KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
GOOD = {"Authorization": f"Bearer {KEY}"}


# --- signing ---


def _headers(method="GET", path="/status", body=b"", scope="admin", now=None, nonce=None):
    return signing.sign(
        SECRET, method, path, body, scope=scope, actor="cli:me", now=now, nonce=nonce
    )


def test_a_signed_call_is_accepted_once_and_its_replay_refused():
    verifier = signing.Verifier(SECRET, started_at=time.time() - 5)
    headers = _headers()
    assert verifier.verify("GET", "/status", b"", headers) == ("admin", "cli:me")
    with pytest.raises(signing.SignatureError, match="replayed"):
        verifier.verify("GET", "/status", b"", headers)


@pytest.mark.parametrize(
    "change",
    [
        {"method": "POST"},
        {"path": "/app/stop"},
        {"body": b'{"x": 1}'},
    ],
)
def test_a_tampered_call_is_refused(change):
    verifier = signing.Verifier(SECRET, started_at=time.time() - 5)
    headers = _headers()
    call = {"method": "GET", "path": "/status", "body": b"", **change}
    with pytest.raises(signing.SignatureError, match="bad signature"):
        verifier.verify(call["method"], call["path"], call["body"], headers)


def test_a_raised_scope_claim_is_refused():
    verifier = signing.Verifier(SECRET, started_at=time.time() - 5)
    headers = {**_headers(scope="read"), signing.H_SCOPE: "owner"}
    with pytest.raises(signing.SignatureError, match="bad signature"):
        verifier.verify("GET", "/status", b"", headers)


def test_another_secret_and_an_unsigned_call_are_refused():
    verifier = signing.Verifier(SECRET, started_at=time.time() - 5)
    other = signing.sign("x" * 40, "GET", "/status", b"", scope="admin", actor="a")
    with pytest.raises(signing.SignatureError, match="bad signature"):
        verifier.verify("GET", "/status", b"", other)
    with pytest.raises(signing.SignatureError, match="unsigned"):
        verifier.verify("GET", "/status", b"", {})


def test_a_call_older_than_30_seconds_or_older_than_the_helper_is_refused():
    now = time.time()
    verifier = signing.Verifier(SECRET, started_at=now - 100)
    with pytest.raises(signing.SignatureError, match="expired"):
        verifier.verify("GET", "/status", b"", _headers(now=now - 31), now=now)
    assert verifier.verify("GET", "/status", b"", _headers(now=now - 29), now=now)
    fresh = signing.Verifier(SECRET, started_at=now)
    with pytest.raises(signing.SignatureError, match="expired"):
        fresh.verify("GET", "/status", b"", _headers(now=now - 2), now=now)


def test_the_nonce_is_recorded_only_for_a_valid_signature():
    verifier = signing.Verifier(SECRET, started_at=time.time() - 5)
    headers = _headers(nonce="n1")
    bad = {**headers, signing.H_SIGNATURE: "0" * 64}
    with pytest.raises(signing.SignatureError, match="bad signature"):
        verifier.verify("GET", "/status", b"", bad)
    assert verifier.verify("GET", "/status", b"", headers)


# --- the helper ---


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project directory for the helper: .env, .env.example, a database and a backup."""
    from app.admin.jobs import JobRegistry
    from app.host import helper, ops

    (tmp_path / "data" / "backups").mkdir(parents=True)
    (tmp_path / ".env").write_text(
        f"DATABASE_URL=sqlite+aiosqlite:///./data/channelagent.db\nAPI_SERVER_PORT=1\n"
        f"HOST_HELPER_ENABLED=true\nHOST_HELPER_SECRET={SECRET}\nEMAIL_USERNAME=a\n"
    )
    shutil.copy(Path(__file__).resolve().parents[1] / ".env.example", tmp_path / ".env.example")
    monkeypatch.setattr(ops, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(helper, "jobs", JobRegistry(prefix="host-"))
    monkeypatch.setattr(ops, "STOP_GRACE_SECONDS", 0)
    monkeypatch.setattr(ops, "POLL_SECONDS", 0.01)
    return tmp_path


def _helper_client(secret=SECRET):
    from app.host.helper import build_app

    transport = httpx.ASGITransport(app=build_app(secret), raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://helper")


async def _signed(client, method, path, body=None, scope="owner"):
    content = json.dumps(body).encode() if body is not None else b""
    headers = signing.sign(SECRET, method, path, content, scope=scope, actor="cli:me")
    return await client.request(method, path, content=content, headers=headers)


def _audit(project) -> list[dict]:
    path = project / "logs" / "host-helper-audit.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def test_the_helper_refuses_unsigned_replayed_and_unknown_calls(project, monkeypatch):
    from app.host import ops

    monkeypatch.setattr(
        ops, "app_state", lambda: {"mode": "native", "running": False, "reasons": []}
    )
    async with _helper_client() as client:
        assert (await client.get("/status")).status_code == 401
        headers = signing.sign(SECRET, "GET", "/status", b"", scope="admin", actor="cli:me")
        first = await client.get("/status", headers=headers)
        replay = await client.get("/status", headers=headers)
        outside = await _signed(client, "POST", "/shell", {"cmd": "id"})
        traversal = await _signed(client, "POST", "/backups/a;rm -rf/restore", {})
    assert first.status_code == 200 and first.json()["running"] is False
    assert replay.status_code == 401 and "replayed" in replay.json()["detail"]
    assert outside.status_code == 403 and "not in the helper's table" in outside.json()["detail"]
    assert traversal.status_code == 403
    phases = [(r["phase"], r.get("result")) for r in _audit(project)]
    assert phases.count(("refused", "unsigned call")) == 1
    assert phases.count(("refused", "replayed call")) == 1
    assert phases.count(("refused", "operation not in the table")) == 2


async def test_the_helper_rechecks_the_scope_claim(project):
    async with _helper_client() as client:
        refused = await _signed(client, "POST", "/app/stop", scope="admin")
        also = await _signed(client, "GET", "/status", scope="read")
    assert refused.status_code == 403 and refused.json()["detail"] == "Insufficient scope"
    assert also.status_code == 403


async def test_a_job_call_has_two_audit_lines_before_and_after_it_ends(project, monkeypatch):
    from app.host import helper, ops

    async def stop(job):
        job.update(0.5, "stopping")
        await asyncio.sleep(0.05)
        return {"stopped": True}

    monkeypatch.setattr(ops, "stop", stop)
    async with _helper_client() as client:
        response = await _signed(client, "POST", "/app/stop")
        job = response.json()
        assert response.status_code == 202 and job["id"].startswith("host-")
        await helper.jobs.get(job["id"]).task
        done = await _signed(client, "GET", f"/jobs/{job['id']}", scope="admin")
    assert done.json()["status"] == "done" and done.json()["result"] == {"stopped": True}
    lines = [r for r in _audit(project) if r.get("op") == "app.stop"]
    assert [r["phase"] for r in lines] == ["before", "after"]
    assert lines[1]["result"] == "done" and lines[1]["job"] == job["id"]
    assert lines[0]["call"] == lines[1]["call"] and lines[0]["actor"] == "cli:me"
    assert stat.S_IMODE((project / "logs" / "host-helper-audit.jsonl").stat().st_mode) == 0o600


async def test_a_stop_restore_or_rekey_job_cannot_be_cancelled_half_way(project, monkeypatch):
    from app.host import helper, ops

    async def slow(job):
        await asyncio.sleep(5)

    monkeypatch.setattr(ops, "stop", slow)
    async with _helper_client() as client:
        job = (await _signed(client, "POST", "/app/stop")).json()
        cancel = await _signed(client, "POST", f"/jobs/{job['id']}/cancel", scope="admin")
    assert cancel.status_code == 409 and "cannot be stopped half-way" in cancel.json()["detail"]
    helper.jobs.get(job["id"]).task.cancel()


async def test_a_restore_names_a_listed_backup_only(project):
    async with _helper_client() as client:
        missing = await _signed(client, "POST", "/backups/channelagent-x.db/restore", {})
        typed = await _signed(client, "POST", "/backups/a.db/restore", {"allow_unreadable": [1]})
        extra = await _signed(client, "POST", "/rekey", {"allow_unreadable": False, "x": 1})
    assert missing.status_code == 422 and "No backup named" in missing.json()["detail"]
    assert typed.status_code == 422 and extra.status_code == 422


async def test_a_secret_set_through_the_helper_never_reaches_the_audit(project):
    async with _helper_client() as client:
        secret_value = "Tq9vT3mK9xW2pL7nR4bY6cH1dF5gJ0sB"
        response = await _signed(
            client, "PATCH", "/config", {"key": "API_SERVER_KEY", "value": secret_value}
        )
        plain = await _signed(client, "PATCH", "/config", {"key": "EMAIL_USERNAME", "value": "b"})
        listing = await _signed(client, "GET", "/config", scope="admin")
    assert response.status_code == 200 and plain.status_code == 200
    assert secret_value in (project / ".env").read_text()
    text = (project / "logs" / "host-helper-audit.jsonl").read_text()
    assert secret_value not in text and "(secret)" in text
    entries = {e["key"]: e for e in listing.json()}
    assert entries["API_SERVER_KEY"]["value"] is None and entries["EMAIL_USERNAME"]["value"] == "b"


async def test_the_encryption_key_is_not_set_through_the_helper(project):
    async with _helper_client() as client:
        response = await _signed(
            client, "PATCH", "/config", {"key": "ENCRYPTION_KEY", "value": "x" * 43 + "="}
        )
    assert response.status_code == 409


async def test_a_body_over_the_limit_is_refused(project):
    from app.host.helper import MAX_BODY_BYTES

    async with _helper_client() as client:
        response = await client.post("/rekey", content=b"x" * (MAX_BODY_BYTES + 1))
    assert response.status_code == 413


def test_the_helper_does_not_start_when_off_or_with_a_weak_secret(project):
    from app.host import helper

    env = project / ".env"
    env.write_text(env.read_text().replace("HOST_HELPER_ENABLED=true", "HOST_HELPER_ENABLED=false"))
    assert helper.main(["--check"]) == 2
    env.write_text(
        env.read_text()
        .replace("HOST_HELPER_ENABLED=false", "HOST_HELPER_ENABLED=true")
        .replace(SECRET, "changeme-changeme-changeme")
    )
    assert helper.main(["--check"]) == 2
    env.write_text(env.read_text().replace("changeme-changeme-changeme", SECRET))
    assert helper.main(["--check"]) == 0


# --- the operations ---


FAKE_START_SH = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> calls.log
env > "env-$1.log"
case "$1" in
  --restore) cp data/backups/"$2" data/channelagent.db; echo "Restored" ;;
  --rekey) echo "rekey"; exit "$(cat rekey.exit 2>/dev/null || echo 0)" ;;
esac
exit 0
"""


@pytest.fixture
def fake_script(project, monkeypatch):
    from app.host import ops

    (project / "start.sh").write_text(FAKE_START_SH)
    state = {"running": True, "starts": 0}

    async def fake_state():
        return {"mode": "native", "running": state["running"], "reasons": []}

    real_run = ops.run_script

    async def run_script(*args, **kw):
        result = await real_run(*args, **kw)
        if args[0] == "--stop":
            state["running"] = False
        if args[0] in ("--native", "--docker"):
            state["running"] = True
            state["starts"] += 1
        return result

    monkeypatch.setattr(ops, "_state", fake_state)
    monkeypatch.setattr(ops, "run_script", run_script)
    monkeypatch.setattr(ops, "_api_answers", lambda: state["running"])
    return state


def _calls(project) -> list[str]:
    return (project / "calls.log").read_text().splitlines()


def _db(path: Path, users: int) -> None:
    import sqlite3

    con = sqlite3.connect(path)
    con.execute("create table users (id integer)")
    con.executemany("insert into users values (?)", [(i,) for i in range(users)])
    con.commit()
    con.close()


async def test_a_restore_stops_restores_and_starts_again(project, fake_script):
    from app.admin.jobs import Job
    from app.host import ops

    _db(project / "data" / "channelagent.db", 3)
    _db(project / "data" / "backups" / "channelagent-manual-20260927-100000.db", 7)
    result = await ops.restore(
        Job(id="h", kind="k"), "channelagent-manual-20260927-100000.db", False
    )
    assert _calls(project) == [
        "--stop",
        "--restore channelagent-manual-20260927-100000.db --yes",
        "--native --detach",
    ]
    assert result["backup_rows"]["users"] == result["database_rows_after"]["users"] == 7
    assert result["rows_equal"] and result["restarted"]


async def test_the_script_runs_with_a_minimal_environment_and_start_sh_local(project, fake_script):
    from app.host import ops

    os.environ["API_SERVER_KEY_LEAK_PROBE"] = "must-not-pass"
    try:
        code, _ = await ops.run_script("--stop")
    finally:
        del os.environ["API_SERVER_KEY_LEAK_PROBE"]
    env = dict(
        line.split("=", 1)
        for line in (project / "env---stop.log").read_text().splitlines()
        if "=" in line
    )
    assert code == 0 and env["START_SH_LOCAL"] == "1"
    assert "API_SERVER_KEY_LEAK_PROBE" not in env and "ENCRYPTION_KEY" not in env


async def test_a_rekey_that_leaves_unsafe_data_does_not_start_the_application(project, fake_script):
    from app.admin.jobs import Job, JobError
    from app.host import ops

    (project / "rekey.exit").write_text("3")
    with pytest.raises(JobError, match="left stopped"):
        await ops.rekey(Job(id="h", kind="k"), False, False)
    assert _calls(project) == ["--stop", "--rekey --yes"]
    assert fake_script["running"] is False


async def test_a_refused_rekey_starts_the_application_again(project, fake_script):
    from app.admin.jobs import Job, JobError
    from app.host import ops

    (project / "rekey.exit").write_text("2")
    with pytest.raises(JobError, match="Rekey failed: rekey"):
        await ops.rekey(Job(id="h", kind="k"), True, True)
    assert _calls(project) == [
        "--stop",
        "--rekey --yes --allow-unreadable --dry-run",
        "--native --detach",
    ]


async def test_start_uses_the_mode_start_sh_last_used(project, fake_script):
    from app.admin.jobs import Job
    from app.host import ops

    fake_script["running"] = False
    (project / ".run-mode").write_text("docker\n")
    await ops.start(Job(id="h", kind="k"))
    assert _calls(project) == ["--docker --detach"]


# --- the Admin API ---


@pytest.fixture
async def api(fresh_db, project, monkeypatch):
    """The Admin API with the helper enabled and served in process."""
    from app.admin.jobs import registry
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db
    from app.host import client as host_client

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    monkeypatch.setenv("HOST_HELPER_ENABLED", "true")
    monkeypatch.setenv("HOST_HELPER_SECRET", SECRET)
    get_settings.cache_clear()
    deps.reset_failure_state()
    registry.clear()
    await init_db()

    def helper_client():
        from app.host.helper import build_app

        transport = httpx.ASGITransport(app=build_app(SECRET), raise_app_exceptions=False)
        return httpx.AsyncClient(transport=transport, base_url="http://helper")

    monkeypatch.setattr(host_client, "_client", helper_client)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=GOOD) as c:
        yield c
    registry.clear()
    get_settings.cache_clear()


async def test_a_host_stop_goes_through_the_helper_and_is_recorded(api, project, monkeypatch):
    from app.host import helper, ops

    monkeypatch.setattr(ops, "stop", lambda job: asyncio.sleep(0, {"stopped": True}))
    response = await api.post("/host/stop")
    job = response.json()
    assert response.status_code == 202 and job["id"].startswith("host-")
    await helper.jobs.get(job["id"]).task
    followed = await api.get(f"/jobs/{job['id']}")
    assert followed.status_code == 200 and followed.json()["status"] == "done"
    events = (await api.get("/admin-events", params={"action": "host.stop"})).json()
    assert len(events) == 1 and job["id"] in str(events[0]["details"])
    assert [r["phase"] for r in _audit(project) if r.get("op") == "app.stop"] == [
        "before",
        "after",
    ]


async def test_a_restore_name_that_is_not_a_file_name_is_refused_by_the_api(api):
    response = await api.post("/backups/a%3Bb.db/restore", json={})
    assert response.status_code == 422


async def test_the_host_routes_need_the_owner_scope(api):
    from app.api.app import app
    from app.api.scopes import Principal, Scope, get_principal

    app.dependency_overrides[get_principal] = lambda: Principal("api", Scope.ADMIN)
    try:
        codes = [
            (await api.post(path, json={})).status_code
            for path in ("/host/stop", "/host/start", "/host/restart", "/host/rekey")
        ]
        codes.append((await api.post("/backups/x.db/restore", json={})).status_code)
    finally:
        app.dependency_overrides.clear()
    assert codes == [403] * 5


async def test_a_helper_that_is_off_or_has_another_secret_is_a_409(api, monkeypatch):
    from app.config import get_settings
    from app.host import client as host_client

    def other_secret():
        from app.host.helper import build_app

        transport = httpx.ASGITransport(app=build_app("y" * 40), raise_app_exceptions=False)
        return httpx.AsyncClient(transport=transport, base_url="http://helper")

    monkeypatch.setattr(host_client, "_client", other_secret)
    refused = await api.get("/host/status")
    assert refused.status_code == 409 and "HOST_HELPER_SECRET" in refused.json()["detail"]
    monkeypatch.setenv("HOST_HELPER_ENABLED", "false")
    get_settings.cache_clear()
    off = await api.post("/host/stop")
    assert off.status_code == 409 and "helper is off" in off.json()["detail"]
    assert (await api.get("/admin-events", params={"action": "host.stop"})).json() == []


async def test_config_and_models_go_through_the_helper_without_a_local_env(
    api, project, monkeypatch, tmp_path
):
    from app.config import get_settings

    monkeypatch.setenv("ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "absent-models"))
    get_settings.cache_clear()
    listing = await api.get("/config")
    changed = await api.patch("/config", json={"key": "EMAIL_USERNAME", "value": "host"})
    assert listing.status_code == 200 and changed.status_code == 200
    assert "EMAIL_USERNAME=host" in (project / ".env").read_text()
    events = (await api.get("/admin-events", params={"action": "config.set"})).json()
    assert len(events) == 1
    models = await api.get("/models")
    assert models.status_code == 409 and "models directory" in models.json()["detail"]
    ops_lines = [r["op"] for r in _audit(project) if r["phase"] == "before"]
    assert ops_lines == ["config.get", "config.set", "models.list"]


async def test_without_the_helper_config_still_says_where_to_edit_it(api, monkeypatch, tmp_path):
    from app.config import get_settings

    monkeypatch.setenv("ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("HOST_HELPER_ENABLED", "false")
    get_settings.cache_clear()
    response = await api.get("/config")
    assert response.status_code == 409 and "--config" in response.json()["detail"]


async def test_models_are_listed_through_the_helper_with_their_dates(
    api, project, monkeypatch, tmp_path
):
    """Measured in Docker (2026-09-27): a listing with a model in it answered 500 in the
    helper (a datetime in the JSON), 409 through the API."""
    from app.api import models_routes
    from app.config import get_settings

    models = tmp_path / "models"
    models.mkdir()
    (models / "tiny.gguf").write_bytes(b"GGUF" + b"\0" * 60)
    monkeypatch.setenv("MODELS_DIR", str(models))
    get_settings.cache_clear()
    monkeypatch.setattr(models_routes, "_on_host", lambda: True)
    response = await api.get("/models")
    assert response.status_code == 200, response.text
    assert [m["name"] for m in response.json()] == ["tiny.gguf"]
    assert response.json()[0]["modified_at"]
    assert [r["op"] for r in _audit(project) if r["phase"] == "before"] == ["models.list"]


# --- the transport over a Unix socket ---


def test_a_signed_call_over_a_unix_socket_of_mode_660(project, monkeypatch):
    import uvicorn

    from app.config import get_settings
    from app.host import client as host_client
    from app.host.helper import bind_socket, build_app

    short = Path(tempfile.mkdtemp(prefix="hh", dir="/tmp"))
    sock = short / "run" / "helper.sock"
    old_umask = os.umask(0o077)  # what harden_process sets, and a cautious shell too
    try:
        listening = bind_socket(sock)
    finally:
        os.umask(old_umask)
    server = uvicorn.Server(uvicorn.Config(build_app(SECRET), log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listening]}, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    try:
        assert stat.S_IMODE(sock.stat().st_mode) == 0o660
        assert stat.S_IMODE(sock.parent.stat().st_mode) == 0o750
        monkeypatch.setenv("HOST_HELPER_ENABLED", "true")
        monkeypatch.setenv("HOST_HELPER_SECRET", SECRET)
        monkeypatch.setenv("HOST_HELPER_URL", f"unix://{sock}")
        get_settings.cache_clear()
        from app.host import ops

        monkeypatch.setattr(
            ops, "app_state", lambda: {"mode": "native", "running": True, "reasons": ["x"]}
        )
        body = asyncio.run(host_client.call("GET", "/status", scope="admin", actor="cli:me"))
        assert body["running"] is True
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        shutil.rmtree(short, ignore_errors=True)
        get_settings.cache_clear()


def test_the_helper_url_is_never_another_machine():
    from app.settings_rules import helper_url_problem

    assert helper_url_problem("http://127.0.0.1:8701") is None
    assert helper_url_problem("http://host.docker.internal:8701") is None
    assert helper_url_problem("unix:///run/channelagent-host/helper.sock") is None
    assert helper_url_problem("http://10.0.0.2:8701")
    assert helper_url_problem("https://localhost:8701")


def test_the_helper_module_runs_as_a_program():
    import subprocess

    out = subprocess.run(
        [sys.executable, "-m", "app.host.helper", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert out.returncode == 0 and "host helper" in out.stdout
