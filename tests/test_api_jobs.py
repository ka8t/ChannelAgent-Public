"""Tests: long operations are jobs (id, status, progress, cancel), backups are
the first of them, and the acting client is recorded in the admin events.
"""

import asyncio
import sqlite3

import httpx
import pytest

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
GOOD = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
async def client(fresh_db, monkeypatch):
    from app.admin.jobs import registry
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    registry.clear()
    await init_db()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=GOOD) as c:
        yield c
    registry.clear()
    deps.reset_failure_state()


async def _wait(client, job_id, final=("done", "failed", "cancelled"), tries=100):
    for _ in range(tries):
        job = (await client.get(f"/jobs/{job_id}")).json()
        if job["status"] in final:
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {job}")


async def test_a_backup_is_a_job_that_ends_with_a_verified_file(client):
    from app.config import get_settings

    response = await client.post("/backups")
    assert response.status_code == 202
    job = await _wait(client, response.json()["id"])
    assert job["status"] == "done" and job["progress"] == 1.0
    name = job["result"]["name"]
    assert job["result"]["size_bytes"] > 0
    listing = (await client.get("/backups")).json()
    assert name in [b["name"] for b in listing]
    db = get_settings().database_url.split("///", 1)[1]
    backup = db.rsplit("/", 1)[0] + "/backups/" + name
    con = sqlite3.connect(backup)
    try:
        assert con.execute("pragma integrity_check").fetchone()[0] == "ok"
    finally:
        con.close()


async def test_a_backup_is_recorded_as_an_admin_event(client):
    job = await _wait(client, (await client.post("/backups")).json()["id"])
    events = (await client.get("/admin-events", params={"action": "backup.create"})).json()
    assert len(events) == 1
    assert events[0]["actor"] == "api" and events[0]["target_type"] == "backup"
    assert job["result"]["name"] in str(events[0]["details"])


async def test_a_running_job_can_be_cancelled(client, monkeypatch):
    from app.api import operations

    def slow(_path, _label):
        import time

        time.sleep(2)

    monkeypatch.setattr(operations, "make_backup", slow)
    started = (await client.post("/backups")).json()
    assert started["status"] == "running"
    cancelled = await client.post(f"/jobs/{started['id']}/cancel")
    assert cancelled.status_code == 200
    assert (await _wait(client, started["id"]))["status"] == "cancelled"
    # a finished job cannot be cancelled again
    assert (await client.post(f"/jobs/{started['id']}/cancel")).status_code == 409


async def test_a_failing_job_reports_a_safe_error(client, monkeypatch):
    from app.api import operations

    def broken(_path, _label):
        raise RuntimeError("/secret/dir SELECT token")

    monkeypatch.setattr(operations, "make_backup", broken)
    job = await _wait(client, (await client.post("/backups")).json()["id"])
    assert job["status"] == "failed"
    assert "secret" not in job["error"] and "error id" in job["error"]


async def test_an_unknown_job_is_404_and_the_list_is_newest_first(client):
    assert (await client.get("/jobs/nope")).status_code == 404
    first = (await client.post("/backups")).json()["id"]
    await _wait(client, first)
    second = (await client.post("/backups")).json()["id"]
    await _wait(client, second)
    assert [j["id"] for j in (await client.get("/jobs")).json()][:2] == [second, first]


async def test_the_cli_label_is_the_recorded_actor(client):
    ok = await client.post("/users", json={"display_name": "a"}, headers={"X-Client": "cli:alice"})
    assert ok.status_code == 201
    forged = await client.post(
        "/users", json={"display_name": "b"}, headers={"X-Client": "cli:bad name;drop"}
    )
    assert forged.status_code == 201
    other = await client.post("/users", json={"display_name": "c"}, headers={"X-Client": "root"})
    assert other.status_code == 201
    events = (await client.get("/admin-events", params={"action": "user.create"})).json()
    assert sorted(e["actor"] for e in events) == ["api", "api", "cli:alice"]
