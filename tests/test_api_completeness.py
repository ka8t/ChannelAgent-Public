"""Tests: what a UI and a script generated from the API rely on: who am I,
service status, paging with a total, tags, documented errors and a declared version.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from app.api.scopes import Principal, Scope, get_principal
from app.api.version import API_VERSION

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
GOOD = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
async def client(fresh_db, monkeypatch):
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    # Nothing listens here, so the engine reads as down unless a test starts a mock.
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:9")
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=GOOD) as c:
        yield c
    app.dependency_overrides.clear()
    deps.reset_failure_state()


# --- whoami and status ---


async def test_whoami_reports_the_actor_the_scope_and_the_version(client):
    body = (await client.get("/whoami")).json()
    assert body == {"actor": "api", "scope": "owner", "api_version": API_VERSION}


async def test_whoami_follows_the_scope_of_the_caller(client):
    from app.api.app import app

    app.dependency_overrides[get_principal] = lambda: Principal("someone", Scope.READ)
    body = (await client.get("/whoami")).json()
    assert (body["actor"], body["scope"]) == ("someone", "read")


async def test_status_with_the_engine_down_answers_fast_and_says_so(client):
    started = time.monotonic()
    response = await client.get("/status")
    assert response.status_code == 200
    assert time.monotonic() - started < 3
    body = response.json()
    assert set(body) == {
        "api_version",
        "started_at",
        "uptime_seconds",
        "components",
        "database",
        "engine",
        "memory",
    }
    assert body["engine"] == {
        "reachable": False,
        "model": None,
        "n_ctx": None,
        "slots_total": None,
        "slots_busy": None,
    }
    assert body["api_version"] == API_VERSION
    assert body["uptime_seconds"] >= 0
    assert body["database"]["kind"] == "sqlite"
    assert body["database"]["size_bytes"] > 0
    assert body["database"]["revision"]


async def test_status_reports_the_engine_model_file_name_and_context(client, monkeypatch):
    class Props(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/slots":  # Two slots, one serving a request
                body = [{"id": 0, "is_processing": True}, {"id": 1, "is_processing": False}]
            else:
                body = {
                    "model_path": "/private/dir/models/tiny.gguf",
                    "default_generation_settings": {"n_ctx": 4096},
                }
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer(("127.0.0.1", 0), Props)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    from app.config import get_settings

    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    get_settings.cache_clear()
    try:
        engine = (await client.get("/status")).json()["engine"]
    finally:
        server.shutdown()
    # only the file name, never the directory layout
    assert engine == {
        "reachable": True,
        "model": "tiny.gguf",
        "n_ctx": 4096,
        "slots_total": 2,
        "slots_busy": 1,
    }


async def test_status_lists_the_components_with_their_last_success(client):
    from app import health

    health.register("telegram", 60)
    try:
        components = (await client.get("/status")).json()["components"]
    finally:
        health.unregister("telegram")
    assert components["telegram"]["healthy"] is True
    assert components["telegram"]["seconds_since_success"] >= 0


# --- paging with a total ---


async def _make(client, n):
    for i in range(n):
        assert (await client.post("/users", json={"display_name": f"u{i}"})).status_code == 201


async def test_users_are_paged_and_the_total_is_in_a_header(client):
    await _make(client, 5)
    page = await client.get("/users", params={"limit": 2, "offset": 1})
    assert [u["display_name"] for u in page.json()] == ["u1", "u2"]
    assert page.headers["X-Total-Count"] == "5"
    everything = await client.get("/users")
    assert len(everything.json()) == 5 and everything.headers["X-Total-Count"] == "5"


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 1001}, {"offset": -1}])
async def test_a_bad_page_is_refused(client, params):
    assert (await client.get("/users", params=params)).status_code == 422


async def test_the_other_simple_lists_are_paged_too(client):
    await _make(client, 1)
    for url in ("/requests", "/users/1/channels", "/users/1/agents"):
        response = await client.get(url, params={"limit": 1, "offset": 0})
        assert response.status_code == 200, url
        assert "X-Total-Count" in response.headers, url
        assert (await client.get(url, params={"limit": 0})).status_code == 422, url


# --- the OpenAPI contract ---


def _operations():
    from app.api.app import app

    spec = app.openapi()
    return spec, [
        (method.upper(), path, op)
        for path, item in spec["paths"].items()
        for method, op in item.items()
        if method in {"get", "post", "put", "patch", "delete"}
    ]


def test_every_operation_has_a_tag_and_documents_401_403_429():
    _spec, operations = _operations()
    assert len(operations) >= 27
    for method, path, op in operations:
        assert op.get("tags"), (method, path)
        for code in ("401", "403", "429"):
            assert code in op["responses"], (method, path, code)


def test_operations_with_an_id_document_404_and_changes_document_409():
    _spec, operations = _operations()
    for method, path, op in operations:
        if "{" in path:
            assert "404" in op["responses"], (method, path)
        if method != "GET":
            assert "409" in op["responses"], (method, path)


def test_the_declared_version_is_the_openapi_version():
    spec, _ops = _operations()
    assert spec["info"]["version"] == API_VERSION


async def test_the_new_routes_need_no_more_than_read(client):
    from app.api.app import app

    app.dependency_overrides[get_principal] = lambda: Principal("r", Scope.READ)
    assert (await client.get("/whoami")).status_code == 200
    assert (await client.get("/status")).status_code == 200
