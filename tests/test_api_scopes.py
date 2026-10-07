"""Tests: every Admin API route declares a scope (default deny), and the
requests are protected against a foreign Host, a foreign Origin, an oversized body,
a slow request and a leaking error.
"""

import asyncio
import logging
import re

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from app.api.protect import _host_name
from app.api.scopes import (
    Principal,
    Scope,
    _api_routes,
    declared_scopes,
    get_principal,
    require,
    verify_scopes,
)

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
GOOD = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
async def client(fresh_db, monkeypatch):
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=GOOD) as c:
        yield c
    app.dependency_overrides.clear()
    deps.reset_failure_state()


def _real_routes() -> list[APIRoute]:
    from app.api.app import app

    return [r for r in _api_routes(app.routes) if isinstance(r, APIRoute)]


# --- default deny ---


def test_a_route_without_a_scope_stops_the_startup():
    app = FastAPI()

    @app.get("/unguarded")
    async def unguarded():
        return {}

    with pytest.raises(RuntimeError, match=r"GET /unguarded"):
        verify_scopes(app)


def test_a_route_with_two_scopes_is_refused_too():
    app = FastAPI()

    @app.get("/both", dependencies=[require(Scope.READ), require(Scope.OWNER)])
    async def both():
        return {}

    with pytest.raises(RuntimeError, match=r"GET /both"):
        verify_scopes(app)


def test_the_real_application_declares_a_scope_on_every_route():
    from app.api.app import app

    verify_scopes(app)
    assert len(_real_routes()) >= 25


def test_the_sensitive_routes_keep_their_scope():
    by_route = {
        (m, r.path): declared_scopes(r)[0] for r in _real_routes() for m in r.methods
    }
    assert by_route[("DELETE", "/users/{user_id}")] == Scope.OWNER
    assert by_route[("GET", "/logs")] == Scope.ADMIN
    assert by_route[("GET", "/admin-events")] == Scope.ADMIN
    assert by_route[("POST", "/users/{user_id}/channels/{channel_identity_id}/permissions")] == (
        Scope.ADMIN
    )
    assert by_route[("GET", "/users")] == Scope.READ


# --- the authorization matrix ---


async def test_authorization_matrix_routes_by_scopes(client):
    """For every route and every scope: below the required scope the answer is exactly
    403, at or above it is never 401 or 403. No response body ever holds the key."""
    from app.api.app import app

    checked = 0
    for route in _real_routes():
        required = declared_scopes(route)[0]
        url = re.sub(r"\{[^}]+\}", "1", route.path)
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            for scope in Scope:
                app.dependency_overrides[get_principal] = lambda s=scope: Principal("t", s)
                body = {} if method in {"POST", "PUT", "PATCH"} else None
                response = await client.request(method, url, json=body)
                if scope < required:
                    assert response.status_code == 403, (method, route.path, scope.name)
                else:
                    assert response.status_code not in (401, 403), (method, route.path, scope.name)
                    assert response.status_code < 500, (method, route.path, scope.name)
                assert KEY not in response.text
                checked += 1
    print(f"matrix: {len(_real_routes())} routes x {len(Scope)} scopes, {checked} calls")
    assert checked == sum(len(r.methods - {"HEAD", "OPTIONS"}) for r in _real_routes()) * len(Scope)


async def test_the_api_key_holder_is_the_owner(client):
    assert (await client.delete("/users/1")).status_code == 404  # allowed, then not found


# --- Host and Origin ---


def test_host_header_parsing():
    assert _host_name("t") == "t"
    assert _host_name("T:8700") == "t"
    assert _host_name("[::1]:8700") == "::1"
    assert _host_name("127.0.0.1:8700") == "127.0.0.1"
    assert _host_name("") == ""


async def test_a_foreign_host_is_refused_before_authentication(client):
    response = await client.get("/users", headers={"Host": "evil.example"})
    assert response.status_code == 421
    assert (await client.get("/users", headers={"Host": "t:8700"})).status_code == 200


async def test_a_foreign_origin_on_a_state_changing_request_is_refused(client):
    forged = await client.post("/users", json={}, headers={"Origin": "https://evil.example"})
    assert forged.status_code == 403
    assert (await client.post("/users", json={}, headers={"Origin": "null"})).status_code == 403
    same_origin = await client.post("/users", json={}, headers={"Origin": "http://t"})
    assert same_origin.status_code == 201
    # a safe method is not checked, and a client with no Origin (curl, the script) is fine
    safe = await client.get("/users", headers={"Origin": "https://evil.example"})
    assert safe.status_code == 200
    assert (await client.post("/users", json={})).status_code == 201


async def test_an_empty_allow_list_falls_back_to_the_loopback_names(monkeypatch):
    from app.api.protect import allowed_hosts
    from app.config import get_settings

    monkeypatch.setenv("ALLOWED_HOSTS", "")
    get_settings.cache_clear()
    assert allowed_hosts() == {"localhost", "127.0.0.1", "::1"}


# --- size and time ---


async def test_an_oversized_body_is_refused(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("API_MAX_BODY_BYTES", "200")
    get_settings.cache_clear()
    big = {"display_name": "x" * 2000}
    assert (await client.post("/users", json=big)).status_code == 413
    assert (await client.post("/users", json={"display_name": "ok"})).status_code == 201


async def test_an_oversized_body_without_a_length_is_refused_too(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("API_MAX_BODY_BYTES", "200")
    get_settings.cache_clear()

    async def chunks():
        for _ in range(10):
            yield b'{"display_name": "' + b"x" * 100 + b'"}'

    response = await client.post(
        "/users", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413


async def test_a_slow_request_is_cut(client, monkeypatch):
    from app.admin import service
    from app.config import get_settings

    async def slow(*_a, **_k):
        await asyncio.sleep(3)

    monkeypatch.setattr(service, "list_users", slow)
    monkeypatch.setenv("API_REQUEST_TIMEOUT_SECONDS", "1")
    get_settings.cache_clear()
    assert (await client.get("/users")).status_code == 504


# --- errors and schemas ---


async def test_an_internal_error_returns_an_id_and_never_the_exception_text(
    client, monkeypatch, caplog
):
    from app.admin import service

    async def boom(*_a, **_k):
        raise RuntimeError("/etc/secret/path SELECT * FROM users WHERE token='topsecret'")

    monkeypatch.setattr(service, "create_user", boom)
    with caplog.at_level(logging.ERROR, logger="channelagent.api"):
        response = await client.post("/users", json={})
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error"
    assert re.fullmatch(r"[0-9a-f]{12}", body["error_id"])
    assert "secret" not in response.text and "SELECT" not in response.text
    assert any(body["error_id"] in r.getMessage() for r in caplog.records)


async def test_unknown_fields_are_refused(client):
    response = await client.post("/users", json={"display_name": "a", "is_admin": True})
    assert response.status_code == 422


async def test_a_declared_length_over_the_limit_is_refused_before_the_application_runs(
    monkeypatch,
):
    from app.api.protect import ProtectMiddleware
    from app.config import get_settings

    monkeypatch.setenv("API_MAX_BODY_BYTES", "200")
    get_settings.cache_clear()
    reached, sent = [], []

    async def application(scope, receive, send):
        reached.append(1)

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/users",
        "headers": [(b"host", b"t"), (b"content-length", b"5000")],
    }
    await ProtectMiddleware(application)(scope, receive, send)
    assert sent[0]["status"] == 413
    assert reached == []
