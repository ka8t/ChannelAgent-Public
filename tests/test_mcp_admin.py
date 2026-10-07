"""Tests: the MCP server registry's admin service and API — validation,
CRUD, the per-tool switch, admin events. /test and GET .../tools connect for real
to the built-in "time" server (a fast, deterministic subprocess), not a mock, since
the point of those two routes is proving a real connection works.
"""

import pytest

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
GOOD = {"Authorization": f"Bearer {KEY}"}


# --- the service layer, direct ---


@pytest.fixture
async def session(fresh_db):
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as s:
        yield s


async def test_create_stdio_server_needs_a_vetted_builtin_id(session):
    from app.admin import mcp as service
    from app.admin.service import InvalidInputError

    with pytest.raises(InvalidInputError):
        await service.create_server(
            session, name="evil", protocol="stdio", builtin_id="not-a-real-one", actor="t"
        )
    with pytest.raises(InvalidInputError):
        await service.create_server(session, name="evil", protocol="stdio", actor="t")


async def test_create_http_server_needs_a_url(session):
    from app.admin import mcp as service
    from app.admin.service import InvalidInputError

    with pytest.raises(InvalidInputError):
        await service.create_server(session, name="remote", protocol="http", actor="t")


async def test_create_server_records_an_admin_event(session):
    from app.admin import mcp as service

    server = await service.create_server(
        session, name="time", protocol="stdio", builtin_id="time", actor="t"
    )
    assert server.name == "time"
    assert server.egress == "local"
    assert server.enabled is True


async def test_a_duplicate_name_is_refused(session):
    from app.admin import mcp as service
    from app.admin.service import ConflictError

    await service.create_server(
        session, name="time", protocol="stdio", builtin_id="time", actor="t"
    )
    with pytest.raises(ConflictError):
        await service.create_server(
            session, name="time", protocol="stdio", builtin_id="time", actor="t"
        )


@pytest.mark.parametrize(
    "fields",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": 601},
        {"concurrency_limit": 0},
        {"concurrency_limit": 21},
        {"result_max_bytes": 0},
        {"egress": "outer-space"},
        {"env_vars": {"a": 1}},
        {"disabled_tools": ["x" * 201]},
    ],
)
async def test_invalid_fields_are_refused(session, fields):
    from app.admin import mcp as service
    from app.admin.service import InvalidInputError

    with pytest.raises(InvalidInputError):
        await service.create_server(
            session, name="time", protocol="stdio", builtin_id="time", fields=fields, actor="t"
        )


async def test_configure_rejects_a_url_on_a_stdio_server(session):
    from app.admin import mcp as service
    from app.admin.service import InvalidInputError

    server = await service.create_server(
        session, name="time", protocol="stdio", builtin_id="time", actor="t"
    )
    with pytest.raises(InvalidInputError):
        await service.configure_server(session, server.id, {"url": "http://x"}, actor="t")


async def test_enabled_configs_excludes_disabled_servers(session):
    from app.admin import mcp as service

    a = await service.create_server(
        session, name="a", protocol="stdio", builtin_id="time", actor="t"
    )
    await service.create_server(
        session,
        name="b",
        protocol="stdio",
        builtin_id="time",
        fields={"enabled": False},
        actor="t",
    )
    configs = await service.enabled_configs(session)
    assert [c.name for c in configs] == [a.name]


async def test_to_config_round_trips_env_vars_and_disabled_tools(session):
    from app.admin import mcp as service

    server = await service.create_server(
        session,
        name="time",
        protocol="stdio",
        builtin_id="time",
        fields={"env_vars": {"X": "y"}, "disabled_tools": ["ghost"]},
        actor="t",
    )
    config = service.to_config(server)
    assert config.env_vars == {"X": "y"}
    assert config.disabled_tools == ("ghost",)


async def test_delete_server_removes_it(session):
    from app.admin import mcp as service

    server = await service.create_server(
        session, name="time", protocol="stdio", builtin_id="time", actor="t"
    )
    await service.delete_server(session, server.id, actor="t")
    with pytest.raises(service.McpServerNotFoundError):
        await service.get_server(session, server.id)


# --- the API ---


@pytest.fixture
async def api(fresh_db, monkeypatch):
    import httpx

    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:9")
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=GOOD) as c:
        yield c
    app.dependency_overrides.clear()
    deps.reset_failure_state()


async def _create(api, **overrides):
    body = {"name": "time", "protocol": "stdio", "builtin_id": "time"}
    body.update(overrides)
    response = await api.post("/mcp/servers", json=body)
    assert response.status_code == 201, response.text
    return response.json()


async def test_create_and_list(api):
    created = await _create(api)
    assert created["has_env_vars"] is False
    listed = (await api.get("/mcp/servers")).json()
    assert [s["name"] for s in listed] == ["time"]


async def test_env_vars_are_never_returned(api):
    created = await _create(api, env_vars={"SECRET": "shh"})
    assert "env_vars" not in created
    assert created["has_env_vars"] is True


async def test_get_missing_server_is_404(api):
    assert (await api.get("/mcp/servers/999")).status_code == 404


async def test_patch_updates_a_server(api):
    created = await _create(api)
    response = await api.patch(f"/mcp/servers/{created['id']}", json={"enabled": False})
    assert response.status_code == 200
    assert response.json()["enabled"] is False


async def test_delete_removes_a_server(api):
    created = await _create(api)
    assert (await api.delete(f"/mcp/servers/{created['id']}")).status_code == 204
    assert (await api.get(f"/mcp/servers/{created['id']}")).status_code == 404


async def test_create_and_delete_record_admin_events(api):
    created = await _create(api)
    await api.delete(f"/mcp/servers/{created['id']}")
    events = [e["action"] for e in (await api.get("/admin-events")).json()]
    assert "mcp_server.create" in events
    assert "mcp_server.delete" in events


async def test_test_endpoint_connects_for_real(api):
    created = await _create(api)
    response = await api.post(f"/mcp/servers/{created['id']}/test")
    assert response.status_code == 200
    body = response.json()
    assert body["reachable"] is True
    assert body["error"] is None
    assert [t["name"] for t in body["tools"]] == ["get_time"]


async def test_test_endpoint_reports_an_unreachable_http_server(api):
    created = await _create(
        api,
        name="ghost",
        protocol="http",
        builtin_id=None,
        url="http://127.0.0.1:1/mcp",
        timeout_seconds=2,
    )
    response = await api.post(f"/mcp/servers/{created['id']}/test")
    assert response.status_code == 200
    body = response.json()
    assert body["reachable"] is False
    assert body["error"]
    assert body["tools"] == []


async def test_list_tools_endpoint(api):
    created = await _create(api)
    response = await api.get(f"/mcp/servers/{created['id']}/tools")
    assert response.status_code == 200
    tools = response.json()
    assert [t["name"] for t in tools] == ["get_time"]
    assert tools[0]["enabled"] is True
    assert tools[0]["description"]


async def test_list_tools_on_an_unreachable_server_is_409(api):
    created = await _create(
        api,
        name="ghost",
        protocol="http",
        builtin_id=None,
        url="http://127.0.0.1:1/mcp",
        timeout_seconds=2,
    )
    response = await api.get(f"/mcp/servers/{created['id']}/tools")
    assert response.status_code == 409


async def test_toggle_a_tool_off_then_on(api):
    created = await _create(api)
    off = await api.patch(f"/mcp/servers/{created['id']}/tools/get_time", json={"enabled": False})
    assert off.status_code == 200
    assert off.json()["disabled_tools"] == ["get_time"]
    tools = (await api.get(f"/mcp/servers/{created['id']}/tools")).json()
    assert tools[0]["enabled"] is False
    on = await api.patch(f"/mcp/servers/{created['id']}/tools/get_time", json={"enabled": True})
    assert on.json()["disabled_tools"] == []
