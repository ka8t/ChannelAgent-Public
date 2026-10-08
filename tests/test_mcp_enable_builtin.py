"""Tests: POST /mcp/builtins/{builtin_id}/enable, one call that declares a built-in
server when needed, enables it, approves its new definitions, grants it to every agent of the
user and adds its tools to an agent, adding and never replacing. Servers are real stdio
subprocesses (the built-in "time", "calc" and "search" servers).
"""

import pytest
from sqlalchemy import select

from app.api.scopes import Principal, Scope, get_principal
from app.db.models import AdminEvent, McpGrant, McpServer

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
GOOD = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
async def api(fresh_db, monkeypatch):
    import httpx

    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("SEARXNG_URL", "")
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=GOOD) as c:
        c.app = app
        yield c
    app.dependency_overrides.clear()
    deps.reset_failure_state()
    get_settings.cache_clear()


async def _user(api, name="Sam") -> int:
    return (await api.post("/users", json={"display_name": name})).json()["id"]


async def _agent(api, user: int, name="a", tools=None) -> int:
    body = {"name": name, **({"tools": tools} if tools is not None else {})}
    r = await api.post(f"/users/{user}/agents", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _rows(model, **where):
    from app.db.session import session_scope

    async with session_scope() as session:
        stmt = select(model).filter_by(**where)
        return list((await session.execute(stmt)).scalars())


async def _enable(api, builtin: str, **body):
    return await api.post(f"/mcp/builtins/{builtin}/enable", json=body)


async def test_one_call_declares_approves_grants_and_lists_the_tool(api):
    user = await _user(api)
    agent = await _agent(api, user)
    r = await _enable(api, "time", user_id=user, agent_id=agent)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["created"] is True
    assert body["server"] == "time"
    assert body["approved"] == ["get_time"]
    assert body["granted"] is True
    assert body["added_tools"] == ["mcp__time__get_time"]
    assert body["missing_setting"] is None

    [server] = await _rows(McpServer, builtin_id="time")
    assert server.enabled and server.egress == "local"
    assert sorted(server.approved_definitions) == ["get_time"]
    [grant] = await _rows(McpGrant, user_id=user)
    assert (grant.agent_id, grant.server_name, grant.tool_name) == (None, "time", None)
    shown = (await api.get(f"/agents/{agent}")).json()
    assert shown["tools"] == ["mcp__time__get_time"]
    assert len(await _rows(AdminEvent, action="mcp_builtin.enable")) == 1

    # The tool is now offered to a turn of that agent: the catalogue's own rules agree.
    exposure = (await api.get(f"/agents/{agent}/exposure")).json()
    assert [(t["name"], t["counted"]) for t in exposure["tools"]] == [
        ("mcp__time__get_time", True)
    ]


async def test_internet_builtins_are_declared_with_egress_internet(api):
    user = await _user(api)
    r = await _enable(api, "search", user_id=user)
    assert r.status_code == 200, r.text
    [server] = await _rows(McpServer, builtin_id="search")
    assert server.egress == "internet"


async def test_a_second_call_changes_nothing(api):
    user = await _user(api)
    agent = await _agent(api, user)
    await _enable(api, "time", user_id=user, agent_id=agent)
    again = (await _enable(api, "time", user_id=user, agent_id=agent)).json()
    assert (again["created"], again["enabled"], again["approved"], again["granted"]) == (
        False, False, [], False,
    )  # fmt: skip
    assert again["added_tools"] == []
    assert len(await _rows(McpServer, builtin_id="time")) == 1
    assert len(await _rows(McpGrant, user_id=user)) == 1
    assert (await api.get(f"/agents/{agent}")).json()["tools"] == ["mcp__time__get_time"]


async def test_it_adds_to_the_grants_and_tools_already_there(api):
    user = await _user(api)
    agent = await _agent(api, user, tools=["mcp__other__thing"])
    await _enable(api, "time", user_id=user)
    put = await api.put(
        "/mcp/grants",
        json={"user_id": user, "grants": [
            {"server_name": "time", "agent_id": agent, "tool_name": "get_time"},
        ]},
    )  # fmt: skip
    assert put.status_code == 200, put.text
    r = await _enable(api, "calc", user_id=user, agent_id=agent)
    assert r.status_code == 200, r.text
    grants = {(g.agent_id, g.server_name, g.tool_name) for g in await _rows(McpGrant, user_id=user)}
    assert grants == {(agent, "time", "get_time"), (None, "calc", None)}
    assert (await api.get(f"/agents/{agent}")).json()["tools"] == [
        "mcp__other__thing",
        "mcp__calc__calculate",
        "mcp__calc__convert",
    ]


async def test_a_server_already_declared_under_another_name_is_reused_and_reenabled(api):
    r = await api.post(
        "/mcp/servers",
        json={"name": "clock", "protocol": "stdio", "builtin_id": "time", "enabled": False},
    )
    assert r.status_code == 201, r.text
    user = await _user(api)
    agent = await _agent(api, user)
    body = (await _enable(api, "time", user_id=user, agent_id=agent)).json()
    assert (body["server"], body["created"], body["enabled"]) == ("clock", False, True)
    assert body["added_tools"] == ["mcp__clock__get_time"]
    [server] = await _rows(McpServer, builtin_id="time")
    assert (server.name, server.enabled) == ("clock", True)


async def test_a_tool_turned_off_on_the_server_is_not_added_to_the_agent(api):
    user = await _user(api)
    agent = await _agent(api, user)
    await _enable(api, "calc", user_id=user)
    [server] = await _rows(McpServer, builtin_id="calc")
    r = await api.patch(f"/mcp/servers/{server.id}", json={"disabled_tools": ["convert"]})
    assert r.status_code == 200, r.text
    body = (await _enable(api, "calc", user_id=user, agent_id=agent)).json()
    assert body["added_tools"] == ["mcp__calc__calculate"]


async def test_a_changed_definition_is_refused_and_nothing_is_written(api):
    from app.db.session import session_scope

    user = await _user(api)
    await _enable(api, "time", user_id=user)
    async with session_scope() as session:
        server = (await session.execute(select(McpServer))).scalar_one()
        entry = dict(server.approved_definitions["get_time"])
        entry["sha256"] = "0" * 64
        server.approved_definitions = {"get_time": entry}
        server.enabled = False
        await session.commit()
    other = await _user(api, "Kim")
    agent = await _agent(api, other)
    r = await _enable(api, "time", user_id=other, agent_id=agent)
    assert r.status_code == 409
    assert "get_time" in r.json()["detail"] and "list-tools" in r.json()["detail"]
    [server] = await _rows(McpServer, builtin_id="time")
    assert server.enabled is False
    assert server.approved_definitions["get_time"]["sha256"] == "0" * 64
    assert await _rows(McpGrant, user_id=other) == []
    assert (await api.get(f"/agents/{agent}")).json()["tools"] in (None, [])


async def test_the_empty_required_setting_is_named_with_its_command(api, monkeypatch):
    from app.config import get_settings

    user = await _user(api)
    body = (await _enable(api, "search", user_id=user)).json()
    assert body["missing_setting"] == {
        "name": "SEARXNG_URL",
        "needs": "the address of a SearXNG instance; SEARXNG_MANAGED=true makes start.sh run one",
        "command": "./start.sh --config SEARXNG_URL=...",
    }
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")
    get_settings.cache_clear()
    body = (await _enable(api, "search", user_id=user)).json()
    assert body["missing_setting"] is None


@pytest.mark.parametrize(
    ("path_builtin", "body", "status"),
    [
        ("shell", {"user_id": 1}, 404),
        ("time", {"user_id": 999}, 404),
    ],
)
async def test_unknown_builtin_or_user_is_refused(api, path_builtin, body, status):
    await _user(api)
    r = await _enable(api, path_builtin, **body)
    assert r.status_code == status
    assert await _rows(McpServer) == []


async def test_an_agent_of_another_user_is_refused(api):
    owner = await _user(api)
    agent = await _agent(api, owner)
    other = await _user(api, "Kim")
    r = await _enable(api, "time", user_id=other, agent_id=agent)
    assert r.status_code == 422
    assert await _rows(McpServer) == []
    assert await _rows(McpGrant) == []


async def test_it_needs_the_admin_scope(api):
    user = await _user(api)
    api.app.dependency_overrides[get_principal] = lambda: Principal("tester", Scope.OPERATE)
    r = await _enable(api, "time", user_id=user)
    assert r.status_code == 403
    api.app.dependency_overrides.clear()
    assert await _rows(McpServer) == []


async def test_the_tool_limit_of_an_agent_still_holds(api):
    user = await _user(api)
    agent = await _agent(api, user, tools=[f"mcp__x__t{i}" for i in range(99)])
    r = await _enable(api, "calc", user_id=user, agent_id=agent)
    assert r.status_code == 422
    assert "at most 100 tools" in r.json()["detail"]
    assert len((await api.get(f"/agents/{agent}")).json()["tools"]) == 99
    assert await _rows(McpServer) == []


async def test_a_grant_for_one_agent_does_not_count_as_a_grant_for_every_agent(api):
    user = await _user(api)
    agent = await _agent(api, user)
    await _enable(api, "time", user_id=user)
    put = await api.put(
        "/mcp/grants",
        json={"user_id": user, "grants": [{"server_name": "time", "agent_id": agent}]},
    )
    assert put.status_code == 200, put.text
    body = (await _enable(api, "time", user_id=user)).json()
    assert body["granted"] is True
    grants = {(g.agent_id, g.server_name, g.tool_name) for g in await _rows(McpGrant, user_id=user)}
    assert grants == {(agent, "time", None), (None, "time", None)}
