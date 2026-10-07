"""Tests: admin export of one user's conversation, itself audited.

- JSON and Markdown; the export holds N messages and N equals the database count, for the
  user or one of their agents; the text only when asked (otherwise its length).
- One admin event per export; a scope below admin gets 403; an unknown user or another
  user's agent 404; a bad format 422.
- A channel user's text is fenced in the Markdown and escaped in the UI.
"""

import json
import sqlite3

import httpx
import pytest

from app.admin import service
from app.db.models import Channel, Direction

KEY = "Eq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
AUTH = {"Authorization": f"Bearer {KEY}"}
PAYLOAD = "<script>alert(1)</script> ``` fence"


def _sql(query, *args):
    from app.config import get_settings

    con = sqlite3.connect(get_settings().database_url.split("///", 1)[1])
    try:
        return con.execute(query, args).fetchall()
    finally:
        con.close()


@pytest.fixture
async def api(fresh_db, monkeypatch):
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db, session_scope

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    async with session_scope() as s:
        alice, bob = await service.create_user(s, "Alice"), await service.create_user(s, "Bob")
        one = await service.create_agent(s, alice.id, "one")
        two = await service.create_agent(s, alice.id, "two")
        other = await service.create_agent(s, bob.id, "bob's")
        for agent, direction, text in (
            (one, Direction.INBOUND, "hello"),
            (one, Direction.OUTBOUND, "hi Alice"),
            (two, Direction.INBOUND, PAYLOAD),
            (two, Direction.OUTBOUND, "noted"),
            (other, Direction.INBOUND, "Bob here"),
        ):
            await service.record_action(
                s, user_id=agent.user_id, agent_id=agent.id, channel=Channel.TELEGRAM,
                direction=direction, text=text,
            )  # fmt: skip
        await s.commit()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=AUTH) as c:
        yield c
    get_settings.cache_clear()


def _events():
    return _sql(
        "select actor, target_id, details from admin_events where action = ?", "conversation.export"
    )


async def test_the_export_holds_every_message_of_the_user_and_nothing_else(api):
    body = (await api.get("/users/1/conversations/export")).json()
    count = _sql("select count(*) from action_logs where user_id = 1")[0][0]
    assert body["about"]["messages"] == len(body["messages"]) == count == 4
    assert {m["agent_id"] for m in body["messages"]} == {1, 2}
    assert all("text" not in m and m["length"] > 0 for m in body["messages"])
    per_agent = (await api.get("/users/1/conversations/export", params={"agent_id": 2})).json()
    assert (
        len(per_agent["messages"])
        == _sql("select count(*) from action_logs where agent_id = 2")[0][0]
        == 2
    )


async def test_the_text_is_there_only_when_asked(api):
    body = (await api.get("/users/1/conversations/export", params={"include_text": "true"})).json()
    assert [m["text"] for m in body["messages"]] == ["hello", "hi Alice", PAYLOAD, "noted"]


async def test_each_export_is_one_admin_event(api):
    await api.get("/users/1/conversations/export")
    await api.get(
        "/users/1/conversations/export", params={"format": "markdown", "include_text": "true"}
    )
    from app.db.session import session_scope

    rows = _events()
    assert len(rows) == 2 and all(r[0] == "api" and r[1] == 1 for r in rows)
    async with session_scope() as db:  # the details are encrypted at rest: read them decrypted
        events = await service.search_admin_events(db, action="conversation.export")
    assert json.loads(events[0].details) == {
        "agent_id": None, "format": "markdown", "with_text": True, "messages": 4
    }  # fmt: skip


async def test_the_markdown_fences_the_text(api):
    response = await api.get(
        "/users/1/conversations/export", params={"format": "markdown", "include_text": "true"}
    )
    assert response.status_code == 200 and response.headers["content-type"].startswith(
        "text/markdown"
    )
    text = response.text
    assert text.startswith("# Conversation of Alice") and "4 message(s)." in text
    assert f"````text\n{PAYLOAD}\n````" in text, "a text holding ``` gets a longer fence"
    assert "```text\nhello\n```" in text


async def test_refusals(api):
    from app.api.app import app
    from app.api.scopes import Principal, Scope, get_principal

    assert (await api.get("/users/9/conversations/export")).status_code == 404
    assert (
        await api.get("/users/1/conversations/export", params={"agent_id": 3})
    ).status_code == 404
    assert (
        await api.get("/users/1/conversations/export", params={"format": "pdf"})
    ).status_code == 422
    app.dependency_overrides[get_principal] = lambda: Principal("api", Scope.OPERATE)
    try:
        assert (await api.get("/users/1/conversations/export")).status_code == 403
    finally:
        app.dependency_overrides.clear()
    assert _events() == []


async def test_the_ui_offers_the_export_and_shows_it_as_text(api, monkeypatch):
    import re

    from app.server import root
    from app.ui import security

    security.sessions.clear()
    security.login_limiter.clear()
    transport = httpx.ASGITransport(app=root, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="https://t") as ui:
        page = await ui.get("/ui/login")
        csrf = re.search(r'name="csrf" value="([^"]+)"', page.text).group(1)
        await ui.post("/ui/login", data={"csrf": csrf, "key": KEY})
        users = (await ui.get("/ui/op/list-users")).text
        assert 'href="/ui/op/export-conversation?user_id=1"' in users
        logs = (await ui.get("/ui/logs", params={"user_id": 1})).text
        assert (
            "/ui/op/export-conversation?user_id=1&amp;format=markdown&amp;include_text=true" in logs
        )
        export = await ui.get(
            "/ui/op/export-conversation",
            params={"user_id": 1, "format": "markdown", "include_text": "true"},
        )
    assert (
        "&lt;script&gt;alert(1)&lt;/script&gt;" in export.text
        and "<script>alert" not in export.text
    )
    security.sessions.clear()
