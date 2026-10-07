"""Tests: reading RSS and Atom feeds, and what a scheduled task remembers.

The parser (RSS 2.0, RSS 1.0, Atom, HTML reduced to text, dates, ids, entity declarations
refused), the fetch through the real guard with a scripted network (a resolver and an httpx
transport that answer chosen addresses and documents), the built-in server's tool, the list of
feeds for the builder, and, through a real task turn with a scripted engine, the application's
filter: a second run of the same task gets 0 of the items the first one delivered.
"""

import json
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
from sqlalchemy import func, select

from app import feed_memory, feeds
from app.security.outbound import Guard
from app.web_fetch import FetchError, Limits, fetch_bytes

RSS = b"""<?xml version="1.0" encoding="ISO-8859-1"?>
<rss version="2.0"><channel><title>AI &amp; Code news</title>
<item><title>Older release</title><link>https://example.org/a</link>
<pubDate>Mon, 21 Sep 2026 08:00:00 +0000</pubDate><description>First.</description></item>
<item><title>Model \xe9t\xe9 released</title><link>https://example.org/b</link>
<pubDate>Thu, 24 Sep 2026 09:30:00 +0200</pubDate>
<description>&lt;p&gt;A &lt;b&gt;new&lt;/b&gt; open model.&lt;/p&gt;</description></item>
<item><title>No date</title><link>https://example.org/c</link></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Releases</title>
<entry><title>v2.0</title><link rel="self" href="https://example.org/self"/>
<link rel="alternate" href="https://example.org/v2"/><updated>2026-09-27T10:00:00Z</updated>
<summary type="html">&lt;i&gt;Big&lt;/i&gt; release</summary></entry>
<entry><title>v1.0</title><link href="https://example.org/v1"/>
<published>2026-01-01T00:00:00Z</published><content>Old</content></entry>
</feed>"""

RSS1 = b"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
 xmlns="http://purl.org/rss/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/">
<channel><title>arXiv cs.AI</title></channel>
<item><title>A paper</title><link>https://example.org/p1</link>
<description>Abstract</description><dc:date>2026-09-25T00:00:00Z</dc:date></item>
</rdf:RDF>"""

LAUGHS = b"""<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">
<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">]>
<rss><channel><item><title>&lol2;</title></item></channel></rss>"""


# --- the parser ---


def test_rss_two_items_newest_first_with_text_dates_and_ids():
    title, items = feeds.parse(RSS)
    assert title == "AI & Code news"
    assert [i.title for i in items] == ["Model été released", "Older release", "No date"]
    newest = items[0]
    assert newest.summary == "A new open model."
    assert newest.published == datetime(2026, 9, 24, 7, 30, tzinfo=UTC)
    assert newest.link == "https://example.org/b" and len(newest.id) == 16
    assert newest.id == feeds.item_id("https://example.org/b", "", None), "the id is the link's"
    assert items[2].published is None


def test_atom_takes_the_alternate_link_and_published_or_updated():
    title, items = feeds.parse(ATOM)
    assert title == "Releases"
    assert [(i.title, i.link) for i in items] == [
        ("v2.0", "https://example.org/v2"), ("v1.0", "https://example.org/v1"),
    ]  # fmt: skip
    assert items[0].summary == "Big release"
    assert items[0].published == datetime(2026, 9, 27, 10, 0, tzinfo=UTC)


def test_rss_one_is_read():
    title, items = feeds.parse(RSS1)
    assert title == "arXiv cs.AI" and [(i.title, i.summary) for i in items] == [
        ("A paper", "Abstract")
    ]


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (LAUGHS, "declares entities"),
        (b"<html><body>Not a feed</body></html>", "not an RSS or Atom feed"),
        (b"plain words", "not a readable XML feed"),
        (b"<rss><channel><item>", "not a readable XML feed"),
    ],
)
def test_what_is_not_a_feed_is_refused_with_the_reason(data, reason):
    with pytest.raises(feeds.FeedError, match=reason):
        feeds.parse(data)


def test_select_filters_by_date_and_caps_the_count():
    _title, items = feeds.parse(RSS)
    since = datetime(2026, 9, 22, tzinfo=UTC)
    assert [i.title for i in feeds.select(items, since, 20)] == ["Model été released"]
    assert len(feeds.select(items, None, 1)) == 1
    many = items * 30
    assert len(feeds.select(many, None, 500)) == feeds.MAX_ITEMS
    assert len(feeds.select(many, None, 0)) == 1


def test_render_and_split_blocks_round_trip():
    _title, items = feeds.parse(RSS)
    text = feeds.render("AI", "https://example.org/rss", items)
    head, blocks = feeds.split_blocks(text)
    assert head.startswith("Feed: AI\nSource: https://example.org/rss\nItems: 3")
    assert [i for i, _b in blocks] == [i.id for i in items]
    assert blocks[0][1].startswith("### Model été released\nid: ")


# --- the fetch, through the real guard ---

PUBLIC = "93.184.216.34"
DNS = {"example.org": [PUBLIC], "inside.example.org": ["10.0.0.5"]}


def resolver(host, port):
    if host not in DNS:
        raise OSError("no such host")
    return DNS[host]


async def get(url, response):
    limits = Limits(allowed_hosts=frozenset({"*"}))
    sent = []

    def answer(request):
        sent.append(str(request.url))
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        guard = Guard(allowed_hosts=limits.allowed_hosts, resolver=resolver)
        try:
            return await fetch_bytes(
                url, limits, feeds.FEED_TYPES, guard=guard, client=client
            ), sent
        except FetchError as exc:
            return exc, sent


def feed_response(data=RSS, kind="application/rss+xml"):
    return httpx.Response(200, content=data, headers={"content-type": kind})


async def test_a_public_feed_is_fetched_as_bytes():
    (final, data), sent = await get("https://example.org/rss", feed_response())
    assert final == "https://example.org/rss" and data == RSS and len(sent) == 1
    assert len(feeds.parse(data)[1]) == 3


async def test_a_private_address_is_refused_and_nothing_is_sent():
    result, sent = await get("https://inside.example.org/rss", feed_response())
    assert isinstance(result, FetchError) and "non-public address" in str(result)
    assert sent == []


@pytest.mark.parametrize("url", ["http://example.org/rss", "https://127.0.0.1/rss"])
async def test_http_and_a_loopback_address_are_refused(url):
    result, sent = await get(url, feed_response())
    assert isinstance(result, FetchError) and sent == []


async def test_a_web_page_is_not_a_feed():
    result, _sent = await get("https://example.org/", feed_response(b"<html/>", "text/html"))
    assert isinstance(result, FetchError) and "not a feed (text/html)" in str(result)


async def test_no_allowed_host_refuses_before_any_request():
    with pytest.raises(FetchError, match="set WEB_FETCH_ALLOWED_HOSTS"):
        await fetch_bytes("https://example.org/rss", Limits(allowed_hosts=frozenset()),
                          feeds.FEED_TYPES)  # fmt: skip


# --- the built-in server's tool ---


async def test_read_feed_tool_renders_the_items(monkeypatch):
    from app.mcp.builtin_servers import feeds_server

    async def fake_fetch(url, limits, types):
        return url, RSS

    monkeypatch.setattr(feeds_server, "fetch_bytes", fake_fetch)
    text = await feeds_server.read_feed("https://example.org/rss", since="2026-09-22")
    assert "Items: 1" in text and "### Model été released" in text
    assert (await feeds_server.read_feed("https://x.org/rss", since="yesterday")).startswith(
        "error: since is a date"
    )


async def test_read_feed_tool_reports_errors_as_text(monkeypatch):
    from app.mcp.builtin_servers import feeds_server

    async def fake_fetch(url, limits, types):
        return url, b"<html/>"

    monkeypatch.setattr(feeds_server, "fetch_bytes", fake_fetch)
    assert (await feeds_server.read_feed("https://example.org/")).startswith("error: not an RSS")


def test_the_feeds_server_is_read_only_open_world_and_gets_only_the_fetch_settings():
    from app.mcp import builtin
    from app.mcp.builtin_servers import feeds_server

    assert builtin.REGISTRY["feeds"] == "app.mcp.builtin_servers.feeds_server"
    assert builtin.BUILTIN_SETTINGS["feeds"] == (
        "WEB_FETCH_ALLOWED_HOSTS", "WEB_FETCH_MAX_BYTES", "WEB_FETCH_TIMEOUT_SECONDS",
        "WEB_FETCH_USER_AGENT",
    )  # fmt: skip
    tool = feeds_server.mcp._tool_manager.get_tool("read_feed")
    assert (tool.annotations.readOnlyHint, tool.annotations.openWorldHint) == (True, True)


# --- a task's memory, through real task turns ---


class _Scripted(BaseHTTPRequestHandler):
    responses: list = []
    requests: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.loads(raw)
        type(self).requests.append(body)
        if body.get("tool_choice") == "required":
            payload = _message(tool_calls=[_call("_capability_probe", {})])
        else:
            payload = type(self).responses.pop(0)
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _message(content="", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


def _call(name, arguments):
    return {"id": "c1", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}  # fmt: skip


KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"


@pytest.fixture
async def env(fresh_db, monkeypatch):
    """Sam (Telegram 555) with the built-in `feeds` server approved and granted, an agent
    with its tool, and a task holding a standing approval for it. The server's network
    fetch is replaced by the RSS document above; task results are captured."""
    from app import tools as tools_module
    from app.api import deps
    from app.api.app import app
    from app.channels import notify
    from app.config import get_settings
    from app.db.session import init_db
    from app.mcp.manager import ManagedServer

    _Scripted.requests, _Scripted.responses = [], []
    server = HTTPServer(("127.0.0.1", 0), _Scripted)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    tools_module.reset_capability_check()
    await init_db()
    real_call = ManagedServer.call_tool
    fetched = []

    async def call_tool(self, name, arguments):
        if self.config.builtin_id == "feeds" and name == "read_feed":
            fetched.append(arguments["url"])
            title, items = feeds.parse(RSS)
            return feeds.render(title, arguments["url"], items), False
        return await real_call(self, name, arguments)

    monkeypatch.setattr(ManagedServer, "call_tool", call_tool)
    delivered = []

    async def sender(external_id, text):
        delivered.append(text)

    notify.register_sender(Channel.TELEGRAM, sender)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    headers = {"Authorization": f"Bearer {KEY}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as api:
        user = (await api.post("/users", json={"display_name": "Sam"})).json()["id"]
        channel = {"channel": "telegram", "identifier": "555"}
        identity = (await api.post(f"/users/{user}/channels", json=channel)).json()["id"]
        await api.post(f"/users/{user}/channels/{identity}/permissions", json={"kind": "chat"})
        created = await api.post(
            "/mcp/servers", json={"name": "feeds", "protocol": "stdio", "builtin_id": "feeds"}
        )
        approved = await api.post(
            f"/mcp/servers/{created.json()['id']}/approve-definitions", json={}
        )
        assert approved.status_code == 200, approved.text
        await api.put("/mcp/grants", json={"user_id": user, "grants": [{"server_name": "feeds"}]})
        agent = await api.post(
            f"/users/{user}/agents", json={"name": "news", "tools": ["mcp__feeds__read_feed"]}
        )
        task = await api.post("/tasks", json={
            "user_id": user, "agent_id": agent.json()["id"], "prompt": "Digest of the feed.",
            "kind": "daily", "expr": "09:00", "channel_identity_id": identity,
            "standing_tools": ["mcp__feeds__read_feed"]})  # fmt: skip
        assert task.status_code == 201, task.text
        api.task_id, api.delivered, api.fetched = task.json()["id"], delivered, fetched
        yield api
    from app.mcp.manager import manager as shared_manager

    notify.unregister_sender(Channel.TELEGRAM)
    await shared_manager.reset()
    app.dependency_overrides.clear()
    server.shutdown()
    get_settings.cache_clear()
    deps.reset_failure_state()
    tools_module.reset_capability_check()


from app.db.models import Channel, McpCall, TaskFeedItem  # noqa: E402


def _run_script():
    _Scripted.responses = [
        _message(tool_calls=[_call("mcp__feeds__read_feed", {"url": "https://example.org/rss"})]),
        _message(content="Here is your digest."),
    ]


def _tool_text() -> str:
    return next(m for m in _Scripted.requests[-1]["messages"] if m.get("role") == "tool")["content"]


async def _remembered(task_id) -> int:
    from app.db.session import session_scope

    async with session_scope() as session:
        stmt = select(func.count()).select_from(TaskFeedItem).where(TaskFeedItem.task_id == task_id)
        return (await session.execute(stmt)).scalar_one()


async def test_the_second_run_of_a_task_gets_none_of_the_items_the_first_delivered(env):
    from app import tasks

    _run_script()
    first = await tasks.execute(env.task_id, trigger="manual")
    first_text = _tool_text()
    assert first["status"] == "ok" and "Items: 3" in first_text
    assert first_text.count("\n### ") == 3
    assert await _remembered(env.task_id) == 3

    _run_script()
    second = await tasks.execute(env.task_id, trigger="manual")
    second_text = _tool_text()
    print(second_text)
    assert second["status"] == "ok"
    assert "Items: 0" in second_text and "### " not in second_text
    assert "(3 items this task already delivered were left out)" in second_text
    assert feed_memory.NOTHING_NEW in second_text
    assert env.fetched == ["https://example.org/rss"] * 2
    from app.db.session import session_scope

    async with session_scope() as session:
        decisions = (await session.execute(select(McpCall.decision))).scalars().all()
    assert decisions == ["standing", "standing"]


async def test_an_undelivered_run_remembers_nothing(env):
    from app import tasks
    from app.channels import notify

    notify.unregister_sender(Channel.TELEGRAM)  # no running Telegram adapter
    _run_script()
    result = await tasks.execute(env.task_id, trigger="manual")
    assert result["status"] == "undelivered"
    assert await _remembered(env.task_id) == 0


async def test_a_chat_turn_is_not_filtered():
    text = feeds.render("t", "https://example.org/rss", feeds.parse(RSS)[1])
    assert await feed_memory.filter_result("feeds", text) == text


async def test_another_server_is_not_filtered():
    token = feed_memory.start(1)
    try:
        items = feeds.parse(RSS)[1]
        feed_memory.current_task.get()["seen"] = {i.id for i in items}  # all delivered before
        text = feeds.render("t", "https://example.org/rss", items)
        assert await feed_memory.filter_result("web", text) == text
        assert "Items: 0" in await feed_memory.filter_result("feeds", text)
    finally:
        feed_memory.current_task.reset(token)


async def test_remember_keeps_the_newest_and_deleting_the_task_forgets(env, monkeypatch):
    from app import tasks
    from app.db.session import session_scope

    monkeypatch.setattr(feed_memory, "MAX_REMEMBERED", 3)
    await feed_memory.remember(env.task_id, ["a" * 16, "b" * 16])
    assert await feed_memory.remember(env.task_id, ["b" * 16, "c" * 16, "d" * 16]) == 2
    async with session_scope() as session:
        stmt = select(TaskFeedItem.item_id).order_by(TaskFeedItem.id)
        kept = (await session.execute(stmt)).scalars().all()
        assert kept == ["b" * 16, "c" * 16, "d" * 16]
        await tasks.delete_task(session, env.task_id, actor="test")
        await session.commit()
    assert await _remembered(env.task_id) == 0


# --- the list of feeds for the builder ---


async def test_feed_list_through_the_api(env):
    created = await env.post(
        "/feeds",
        json={"name": "AI news", "url": "https://example.org/ai.rss", "topics": "ai,  models"},
    )
    assert created.status_code == 201 and created.json()["topics"] == "ai, models"
    taken = await env.post("/feeds", json={"name": "AI news", "url": "https://x.org/f"})
    assert taken.status_code == 409
    plain = await env.post("/feeds", json={"name": "Plain", "url": "http://example.org/f.rss"})
    assert plain.status_code == 422 and "https URL" in plain.json()["detail"]
    assert [f["name"] for f in (await env.get("/feeds")).json()] == ["AI news"]
    from app.api.app import app
    from app.api.scopes import Principal, Scope, get_principal

    app.dependency_overrides[get_principal] = lambda: Principal("op", Scope.OPERATE)
    below = await env.post("/feeds", json={"name": "B", "url": "https://x.org/f"})
    assert below.status_code == 403
    assert (await env.get("/feeds")).status_code == 200
    app.dependency_overrides.clear()
    gone = await env.delete(f"/feeds/{created.json()['id']}")
    assert gone.status_code == 204 and (await env.get("/feeds")).json() == []


async def test_a_feed_served_as_octet_stream_is_read():
    # blog.python.org serves its Atom feed as application/octet-stream (measured 2026-09-28).
    (_final, data), _sent = await get(
        "https://example.org/atom", feed_response(ATOM, "application/octet-stream")
    )
    assert [i.title for i in feeds.parse(data)[1]] == ["v2.0", "v1.0"]


async def test_each_run_of_a_task_starts_from_an_empty_conversation(env):
    from app import tasks

    _run_script()
    await tasks.execute(env.task_id, trigger="manual")
    _run_script()
    await tasks.execute(env.task_id, trigger="manual")
    turn = [r for r in _Scripted.requests if r.get("tool_choice") is None]
    second_run_first_request = turn[2]
    texts = [str(m.get("content")) for m in second_run_first_request["messages"]]
    assert not any("Here is your digest." in t for t in texts), "the first run's reply is gone"
    assert sum("Digest of the feed." in t for t in texts) == 1


async def test_nothing_new_is_said_only_when_every_item_was_delivered():
    items = feeds.parse(RSS)[1]
    text = feeds.render("t", "https://example.org/rss", items)
    token = feed_memory.start(1)
    try:
        feed_memory.current_task.get()["seen"] = {items[0].id}
        some = await feed_memory.filter_result("feeds", text)
        assert "Items: 2" in some and feed_memory.NOTHING_NEW not in some
    finally:
        feed_memory.current_task.reset(token)
