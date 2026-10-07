"""Tests for the built-in MCP server "search": a web search through the owner's SearXNG,
only that address, the query as words only, no redirect followed, results capped and links to
the inside dropped."""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from app.search import MAX_RESULTS, SearchError, web_search


class _SearXNG(BaseHTTPRequestHandler):
    """Answers from `reply`: (status, content type, body bytes, extra headers)."""

    reply: tuple = (200, "application/json", b"{}", {})
    requests: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        type(self).requests.append((self.path, self.headers.get("Host")))
        status, kind, body, extra = type(self).reply
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for name, value in extra.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


def _results(*items):
    return json.dumps({"query": "q", "results": list(items)}).encode()


def _item(n, url=None, content="An extract."):
    return {"title": f"Result {n}", "url": url or f"https://example.org/{n}", "content": content}


@pytest.fixture
def searxng():
    _SearXNG.reply = (200, "application/json", _results(), {})
    _SearXNG.requests = []
    httpd = HTTPServer(("127.0.0.1", 0), _SearXNG)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield {"SEARXNG_URL": f"http://127.0.0.1:{httpd.server_port}", "SEARCH_MAX_RESULTS": "3"}
    httpd.shutdown()


async def test_results_are_titles_links_and_snippets(searxng):
    _SearXNG.reply = (200, "application/json", _results(_item(1), _item(2, content="")), {})
    text = await web_search("ram vs vram", searxng)
    assert text == (
        "Results for 'ram vs vram' (read a page with fetch_page):\n"
        "1. Result 1\n   https://example.org/1\n   An extract.\n"
        "2. Result 2\n   https://example.org/2"
    )


async def test_the_query_is_sent_as_words_only_to_the_configured_address(searxng):
    await web_search("C++ & Rust #1 ?format=html&engines=x", searxng)
    ((path, host),) = _SearXNG.requests
    parts = urlsplit(path)
    assert parts.path == "/search"
    assert parse_qs(parts.query) == {
        "q": ["C++ & Rust #1 ?format=html&engines=x"],
        "format": ["json"],
    }
    assert host == searxng["SEARXNG_URL"].removeprefix("http://")


async def test_the_count_is_capped_and_long_texts_cut(searxng):
    items = [_item(n, content="word " * 200) for n in range(30)]
    _SearXNG.reply = (200, "application/json", _results(*items), {})
    text = await web_search("x", searxng)
    assert text.count("https://example.org/") == 3
    assert max(len(line) for line in text.splitlines()) <= 3 + 300
    text = await web_search("x", {**searxng, "SEARCH_MAX_RESULTS": "500"})
    assert text.count("https://example.org/") == MAX_RESULTS


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8700/users",
        "http://192.168.1.10/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "http://localhost:8080/",
        "http://printer.localhost/",
        "javascript:alert(1)",
        "file:///etc/passwd",
        "ftp://example.org/file",
    ],
)
async def test_a_result_leading_to_the_inside_or_not_http_is_dropped(searxng, url):
    _SearXNG.reply = (200, "application/json", _results(_item(1, url=url), _item(2)), {})
    text = await web_search("x", searxng)
    assert url not in text
    assert "1. Result 2\n   https://example.org/2" in text


async def test_only_unsafe_results_give_no_result(searxng):
    _SearXNG.reply = (200, "application/json", _results(_item(1, url="http://10.0.0.1/")), {})
    assert await web_search("x", searxng) == "no result for 'x'"


@pytest.mark.parametrize(
    "reply, message",
    [
        ((403, "text/html", b"Forbidden", {}), "json format"),
        ((500, "application/json", b"{}", {}), "answered 500"),
        ((302, "text/html", b"", {"Location": "http://169.254.169.254/"}), "answered 302"),
        ((200, "text/html", b"<html>results</html>", {}), "did not answer JSON (text/html)"),
        ((200, "application/json", b"not json", {}), "not readable"),
        ((200, "application/json", b'{"results": "x"}', {}), "not readable"),
        ((200, "application/json", b"[1]", {}), "not readable"),
        ((200, "application/json", b" " * (1024 * 1024 + 1), {}), "too large"),
    ],
)
async def test_a_bad_answer_is_refused_and_a_redirect_not_followed(searxng, reply, message):
    _SearXNG.reply = reply
    with pytest.raises(SearchError, match=re.escape(message)):
        await web_search("x", searxng)
    assert len(_SearXNG.requests) == 1


@pytest.mark.parametrize(
    "query, environ, message",
    [
        ("  ", {"SEARXNG_URL": "http://127.0.0.1:1"}, "empty"),
        ("x" * 301, {"SEARXNG_URL": "http://127.0.0.1:1"}, "longer than 300"),
        ("x", {}, "SEARXNG_URL is empty"),
        ("x", {"SEARXNG_URL": "ftp://127.0.0.1/"}, "http(s) address"),
        ("x", {"SEARXNG_URL": "http://127.0.0.1/?engines=x"}, "no query"),
        ("x", {"SEARXNG_URL": "http://user:secret@127.0.0.1:1"}, "credentials"),
        ("x", {"SEARXNG_URL": "http://127.0.0.1:1"}, "could not be reached"),
    ],
)
async def test_a_bad_query_or_configuration_is_refused(query, environ, message):
    with pytest.raises(SearchError, match=re.escape(message)):
        await web_search(query, environ)


async def test_a_slow_engine_is_cut_off(searxng, monkeypatch):
    import asyncio

    import app.search as search

    real_send = search.httpx.AsyncClient.send

    async def slow_send(self, *args, **kwargs):
        await asyncio.sleep(5)
        return await real_send(self, *args, **kwargs)

    monkeypatch.setattr(search.httpx.AsyncClient, "send", slow_send)
    with pytest.raises(SearchError, match="more than 0 s"):
        await web_search("x", {**searxng, "WEB_FETCH_TIMEOUT_SECONDS": "0.2"})


# --- the server ---


def test_the_search_server_is_read_only_open_world_and_gets_only_its_settings(monkeypatch):
    from app.config import get_settings
    from app.mcp import builtin
    from app.mcp.builtin_servers import search_server

    assert builtin.REGISTRY["search"] == "app.mcp.builtin_servers.search_server"
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")
    get_settings.cache_clear()
    try:
        assert builtin.builtin_environment("search") == {
            "SEARXNG_URL": "http://127.0.0.1:8888",
            "SEARCH_MAX_RESULTS": "8",
            "WEB_FETCH_TIMEOUT_SECONDS": "15",
        }
    finally:
        get_settings.cache_clear()
    tool = search_server.mcp._tool_manager.get_tool("web_search")
    assert (tool.annotations.readOnlyHint, tool.annotations.openWorldHint) == (True, True)


async def test_a_call_round_trips_through_the_real_search_server(searxng, monkeypatch):
    from app.config import get_settings
    from app.db.models import McpTransport
    from app.mcp.manager import ManagedServer, ServerConfig

    _SearXNG.reply = (200, "application/json", _results(_item(1)), {})
    monkeypatch.setenv("SEARXNG_URL", searxng["SEARXNG_URL"])
    get_settings.cache_clear()
    server = ManagedServer(
        ServerConfig(
            name="search", protocol=McpTransport.STDIO, builtin_id="search", timeout_seconds=20,
            concurrency_limit=1, result_max_bytes=100_000,
        )
    )  # fmt: skip
    try:
        assert [t.name for t in await server.list_tools()] == ["web_search"]
        text, is_error = await server.call_tool("web_search", {"query": "ram"})
        assert is_error is False and "1. Result 1\n   https://example.org/1" in text
    finally:
        await server.disconnect()
        get_settings.cache_clear()


async def test_the_tool_reports_a_refusal_as_text(monkeypatch):
    from app.mcp.builtin_servers import search_server

    monkeypatch.delenv("SEARXNG_URL", raising=False)
    assert await search_server.web_search("ram") == (
        "error: no search engine is configured (SEARXNG_URL is empty)"
    )
