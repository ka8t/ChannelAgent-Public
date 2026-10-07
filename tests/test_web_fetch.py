"""Tests: reading a web page. The guard (https only, allowed hosts, public addresses
after resolution and on every redirect), the size, time and type limits, the Markdown
extraction and the cache, all with the real guard and a scripted network (a resolver that
answers chosen addresses, an httpx transport that answers chosen pages). Then the built-in
MCP server "web" over its real stdio subprocess: what it is given of the settings, and its
classes for the read-then-write rule.
"""

import asyncio
import sys

import httpx
import pytest

from app import web_fetch
from app.security.outbound import Guard
from app.web_fetch import FetchError, Limits, fetch_markdown

PUBLIC = "93.184.216.34"
ARTICLE = """<html><head><title>Otters</title></head><body>
<nav><a href="/">Home</a> | <a href="/shop">Shop</a></nav>
<article><h1>Sea otters</h1>
<p>Sea otters hold hands while they sleep so that they do not drift apart. They live along
the coasts of the northern Pacific Ocean and eat sea urchins, crabs and clams.</p>
<p>An adult sea otter has the densest fur of any animal: up to a million hairs per square inch.
See <a href="https://example.org/fur">the fur study</a> for the details.</p>
<h2>Where they live</h2>
<p>From Japan to California, in kelp forests close to the shore, rarely more than a kilometre
out at sea. Their numbers fell to about two thousand in 1911 and have grown back since.</p>
</article><footer>Copyright 2026, all rights reserved. Cookie settings.</footer>
</body></html>"""

DNS = {"example.org": [PUBLIC], "news.example.org": [PUBLIC], "inside.example.org": ["10.0.0.5"],
       "sneaky.example.net": [PUBLIC, "127.0.0.1"], "other.example.net": [PUBLIC]}  # fmt: skip


def resolver(host, port):
    if host not in DNS:
        raise OSError("no such host")
    return DNS[host]


def _limits(hosts=("example.org",), **extra) -> Limits:
    return Limits(allowed_hosts=frozenset(hosts), **{"cache_seconds": 0, **extra})


def _guard(limits) -> Guard:
    return Guard(allowed_hosts=limits.allowed_hosts, resolver=resolver)


class Web:
    """Pages by (host, path); counts the requests that reached the network."""

    def __init__(self, pages):
        self.pages, self.requests = pages, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        host = request.headers["host"]
        self.requests.append((host, request.url.path, str(request.url.host)))
        answer = self.pages.get((host, request.url.path))
        if answer is None:
            return httpx.Response(404)
        return answer() if callable(answer) else answer


def html(body=ARTICLE, **headers):
    return httpx.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8",
                                                   **headers})  # fmt: skip


async def fetch(url, pages, limits=None):
    limits = limits or _limits()
    web = Web(pages)
    async with httpx.AsyncClient(transport=httpx.MockTransport(web)) as client:
        try:
            return await fetch_markdown(url, limits, guard=_guard(limits), client=client), web
        except FetchError as exc:
            return exc, web


@pytest.fixture(autouse=True)
def _no_cache():
    web_fetch.clear_cache()
    yield
    web_fetch.clear_cache()


# --- refusals ---


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://example.org/page", "must use https"),
        ("https://127.0.0.1/page", "not allowed"),
        ("https://inside.example.org/page", "non-public address"),
        ("https://sneaky.example.net/page", "not allowed"),
        ("https://user:pw@example.org/page", "credentials"),
        ("ftp://example.org/page", "only http(s)"),
        ("", "url is the https address"),
        ("https://nowhere.example.org/", "could not be resolved"),
    ],
)
async def test_bad_addresses_are_refused_before_any_request(url, reason):
    result, web = await fetch(url, {("example.org", "/page"): html()})
    assert isinstance(result, FetchError) and reason in str(result), result
    assert web.requests == []


async def test_a_loopback_address_is_refused_even_when_any_host_is_allowed():
    for url in ("https://127.0.0.1/", "https://[::1]/", "https://169.254.169.254/latest",
                "https://10.1.2.3/", "https://sneaky.example.net/"):  # fmt: skip
        result, web = await fetch(url, {}, _limits(hosts=("*",)))
        assert isinstance(result, FetchError), url
        assert "non-public address" in str(result), (url, result)
        assert web.requests == []


async def test_a_redirect_to_a_private_address_is_refused_at_the_hop():
    moved = httpx.Response(302, headers={"location": "https://inside.example.org/x"})
    pages = {("example.org", "/go"): moved}
    result, web = await fetch("https://example.org/go", pages, _limits(hosts=("example.org",)))
    assert isinstance(result, FetchError) and "non-public address" in str(result)
    assert [r[0] for r in web.requests] == ["example.org"], "the private hop is never requested"


async def test_a_redirect_to_http_or_to_a_host_not_allowed_is_refused():
    for location, reason in (("http://example.org/x", "must use https"),
                             ("https://other.example.net/x", "not allowed")):  # fmt: skip
        pages = {("example.org", "/go"): httpx.Response(302, headers={"location": location})}
        result, _web = await fetch("https://example.org/go", pages)
        assert isinstance(result, FetchError) and reason in str(result), location


async def test_too_many_redirects_are_refused():
    pages = {("example.org", "/loop"): httpx.Response(302, headers={"location": "/loop"})}
    result, web = await fetch("https://example.org/loop", pages)
    assert isinstance(result, FetchError) and "too many redirects" in str(result)
    assert len(web.requests) == web_fetch.MAX_REDIRECTS + 1


async def test_a_page_over_the_size_limit_is_refused_declared_or_not():
    limits = _limits(max_bytes=500)
    small_but_declared_big = html("<p>tiny</p>", **{"content-length": "999999"})
    declared, _ = await fetch(
        "https://example.org/big", {("example.org", "/big"): small_but_declared_big}, limits
    )
    assert isinstance(declared, FetchError) and "larger than 500 bytes" in str(declared)

    async def chunks():
        yield b"<p>" + b"x" * 400
        yield b"y" * 400

    def undeclared():
        return httpx.Response(200, headers={"content-type": "text/html"}, content=chunks())

    streamed, _ = await fetch("https://example.org/big", {("example.org", "/big"): undeclared},
                              limits)  # fmt: skip
    assert isinstance(streamed, FetchError) and "larger than 500 bytes" in str(streamed)


async def test_what_is_not_a_web_page_or_not_found_is_refused():
    kind = {"content-type": "application/octet-stream"}
    binary = httpx.Response(200, content=b"\x00\x01", headers=kind)
    result, _ = await fetch("https://example.org/f", {("example.org", "/f"): binary})
    assert isinstance(result, FetchError) and "not a web page" in str(result)
    missing, _ = await fetch("https://example.org/none", {})
    assert isinstance(missing, FetchError) and "answered 404" in str(missing)


async def test_no_host_allowed_means_no_page():
    result, web = await fetch("https://example.org/page", {("example.org", "/page"): html()},
                              _limits(hosts=()))  # fmt: skip
    assert isinstance(result, FetchError) and "WEB_FETCH_ALLOWED_HOSTS" in str(result)
    assert web.requests == []


async def test_a_slow_page_is_cut_at_the_time_limit():
    async def slow(request):
        await asyncio.sleep(5)
        return html()

    limits = _limits(timeout_seconds=0.2)
    async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
        with pytest.raises(FetchError, match="took more than"):
            await fetch_markdown("https://example.org/p", limits, guard=_guard(limits),
                                 client=client)  # fmt: skip


# --- a page ---


async def test_a_page_becomes_markdown_without_menus_or_footer_through_the_pinned_address():
    result, web = await fetch("https://news.example.org/otters",
                              {("news.example.org", "/otters"): html()},
                              _limits(hosts=("example.org",)))  # fmt: skip
    final, markdown = result
    assert final == "https://news.example.org/otters"
    assert "hold hands while they sleep" in markdown and "Where they live" in markdown
    assert "Cookie settings" not in markdown and "Shop" not in markdown
    assert len(markdown) >= 300
    assert web.requests == [("news.example.org", "/otters", PUBLIC)], "connected to the checked IP"


async def test_a_redirect_inside_the_allowed_hosts_is_followed():
    pages = {("example.org", "/old"): httpx.Response(301, headers={"location": "/new"}),
             ("example.org", "/new"): html()}  # fmt: skip
    (final, markdown), web = await fetch("https://example.org/old", pages)
    assert final == "https://example.org/new" and "Sea otters" in markdown
    assert [r[1] for r in web.requests] == ["/old", "/new"]


async def test_plain_text_is_returned_as_it_is_and_a_blank_page_is_refused():
    text = httpx.Response(
        200, text="  line one\nline two  ", headers={"content-type": "text/plain"}
    )
    (_final, markdown), _ = await fetch("https://example.org/t", {("example.org", "/t"): text})
    assert markdown == "line one\nline two"
    blank, _ = await fetch("https://example.org/b",
                           {("example.org", "/b"): html("<html><body></body></html>")})  # fmt: skip
    assert isinstance(blank, FetchError) and "no readable text" in str(blank)


async def test_a_page_read_again_comes_from_the_cache_until_it_expires(monkeypatch):
    limits = _limits(cache_seconds=60)
    web = Web({("example.org", "/c"): html})
    clock = [1000.0]
    monkeypatch.setattr(web_fetch.time, "monotonic", lambda: clock[0])
    async with httpx.AsyncClient(transport=httpx.MockTransport(web)) as client:
        for _ in range(3):
            await fetch_markdown("https://example.org/c", limits, guard=_guard(limits),
                                 client=client)  # fmt: skip
        assert len(web.requests) == 1
        clock[0] += 61
        await fetch_markdown("https://example.org/c", limits, guard=_guard(limits), client=client)
    assert len(web.requests) == 2


def test_the_limits_are_read_from_the_environment():
    limits = Limits.from_environment({"WEB_FETCH_ALLOWED_HOSTS": " Example.org , ,wiki.org ",
                                      "WEB_FETCH_MAX_BYTES": "1000",
                                      "WEB_FETCH_TIMEOUT_SECONDS": "3",
                                      "WEB_FETCH_CACHE_SECONDS": "0"})  # fmt: skip
    assert limits == Limits(frozenset({"example.org", "wiki.org"}), 1000, 3.0, 0.0)
    assert Limits.from_environment({}).allowed_hosts == frozenset()


# --- the built-in MCP server ---


def test_the_web_server_is_given_only_its_settings(monkeypatch):
    from app.config import get_settings
    from app.mcp.builtin import builtin_environment

    monkeypatch.setenv("WEB_FETCH_ALLOWED_HOSTS", "example.org")
    monkeypatch.setenv("ENCRYPTION_KEY", "must-not-leak")
    get_settings.cache_clear()
    try:
        env = builtin_environment("web")
        assert set(env) == {"WEB_FETCH_ALLOWED_HOSTS", "WEB_FETCH_MAX_BYTES",
                            "WEB_FETCH_TIMEOUT_SECONDS", "WEB_FETCH_CACHE_SECONDS",
                            "WEB_FETCH_USER_AGENT", "WEB_FETCH_BROWSER",
                            "PLAYWRIGHT_BROWSERS_PATH"}  # fmt: skip
        assert env["WEB_FETCH_ALLOWED_HOSTS"] == "example.org"
        assert "must-not-leak" not in env.values()
        assert builtin_environment("time") == {}
    finally:
        get_settings.cache_clear()


def test_the_server_module_never_reads_the_application_settings():
    import subprocess

    code = "import sys, app.mcp.builtin_servers.web_server; print('app.config' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_fetch_page_is_untrusted_and_outbound_so_a_second_page_asks_first():
    from app.mcp.builtin_servers.web_server import mcp
    from app.mcp.policy import Policy, default_classes, default_policy

    (tool,) = mcp._tool_manager.list_tools()
    definition = {"annotations": tool.annotations.model_dump(exclude_none=True)}
    assert default_classes(definition, "internet") == {
        "private": False, "untrusted": True, "outbound": True,
    }  # fmt: skip
    assert default_policy(definition) == Policy.CONFIRM, "open-world: asks by default (M4)"


async def test_the_real_server_refuses_a_loopback_address_with_its_settings(monkeypatch):
    from app.config import get_settings
    from app.db.models import McpTransport
    from app.mcp.manager import ManagedServer, ServerConfig

    monkeypatch.setenv("WEB_FETCH_ALLOWED_HOSTS", "*")
    get_settings.cache_clear()
    server = ManagedServer(ServerConfig(name="web", protocol=McpTransport.STDIO, builtin_id="web",
                                        timeout_seconds=20, concurrency_limit=1,
                                        result_max_bytes=1_000_000))  # fmt: skip
    try:
        assert [t.name for t in await server.list_tools()] == ["fetch_page"]
        text, _ = await server.call_tool("fetch_page", {"url": "https://127.0.0.1/admin"})
        assert text == "error: host 127.0.0.1 points at a non-public address"
        text, _ = await server.call_tool("fetch_page", {"url": "http://example.org/"})
        assert text == "error: the URL must use https"
    finally:
        await server.disconnect()
        get_settings.cache_clear()


async def test_the_tool_cuts_a_long_page_and_names_its_source(monkeypatch):
    from app.mcp.builtin_servers import web_server

    async def long_page(url, limits):
        return "https://example.org/long", "word " * 10_000

    monkeypatch.setattr(web_server, "fetch_markdown", long_page)
    text = await web_server.fetch_page("https://example.org/long")
    assert text.startswith("Source: https://example.org/long\n\n")
    assert text.endswith("[... the page goes on, cut here]")
    assert len(text) < web_server.MAX_RESULT_CHARS + 100

    async def refused(url, limits):
        raise FetchError("the URL must use https")

    monkeypatch.setattr(web_server, "fetch_markdown", refused)
    assert await web_server.fetch_page("http://x") == "error: the URL must use https"
