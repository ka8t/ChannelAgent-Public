"""Rendering a web page in a headless browser, for a page whose text only exists after
its scripts run, or that refuses a plain request.

The browser never reaches the network by itself. Every request it wants to make goes through
`page.route`: the ones that carry no text (images, media, fonts, style sheets, pings) are
aborted, and web sockets are closed before they connect (`route_web_socket`); the others are
fetched by this module through the outbound guard
(https, an allowed host, every resolved address public, the connection pinned to the checked
address) and handed back to the browser with `route.fulfill`. What the guard refuses is aborted
and counted. As a second barrier, the browser is given a proxy that leads nowhere
(127.0.0.1:1), so a request the route would miss fails instead of going out; WebRTC is kept off
non-proxied UDP, DNS prefetching is off, service workers are blocked, downloads refused, and
each page gets a fresh context (no cookies kept).

The browser program is Playwright's headless Chromium, installed inside this repository
(`vendor/playwright`, PLAYWRIGHT_BROWSERS_PATH), never in a user cache.
"""

import asyncio
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from app.security.outbound import Guard, OutboundError

SKIPPED_TYPES = {"image", "media", "font", "stylesheet", "ping", "websocket", "manifest", "other"}
CHROMIUM_ARGS = [
    "--proxy-server=http://127.0.0.1:1",
    "--proxy-bypass-list=<-loopback>",
    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    "--dns-prefetch-disable",
    "--disable-background-networking",
    "--disable-component-update",
    "--no-first-run",
]
SETTLE_SECONDS = 1.5  # after the load event, time for the page's scripts to write the text


class RenderError(Exception):
    """The page could not be rendered; the message is safe to show."""


@dataclass
class Traffic:
    """What the browser asked for: fetched through the guard, refused by it, skipped."""

    fetched: list[str] = field(default_factory=list)
    refused: list[tuple[str, str]] = field(default_factory=list)
    skipped: int = 0
    bytes: int = 0


async def _fetch_for_browser(client, guard: Guard, request, limit: int, traffic: Traffic):
    """One request of the browser, made by us through the guard: (status, headers, body)."""
    target = await asyncio.to_thread(guard.prepare, request.url)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "cookie")}
    headers["Host"] = target.host_header
    extensions = {"sni_hostname": target.sni} if target.sni else {}
    outgoing = client.build_request(
        request.method, target.url, headers=headers, content=request.post_data_buffer,
        extensions=extensions,
    )  # fmt: skip
    response = await client.send(outgoing, stream=True)
    try:
        chunks = []
        async for chunk in response.aiter_bytes():
            traffic.bytes += len(chunk)
            if traffic.bytes > limit:
                raise RenderError(f"the page and its scripts are larger than {limit} bytes")
            chunks.append(chunk)
    finally:
        await response.aclose()
    # The body is handed over decoded: its encoding and length headers would no longer match,
    # and no cookie is given to the browser (a fresh context per page, nothing kept).
    dropped = ("content-length", "content-encoding", "transfer-encoding", "set-cookie")
    kept = {k: v for k, v in response.headers.items() if k.lower() not in dropped}
    return response.status_code, kept, b"".join(chunks)


async def render_html(
    url: str,
    guard: Guard,
    *,
    user_agent: str,
    max_bytes: int,
    timeout_seconds: float,
    client: httpx.AsyncClient | None = None,
) -> tuple[str, str, Traffic]:
    """(final URL, HTML after the scripts ran, traffic) of `url` rendered in headless Chromium."""
    try:
        from playwright.async_api import Error as PlaywrightError
        from playwright.async_api import async_playwright
    except ImportError:
        raise RenderError("the browser is not installed (playwright)") from None
    try:
        guard.prepare(url)
    except OutboundError as exc:
        raise RenderError(str(exc)) from None
    traffic = Traffic()
    own = client is None
    # Redirects are answered to the browser as they are: its next request comes back through
    # the route, and so through the guard, one hop at a time.
    client = client or httpx.AsyncClient(follow_redirects=False, trust_env=False, timeout=15)
    failure: list[Exception] = []

    async def route(route):
        request = route.request
        if request.resource_type in SKIPPED_TYPES or urlsplit(request.url).scheme not in (
            "https", "http",
        ):  # fmt: skip
            traffic.skipped += 1
            await route.abort()
            return
        try:
            status, headers, body = await _fetch_for_browser(
                client, guard, request, max_bytes * 5, traffic
            )
        except OutboundError as exc:
            traffic.refused.append((request.url, str(exc)))
            await route.abort("blockedbyclient")
            return
        except RenderError as exc:
            failure.append(exc)
            await route.abort("blockedbyclient")
            return
        except httpx.HTTPError:
            await route.abort("failed")
            return
        traffic.fetched.append(request.url)
        await route.fulfill(status=status, headers=headers, body=body)

    async def web_socket(ws):
        traffic.skipped += 1
        await ws.close(code=1008, reason="not allowed")

    started = time.monotonic()
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=CHROMIUM_ARGS)
            try:
                context = await browser.new_context(
                    user_agent=user_agent, service_workers="block", accept_downloads=False,
                    java_script_enabled=True,
                )  # fmt: skip
                await context.route("**/*", route)
                # Web sockets are not requests: `route` never sees them. Each one is closed
                # here without ever connecting (a test counts 0 connections without the proxy).
                await context.route_web_socket("**/*", web_socket)
                page = await context.new_page()
                remaining = timeout_seconds - (time.monotonic() - started)
                response = await page.goto(url, wait_until="load", timeout=remaining * 1000)
                await asyncio.sleep(min(SETTLE_SECONDS, max(0.0, remaining / 4)))
                if failure:
                    raise failure[0]
                html = await page.content()
                final = page.url
                status = response.status if response else None
            finally:
                await browser.close()
    except PlaywrightError as exc:
        if failure:
            raise failure[0] from None
        text = str(exc).splitlines()[0][:200]
        if "Executable doesn't exist" in text:
            raise RenderError("the browser is not installed: ./start.sh installs it when "
                              "WEB_FETCH_BROWSER=true") from None  # fmt: skip
        raise RenderError(f"the browser could not render the page ({text})") from None
    finally:
        if own:
            await client.aclose()
    if status is not None and status >= 400:
        raise RenderError(f"the server answered {status} to the browser too")
    return final, html, traffic
