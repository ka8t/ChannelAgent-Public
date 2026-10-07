"""Reading a web page: the page fetched through the
outbound guard, its main text extracted and turned into Markdown.

- The guard (`app.security.outbound.Guard`) on the first request and on every
  redirect, one hop at a time: https only, a host of WEB_FETCH_ALLOWED_HOSTS ("*" for any),
  every resolved address public (no private, loopback, link-local, reserved or multicast),
  the connection pinned to the address that was checked.
- Limits: WEB_FETCH_MAX_BYTES (the body is read in chunks and refused past the limit, a
  larger Content-Length is refused before reading), WEB_FETCH_TIMEOUT_SECONDS for the whole
  page, redirects at most MAX_REDIRECTS, only text/html, application/xhtml+xml, text/plain.
- The extraction: trafilatura (main content, no menus or footers, Markdown with links and
  tables), run in a thread; plain text is returned as it is.
- A cache in memory: a page read again within WEB_FETCH_CACHE_SECONDS is not fetched again.

This module never reads `.env` (no `app.config`): it runs inside the built-in MCP server,
which is started with a cleared environment holding only the WEB_FETCH_* values.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from app.security.outbound import Guard, OutboundError

MAX_REDIRECTS = 5
_REDIRECTS = {301, 302, 303, 307, 308}
_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
CACHE_MAX_PAGES = 100
# The request of a current desktop browser: a site that filters clients by their name
# serves it; WEB_FETCH_USER_AGENT replaces the name.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)
BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
}
# Signs of an anti-bot barrier: a CAPTCHA, a "checking your browser" page, a bot
# manager's challenge. Reported to the user, never passed.
CHALLENGE_SIGNS = (
    "cf-chl-", "challenge-platform", "cdn-cgi/challenge", "just a moment...",
    "attention required! | cloudflare", "g-recaptcha", "recaptcha/api", "h-captcha",
    "hcaptcha.com", "checking your browser", "px-captcha", "_incapsula_resource",
    "captcha-delivery.com", "are you a robot", "verify you are human",
)  # fmt: skip
CHALLENGE_SNIFF_BYTES = 64 * 1024
MIN_TEXT_CHARS = 300  # less than this from a plain request: the page may be built by scripts
CHALLENGE_MESSAGE = "this page is protected against automated reading; open it yourself"


class FetchError(Exception):
    """The page was refused or could not be read; the message is safe to show."""


class BarrierError(FetchError):
    """An anti-bot barrier: reported, never passed."""


class RefusedError(FetchError):
    """The server answered a refusal (a status other than 200 and the redirects)."""

    def __init__(self, status: int):
        super().__init__(f"the server answered {status}")
        self.status = status


def is_challenge(text: str, headers=None) -> bool:
    if headers is not None and "challenge" in headers.get("cf-mitigated", "").lower():
        return True
    lowered = text[:CHALLENGE_SNIFF_BYTES].lower()
    return any(sign in lowered for sign in CHALLENGE_SIGNS)


@dataclass(frozen=True)
class Limits:
    allowed_hosts: frozenset
    max_bytes: int = 2 * 1024**2
    timeout_seconds: float = 15.0
    cache_seconds: float = 600.0
    user_agent: str = DEFAULT_USER_AGENT
    browser: bool = False  # render in headless Chromium when a plain request gets no text

    @classmethod
    def from_environment(cls, environ=os.environ) -> "Limits":
        hosts = {
            h.strip().lower()
            for h in environ.get("WEB_FETCH_ALLOWED_HOSTS", "").split(",")
            if h.strip()
        }
        return cls(
            allowed_hosts=frozenset(hosts),
            max_bytes=int(environ.get("WEB_FETCH_MAX_BYTES") or 2 * 1024**2),
            timeout_seconds=float(environ.get("WEB_FETCH_TIMEOUT_SECONDS") or 15),
            cache_seconds=float(environ.get("WEB_FETCH_CACHE_SECONDS") or 600),
            user_agent=environ.get("WEB_FETCH_USER_AGENT") or DEFAULT_USER_AGENT,
            browser=environ.get("WEB_FETCH_BROWSER", "").lower() == "true",
        )


_cache: dict[str, tuple[float, str, str]] = {}


def clear_cache() -> None:
    _cache.clear()


def _to_markdown(body: str, kind: str, url: str) -> str:
    if kind == "text/plain":
        return body.strip()
    import trafilatura

    text = trafilatura.extract(
        body, url=url, output_format="markdown", include_links=True, include_tables=True
    )
    return (text or "").strip()


async def _get(
    client: httpx.AsyncClient,
    guard: Guard,
    url: str,
    limits: Limits,
    types: tuple[str, ...] = _TYPES,
    raw: bool = False,
):
    """The final URL, its media type and its body, redirects followed through the guard.
    `types` are the media types accepted; `raw` returns the body as bytes (a feed's XML
    declares its own encoding)."""
    for _ in range(MAX_REDIRECTS + 1):
        try:
            target = await asyncio.to_thread(guard.prepare, url)
        except OutboundError as exc:
            raise FetchError(str(exc)) from None
        headers = {"Host": target.host_header, "User-Agent": limits.user_agent, **BROWSER_HEADERS}
        extensions = {"sni_hostname": target.sni} if target.sni else {}
        request = client.build_request("GET", target.url, headers=headers, extensions=extensions)
        response = await client.send(request, stream=True)
        try:
            if response.status_code in _REDIRECTS:
                location = response.headers.get("location")
                if not location:
                    raise FetchError("the server redirected without saying where")
                url = urljoin(url, location)
                continue
            if response.status_code != 200:
                sniffed = b""
                async for chunk in response.aiter_bytes():
                    sniffed += chunk
                    if len(sniffed) >= CHALLENGE_SNIFF_BYTES:
                        break
                if is_challenge(sniffed.decode("utf-8", "replace"), response.headers):
                    raise BarrierError(CHALLENGE_MESSAGE)
                raise RefusedError(response.status_code)
            kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if kind not in types:
                what = "a web page" if types is _TYPES else "a feed"
                raise FetchError(f"not {what} ({kind or 'no content type'})")
            declared = response.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > limits.max_bytes:
                raise FetchError(f"the page is larger than {limits.max_bytes} bytes")
            chunks, size = [], 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > limits.max_bytes:
                    raise FetchError(f"the page is larger than {limits.max_bytes} bytes")
                chunks.append(chunk)
            data = b"".join(chunks)
            body = data.decode(response.encoding or "utf-8", errors="replace")
            if kind != "text/plain" and is_challenge(body, response.headers):
                raise BarrierError(CHALLENGE_MESSAGE)
            return url, kind, data if raw else body
        finally:
            await response.aclose()
    raise FetchError("too many redirects")


async def fetch_markdown(
    url: str,
    limits: Limits,
    *,
    guard: Guard | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[str, str]:
    """(final URL, Markdown) of the page at `url`, or FetchError with the reason."""
    if not isinstance(url, str) or not url.strip():
        raise FetchError("url is the https address of the page")
    url = url.strip()
    now = time.monotonic()
    cached = _cache.get(url)
    if cached and limits.cache_seconds and now - cached[0] < limits.cache_seconds:
        return cached[1], cached[2]
    if not limits.allowed_hosts:
        raise FetchError("no host is allowed: set WEB_FETCH_ALLOWED_HOSTS")
    guard = guard or Guard(allowed_hosts=limits.allowed_hosts)
    own = client is None
    client = client or httpx.AsyncClient(follow_redirects=False, trust_env=False)
    markdown, reason = "", "no readable text on the page"
    try:
        async with asyncio.timeout(limits.timeout_seconds):
            final, kind, body = await _get(client, guard, url, limits)
        markdown = await asyncio.to_thread(_to_markdown, body, kind, final)
    except TimeoutError:
        raise FetchError(f"the page took more than {limits.timeout_seconds:.0f} s") from None
    except httpx.HTTPError as exc:
        raise FetchError(f"the page could not be read ({type(exc).__name__})") from None
    except RefusedError as exc:
        # A refusal to a plain request (403) may be a filter on non-browser clients: the
        # browser gets a chance; any other status is the answer.
        if exc.status != 403 or not limits.browser:
            raise
        final, reason = url, str(exc)
    finally:
        if own:
            await client.aclose()
    # A page built by scripts gives a plain request its menu at most: under MIN_TEXT_CHARS the
    # browser renders it and the longer text is kept (quotes.toscrape.com/js/: 29 characters).
    if len(markdown) < MIN_TEXT_CHARS and limits.browser:
        rendered, rendered_final = await _rendered(url, limits, guard)
        if len(rendered) > len(markdown):
            markdown, final = rendered, rendered_final
    if not markdown:
        raise FetchError(reason)
    if limits.cache_seconds:
        if len(_cache) >= CACHE_MAX_PAGES:
            _cache.pop(next(iter(_cache)))
        _cache[url] = (now, final, markdown)
    return final, markdown


async def fetch_bytes(
    url: str,
    limits: Limits,
    types: tuple[str, ...],
    *,
    guard: Guard | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[str, bytes]:
    """(final URL, body bytes) of a document of one of `types` (a feed), through the
    same guard and limits as a page; no browser and no cache (a feed changes)."""
    if not isinstance(url, str) or not url.strip():
        raise FetchError("url is the https address of the feed")
    if not limits.allowed_hosts:
        raise FetchError("no host is allowed: set WEB_FETCH_ALLOWED_HOSTS")
    guard = guard or Guard(allowed_hosts=limits.allowed_hosts)
    own = client is None
    client = client or httpx.AsyncClient(follow_redirects=False, trust_env=False)
    try:
        async with asyncio.timeout(limits.timeout_seconds):
            final, _kind, data = await _get(client, guard, url.strip(), limits, types, raw=True)
    except TimeoutError:
        raise FetchError(f"the feed took more than {limits.timeout_seconds:.0f} s") from None
    except httpx.HTTPError as exc:
        raise FetchError(f"the feed could not be read ({type(exc).__name__})") from None
    finally:
        if own:
            await client.aclose()
    return final, data


async def _rendered(url: str, limits: Limits, guard: Guard) -> tuple[str, str]:
    """(Markdown, final URL) of the page rendered in headless Chromium."""
    from app.web_render import RenderError, render_html

    try:
        final, html, _traffic = await render_html(
            url, guard, user_agent=limits.user_agent, max_bytes=limits.max_bytes,
            timeout_seconds=limits.timeout_seconds * 2,
        )  # fmt: skip
    except RenderError as exc:
        raise FetchError(str(exc)) from None
    if is_challenge(html):
        raise BarrierError(CHALLENGE_MESSAGE)
    return await asyncio.to_thread(_to_markdown, html, "text/html", final), final
