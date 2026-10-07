"""Web search through the owner's own SearXNG instance, for the built-in MCP server "search".
Each result is a title, a link and a snippet; a page is then read with `fetch_page`
(the server "web"), through its own guard.

- The request goes to SEARXNG_URL and nowhere else: the model gives only the words of the
  query, sent URL-encoded as `q`, so it cannot choose the host, the path or another parameter.
  It goes through the outbound guard (app/security/outbound.py) restricted to that one host;
  a private address is accepted there only because the administrator configured it (SearXNG
  runs next to the application). No redirect is followed, the answer must be JSON (SearXNG's
  `search.formats` must list `json`) and is read up to MAX_RESPONSE_BYTES.
- A result whose link is not http(s), or whose host is `localhost` or a non-public address
  literal, is dropped: a search must not hand the model a way to the inside.
- At most SEARCH_MAX_RESULTS results (capped at MAX_RESULTS), titles and snippets cut.

Every refusal raises SearchError with a message the model can act on.
"""

import asyncio
import ipaddress
import json
import os
from urllib.parse import urlencode, urlsplit

import httpx

from app.security.outbound import Guard, OutboundError, is_public

DEFAULT_RESULTS = 8
MAX_RESULTS = 20
MAX_QUERY_CHARS = 300
MAX_RESPONSE_BYTES = 1024 * 1024
TITLE_CHARS = 200
SNIPPET_CHARS = 300


class SearchError(Exception):
    """The search was refused or failed; the message is safe to show."""


def _settings(environ=os.environ) -> tuple[str, int, float]:
    base = environ.get("SEARXNG_URL", "").strip().rstrip("/")
    count = int(environ.get("SEARCH_MAX_RESULTS") or DEFAULT_RESULTS)
    timeout = float(environ.get("WEB_FETCH_TIMEOUT_SECONDS") or 15)
    return base, max(1, min(count, MAX_RESULTS)), timeout


def _link_is_safe(link: str) -> bool:
    parts = urlsplit(link)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    host = parts.hostname.lower()
    if host == "localhost" or host.endswith(".localhost"):
        return False
    try:
        return is_public(str(ipaddress.ip_address(host)))
    except ValueError:
        return True  # a name: fetch_page resolves it and checks every address itself


def _cut(text, size: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= size else text[: size - 3] + "..."


def render(query: str, results: list, count: int) -> str:
    lines, shown = [], 0
    for item in results:
        if not isinstance(item, dict):
            continue
        link = str(item.get("url") or "")
        if not _link_is_safe(link):
            continue
        shown += 1
        lines.append(f"{shown}. {_cut(item.get('title') or link, TITLE_CHARS)}")
        lines.append(f"   {link}")
        snippet = _cut(item.get("content"), SNIPPET_CHARS)
        if snippet:
            lines.append(f"   {snippet}")
        if shown == count:
            break
    if not lines:
        return f"no result for {query!r}"
    return f"Results for {query!r} (read a page with fetch_page):\n" + "\n".join(lines)


async def web_search(
    query: str, environ=os.environ, client: httpx.AsyncClient | None = None, resolver=None
) -> str:
    words = " ".join(str(query or "").split())
    if not words:
        raise SearchError("the query is empty")
    if len(words) > MAX_QUERY_CHARS:
        raise SearchError(f"a query longer than {MAX_QUERY_CHARS} characters")
    base, count, timeout = _settings(environ)
    if not base:
        raise SearchError("no search engine is configured (SEARXNG_URL is empty)")
    parts = urlsplit(base)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.query:
        raise SearchError("SEARXNG_URL must be an http(s) address with a host and no query")
    guard = Guard(
        allowed_hosts=frozenset({parts.hostname.lower()}),
        allow_http=parts.scheme == "http",
        allow_private=True,
        **({"resolver": resolver} if resolver else {}),
    )
    url = f"{base}/search?" + urlencode({"q": words, "format": "json"})
    try:
        target = await asyncio.to_thread(guard.prepare, url)
    except OutboundError as exc:
        raise SearchError(str(exc)) from None
    own = client is None
    client = client or httpx.AsyncClient(follow_redirects=False, trust_env=False)
    headers = {"Host": target.host_header, "Accept": "application/json"}
    extensions = {"sni_hostname": target.sni} if target.sni else {}
    try:
        async with asyncio.timeout(timeout):
            request = client.build_request(
                "GET", target.url, headers=headers, extensions=extensions
            )
            response = await client.send(request, stream=True)
            try:
                if response.status_code == 403:
                    raise SearchError(
                        "the search engine refused (403): its settings must allow the json format"
                    )
                if response.status_code != 200:
                    raise SearchError(f"the search engine answered {response.status_code}")
                kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if kind != "application/json":
                    what = kind or "no type"
                    raise SearchError(f"the search engine did not answer JSON ({what})")
                data = b""
                async for chunk in response.aiter_bytes():
                    data += chunk
                    if len(data) > MAX_RESPONSE_BYTES:
                        raise SearchError("the search engine's answer is too large")
            finally:
                await response.aclose()
    except TimeoutError:
        raise SearchError(f"the search took more than {timeout:.0f} s") from None
    except httpx.HTTPError as exc:
        reason = type(exc).__name__
        raise SearchError(f"the search engine could not be reached ({reason})") from None
    finally:
        if own:
            await client.aclose()
    try:
        results = json.loads(data).get("results") or []
    except (ValueError, AttributeError):
        raise SearchError("the search engine's answer is not readable") from None
    if not isinstance(results, list):
        raise SearchError("the search engine's answer is not readable")
    return render(words, results, count)
