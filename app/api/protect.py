"""Request protections in front of the Admin API (.

- Host allow-list: a request whose Host is not one of ALLOWED_HOSTS is refused with
  421 before authentication. A web page cannot reach a loopback API through DNS
  rebinding, because the browser sends the attacker's name as Host.
- Origin check: a state-changing request that carries an Origin from another site is
  refused with 403. Scripts and curl send no Origin and are not affected.
- Body size limit (413) and request time limit (504).

Settings are read on every request (they are cached), not at import.
"""

import asyncio
import json
from urllib.parse import urlsplit

from app.config import get_settings

DEFAULT_HOSTS = ("localhost", "127.0.0.1", "::1")
_STATE_CHANGING = {"POST", "PUT", "PATCH", "DELETE"}


def allowed_hosts() -> set[str]:
    raw = get_settings().allowed_hosts or ""
    hosts = {h.strip().lower() for h in raw.split(",") if h.strip()}
    return hosts or set(DEFAULT_HOSTS)


def _host_name(value: str) -> str:
    """The host part of a Host header value: no port, no brackets."""
    value = value.strip().lower()
    if value.startswith("["):
        return value[1:].split("]", 1)[0]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


class ProtectMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        settings = get_settings()
        hosts = allowed_hosts()
        if _host_name(headers.get("host", "")) not in hosts:
            await _reply(send, 421, "Host not allowed")
            return
        origin = headers.get("origin")
        if scope["method"] in _STATE_CHANGING and origin is not None:
            origin_host = (urlsplit(origin).hostname or "").lower()
            if origin_host not in hosts:
                await _reply(send, 403, "Origin not allowed")
                return
        limit = settings.api_max_body_bytes
        declared = headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await _reply(send, 413, "Request body too large")
            return

        received = 0
        too_large = False

        async def counted_receive():
            # Past the limit the application is told the client went away, and whatever
            # it answers is dropped: FastAPI turns an exception raised while reading the
            # body into a 400, so raising here would never reach this middleware.
            nonlocal received, too_large
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    too_large = True
                    return {"type": "http.disconnect"}
            return message

        started = False

        async def tracked_send(message):
            nonlocal started
            if too_large:
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await asyncio.wait_for(
                self.app(scope, counted_receive, tracked_send),
                timeout=settings.api_request_timeout_seconds,
            )
        except TimeoutError:
            if not started:
                await _reply(send, 504, "Request took too long")
            return
        except Exception:
            if not too_large:
                raise
        if too_large and not started:
            await _reply(send, 413, "Request body too large")


async def _reply(send, status_code: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
