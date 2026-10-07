"""Sessions, CSRF, login limiter and response headers of the admin UI (
.

- Sign-in: a named administrator's name, password and code get an API token that
  the session keeps on the server; until a named owner exists, the admin key typed once does
  too (D10 option a). The browser holds only a random session id in a `__Host-` cookie
  (`Secure`, `HttpOnly`, `SameSite=Strict`, path `/`); neither the key nor the token goes back
  to it. A session
  ends after 30 minutes without a request or 8 hours after sign-in; its id is replaced at
  sign-in, so an id planted before sign-in is worthless after it.
- CSRF: every session has a random token; every POST must carry it (`csrf` form field),
  compared in constant time. Cross-site POSTs are also stopped by `SameSite=Strict` and by the
  Origin check of `ProtectMiddleware`.
- Login limiter: 5 wrong keys from one address in a minute, then 429 for that address.
- Headers on every UI response: a Content-Security-Policy with a per-request nonce and no
  inline script, `nosniff`, `Referrer-Policy: same-origin` (not `no-referrer`: Chrome then
  sends `Origin: null` on the UI's own posts), `no-store`, and no framing.

Sessions live in memory (one process serves the UI): a restart signs everyone out.
"""

from __future__ import annotations

import secrets
import time
from collections import deque
from dataclasses import dataclass, field

COOKIE = "__Host-ca_session"
IDLE_SECONDS = 30 * 60
ABSOLUTE_SECONDS = 8 * 3600
MAX_SESSIONS = 1000
LOGIN_FAILURE_LIMIT = 5
LOGIN_WINDOW_SECONDS = 60


def _now() -> float:
    return time.monotonic()


@dataclass
class Session:
    id: str
    csrf: str
    authenticated: bool
    # A named administrator's sign-in: the API token this session acts with, kept on
    # the server only, and its id to revoke it at sign-out. None with the interim API key.
    token: str | None = None
    token_id: int | None = None
    # The outcome of the last action, shown once on the page it redirects to.
    flash: dict | None = None
    # Through a lambda, so the clock is looked up when a session is made (tests move it).
    created: float = field(default_factory=lambda: _now())
    last_seen: float = field(default_factory=lambda: _now())

    def expired(self, now: float) -> bool:
        return now - self.last_seen > IDLE_SECONDS or now - self.created > ABSOLUTE_SECONDS


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, authenticated: bool) -> Session:
        self._prune()
        session = Session(
            id=secrets.token_urlsafe(32),
            csrf=secrets.token_urlsafe(32),
            authenticated=authenticated,
        )
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str | None) -> Session | None:
        if not session_id:
            return None
        session = self._sessions.get(session_id)
        if session is None:
            return None
        now = _now()
        if session.expired(now):
            del self._sessions[session_id]
            return None
        session.last_seen = now
        return session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def clear(self) -> None:
        self._sessions.clear()

    def _prune(self) -> None:
        now = _now()
        for sid in [s.id for s in self._sessions.values() if s.expired(now)]:
            del self._sessions[sid]
        while len(self._sessions) >= MAX_SESSIONS:
            oldest = min(self._sessions.values(), key=lambda s: s.last_seen)
            del self._sessions[oldest.id]


sessions = SessionStore()


def csrf_ok(session: Session | None, token: str | None) -> bool:
    return bool(session and token) and secrets.compare_digest(session.csrf, token)


class LoginLimiter:
    def __init__(self) -> None:
        self._failures: dict[str, deque[float]] = {}

    def _recent(self, address: str, now: float) -> deque[float]:
        window = self._failures.get(address, deque())
        while window and now - window[0] > LOGIN_WINDOW_SECONDS:
            window.popleft()
        return window

    def blocked(self, address: str) -> bool:
        return len(self._recent(address, _now())) >= LOGIN_FAILURE_LIMIT

    def failed(self, address: str) -> None:
        now = _now()
        window = self._recent(address, now)
        window.append(now)
        self._failures[address] = window
        if len(self._failures) > 1024:
            del self._failures[next(iter(self._failures))]

    def clear(self) -> None:
        self._failures.clear()


login_limiter = LoginLimiter()


def cookie_header(session_id: str, max_age: int = ABSOLUTE_SECONDS) -> str:
    return f"{COOKIE}={session_id}; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age={max_age}"


def csp(nonce: str) -> str:
    return (
        "default-src 'none'; "
        f"script-src 'nonce-{nonce}'; "
        "style-src 'self'; img-src 'self'; connect-src 'self'; "
        "form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
    )


class HeadersMiddleware:
    """A nonce per request (`request.state.csp_nonce`), and the security headers on every
    response of the UI, whatever produced it."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        nonce = secrets.token_urlsafe(16)
        scope.setdefault("state", {})["csp_nonce"] = nonce

        async def with_headers(message):
            if message["type"] == "http.response.start":
                headers = [
                    (k, v) for k, v in message.get("headers", []) if k.lower() not in _SET_HERE
                ]
                headers += [
                    (b"content-security-policy", csp(nonce).encode()),
                    (b"x-content-type-options", b"nosniff"),
                    # same-origin, not no-referrer: with no-referrer Chrome sends `Origin: null`
                    # on the UI's own form posts, which the Origin check refuses (measured,
                    # 2026-09-27). No referrer ever leaves for another site either way.
                    (b"referrer-policy", b"same-origin"),
                    (b"cache-control", b"no-store"),
                    (b"x-frame-options", b"DENY"),
                ]
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, with_headers)


_SET_HERE = {
    b"content-security-policy",
    b"x-content-type-options",
    b"referrer-policy",
    b"cache-control",
    b"x-frame-options",
}
