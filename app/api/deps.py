"""Shared FastAPI dependencies for the Admin API: a DB session per request, and the
authentication: the API_SERVER_KEY bearer key until a named owner exists, the named
administrators' tokens and their sign-in (app/admin/accounts.py).
"""

import base64
import logging
import secrets
import time
from collections import deque
from collections.abc import AsyncIterator
from contextvars import ContextVar

from fastapi import Header, HTTPException, Request, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.actor import actor_from_header, set_actor
from app.api.scopes import Principal, Scope
from app.config import get_settings
from app.db.session import get_sessionmaker

# The key rule lives in app.settings_rules, which `./start.sh --config KEY=VALUE` also runs:
# the API and the command that writes the key can never disagree.
from app.settings_rules import MIN_API_KEY_LENGTH, api_key_is_acceptable  # noqa: F401

logger = logging.getLogger("channelagent.api")

# Failed-attempt limiter: at most FAILURE_LIMIT failures per source
# address inside FAILURE_WINDOW_SECONDS, then 429 for everything from that
# address (a right key included, otherwise a guesser learns nothing from
# being blocked but keeps guessing). In memory on purpose: a restart
# clears it, and one process serves the API.
FAILURE_LIMIT = 10
FAILURE_WINDOW_SECONDS = 60
MAX_TRACKED_ADDRESSES = 1024
_failures: dict[str, deque[float]] = {}


def _now() -> float:
    return time.monotonic()


def reset_failure_state() -> None:
    _failures.clear()


def _source(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _recent_failures(address: str, now: float) -> deque[float]:
    window = _failures.get(address)
    if window is None:
        return deque()
    while window and now - window[0] > FAILURE_WINDOW_SECONDS:
        window.popleft()
    if not window:
        del _failures[address]
        return deque()
    return window


def _record_failure(address: str, now: float) -> None:
    window = _recent_failures(address, now)
    window.append(now)
    _failures[address] = window
    while len(_failures) > MAX_TRACKED_ADDRESSES:
        del _failures[next(iter(_failures))]


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        yield session


# The in-process transport of the script (`--transport inprocess`, application stopped): the
# operating-system user already holds `.env` and the database, so the static key keeps working
# there after a named owner exists (recovery). Set by app/admin/client.py only; the UI,
# which also calls the API in process, never sets it.
INPROCESS_CLI: ContextVar[bool] = ContextVar("inprocess_cli", default=False)

SIGN_IN_PATH = "/auth/token"
KEY_DISABLED = (
    "The API key is disabled since a named owner exists: sign in with a named account "
    "(./start.sh --admin sign-in --name NAME)"
)


def _header(scope, name: bytes) -> str | None:
    for key, value in scope["headers"]:
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def _basic_credentials(authorization: str) -> tuple[str, str] | None:
    try:
        decoded = base64.b64decode(authorization[len("Basic ") :], validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    name, sep, password = decoded.partition(":")
    return (name, password) if sep else None


async def _named(address: str, authorization: str, path: str, code: str | None, now: float):
    """A named administrator: a bearer token anywhere, or a name and password (and code) in
    a Basic header on the sign-in route only."""
    from app.admin import accounts
    from app.db.session import session_scope

    async with session_scope() as session:
        if authorization.startswith("Bearer "):
            found = await accounts.resolve_token(session, authorization[len("Bearer ") :])
            if found is not None:
                account, token = found
                scope = min(accounts.scope_rank(account.scope), accounts.scope_rank(token.scope))
                return Principal(accounts.actor_of(account), Scope(scope), account.id, token.id)
        elif authorization.startswith("Basic ") and path == SIGN_IN_PATH:
            credentials = _basic_credentials(authorization)
            if credentials is not None:
                try:
                    account = await accounts.check_credentials(session, *credentials, code)
                except accounts.SignInError:
                    account = None
                if account is not None:
                    await session.commit()  # the TOTP step used: the same code never twice
                    return Principal(
                        accounts.actor_of(account), Scope(accounts.scope_rank(account.scope)),
                        account.id,
                    )  # fmt: skip
    _record_failure(address, now)
    logger.warning("Admin API: failed authentication from %s.", address)
    return status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key"


STORE_UNAVAILABLE = (
    status.HTTP_503_SERVICE_UNAVAILABLE,
    "The administrators' store cannot be read; try again later",
)


async def authenticate(
    address: str,
    authorization: str | None,
    path: str = "",
    code: str | None = None,
    client: str | None = None,
) -> Principal | tuple[int, str]:
    """Who the caller is (a Principal), or the refusal (status, detail). One function for the
    middleware that runs before any body is read (`AuthenticateMiddleware`) and for the route
    dependency (`verify_api_key`), so the two cannot disagree. Fails closed: when the database
    cannot say whether the static key is still valid or whose a token is, the answer is 503,
    never an access."""
    try:
        return await _authenticate(address, authorization, path, code, client)
    except SQLAlchemyError:
        logger.error("Admin API: the administrators' store cannot be read.", exc_info=True)
        return STORE_UNAVAILABLE


async def _authenticate(
    address: str,
    authorization: str | None,
    path: str,
    code: str | None,
    client: str | None,
) -> Principal | tuple[int, str]:
    settings = get_settings()
    if not settings.api_server_key:
        return status.HTTP_503_SERVICE_UNAVAILABLE, "API_SERVER_KEY is not configured"
    now = _now()
    if len(_recent_failures(address, now)) >= FAILURE_LIMIT:
        logger.warning("Admin API: request from %s blocked, too many failed attempts.", address)
        return status.HTTP_429_TOO_MANY_REQUESTS, "Too many failed attempts, try again later"
    expected = f"Bearer {settings.api_server_key}"
    # Constant-time comparison — this guards a real secret, not just a
    # display value, so a naive `==` (early-exit on first mismatched
    # byte) would leak timing information about how many leading
    # characters of the token a guess got right.
    if authorization is not None and secrets.compare_digest(authorization, expected):
        if not INPROCESS_CLI.get():
            from app.admin import accounts
            from app.db.session import session_scope

            async with session_scope() as session:
                disabled = await accounts.named_owner_exists(session)
            if disabled:
                _record_failure(address, now)
                logger.warning("Admin API: the disabled API key was used from %s.", address)
                return status.HTTP_401_UNAUTHORIZED, KEY_DISABLED
        # The static key holder is the owner (no named account): the client says who it is.
        return Principal(actor=actor_from_header(client), scope=Scope.OWNER)
    if authorization is None:
        _record_failure(address, now)
        logger.warning("Admin API: failed authentication from %s.", address)
        return status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key"
    return await _named(address, authorization, path, code, now)


class AuthenticateMiddleware:
    """Refuses a request without valid credentials before the application reads its body.
    FastAPI reads and validates a JSON body before it runs the route's dependencies, so
    with the dependency alone an invalid body sent without any key got a 422, not a 401
    (measured on 18 write routes, 2026-09-27). Every
    route of the API needs credentials: there is no public path. The caller found here is
    kept in the request's state for the route (`verify_api_key`)."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        found = await authenticate(
            client[0] if client else "unknown",
            _header(scope, b"authorization"),
            scope.get("path", ""),
            _header(scope, b"x-totp"),
            _header(scope, b"x-client"),
        )
        if not isinstance(found, Principal):
            from app.api.protect import _reply

            await _reply(send, *found)
            return
        scope.setdefault("state", {})["principal"] = found
        await self.app(scope, receive, send)


async def verify_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_totp: str | None = Header(default=None, include_in_schema=False),
) -> None:
    """Applied to every route via the app-level `dependencies=` list in
    app/api/app.py, not per-router — a new endpoint added later can't
    accidentally ship without this check. `AuthenticateMiddleware` authenticates first and
    keeps the caller in the request's state; a route mounted without the middleware is still
    guarded here. Sets who the caller is for the admin events.
    """
    found = getattr(request.state, "principal", None)
    if found is None:
        found = await authenticate(
            _source(request), authorization, request.url.path, x_totp,
            request.headers.get("x-client"),
        )  # fmt: skip
    if not isinstance(found, Principal):
        raise HTTPException(*found)
    set_actor(found.actor)
    request.state.principal = found
