"""Scopes: what a caller of the Admin API may do (.

Four scopes, each includes the ones below it:

    READ     status, lists, counts. No conversation text, no secrets.
    OPERATE  users, access requests, agents, conversation reset.
    ADMIN    permissions, logs and admin events (conversation text).
    OWNER    purging a user, and everything else.

Every route declares the scope it needs with `require(...)`, and the application
refuses to start when one does not (`verify_scopes`): a route added later without
a scope cannot ship (default deny). A named administrator's token carries its own
scope; the static API key acts as OWNER until the first named owner exists.
"""

from dataclasses import dataclass
from enum import IntEnum

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.routing import APIRoute


class Scope(IntEnum):
    READ = 1
    OPERATE = 2
    ADMIN = 3
    OWNER = 4


@dataclass(frozen=True)
class Principal:
    actor: str
    scope: Scope
    # A named administrator: their account, and the token used (None at sign-in).
    # Both None for the static API_SERVER_KEY.
    account_id: int | None = None
    token_id: int | None = None


def get_principal(request: Request) -> Principal:
    """Set by the authentication (app/api/deps.py), which runs before any route dependency."""
    principal = getattr(request.state, "principal", None)
    if principal is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")
    return principal


class _Requirement:
    """The route dependency that enforces one scope. A class, not a closure, so
    `verify_scopes` can recognise it and read the scope it declares.
    """

    def __init__(self, scope: Scope) -> None:
        self.scope = scope

    async def __call__(self, principal: Principal = Depends(get_principal)) -> None:
        if principal.scope < self.scope:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient scope")


def require(scope: Scope):
    return Depends(_Requirement(scope))


def declared_scopes(route: APIRoute) -> list[Scope]:
    return [
        d.dependency.scope for d in route.dependencies if isinstance(d.dependency, _Requirement)
    ]


def _api_routes(routes):
    """Every route, looking inside included routers: FastAPI wraps an included router
    in an object that holds it, and the scope is declared on the routes inside.
    """
    for route in routes:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            yield from _api_routes(inner.routes)
        else:
            yield route


def verify_scopes(app: FastAPI) -> None:
    """Fail closed: every route must declare exactly one scope."""
    missing = []
    for route in _api_routes(app.routes):
        if not isinstance(route, APIRoute):
            missing.append(f"{getattr(route, 'path', type(route).__name__)} (not an API route)")
        elif len(declared_scopes(route)) != 1:
            methods = ",".join(sorted(route.methods or []))
            missing.append(f"{methods} {route.path}")
    if missing:
        raise RuntimeError(
            "Admin API routes without exactly one declared scope: " + "; ".join(missing)
        )
