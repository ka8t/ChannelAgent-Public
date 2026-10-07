"""The Admin API call trail: one `api_calls` row per request, refused ones included.

Outermost middleware of the API application, so a request refused by the Host check (421), the
key (401), a scope (403) or a size limit (413) is recorded like an accepted one. Recorded: the
actor (the client's label once the key is checked, `unauthenticated` before), the source
address, the method, the path without its query string (a query can hold a search keyword), the
status and the duration. Never a body. A failure to write is logged and the response is
unchanged: the trail must not break the API.
"""

import logging
import time

from app.db.models import ApiCall

logger = logging.getLogger("channelagent.api")

UNAUTHENTICATED = "unauthenticated"
MAX_PATH = 300


def _actor(scope) -> str:
    principal = (scope.get("state") or {}).get("principal")
    actor = getattr(principal, "actor", None)
    return str(actor)[:32] if actor else UNAUTHENTICATED


async def record_call(actor: str, source: str, method: str, path: str, status: int, ms: int):
    from app.db.session import session_scope

    try:
        async with session_scope() as session:
            session.add(
                ApiCall(
                    actor=actor, source=source[:64], method=method[:8], path=path[:MAX_PATH],
                    status=status, duration_ms=ms,
                )
            )  # fmt: skip
            await session.commit()
    except Exception:
        logger.warning("The API call trail could not be written", exc_info=True)


class AuditMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.monotonic()
        status = {"code": 500}

        async def watched_send(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        scope.setdefault("state", {})
        try:
            await self.app(scope, receive, watched_send)
        finally:
            client = scope.get("client")
            await record_call(
                _actor(scope),
                client[0] if client else "unknown",
                scope.get("method", ""),
                scope.get("path", ""),
                status["code"],
                int((time.monotonic() - started) * 1000),
            )
