"""Identification and authorization matrix of the Admin API ("vérifie
que toutes les commandes passées à l'api ont bien un token d'identification et
d'autorisation"). For every route of the real application, in process, on a throwaway
database:

  no token          -> expected 401
  a wrong token     -> expected 401
  invalid JSON body without a token (write routes) -> expected 401, not 422
  a valid token whose scope is one below the route's (simulated: the single key is OWNER
                       until named administrators exist, S3)  -> expected 403
  a valid token with the route's own scope -> expected neither 401 nor 403

Also lists every route of the application that is not a normal API route (FastAPI mounts
/docs and /openapi.json outside the app-level dependency).

    .venv/bin/python scripts/api_auth_matrix.py
Exit 0 when every route behaves as expected.
"""

import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
WORK = Path(tempfile.mkdtemp(prefix="authmatrix-"))
KEY = "matrix-" + "k7Qz" * 8
os.environ.update(
    DATABASE_URL=f"sqlite+aiosqlite:///{WORK}/m.db",
    CHECKPOINT_DB_PATH=str(WORK / "cp.db"),
    API_SERVER_KEY=KEY,
    MIGRATION_BACKUPS_KEEP="0",
    LLAMA_SERVER_URL="http://127.0.0.1:9",
)

import httpx  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402

from app.api import deps  # noqa: E402
from app.api.app import app  # noqa: E402
from app.api.scopes import (  # noqa: E402
    Principal,
    Scope,
    _api_routes,
    declared_scopes,
    get_principal,
)
from app.db.session import init_db  # noqa: E402

SAMPLE = {"kind": "chat", "name": "x.gguf", "tool": "t", "channel": "telegram"}


def _path(template: str) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: SAMPLE.get(m.group(1), "1"), template)


async def main() -> int:
    await init_db()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    good = {"Authorization": f"Bearer {KEY}"}
    bad = {"Authorization": "Bearer wrong-" + "x" * 30}
    rows, problems = [], []
    # The same walk as verify_scopes: included routers are unwrapped to their routes.
    every = list(_api_routes(app.routes))
    others = [r for r in every if not isinstance(r, APIRoute)]
    routes = [r for r in every if isinstance(r, APIRoute)]
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as c:
        for route in routes:
            scopes = declared_scopes(route)
            if len(scopes) != 1:
                problems.append(f"{route.path}: {len(scopes)} declared scopes")
                continue
            scope = scopes[0]
            for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
                path = _path(route.path)
                body = {} if method in ("POST", "PUT", "PATCH") else None

                async def send(headers, raw=None, _m=method, _p=path, _b=body):
                    deps.reset_failure_state()
                    if raw is not None:
                        return (
                            await c.request(
                                _m,
                                _p,
                                headers={**headers, "Content-Type": "application/json"},
                                content=raw,
                            )
                        ).status_code
                    return (await c.request(_m, _p, headers=headers, json=_b)).status_code

                none, wrong = await send({}), await send(bad)
                broken = await send({}, raw="{not json") if body is not None else None
                lower = None
                if scope > Scope.READ:
                    app.dependency_overrides[get_principal] = lambda s=scope: Principal(
                        "matrix", Scope(s - 1)
                    )
                    lower = await send(good)
                app.dependency_overrides[get_principal] = lambda s=scope: Principal("matrix", s)
                right = await send(good)
                app.dependency_overrides.clear()
                ok = (
                    none == 401
                    and wrong == 401
                    and broken in (None, 401)
                    and lower in (None, 403)
                    and right not in (401, 403)
                )
                rows.append((method, route.path, scope.name, none, wrong, broken, lower, right, ok))
                if not ok:
                    problems.append(f"{method} {route.path}")
    print(f"{'method':6} {'path':58} {'scope':8} none wrong badjson lower right")
    for method, path, scope, none, wrong, broken, lower, right, ok in rows:
        print(
            f"{method:6} {path:58} {scope:8} {none:4} {wrong:5} {str(broken or '-'):7} "
            f"{str(lower or '-'):5} {right}{'' if ok else '   <-- PROBLEM'}"
        )
    by_scope = {}
    for row in rows:
        by_scope[row[2]] = by_scope.get(row[2], 0) + 1
    print(f"\n{len(rows)} operations on {len(routes)} API routes; by scope: {by_scope}")
    print(f"refused without a token: {sum(r[3] == 401 for r in rows)} of {len(rows)}")
    print(f"refused with a wrong token: {sum(r[4] == 401 for r in rows)} of {len(rows)}")
    writes = [r for r in rows if r[5] is not None]
    refused = sum(r[5] == 401 for r in writes)
    print(f"invalid JSON without a token refused as 401: {refused} of {len(writes)}")
    lowered = [r for r in rows if r[6] is not None]
    below = sum(r[6] == 403 for r in lowered)
    print(
        f"refused one scope below: {below} of {len(lowered)}"
        f" (READ routes have no lower scope: {len(rows) - len(lowered)})"
    )
    print(
        f"accepted with their own scope: {sum(r[7] not in (401, 403) for r in rows)} of {len(rows)}"
    )
    print(f"routes outside the API routes: {[getattr(r, 'path', r) for r in others]}")
    print("problems:", problems or "none")
    return 1 if problems or others else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
