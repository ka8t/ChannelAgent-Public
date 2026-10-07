"""The admin UI (decision D8): server-rendered pages (Jinja2, autoescaped), one small
vanilla JavaScript file, every asset served from `app/ui/static/`, nothing from another site.

The UI is a client of the Admin API, like `start.sh`: every page reads and every form writes
through the API's own routes, called in process with the signed-in administrator's token
(or the server's key before a named owner exists, actor `ui:web`), so the UI cannot do anything
the API does not allow, and the admin events say who did it. Its forms and result views are
generated from the API's manifest (the same one the script's commands come from), so a new
route has a screen without UI code: `/ui/op/<command>`. Two screens are written by hand:
access requests and log search.

Sign-in, CSRF, the login limiter and the response headers: `app/ui/security.py`.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import secrets
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import httpx
from jinja2 import Environment, FileSystemLoader
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from app.api.protect import ProtectMiddleware
from app.config import get_settings
from app.ui import parity, security
from app.ui.security import HeadersMiddleware, csrf_ok, login_limiter, sessions

logger = logging.getLogger("channelagent.ui")

HERE = Path(__file__).resolve().parent
ACTOR = "ui:web"
JOB_ID = re.compile(r"(host-)?[0-9a-f]{12}")
FINAL = {"done", "failed", "cancelled"}
# The navigation, by what the administrator does: (section, ((label, href), ...)).
NAV = (
    ("System", (
        ("Overview", "/ui/"),
        ("Status", "/ui/status"),
        ("Configuration", "/ui/config"),
        ("Storage", "/ui/op/storage"),
    )),
    ("People", (
        ("Users", "/ui/op/list-users"),
        ("Access requests", "/ui/requests"),
        ("Administrators", "/ui/op/list-admins"),
    )),
    ("Agents", (
        ("Agents", "/ui/op/list-agents"),
        ("Skills", "/ui/op/list-skills"),
        ("Tool servers", "/ui/op/list-servers"),
        ("Scheduled tasks", "/ui/tasks"),
        ("Routing", "/ui/op/get-routing"),
    )),
    ("Data", (
        ("Backups", "/ui/backups"),
        ("Models", "/ui/models"),
        ("Jobs", "/ui/op/list-jobs"),
    )),
    ("Activity", (
        ("Logs", "/ui/logs"),
        ("Telemetry", "/ui/op/get-telemetry"),
        ("Audit trail", "/ui/op/audit-timeline"),
        ("Admin events", "/ui/op/search-admin-events"),
    )),
    ("Clients", (
        ("Script and UI parity", "/ui/parity"),
        ("All operations", "/ui/operations"),
    )),
)  # fmt: skip
SCREENS = tuple(item for _, items in NAV for item in items if item[1] != "/ui/")
# Actions offered on each row of a list: (command, the row's field, the operation's field).
ROW_ACTIONS = {
    "list-users": [
        ("get-user", "id", "user_id"),
        ("update-user", "id", "user_id"),
        ("list-agents", "id", "user_id"),
        ("list-channel-identities", "id", "user_id"),
        ("export-conversation", "id", "user_id"),
        ("delete-user", "id", "user_id"),
    ],
    "list-agents": [
        ("get-agent", "id", "agent_id"),
        ("update-agent", "id", "agent_id"),
        ("set-agent-skills", "id", "agent_id"),
        ("get-agent-exposure", "id", "agent_id"),
    ],
    "list-skills": [
        ("get-skill", "name", "name"),
        ("list-skill-versions", "name", "name"),
        ("update-skill", "name", "name"),
        ("delete-skill", "name", "name"),
    ],
    "list-servers": [
        ("get-server", "id", "server_id"),
        ("list-tools", "id", "server_id"),
        ("update-server", "id", "server_id"),
    ],
    "list-database-backups": [("restore-backup", "name", "name")],
    "list-models": [("delete-model", "name", "name")],
    "list-jobs": [("get-job", "id", "job_id")],
}

templates = Environment(
    loader=FileSystemLoader(str(HERE / "templates")),
    autoescape=True,  # every value is text unless a template says otherwise; none does
    trim_blocks=True,
    lstrip_blocks=True,
)


def _size(value) -> str:
    """4920739232 -> "4.9 GB (4,920,739,232 bytes)"."""
    if not isinstance(value, int):
        return "-"
    for unit, factor in (("GB", 10**9), ("MB", 10**6), ("KB", 10**3)):
        if value >= factor:
            return f"{value / factor:.1f} {unit} ({value:,} bytes)"
    return f"{value:,} bytes"


templates.filters["size"] = _size


def _size_short(value) -> str:
    """4920739232 -> "4.9 GB"."""
    return _size(value).split(" (")[0]


def _duration(seconds) -> str:
    """93784 -> "1 d 2 h 3 min"; under a minute, in seconds."""
    if not isinstance(seconds, (int, float)):
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} s"
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    parts = [f"{days} d"] if days else []
    parts += [f"{hours} h"] if hours or days else []
    return " ".join(parts + [f"{rest // 60} min"])


templates.filters["size_short"] = _size_short
templates.filters["duration"] = _duration


@lru_cache
def operations() -> dict[str, dict]:
    from app.admin.manifest import manifest
    from app.api.app import app as api

    # Signing in has its own page (/ui/login): its credentials go in headers, not a form.
    return {
        op["command"]: op for op in manifest(api)["operations"] if op["command"] != "sign-in"
    }


def _session(request: Request) -> security.Session | None:
    return sessions.get(request.cookies.get(security.COOKIE))


def _page(
    request: Request, template: str, status: int = 200, session=None, **context
) -> HTMLResponse:
    session = session or _session(request)
    flash = None
    if session and session.flash and status < 400:
        flash, session.flash = session.flash, None
    html = templates.get_template(template).render(
        nonce=request.state.csp_nonce,
        csrf=session.csrf if session else "",
        signed_in=bool(session and session.authenticated),
        screens=SCREENS,
        nav=NAV,
        current=_current(request.url.path),
        api_calls=getattr(request.state, "api_calls", []),
        flash=flash,
        **context,
    )
    return HTMLResponse(html, status_code=status)


def _current(path: str) -> str | None:
    """The navigation entry of this page: its own link, or the screen it belongs to (a job
    page has none)."""
    hrefs = [href for _, items in NAV for _, href in items]
    if path in hrefs:
        return path
    return next((h for h in hrefs if h != "/ui/" and path.startswith(h + "/")), None)


def _to_login() -> RedirectResponse:
    return RedirectResponse("/ui/login", status_code=303)


def _signed_in(request: Request) -> security.Session | None:
    session = _session(request)
    return session if session and session.authenticated else None


@asynccontextmanager
async def api_client(request: Request, headers: dict | None = None):
    """The Admin API in process, with the signed-in administrator's token, or the server's
    key before any named owner exists: the browser holds neither. The Host of the UI's
    request, already checked, is the Host of the API call, and its address the caller's."""
    from app.api.app import app as api

    session = _signed_in(request)
    credential = (session and session.token) or get_settings().api_server_key
    headers = headers or {"Authorization": f"Bearer {credential}", "X-Client": ACTOR}
    host = request.headers.get("host", "127.0.0.1")
    address = request.client.host if request.client else "unknown"
    transport = httpx.ASGITransport(app=api, client=(address, 0))
    async with httpx.AsyncClient(
        transport=transport, base_url=f"http://{host}", headers=headers, timeout=120
    ) as client:
        yield client


async def call_api(request: Request, method: str, path: str, query=None, body=None):
    """One API call; it is also written down on the request, with the same call from the
    terminal, so the page can show exactly what it asked the API (the parity footer)."""
    async with api_client(request) as client:
        response = await client.request(method, path, params=query or None, json=body)
    _note(request, parity.describe_call(method, path, query, body, response.status_code))
    if response.status_code == 204:
        return 204, None
    if "json" in response.headers.get("content-type", ""):
        return response.status_code, response.json()
    return response.status_code, response.text


def _note(request: Request, call: dict) -> None:
    calls = getattr(request.state, "api_calls", None)
    if calls is None:
        calls = request.state.api_calls = []
    calls.append(call)


def _safe_next(value: str | None) -> str | None:
    """Only a page of this UI, never another site (`//host`) or a scheme."""
    if value and re.fullmatch(r"/ui/[A-Za-z0-9/_?=&.%-]*", value) and "//" not in value:
        return value
    return None


# --- sign-in ---


async def _key_allowed() -> bool:
    """The admin key signs in until the first named owner exists."""
    from app.admin import accounts
    from app.db.session import session_scope

    async with session_scope() as db:
        return not await accounts.named_owner_exists(db)


async def login_page(request: Request) -> Response:
    if _signed_in(request):
        return RedirectResponse("/ui/", status_code=303)
    session = _session(request) or sessions.create(authenticated=False)
    response = _page(request, "login.html", session=session, key_allowed=await _key_allowed())
    response.headers["set-cookie"] = security.cookie_header(session.id)
    return response


async def _named_sign_in(request: Request, form) -> tuple[int, dict | str]:
    """POST /auth/token through the API itself, with the name and password in a Basic header
    and the code in X-TOTP, from the browser's address (the API's limiter counts it)."""
    name, password = str(form.get("name", "")).strip(), str(form.get("password", ""))
    basic = base64.b64encode(f"{name}:{password}".encode()).decode("ascii")
    headers = {"Authorization": f"Basic {basic}", "X-Client": ACTOR}
    code = str(form.get("code", "")).strip()
    if code:
        headers["X-TOTP"] = code
    hours = security.ABSOLUTE_SECONDS // 3600
    async with api_client(request, headers=headers) as client:
        response = await client.post("/auth/token", params={"label": ACTOR, "hours": hours})
    body = response.json() if "json" in response.headers.get("content-type", "") else {}
    return response.status_code, body


async def login(request: Request) -> Response:
    address = request.client.host if request.client else "unknown"
    form = await request.form()
    session = _session(request)
    if not csrf_ok(session, form.get("csrf")):
        return _page(request, "error.html", 403, message="Invalid or missing CSRF token.")
    key_allowed = await _key_allowed()

    def refused(status: int, error: str) -> Response:
        return _page(request, "login.html", status, error=error, key_allowed=key_allowed)

    if login_limiter.blocked(address):
        logger.warning("Admin UI: sign-in from %s blocked, too many failed sign-ins.", address)
        return refused(429, "Too many failed sign-ins, try again in a minute.")
    token = token_id = None
    actor = ACTOR
    if form.get("name"):
        status, body = await _named_sign_in(request, form)
        if status == 429:
            return refused(429, "Too many failed sign-ins, try again in a minute.")
        if status != 201:
            login_limiter.failed(address)
            logger.warning("Admin UI: failed sign-in from %s.", address)
            return refused(401, "Wrong name, password or code.")
        token, token_id, actor = body["token"], body["id"], f"adm:{body['account']}"
    else:
        expected = get_settings().api_server_key or ""
        given = str(form.get("key", ""))
        if not expected:
            return refused(503, "API_SERVER_KEY is not configured.")
        if not key_allowed:
            return refused(401, "The admin key is disabled: sign in with your name.")
        if not secrets.compare_digest(given.encode(), expected.encode()):
            login_limiter.failed(address)
            logger.warning("Admin UI: wrong key from %s.", address)
            return refused(401, "Wrong key.")
    sessions.delete(session.id)  # the id changes at sign-in
    signed = sessions.create(authenticated=True)
    signed.token, signed.token_id = token, token_id
    await _record(request, "ui.login", actor)
    response = RedirectResponse("/ui/", status_code=303)
    response.headers["set-cookie"] = security.cookie_header(signed.id)
    return response


async def logout(request: Request) -> Response:
    session = _session(request)
    form = await request.form()
    if not csrf_ok(session, form.get("csrf")):
        return _page(request, "error.html", 403, message="Invalid or missing CSRF token.")
    if session.authenticated:
        actor = ACTOR
        if session.token_id is not None:
            # The token dies with the session: revoked through the API, as the caller.
            async with api_client(request) as client:
                revoked = await client.delete(f"/auth/tokens/{session.token_id}")
            if revoked.status_code == 200:
                actor = f"adm:{revoked.json()['account']}"
        await _record(request, "ui.logout", actor)
    sessions.delete(session.id)
    response = _to_login()
    response.headers["set-cookie"] = security.cookie_header("", max_age=0)
    return response


async def _record(request: Request, action: str, actor: str = ACTOR) -> None:
    from app.admin import service
    from app.db.session import session_scope

    address = request.client.host if request.client else "unknown"
    async with session_scope() as db:
        await service.record_admin_event(
            db, actor=actor, action=action, target_type="ui", details={"address": address}
        )
        await db.commit()


# --- pages ---


async def index(request: Request) -> Response:
    if not _signed_in(request):
        return _to_login()
    _, whoami = await call_api(request, "GET", "/whoami")
    status_code, status = await call_api(request, "GET", "/status")
    return _page(
        request, "index.html", whoami=whoami, api_status=status if status_code == 200 else None,
        groups=_groups(), parity=parity.report(),
    )  # fmt: skip


def _groups() -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = {}
    for op in operations().values():
        groups.setdefault(op["tag"], []).append(op)
    return sorted(groups.items())


async def operations_page(request: Request) -> Response:
    """Every API operation, by tag, each with its screen and its start.sh command."""
    if not _signed_in(request):
        return _to_login()
    return _page(request, "operations.html", groups=_groups())


async def parity_page(request: Request) -> Response:
    """The script and the UI compared, computed now from their own code (app/ui/parity.py)."""
    if not _signed_in(request):
        return _to_login()
    return _page(request, "parity.html", report=parity.report())


def _missing(op: dict, values: dict) -> list[str]:
    return [
        f["name"]
        for f in op["fields"]
        if f["required"] and f["default"] is None and not values.get(f["name"])
    ]


def _request(op: dict, values: dict):
    from app.admin.client import UsageError, request_parts

    namespace = SimpleNamespace(
        **{"field__" + f["name"]: (values.get(f["name"]) or None) for f in op["fields"]}
    )
    try:
        return request_parts(op, namespace), None
    except UsageError as exc:
        return None, str(exc)


def _row_links(command: str, result) -> list[list[tuple[str, str]]] | None:
    actions = ROW_ACTIONS.get(command)
    if not actions or not isinstance(result, list):
        return None
    links = []
    for row in result:
        row_links = []
        for target, source, name in actions:
            if isinstance(row, dict) and row.get(source) is not None:
                row_links.append((target, f"/ui/op/{target}?" + urlencode({name: row[source]})))
        links.append(row_links)
    return links


# Fields whose value is the id of a row the UI can list: shown as a choice instead of a number
# to type. field name -> (list route, or a route per user), the key of the label.
ID_CHOICES = {
    "user_id": ("/users", None),
    "server_id": ("/mcp/servers", None),
    "task_id": ("/tasks", None),
    "request_id": ("/requests", None),
    "agent_id": ("/users/{user_id}/agents", None),
    "channel_identity_id": ("/users/{user_id}/channels", None),
}
LABEL_KEYS = ("name", "display_name", "title", "channel", "kind", "prompt", "status")


def _label(row: dict) -> str:
    keys = ("title",) if row.get("title") else LABEL_KEYS
    parts = [str(row[k])[:60] for k in keys if row.get(k) not in (None, "")][:2]
    return f"#{row.get('id')} " + " - ".join(parts)


async def _choices(request: Request, op: dict) -> dict[str, list[tuple[str, str]]]:
    """For each id field of the operation, the rows it may name (id, label), read through the
    API with the session's own key; a list the session may not read gives no choice."""
    wanted = {f["name"] for f in op["fields"] if f["name"] in ID_CHOICES}
    if not wanted:
        return {}
    out: dict[str, list[tuple[str, str]]] = {}
    users: list[dict] | None = None
    for name in sorted(wanted):
        route, _ = ID_CHOICES[name]
        rows: list[dict] = []
        if "{user_id}" in route:
            if users is None:
                code, data = await call_api(request, "GET", "/users")
                users = data if code == 200 and isinstance(data, list) else []
            for user in users:
                path = route.replace("{user_id}", str(user["id"]))
                code, data = await call_api(request, "GET", path)
                for row in data if code == 200 and isinstance(data, list) else []:
                    rows.append({**row, "title": f"{row.get('name') or row.get('channel')} "
                                                 f"({user.get('display_name')})"})  # fmt: skip
        else:
            code, data = await call_api(request, "GET", route)
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                data = data["items"]
            rows = data if code == 200 and isinstance(data, list) else []
        out[name] = [(str(r["id"]), _label(r)) for r in rows if isinstance(r, dict) and "id" in r]
    return out


def _from_lines(op: dict, values: dict) -> dict:
    """A list of texts typed one per line becomes the JSON list the API takes; a value that is
    already JSON (starts with "[") is kept."""
    out = dict(values)
    for f in op["fields"]:
        raw = out.get(f["name"])
        lines = f["type"] == "array" and f.get("items") == "string"
        if lines and raw and not raw.lstrip().startswith("["):
            out[f["name"]] = json.dumps([line.strip() for line in raw.splitlines() if line.strip()])
    return out


async def operation_page(request: Request) -> Response:
    if not _signed_in(request):
        return _to_login()
    command = request.path_params["command"]
    op = operations().get(command)
    if op is None:
        return _page(request, "error.html", 404, message=f"No operation {command}.")
    values = dict(request.query_params)
    result = None
    if op["method"] == "GET" and not _missing(op, values):
        parts, error = _request(op, values)
        if error:
            result = {"status": 422, "body": {"detail": error}}
        else:
            path, query, body = parts
            code, data = await call_api(request, "GET", path, query, body)
            result = {"status": code, "body": data, "links": _row_links(command, data)}
    choices = await _choices(request, op)
    return _page(request, "operation.html", op=op, values=values, result=result, choices=choices,
                 cli=_cli(op, values))  # fmt: skip


def _cli(op: dict, values: dict) -> str:
    """The form's values as the terminal command, with the field names the script takes."""
    names = {f["name"] for f in op["fields"]}
    shown = {k: v for k, v in _from_lines(op, values).items() if k in names}
    return parity.cli_line(op, shown) or ""


async def operation_submit(request: Request) -> Response:
    session = _session(request)
    form = await request.form()
    if not csrf_ok(session, form.get("csrf")) or not session.authenticated:
        return _page(request, "error.html", 403, message="Invalid or missing CSRF token.")
    command = request.path_params["command"]
    op = operations().get(command)
    if op is None or op["method"] == "GET":
        return _page(request, "error.html", 404, message=f"No operation {command} to submit.")
    values = {k: str(v) for k, v in form.items() if k not in ("csrf", "next")}
    shown = values
    values = _from_lines(op, values)
    missing = _missing(op, values)
    if missing:
        result = {"status": 422, "body": {"detail": "Required: " + ", ".join(missing)}}
        return _page(request, "operation.html", 422, op=op, values=shown, result=result,
                     choices=await _choices(request, op), cli=_cli(op, shown))
    parts, error = _request(op, values)
    if error:
        result = {"status": 422, "body": {"detail": error}}
        return _page(request, "operation.html", 422, op=op, values=shown, result=result,
                     choices=await _choices(request, op), cli=_cli(op, shown))
    path, query, body = parts
    code, data = await call_api(request, op["method"], path, query, body)
    done = {**parity.describe_call(op["method"], path, query, body, code),
            "label": op["command"].replace("-", " ").capitalize()}  # fmt: skip
    if code == 202 and isinstance(data, dict) and JOB_ID.fullmatch(str(data.get("id", ""))):
        session.flash = done
        return RedirectResponse(f"/ui/jobs/{data['id']}", status_code=303)
    target = _safe_next(form.get("next"))
    if target and code < 400:
        session.flash = done
        return RedirectResponse(target, status_code=303)
    result = {"status": code, "body": data, "links": None}
    return _page(request, "operation.html", code if code >= 400 else 200, op=op, values=shown,
                 result=result, choices=await _choices(request, op),
                 cli=_cli(op, shown))  # fmt: skip


async def job_page(request: Request) -> Response:
    if not _signed_in(request):
        return _to_login()
    job_id = request.path_params["job_id"]
    if not JOB_ID.fullmatch(job_id):
        return _page(request, "error.html", 404, message="No such job.")
    code, job = await call_api(request, "GET", f"/jobs/{job_id}")
    return _page(request, "job.html", code if code >= 400 else 200, job_id=job_id, code=code,
                 job=job, final=FINAL)  # fmt: skip


async def job_status(request: Request) -> Response:
    if not _signed_in(request):
        return JSONResponse({"detail": "Not signed in"}, status_code=401)
    job_id = request.path_params["job_id"]
    if not JOB_ID.fullmatch(job_id):
        return JSONResponse({"detail": "No such job"}, status_code=404)
    code, job = await call_api(request, "GET", f"/jobs/{job_id}")
    return JSONResponse(job, status_code=code)


async def requests_page(request: Request) -> Response:
    """Hand-written: the pending requests with the first message of each unknown sender,
    and approve or deny in one click."""
    if not _signed_in(request):
        return _to_login()
    state = request.query_params.get("status", "pending")
    if state not in ("pending", "approved", "denied", "all"):
        state = "pending"
    code, rows = await call_api(request, "GET", "/requests", {"status": state, "limit": 200})
    return _page(request, "requests.html", state=state, code=code, rows=rows)


async def logs_page(request: Request) -> Response:
    """Hand-written: the log search, each message shown as text. Nothing is read (and
    audited) until the form is sent."""
    if not _signed_in(request):
        return _to_login()
    op = operations()["search-logs"]
    values = dict(request.query_params)
    searched = values.pop("run", None) is not None
    result = None
    if searched:
        parts, error = _request(op, values)
        if error:
            result = {"status": 422, "body": {"detail": error}}
        else:
            path, query, _ = parts
            code, data = await call_api(request, "GET", path, query)
            result = {"status": code, "body": data}
    return _page(request, "logs.html", op=op, values=values, result=result, cli=_cli(op, values))


# --- dedicated screens: the same API operations, laid out for their task ---


async def tasks_page(request: Request) -> Response:
    """Hand-written: every user's scheduled tasks, the global pause switch, and run
    now, stop, start again or delete in one click."""
    if not _signed_in(request):
        return _to_login()
    code, rows = await call_api(request, "GET", "/tasks", {"limit": 200})
    pause_code, pause = await call_api(request, "GET", "/tasks/pause")
    return _page(
        request, "tasks.html", code=code, rows=rows,
        paused=pause.get("paused") if pause_code == 200 and isinstance(pause, dict) else None,
    )  # fmt: skip


async def status_page(request: Request) -> Response:
    if not _signed_in(request):
        return _to_login()
    status_code, api_status = await call_api(request, "GET", "/status")
    host_code, host = await call_api(request, "GET", "/host/status")
    schedule_code, schedule = await call_api(request, "GET", "/backups/schedule")
    return _page(
        request, "status.html", api_status=api_status if status_code == 200 else None,
        host=host, host_code=host_code, schedule=schedule if schedule_code == 200 else None,
    )  # fmt: skip


async def backups_page(request: Request) -> Response:
    if not _signed_in(request):
        return _to_login()
    code, backups = await call_api(request, "GET", "/backups")
    schedule_code, schedule = await call_api(request, "GET", "/backups/schedule")
    return _page(
        request, "backups.html", code=code, backups=backups,
        schedule=schedule if schedule_code == 200 else None,
    )  # fmt: skip


async def config_page(request: Request) -> Response:
    if not _signed_in(request):
        return _to_login()
    code, entries = await call_api(request, "GET", "/config")
    return _page(request, "config.html", code=code, entries=entries)


async def models_page(request: Request) -> Response:
    if not _signed_in(request):
        return _to_login()
    code, models = await call_api(request, "GET", "/models")
    return _page(request, "models.html", code=code, models=models)


async def root(_request: Request) -> Response:
    return RedirectResponse("/ui/", status_code=303)


routes = [
    Route("/ui", root),
    Route("/ui/", index),
    Route("/ui/login", login_page, methods=["GET"]),
    Route("/ui/login", login, methods=["POST"]),
    Route("/ui/logout", logout, methods=["POST"]),
    Route("/ui/parity", parity_page),
    Route("/ui/operations", operations_page),
    Route("/ui/op/{command}", operation_page, methods=["GET"]),
    Route("/ui/op/{command}", operation_submit, methods=["POST"]),
    Route("/ui/jobs/{job_id}", job_page),
    Route("/ui/jobs/{job_id}/status", job_status),
    Route("/ui/requests", requests_page),
    Route("/ui/tasks", tasks_page),
    Route("/ui/logs", logs_page),
    Route("/ui/status", status_page),
    Route("/ui/backups", backups_page),
    Route("/ui/config", config_page),
    Route("/ui/models", models_page),
    Mount("/ui/static", StaticFiles(directory=str(HERE / "static")), name="static"),
]

ui_app = Starlette(routes=routes)
# Host, Origin and size checks before anything else; the headers middleware, added
# last, is the outermost, so even those refusals carry the security headers.
ui_app.add_middleware(ProtectMiddleware)
ui_app.add_middleware(HeadersMiddleware)
