"""Command-line client of the Admin API: one command, one API call.

    python -m app.admin.client describe [--json]
    python -m app.admin.client <command> [--flag value ...] [--json]
        [--transport auto|http|inprocess]

The commands, their flags and their help are generated from the API's own routes
(`app/admin/manifest.py`), so a new route is a new command with no client code, and the
script and the UI cannot behave differently: both call the same handlers. `./start.sh
--admin <command> ...` (and `--admin describe`) run this in the virtualenv.

Transports: `http` talks to the running application; `inprocess` calls the same handlers
without a server, for when the application is stopped; `auto` (default) picks `http` when
something listens on the API port. Where the API is: `API_URL` when set (the API may run
on another machine than this script), else this machine on API_SERVER_HOST and
API_SERVER_PORT. With `API_URL` set, `auto` never falls back to `inprocess`: that would
act on this machine's database, not on the one the API serves.

Credentials: once a named owner exists, the API key works only in process.
`sign-in --name NAME` asks the password and the code, and keeps the token it gets, per API
address, in a file of mode 600 (`$XDG_CONFIG_HOME/channelagent/api-tokens.json`, by default
under `~/.config`); every later command over HTTP uses it. `sign-out` revokes it and forgets it.

A route that starts a job (202) is waited for, and
`--no-wait` prints the job instead (an in-process job cannot outlive the command, so it
is always waited for).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import getpass
import json
import os
import re
import socket
import sys
from contextlib import asynccontextmanager
from urllib.parse import quote, urlsplit

import httpx

from app.admin.credentials import bearer, forget_token, save_token, saved_token, token_file

FINAL_JOB = {"done", "failed", "cancelled"}
SIGN_IN = "sign-in"
POLL_SECONDS = 0.3
HOST_JOB_UNREACHABLE_SECONDS = 300


class UsageError(Exception):
    pass


def client_label() -> str:
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name in some containers
        user = "unknown"
    return "cli:" + (re.sub(r"[^A-Za-z0-9._-]", "_", user)[:32] or "unknown")


def api_base_url() -> tuple[str, bool]:
    """(base URL of the Admin API, whether API_URL named it). Without API_URL: this
    machine, where the application listens (a wildcard bind is reached on loopback)."""
    from app.settings_rules import api_url_problem

    url = os.environ.get("API_URL", "").strip()
    if url:
        problem = api_url_problem(url)
        if problem:
            raise UsageError(f"API_URL {problem} (./start.sh --config API_URL=...)")
        return url.rstrip("/"), True
    host = os.environ.get("API_SERVER_HOST", "127.0.0.1")
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{int(os.environ.get('API_SERVER_PORT', '8700'))}", False


def _is_up(base_url: str) -> bool:
    parts = urlsplit(base_url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        with socket.create_connection((parts.hostname, port), timeout=0.5):
            return True
    except OSError:
        return False


@asynccontextmanager
async def open_client(transport: str, authorization: str | None = None, extra=None):
    """A client of the API. Over HTTP it signs with `authorization` when given (sign-in),
    else the token saved for that address, else API_SERVER_KEY; in process with the key
    (the static key stays valid there, app/api/deps.py `INPROCESS_CLI`)."""
    key = os.environ.get("API_SERVER_KEY")
    base_url, named = api_base_url()
    if transport == "auto":
        transport = "http" if named or _is_up(base_url) else "inprocess"
    if transport == "http":
        if authorization is None:
            credential = bearer(base_url)
            if credential:
                authorization = f"Bearer {credential}"
            else:
                raise UsageError(
                    "no credentials: sign in (./start.sh --admin sign-in --name NAME), or set "
                    "API_SERVER_KEY (./start.sh --config API_SERVER_KEY=...)"
                )
        headers = {"Authorization": authorization, "X-Client": client_label(), **(extra or {})}
        async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=120) as client:
            client.transport_name = "http"  # type: ignore[attr-defined]
            client.api_base = base_url  # type: ignore[attr-defined]
            yield client
        return
    if not key:
        raise UsageError(
            "API_SERVER_KEY is not set (put it in .env, ./start.sh --config API_SERVER_KEY=...)"
        )
    headers = {
        "Authorization": authorization or f"Bearer {key}",
        "X-Client": client_label(),
        **(extra or {}),
    }
    from app.db.session import init_db
    from app.logging_setup import install_redaction
    from app.security.permissions import harden_process

    install_redaction()
    harden_process()
    await init_db()
    from app.api import deps
    from app.api.app import app

    marker = deps.INPROCESS_CLI.set(True)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
        headers=headers,
        timeout=120,
    ) as client:
        client.transport_name = "inprocess"  # type: ignore[attr-defined]
        client.api_base = base_url  # type: ignore[attr-defined]
        try:
            yield client
        finally:
            deps.INPROCESS_CLI.reset(marker)
            # A turn run in process (`--chat`) may have opened MCP connections: closed
            # here like app.main does, not left to the interpreter's shutdown, which printed an
            # anyio "cancel scope in a different task" traceback (measured 2026-10-04).
            mcp = sys.modules.get("app.mcp.manager")
            if mcp is not None:
                await mcp.manager.reset()
            graph = sys.modules.get("app.graph")
            if graph is not None:
                await graph.close_graph()


# --- the parser, generated from the manifest ---


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _dest(name: str) -> str:
    return "field__" + name


def build_parser(operations: list[dict]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="start.sh --admin",
        description="Command-line client of the Admin API. `describe` lists the commands.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print the response body as JSON")
    common.add_argument(
        "--transport",
        choices=["auto", "http", "inprocess"],
        default="auto",
        help="how to reach the API",
    )
    common.add_argument(
        "--no-wait", action="store_true", help="print a started job instead of waiting"
    )
    sub.add_parser("describe", parents=[common], help="list every command")
    sub.add_parser(
        "sign-out", parents=[common], help="revoke the saved token of this API and forget it"
    )
    for op in operations:
        p = sub.add_parser(
            op["command"],
            parents=[common],
            help=op["description"],
            description=f"{op['description']}  [{op['method']} {op['path']}, scope {op['scope']}]",
        )
        if op["command"] == SIGN_IN:
            p.add_argument("--name", help="the administrator's name (asked when left out)")
        for f in op["fields"]:
            default = f" (default {f['default']})" if f["default"] is not None else ""
            if f.get("secret"):
                default += " (secret: '-' reads it from standard input, else it is asked)"
            p.add_argument(
                _flag(f["name"]),
                dest=_dest(f["name"]),
                required=f["required"] and f["default"] is None and not f.get("secret"),
                choices=f["enum"],
                metavar=f["type"].upper(),
                help=(f["description"] or f["in"]) + default,
            )
    return parser


def _convert(field: dict, raw: str):
    """The typed value of `raw`; None when the field takes null and `null` was typed."""
    kind = field["type"]
    if field.get("nullable") and raw.lower() == "null":
        return None
    try:
        if kind == "integer":
            return int(raw)
        if kind == "number":
            return float(raw)
        if kind == "boolean":
            if raw.lower() in ("true", "1", "yes"):
                return True
            if raw.lower() in ("false", "0", "no"):
                return False
            raise ValueError
        if kind in ("array", "object"):
            return json.loads(raw)
    except ValueError:
        raise UsageError(f"{_flag(field['name'])}: {raw!r} is not a valid {kind}") from None
    return raw


def resolve_secrets(op: dict, args: argparse.Namespace, stdin=None, ask=None) -> None:
    """A secret field is never taken from the command line, where any user of the
    machine sees it (`ps`): `-` reads it from standard input, and when it is left out it is
    asked without echo (or read from standard input when that is not a terminal)."""
    stdin = stdin or sys.stdin
    ask = ask or getpass.getpass
    for f in op["fields"]:
        if not f.get("secret"):
            continue
        dest = _dest(f["name"])
        raw = getattr(args, dest, None)
        if raw is None and not f["required"]:
            continue
        if raw is None:
            raw = ask(f"{f['name']}: ") if stdin.isatty() else "-"
        elif raw != "-":
            raise UsageError(
                f"{_flag(f['name'])} is a secret: give '-' and type it on standard input, or "
                "leave it out to be asked; never on the command line"
            )
        if raw == "-":
            raw = stdin.readline().rstrip("\r\n")
        if not raw:
            raise UsageError(f"{_flag(f['name'])} is empty")
        setattr(args, dest, raw)


def request_parts(op: dict, args: argparse.Namespace) -> tuple[str, dict, dict | None]:
    path, query, body = op["path"], {}, {}
    for f in op["fields"]:
        raw = getattr(args, _dest(f["name"]), None)
        if raw is None:
            continue
        value = _convert(f, raw)
        if value is None and f["in"] != "body":
            continue  # null means "not given" outside a body
        if f["in"] == "path":
            path = path.replace("{" + f["name"] + "}", quote(str(value), safe=""))
        elif f["in"] == "query":
            query[f["name"]] = value
        else:
            body[f["name"]] = value
    has_body = any(f["in"] == "body" for f in op["fields"])
    return path, query, (body if has_body else None)


# --- calling and printing ---


async def wait_for_job(client: httpx.AsyncClient, job: dict) -> dict:
    """Poll until the job ends. A job of the host helper (`host-...`) can stop and
    start the application that answers: while it restarts, the API is unreachable, and the
    client keeps asking for up to HOST_JOB_UNREACHABLE_SECONDS."""
    unreachable = 0.0
    while job["status"] not in FINAL_JOB:
        await asyncio.sleep(POLL_SECONDS)
        host_job = job["id"].startswith("host-")
        try:
            response = await client.get(f"/jobs/{job['id']}")
        except httpx.TransportError:
            unreachable += POLL_SECONDS
            if not host_job or unreachable > HOST_JOB_UNREACHABLE_SECONDS:
                raise
            continue
        if host_job and response.status_code >= 500:
            unreachable += POLL_SECONDS
            if unreachable <= HOST_JOB_UNREACHABLE_SECONDS:
                continue
        response.raise_for_status()
        unreachable = 0.0
        job = response.json()
    return job


def _cell(value) -> str:
    """A table cell: cut at 60 characters so the columns stay readable."""
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _fields(body: dict, indent: str = "") -> list[str]:
    """One object as `key: value` lines, uncut: a nested object is indented under its
    key, a list of rows is a table under it, so `status` shows what `--json` shows."""
    lines = []
    for key, value in body.items():
        if isinstance(value, dict) and value:
            lines.append(f"{indent}{key}:")
            lines += _fields(value, indent + "  ")
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            lines.append(f"{indent}{key}:")
            lines += [(indent + "  " + line).rstrip() for line in render(value).split("\n")]
        elif isinstance(value, (dict, list)):
            lines.append(f"{indent}{key}: {json.dumps(value, ensure_ascii=False)}")
        else:
            lines.append(f"{indent}{key}: {value}")
    return lines


def render(body) -> str:
    if body is None:
        return "ok"
    if isinstance(body, list):
        if not body:
            return "(none)"
        if all(isinstance(r, dict) for r in body):
            columns = list(body[0])
            rows = [
                [_cell(r[c]) if r.get(c) is not None else "" for c in columns] for r in body
            ]
            widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(columns)]
            lines = ["  ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True))]
            lines += ["  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)) for r in rows]
            return "\n".join(lines)
        return "\n".join(_cell(item) for item in body)
    if isinstance(body, dict):
        return "\n".join(_fields(body))
    return str(body)


def _dump(body, as_json: bool) -> str:
    return json.dumps(body, indent=2, ensure_ascii=False) if as_json else render(body)


async def execute(client: httpx.AsyncClient, op: dict, args: argparse.Namespace):
    """One API call, then the job it started if any. Returns (status code, body)."""
    path, query, body = request_parts(op, args)
    response = await client.request(op["method"], path, params=query or None, json=body)
    if response.status_code == 204:
        return 204, None
    try:
        payload = response.json()
    except ValueError:
        payload = response.text
    wait = not args.no_wait or client.transport_name == "inprocess"
    if response.status_code == 202 and wait and isinstance(payload, dict) and "id" in payload:
        payload = await wait_for_job(client, payload)
    return response.status_code, payload


async def amain(argv: list[str], out=None, err=None) -> int:
    out, err = out or sys.stdout, err or sys.stderr
    from app.admin.manifest import manifest
    from app.api.app import app

    m = manifest(app)
    parser = build_parser(m["operations"])
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(out)
        return 2
    if args.command == "describe":
        if args.json:
            print(json.dumps(m, indent=2, ensure_ascii=False), file=out)
        else:
            for o in m["operations"]:
                print(
                    f"{o['command']:<32} {o['method']:<6} {o['path']:<62} "
                    f"{o['scope']:<8} {o['description']}",
                    file=out,
                )
            print(f"{m['count']} commands (API version {m['api_version']})", file=out)
        return 0
    if args.command == "sign-out":
        return await sign_out(args, out, err)
    op = next(o for o in m["operations"] if o["command"] == args.command)
    try:
        if op["command"] == SIGN_IN:
            return await sign_in(op, args, out, err)
        resolve_secrets(op, args)
        async with open_client(args.transport) as client:
            code, body = await execute(client, op, args)
            refused_saved = code == 401 and client.transport_name == "http" and bool(
                saved_token(client.api_base)
            )
    except UsageError as exc:
        print(f"error: {exc}", file=err)
        return 2
    except httpx.HTTPError as exc:
        print(f"error: the API could not be reached ({type(exc).__name__})", file=err)
        return 2
    # A job the command started and waited for ends the command with its outcome; a job
    # that get-job or cancel-job merely describes is data, whatever its status.
    failed = code >= 400 or (
        op["returns_job"]
        and isinstance(body, dict)
        and body.get("status") in ("failed", "cancelled")
    )
    print(_dump(body, args.json), file=err if failed else out)
    if refused_saved:
        print(
            "The saved token was refused (expired or revoked): ./start.sh --admin sign-in", file=err
        )
    return 1 if failed else 0


def _ask_line(prompt: str, secret: bool, stdin=None) -> str:
    """Asked on the terminal (without echo for a secret), or one line of standard input."""
    stdin = stdin or sys.stdin
    if stdin.isatty():
        return getpass.getpass(prompt) if secret else input(prompt)
    return stdin.readline().rstrip("\r\n")


async def sign_in(op: dict, args: argparse.Namespace, out, err) -> int:
    """Name and password in a Basic header, the code in X-TOTP: never on the command line.
    Standard input, when it is not a terminal, gives the password then the code, one a line."""
    name = args.name or _ask_line("name: ", secret=False)
    password = _ask_line("password: ", secret=True)
    code = _ask_line("code (empty when the account has none): ", secret=False).strip()
    if not name or not password:
        raise UsageError("a name and a password are needed")
    basic = base64.b64encode(f"{name}:{password}".encode()).decode("ascii")
    extra = {"X-TOTP": code} if code else None
    async with open_client(args.transport, authorization=f"Basic {basic}", extra=extra) as client:
        status, body = await execute(client, op, args)
        base_url = client.api_base
    if status != 201 or not isinstance(body, dict):
        print(_dump(body, args.json), file=err)
        return 1
    save_token(
        base_url,
        {k: body[k] for k in ("token", "id", "account", "scope", "label", "expires_at")},
    )
    shown = body if args.json else {**body, "token": f"(saved in {token_file()})"}
    print(_dump(shown, args.json), file=out)
    return 0


async def sign_out(args: argparse.Namespace, out, err) -> int:
    base_url, _named = api_base_url()
    saved = saved_token(base_url)
    if saved is None:
        print(f"not signed in to {base_url}", file=out)
        return 0
    try:
        async with open_client("http") as client:
            response = await client.delete(f"/auth/tokens/{saved['id']}")
    except httpx.HTTPError as exc:
        print(f"error: the API could not be reached ({type(exc).__name__})", file=err)
        return 2
    forget_token(base_url)
    revoked = response.status_code == 200
    print(f"signed out of {base_url}" + ("" if revoked else " (the token was already refused)"),
          file=out)  # fmt: skip
    return 0


def main() -> int:
    return asyncio.run(amain(sys.argv[1:]))


if __name__ == "__main__":
    sys.exit(main())
