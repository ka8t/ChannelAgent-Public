"""Interactive admin console: ./start.sh --admin (rebuilt on the Admin API).

The console is a client of the Admin API, like `./start.sh --admin COMMAND` (one call) and
the admin UI: it never opens the database. It reaches the running application over HTTP
(`API_URL`, or this machine's API when it answers) and, when the application is stopped, the
same handlers in process (`app/admin/client.py`). Its menus come from the API's manifest: one
menu per tag (users, requests, agents, audit, ...), one entry per API command, the command's
fields asked one by one with their type. A route added to the API appears here with no console
code. Actions are recorded with the actor `cli:<operating-system user>`, and each route's scope
applies (a refusal is shown, never bypassed).

At a prompt: a number or a name picks an entry, `b` goes back, `q` quits. A command name can
also be typed directly at the first menu. A wrong answer or a refused call prints a message and
the session goes on; only `q`, Ctrl+D or Ctrl+C end it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable
from types import SimpleNamespace

import httpx

from app.admin import client as api

CONFIRM_WORD = "YES"


def _prompt(label: str) -> str:
    return input(f"{label}: ").strip()


def _secret_prompt(label: str) -> str:
    import getpass

    return getpass.getpass(f"{label}: ")


class _BadInput(Exception):
    """An answer that cannot be used. Printed, never a traceback."""


def _manifest() -> list[dict]:
    from app.admin.manifest import manifest
    from app.api.app import app

    return manifest(app)["operations"]


def groups(operations: list[dict]) -> dict[str, list[dict]]:
    """Tag -> its commands, in the manifest's order (tags sorted, commands sorted)."""
    result: dict[str, list[dict]] = {}
    for op in operations:
        result.setdefault(op["tag"] or "other", []).append(op)
    return result


def _pick(answer: str, names: list[str]) -> str | None:
    """A number (1-based) or a name among `names`."""
    if answer.isdigit() and 1 <= int(answer) <= len(names):
        return names[int(answer) - 1]
    return answer if answer in names else None


def _field_label(field: dict) -> str:
    kind = field["type"]
    if field["enum"]:
        kind += ", one of " + "/".join(str(e) for e in field["enum"])
    if field["type"] in ("array", "object"):
        kind += ", as JSON"
    if field.get("nullable"):
        kind += ", null = none"
    if field["required"] and field["default"] is None:
        need = "required"
    elif field["default"] is not None:
        need = f"blank = {field['default']}"
    else:
        need = "blank = none"
    return f"{field['name']} ({kind}, {need})"


def _needs_confirmation(op: dict) -> bool:
    """A deletion, or anything that needs the owner scope (a restore, a key rotation, a stop,
    the configuration, deleting a user)."""
    return op["method"] == "DELETE" or op["scope"] == "owner"


async def run_operation(
    http: httpx.AsyncClient,
    op: dict,
    ask: Callable[[str], str] = _prompt,
    out: Callable[[str], None] = print,
    ask_secret: Callable[[str], str] = _secret_prompt,
) -> bool:
    """Ask the fields of one command, call it, print the result. True when the API answered
    with success. Nothing is sent when an answer is missing or unusable."""
    out(f"{op['command']}: {op['description']}  [{op['method']} {op['path']}, scope {op['scope']}]")
    values: dict[str, str] = {}
    for field in op["fields"]:
        # A secret is typed without echo.
        answer = (ask_secret if field.get("secret") else ask)(_field_label(field))
        if answer:
            values[field["name"]] = answer
        elif field["required"] and field["default"] is None:
            raise _BadInput(f"{field['name']} is required")
    if _needs_confirmation(op):
        if ask(f"Type {CONFIRM_WORD} to run {op['command']}") != CONFIRM_WORD:
            out("Not run.")
            return False
    args = SimpleNamespace(
        no_wait=False,
        **{"field__" + f["name"]: values.get(f["name"]) for f in op["fields"]},
    )
    try:
        code, body = await api.execute(http, op, args)
    except api.UsageError as exc:
        raise _BadInput(str(exc)) from None
    failed = code >= 400 or (
        op["returns_job"]
        and isinstance(body, dict)
        and body.get("status") in ("failed", "cancelled")
    )
    if failed:
        detail = body
        if isinstance(body, dict):
            detail = body.get("error") or body.get("detail") or body
        out(f"Not done (HTTP {code}): {api.render(detail)}")
        return False
    # An object in full (a job, a user, the storage overview); a list as a table.
    out(
        json.dumps(body, indent=2, ensure_ascii=False)
        if isinstance(body, dict)
        else api.render(body)
    )
    return True


async def _run_safely(http, op, ask, out, ask_secret=_secret_prompt) -> None:
    try:
        await run_operation(http, op, ask, out, ask_secret)
    except _BadInput as exc:
        out(f"Invalid input, nothing sent. {exc}")
    except httpx.HTTPError as exc:
        out(f"The API could not be reached ({type(exc).__name__}); nothing was done.")


async def session(
    http: httpx.AsyncClient,
    operations: list[dict],
    ask: Callable[[str], str] = _prompt,
    out: Callable[[str], None] = print,
    ask_secret: Callable[[str], str] = _secret_prompt,
) -> None:
    """The menu loop. Returns on `q`; an ended input (EOFError) goes to the caller. A secret
    field is asked with `ask_secret` (no echo)."""
    menus = groups(operations)
    by_command = {op["command"]: op for op in operations}
    tags = list(menus)
    while True:
        out("")
        for number, tag in enumerate(tags, 1):
            out(f"{number}. {tag} ({len(menus[tag])} commands)")
        out("q. Quit    (or type a command name)")
        answer = ask("Menu")
        if answer == "q":
            return
        if answer in by_command:
            await _run_safely(http, by_command[answer], ask, out, ask_secret)
            continue
        tag = _pick(answer, tags)
        if tag is None:
            out("Unknown option.")
            continue
        while True:
            out("")
            names = [op["command"] for op in menus[tag]]
            for number, op in enumerate(menus[tag], 1):
                out(f"  {number}. {op['command']:<28} {op['scope']:<8} {op['description']}")
            out("  b. Back    q. Quit")
            answer = ask(tag)
            if answer == "q":
                return
            if answer == "b":
                break
            command = _pick(answer, names)
            if command is None:
                out("Unknown option.")
                continue
            await _run_safely(http, by_command[command], ask, out, ask_secret)


async def amain(transport: str = "auto") -> int:
    from app.logging_setup import install_redaction

    install_redaction()  # the key and the tokens never reach a log, in either transport
    print("=== ChannelAgent Admin Console ===")
    try:
        async with api.open_client(transport) as http:
            if http.transport_name == "inprocess":
                where = "in process (the application is stopped)"
                # This machine's files are then the ones used: say if others can read them.
                from app.security.permissions import warn_about_loose_application_files

                warn_about_loose_application_files()
            else:
                where = str(http.base_url)
            print(f"Admin API: {where}. Every action goes through it, as {api.client_label()}.")
            try:
                await session(http, _manifest())
            except (EOFError, KeyboardInterrupt):
                print("\nInput closed, leaving the console.")
    except api.UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The interactive admin console.")
    parser.add_argument("--transport", choices=["auto", "http", "inprocess"], default="auto")
    args = parser.parse_args(argv)
    return asyncio.run(amain(args.transport))


if __name__ == "__main__":
    sys.exit(main())
