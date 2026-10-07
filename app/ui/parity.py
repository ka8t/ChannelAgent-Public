"""What the two clients of the Admin API do, side by side (the parity screen, /ui/parity).

`start.sh --admin` and the UI are both generated from the API's manifest (`app/admin/manifest.py`):
this module proves it live instead of asserting it. It builds the script's real argument parser
(`app.admin.client.build_parser`) and the UI's operation list, and compares them command by
command and flag by flag. It also says which `start.sh` commands do not go through the API and
why, and which API operations each dedicated UI screen reads and writes; the tests keep both
tables true (`tests/test_admin_ui.py`).

It also writes the command line equivalent to a UI call (`cli_line`), shown on every page next
to the API calls the page made, so a screen and a terminal can be compared call for call.
"""

from __future__ import annotations

import argparse
import re
import shlex
from functools import lru_cache

API = "./start.sh --admin"
# The flags every command takes, not fields of the operation.
COMMON_FLAGS = {"--json", "--transport", "--no-wait", "--help", "-h"}

# The commands of start.sh besides `--admin COMMAND`, and how each one reaches the application.
# `via`: "api" = through the Admin API, as the UI; "shared" = the same Python module as the API
# route, without the API (the application may be stopped); "local" = on this machine only,
# before or around the application. `ui`: the screen doing the same thing, if any.
SCRIPT_COMMANDS = (
    {"command": "--docker", "via": "local", "operations": ("host-start",), "ui": "/ui/status",
     "why": "Builds and starts the container: there is no API before it runs. A running "
            "application is restarted from the Status screen through the host helper."},
    {"command": "--native", "via": "local", "operations": ("host-start",), "ui": "/ui/status",
     "why": "Starts the application on this machine: there is no API before it runs."},
    {"command": "--status", "via": "local", "operations": ("status", "host-status"),
     "ui": "/ui/status",
     "why": "Probes the container, the process, the ports and the engine from outside, so it "
            "answers when the API is down. The Status screen asks the API itself."},
    {"command": "--stop", "via": "local", "operations": ("host-stop",), "ui": "/ui/status",
     "why": "Signals the processes of this machine. With API_URL set it calls host-stop "
            "through the API, as the Status screen does."},
    {"command": "--config", "via": "shared", "operations": ("get-config",), "ui": "/ui/config",
     "why": "Reads .env with app/settings_rules.py, the module of GET /config, so it works "
            "with the application stopped."},
    {"command": "--config KEY=VALUE", "via": "shared", "operations": ("set-config",),
     "ui": "/ui/config",
     "why": "Checks and writes with app/settings_rules.py, the module of PATCH /config: "
            "the same rules, the same refusals."},
    {"command": "--config --edit", "via": "shared", "operations": ("set-config",),
     "ui": "/ui/config",
     "why": "One question per variable in a terminal (app/env_wizard.py), written with the "
            "same rules; the Configuration screen changes one variable at a time."},
    {"command": "--chat", "via": "api", "operations": ("chat", "get-chat", "answer-chat"),
     "ui": None,
     "why": "Talks to an agent through the chat routes of the API. No chat screen in the UI."},
    {"command": "--admin", "via": "api", "operations": (), "ui": "/ui/",
     "why": "Alone: interactive menus, one per API tag. With a command: one API call, "
            "generated from the routes (the table below). Every call goes through the API."},
    {"command": "--restore", "via": "local", "operations": ("restore-backup",),
     "ui": "/ui/backups",
     "why": "Restores with the application stopped (app/admin/restore.py). With API_URL set "
            "it calls restore-backup through the API, as the Backups screen does."},
    {"command": "--rekey", "via": "local", "operations": ("host-rekey",), "ui": "/ui/status",
     "why": "Rotates the key with the application stopped (app/admin/rekey_guided.py). With "
            "API_URL set it calls host-rekey through the API, as the Status screen does."},
)

# The commands of the script's client that are not API operations.
CLIENT_ONLY = (
    {"command": "describe", "why": "Prints the manifest the commands are generated from; "
                                   "the UI shows it as All operations and this screen."},
    {"command": "sign-out", "why": "Revokes the saved token (revoke-token) and forgets it; "
                                   "the UI's Sign out does the same for its session."},
)

# The dedicated screens: the API operations each one reads to draw itself, and those its
# buttons send. Every other operation has its generated screen, /ui/op/<command>.
SCREEN_OPERATIONS = {
    "/ui/": {"reads": ("whoami", "status"), "writes": ()},
    "/ui/status": {"reads": ("status", "host-status", "get-backup-schedule"),
                   "writes": ("host-restart", "host-stop", "host-start", "host-rekey")},
    "/ui/backups": {"reads": ("list-database-backups", "get-backup-schedule"),
                    "writes": ("create-database-backup", "restore-backup",
                               "set-backup-schedule", "run-backup-schedule-now")},
    "/ui/config": {"reads": ("get-config",), "writes": ("set-config",)},
    "/ui/models": {"reads": ("list-models",),
                   "writes": ("delete-model", "pull-model", "import-model")},
    "/ui/requests": {"reads": ("list-requests",), "writes": ("approve-request", "deny-request")},
    "/ui/tasks": {"reads": ("list-tasks", "get-pause"),
                  "writes": ("set-pause", "run-task", "update-task", "delete-task")},
    "/ui/logs": {"reads": ("search-logs",), "writes": ()},
    "/ui/login": {"reads": (), "writes": ("sign-in",)},
}


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


@lru_cache
def _manifest() -> dict:
    from app.admin.manifest import manifest
    from app.api.app import app as api

    return manifest(api)


def script_commands() -> dict[str, set[str]]:
    """The commands of `start.sh --admin` with their flags, read from the script's own parser."""
    from app.admin.client import build_parser

    parser = build_parser(_manifest()["operations"])
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return {
        name: {o for a in p._actions for o in a.option_strings if o not in COMMON_FLAGS}
        for name, p in sub.choices.items()
    }


def screens_of(command: str) -> list[str]:
    return [href for href, ops in SCREEN_OPERATIONS.items()
            if command in ops["reads"] or command in ops["writes"]]  # fmt: skip


def report() -> dict:
    """Every API operation with what the script and the UI offer for it, and the differences."""
    from app.ui.app import operations as ui_operations

    ops = _manifest()["operations"]
    script = script_commands()
    ui = ui_operations()
    rows, differences = [], []
    for op in ops:
        command = op["command"]
        fields = {_flag(f["name"]) for f in op["fields"]}
        if command == "sign-in":
            # The name goes in a header: a flag in the script, a field of the sign-in page.
            fields |= {"--name"}
        in_script = command in script
        # sign-in is its own page (/ui/login), every other operation a generated screen.
        in_ui = command in ui or command == "sign-in"
        flags = script.get(command, set())
        problems = []
        if not in_script:
            problems.append("no start.sh command")
        if not in_ui:
            problems.append("no UI screen")
        if in_script and flags != fields:
            missing = sorted(fields - flags)
            extra = sorted(flags - fields)
            if missing:
                problems.append("flags missing in start.sh: " + ", ".join(missing))
            if extra:
                problems.append("flags only in start.sh: " + ", ".join(extra))
        differences += [f"{command}: {p}" for p in problems]
        rows.append({
            **op,
            "script": f"{API} {command}",
            "ui": "/ui/login" if command == "sign-in" else f"/ui/op/{command}",
            "screens": screens_of(command),
            "problems": problems,
        })  # fmt: skip
    api_commands = {op["command"] for op in ops}
    client_only = {c["command"] for c in CLIENT_ONLY}
    for command in sorted(set(script) - api_commands - client_only):
        differences.append(f"{command}: start.sh command with no API operation")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["tag"], []).append(row)
    return {
        "api_version": _manifest()["api_version"],
        "operations": len(ops),
        "script": sum(1 for c in script if c in api_commands),
        "ui": sum(1 for r in rows if not any(p == "no UI screen" for p in r["problems"])),
        "same_flags": sum(1 for r in rows if not r["problems"]),
        "differences": differences,
        "groups": sorted(groups.items()),
        "script_commands": SCRIPT_COMMANDS,
        "client_only": CLIENT_ONLY,
        "screens": SCREEN_OPERATIONS,
    }


# --- one call, written as the terminal would send it ---


@lru_cache
def _matchers() -> list[tuple[str, re.Pattern, dict]]:
    out = []
    for op in _manifest()["operations"]:
        pattern = re.sub(r"\\\{(\w+)\\\}", r"(?P<\1>[^/]+)", re.escape(op["path"]))
        out.append((op["method"], re.compile(pattern + r"\Z"), op))
    # A fixed path wins over a parameter (/tasks/pause before /tasks/{task_id}).
    return sorted(out, key=lambda m: m[2]["path"].count("{"))


def operation_for(method: str, path: str) -> tuple[dict | None, dict]:
    for verb, pattern, op in _matchers():
        match = pattern.match(path)
        if verb == method and match:
            return op, match.groupdict()
    return None, {}


def _text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        import json

        return json.dumps(value, ensure_ascii=False)
    return str(value)


def cli_line(op: dict | None, values: dict) -> str | None:
    """`./start.sh --admin COMMAND --flag value ...` for these values; a secret is `-` (read on
    standard input, never on the command line, as the script requires)."""
    if op is None:
        return None
    parts = [API, op["command"]]
    for f in op["fields"]:
        value = values.get(f["name"])
        if value is None or value == "":
            continue
        parts += [_flag(f["name"]), "-" if f.get("secret") else shlex.quote(_text(value))]
    return " ".join(parts)


def describe_call(method: str, path: str, query: dict | None, body, status: int) -> dict:
    """One API call a page made: the operation it is, and the same call from the terminal."""
    from urllib.parse import unquote

    op, params = operation_for(method, path)
    values = {k: unquote(v) for k, v in params.items()}
    values.update(query or {})
    if isinstance(body, dict):
        values.update(body)
    return {
        "method": method,
        "path": path,
        "status": status,
        "command": op["command"] if op else None,
        "cli": cli_line(op, values),
    }
