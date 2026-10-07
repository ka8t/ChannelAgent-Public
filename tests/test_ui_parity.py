"""The parity of the two clients of the Admin API, start.sh and the admin UI (/ui/parity).

- The report compares the script's real parser with the UI's operations: 0 differences, and a
  flag removed from the script, or an operation missing from the UI, is reported.
- Every start.sh command of its usage is in the table of commands besides --admin COMMAND.
- Each hand-written screen calls and sends exactly the operations its table declares.
- The command line shown for a UI call, parsed by the script's own parser, gives the same API
  request as the UI sent.
- Every page lists its API calls with that command line; an action's outcome shows it once.
"""

import re
import shlex
from argparse import Namespace
from pathlib import Path

from tests.test_admin_ui import KEY, _signed_in, ui  # noqa: F401 - the fixture

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "app" / "ui" / "templates"
SCREEN_TEMPLATES = {
    "/ui/": "index.html",
    "/ui/status": "status.html",
    "/ui/backups": "backups.html",
    "/ui/config": "config.html",
    "/ui/models": "models.html",
    "/ui/requests": "requests.html",
    "/ui/tasks": "tasks.html",
    "/ui/logs": "logs.html",
    "/ui/login": "login.html",
}


def test_the_report_finds_no_difference_between_the_script_and_the_ui():
    from app.ui import parity

    report = parity.report()
    print(f"operations {report['operations']}, script {report['script']}, ui {report['ui']}, "
          f"same {report['same_flags']}, differences {len(report['differences'])}")  # fmt: skip
    assert report["operations"] >= 120
    assert report["script"] == report["ui"] == report["same_flags"] == report["operations"]
    assert report["differences"] == []


def test_a_flag_missing_from_the_script_or_a_screen_missing_from_the_ui_is_reported(monkeypatch):
    from app.ui import app as ui_app
    from app.ui import parity

    real_script = parity.script_commands()
    broken = {**real_script, "get-user": real_script["get-user"] - {"--user-id"}}
    broken["made-up"] = set()
    monkeypatch.setattr(parity, "script_commands", lambda: broken)
    real_ui = ui_app.operations()
    monkeypatch.setattr(
        ui_app, "operations", lambda: {k: v for k, v in real_ui.items() if k != "list-users"}
    )
    differences = parity.report()["differences"]
    assert differences == [
        "get-user: flags missing in start.sh: --user-id",
        "list-users: no UI screen",
        "made-up: start.sh command with no API operation",
    ]


def test_every_command_of_the_start_script_usage_is_in_the_table():
    from app.ui import parity

    usage = (REPO / "start.sh").read_text().split("cat <<'USAGE'")[1].split("USAGE")[0]
    commands = {m.group(1) for m in re.finditer(r"^  (--[a-z-]+)", usage, re.M)}
    known = {c["command"].split()[0] for c in parity.SCRIPT_COMMANDS} | {"--help"}
    print(f"start.sh usage commands: {sorted(commands)}")
    assert len(commands) >= 10 and commands - known == set()
    operations = {o["command"] for o in parity._manifest()["operations"]}
    named = {o for c in parity.SCRIPT_COMMANDS for o in c["operations"]}
    assert named - operations == set(), "every operation named in the table exists"


async def test_each_hand_written_screen_calls_and_sends_what_its_table_declares(ui):  # noqa: F811
    from app.ui import parity

    await _signed_in(ui)
    operations = {o["command"] for o in parity._manifest()["operations"]}
    checked = 0
    for href, ops in parity.SCREEN_OPERATIONS.items():
        template = (TEMPLATES / SCREEN_TEMPLATES[href]).read_text()
        # A form's action, or a command named in a loop or a macro call (`("host-stop", ...)`).
        quoted = re.findall(r'(?<!data-cli=)["\']([a-z]+(?:-[a-z]+)+)["\']', template)
        named = set(quoted) & operations
        sent = set(re.findall(r'action="/ui/op/([a-z-]+)"', template)) | named
        if href == "/ui/login":
            sent = {"sign-in"} if 'action="/ui/login"' in template else set()
        else:
            page = (await ui.get(href, params={"run": "1"} if href == "/ui/logs" else None)).text
            footer = page.split('<footer class="calls">')[1]
            called = set(re.findall(r'<td><a href="/ui/op/([a-z-]+)">', footer))
            assert called == set(ops["reads"]), (href, called)
        # Read from the template: the status screen's host buttons show only with the helper on.
        assert sent == set(ops["writes"]), (href, sent)
        checked += 1
    assert checked == len(parity.SCREEN_OPERATIONS) == len(SCREEN_TEMPLATES)


def _parsed_request(line: str):
    """The API request the script would send for this command line, by its own parser."""
    from app.admin.client import build_parser, request_parts
    from app.ui import parity

    ops = {o["command"]: o for o in parity._manifest()["operations"]}
    words = shlex.split(line)
    assert words[:2] == ["./start.sh", "--admin"]
    args = build_parser(list(ops.values())).parse_args(words[2:])
    return request_parts(ops[args.command], args)


def test_the_command_line_of_a_ui_call_is_the_same_request_for_the_script():
    from app.admin.client import request_parts
    from app.ui import parity
    from app.ui.app import _from_lines

    ops = {o["command"]: o for o in parity._manifest()["operations"]}
    cases = {
        "update-agent": {"agent_id": "7", "name": "it's mine", "is_active": "false",
                         "tools": "mcp__a__one\n mcp__b__two \n"},
        "search-logs": {"user_id": "3", "keyword": "hello world", "limit": "20"},
        "set-backup-schedule": {"enabled": "true", "interval_minutes": "30", "keep": "4"},
        "list-requests": {"status": "pending"},
    }  # fmt: skip
    for command, form in cases.items():
        op = ops[command]
        values = _from_lines(op, form)
        sent = request_parts(op, Namespace(**{"field__" + f["name"]: values.get(f["name"])
                                              for f in op["fields"]}))  # fmt: skip
        line = parity.cli_line(op, values)
        assert _parsed_request(line) == sent, line


def test_a_secret_is_never_written_on_the_command_line():
    from app.ui import parity

    op = next(o for o in parity._manifest()["operations"] if o["command"] == "export-backup")
    assert parity.cli_line(op, {"password": "hunter2-secret"}) == (
        "./start.sh --admin export-backup --password -"
    )


async def test_every_page_lists_its_api_calls_with_the_command_line(ui):  # noqa: F811
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        user = await service.create_user(s, "Sam")
        await s.commit()
        user_id = user.id
    await _signed_in(ui)
    page = (await ui.get("/ui/op/get-user", params={"user_id": str(user_id)})).text
    footer = page.split('<footer class="calls">')[1]
    assert f"./start.sh --admin get-user --user-id {user_id}</code>" in footer
    assert f'<span class="mono">/users/{user_id}</span>' in footer
    assert "./start.sh --admin get-user --user-id " + str(user_id) in page.split("<footer")[0]


async def test_an_action_shows_its_outcome_and_command_once(ui):  # noqa: F811
    csrf = await _signed_in(ui)
    done = await ui.post(
        "/ui/op/set-backup-schedule",
        data={"csrf": csrf, "enabled": "true", "interval_minutes": "45", "keep": "3",
              "next": "/ui/backups"},
    )  # fmt: skip
    assert done.status_code == 303
    line = "./start.sh --admin set-backup-schedule --enabled true --interval-minutes 45 --keep 3"
    first = (await ui.get("/ui/backups")).text.split("<footer")[0]
    again = (await ui.get("/ui/backups")).text.split("<footer")[0]
    assert "Set backup schedule: done." in first and f'<code class="cli">{line}</code>' in first
    assert "Set backup schedule: done." not in again


async def test_the_parity_page_shows_the_counts_and_no_key(ui):  # noqa: F811
    from app.ui import parity

    await _signed_in(ui)
    page = await ui.get("/ui/parity")
    report = parity.report()
    assert page.status_code == 200 and KEY not in page.text
    assert f'<div class="n">{report["operations"]}</div>' in page.text
    assert '<div class="n state-ok">0</div>' in page.text
    assert page.text.count('<span class="badge badge-ok">yes</span>') == report["operations"]
    assert "./start.sh --config KEY=VALUE" in page.text
