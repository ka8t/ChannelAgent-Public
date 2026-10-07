"""Driving the admin console in tests: through the Admin API in process, with scripted
answers built from the manifest, so a test names the values it gives, not the order of the
prompts."""

from app.admin import cli
from app.admin import client as api


def command(cmd: str, /, *, confirm: bool = True, **values) -> list[str]:
    """The answers that run one command from the first menu: its name, then one answer per
    field in the manifest's order (blank when not given), then the confirmation when the
    command asks for one."""
    op = next(o for o in cli._manifest() if o["command"] == cmd)
    answers = [cmd] + ["" if values.get(f["name"]) is None else str(values[f["name"]])
                        for f in op["fields"]]  # fmt: skip
    unknown = set(values) - {f["name"] for f in op["fields"]}
    assert not unknown, f"{cmd} has no field {unknown}"
    if cli._needs_confirmation(op):
        answers.append(cli.CONFIRM_WORD if confirm else "no")
    return answers


async def run_console(monkeypatch, *answers: str) -> str:
    """Run the console's menu loop over the API in process with `answers` (then `q`); returns
    everything it printed."""
    import os

    from app.config import get_settings

    if not os.environ.get("API_SERVER_KEY"):
        monkeypatch.setenv("API_SERVER_KEY", "console-test-key-" + "x" * 20)
    monkeypatch.delenv("API_URL", raising=False)
    get_settings.cache_clear()
    script = iter([*answers, "q"])

    def ask(_label: str) -> str:
        try:
            return next(script)
        except StopIteration:
            raise EOFError from None

    printed: list[str] = []
    async with api.open_client("inprocess") as http:
        await cli.session(http, cli._manifest(), ask=ask, out=printed.append, ask_secret=ask)
    return "\n".join(printed)
