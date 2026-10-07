"""A schedule written in words: "every Thursday at 9", "tous les jours ouvrés à 18h",
"le 1er du mois à 8h" turned into one of the three forms of app/tasks.py.

The model proposes (constrained JSON, the same call as the agent builder), the task parser
decides: the form must parse and fall due, or the phrase is refused with the reason. One
implementation for `/task add <phrase>` (app/channels/dispatch.py) and the Admin API (`POST
/schedules/parse`); the agent builder reads the schedule inside its own specification and
applies the same repairs (`repair`).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from app import tasks
from app.admin.service import InvalidInputError

MAX_PHRASE = 2000
_DAILY = re.compile(r"\d{1,2}:\d{2}")

SYSTEM_PROMPT = """Turn the user's words into a schedule, read in the user's timezone:
- schedule_kind "daily" with schedule_expr "HH:MM" (every day at that time);
- "every" with "30m", "2h" or "1d" (an interval from now);
- "cron" with five fields "minute hour day-of-month month day-of-week". The day of the week is
  the FIFTH field (0 or 7 = Sunday, 1 = Monday ... 4 = Thursday ... 6 = Saturday); the third
  field is a day of the month (1 to 31). Examples: every Thursday at 9:00 is "0 9 * * 4";
  working days (Monday to Friday, "jours ouvrés", weekdays) at 18:00 is "0 18 * * 1-5"; every
  hour from 9 to 17 is "0 9-17 * * *"; the 1st of each month at 8:00 is "0 8 1 * *".
- prompt: when the text also says what to do at each run, that part as the user wrote it
  (without the schedule words); otherwise null.
- When the words hold no schedule, schedule_kind and schedule_expr are null and reason says
  why, in the user's language; otherwise reason is null."""

SCHEMA = {
    "type": "object",
    "properties": {
        "schedule_kind": {"type": ["string", "null"], "enum": ["cron", "every", "daily", None]},
        "schedule_expr": {"type": ["string", "null"]},
        "prompt": {"type": ["string", "null"]},
        "reason": {"type": ["string", "null"]},
    },
    "required": ["schedule_kind", "schedule_expr", "prompt", "reason"],
    "additionalProperties": False,
}


def repair(kind, expr) -> tuple[str | None, str | None]:
    """Deterministic repairs of mistakes measured on the installed model: a five-field
    line given as "daily" is cron, a time given as "cron" is daily."""
    expr = (expr or "").strip() or None
    if expr is not None and kind in ("daily", "cron", None):
        if len(expr.split()) == 5:
            kind = "cron"
        elif _DAILY.fullmatch(expr):
            kind = "daily"
    return kind, expr


def next_runs(kind: str, expr: str, timezone: str | None, now: datetime, count: int = 3) -> list:
    """The next `count` run times, in the user's timezone."""
    zone = ZoneInfo(timezone or "UTC")
    times, after = [], now
    for _ in range(count):
        after = tasks.next_time(kind, expr, timezone, after)
        times.append(after.astimezone(zone))
        after = after + timedelta(seconds=1)
    return times


def describe_runs(times: list) -> str:
    return ", ".join(t.strftime("%a %Y-%m-%d %H:%M") for t in times)


async def parse(text: str, timezone: str | None, now: datetime | None = None) -> dict:
    """{"kind", "expr", "prompt", "next_runs"} for a phrase, or InvalidInputError with the
    reason (no schedule in the words, or a form that does not parse or never falls due)."""
    from app import builder

    if not isinstance(text, str) or not text.strip() or len(text) > MAX_PHRASE:
        raise InvalidInputError(f"a schedule phrase is 1 to {MAX_PHRASE} characters")
    now = now or datetime.now(UTC)
    local = now.astimezone(ZoneInfo(timezone or "UTC")).strftime("%A %Y-%m-%d %H:%M")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Now: {local} ({timezone or 'UTC'}).\nText: {text.strip()}"},
    ]
    data = await builder.ask_model(messages, SCHEMA)
    kind, expr = repair(data.get("schedule_kind"), data.get("schedule_expr"))
    if not kind or not expr:
        reason = data.get("reason") if isinstance(data.get("reason"), str) else None
        raise InvalidInputError(f"no schedule found in the words ({reason or 'none given'})")
    kind, expr = tasks.check_schedule(kind, expr)
    runs = next_runs(kind, expr, timezone, now)  # also refuses a schedule that never falls due
    prompt = data.get("prompt") if isinstance(data.get("prompt"), str) else None
    return {"kind": kind, "expr": expr, "prompt": (prompt or "").strip() or None, "next_runs": runs}
