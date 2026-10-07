"""Liveness signal for the container healthcheck.

While the application runs, `heartbeat()` touches a file every few seconds,
and only while at least one component (adapter, Admin API) is still running.
The Docker healthcheck runs `python -m app.health`, which succeeds when that
file is recent. A process that is alive but has lost all its components, or
whose event loop is stuck, stops touching the file and turns `unhealthy`
after `MAX_AGE_SECONDS`. No network port and no data involved, so it works
with or without the Admin API.

Stdlib only, on purpose: the check starts a fresh Python every interval.
"""

import asyncio
import contextlib
import logging
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger("channelagent")

HEARTBEAT_PATH = Path(tempfile.gettempdir()) / "channelagent.heartbeat"
INTERVAL_SECONDS = 15
MAX_AGE_SECONDS = 60


# Components that must show a recent success. A running task that is stuck on
# the network (a poll that never returns) would otherwise keep the heartbeat alive.
# name -> [max age in seconds, monotonic time of the last success]
_components: dict[str, list[float]] = {}


def register(name: str, max_age: float) -> None:
    """Start expecting a success from `name` at least every `max_age` seconds. The
    registration itself counts as the first one, so a slow start is not a failure.
    """
    _components[name] = [max_age, time.monotonic()]


def unregister(name: str) -> None:
    _components.pop(name, None)


def mark(name: str) -> None:
    """`name` just did what it is for (a poll completed, the API answered)."""
    if name in _components:
        _components[name][1] = time.monotonic()


@contextlib.contextmanager
def working(name: str, seconds: float):
    """While `name` does one piece of work bounded by its own limit, its last success may be
    `seconds` older than usual. An email turn waits for the engine up to
    LLM_TASK_TIMEOUT_SECONDS (600 by default), longer than the email limit (300 s): the
    poll loop marks no success meanwhile, and a legitimate long answer turned the container
    unhealthy (measured 2026-10-04: "Not beating" from 4 min 47 s into a 7 min 32 s turn).
    """
    entry = _components.get(name)
    if entry is None:
        yield
        return
    base = entry[0]
    entry[0] = base + seconds
    try:
        yield
    finally:
        if _components.get(name) is entry:
            entry[0] = base


def snapshot() -> dict[str, dict[str, float | bool]]:
    """Each expected component with the age of its last success and whether that is
    still within its limit."""
    now = time.monotonic()
    return {
        name: {"seconds_since_success": round(now - last, 1), "healthy": now - last <= max_age}
        for name, (max_age, last) in sorted(_components.items())
    }


def stale_components() -> list[str]:
    now = time.monotonic()
    return sorted(n for n, (max_age, last) in _components.items() if now - last > max_age)


def beat(path: Path | None = None) -> None:
    (path or HEARTBEAT_PATH).touch()


async def heartbeat(components: Sequence[asyncio.Task], interval: float = INTERVAL_SECONDS) -> None:
    """Touch the heartbeat file while the application is doing its job: with no
    component at all it is idling on purpose, and counts as alive. Once every
    component has ended it stops, so the container turns unhealthy.
    """
    while not components or any(not task.done() for task in components):
        stale = stale_components()
        if stale:
            logger.warning(
                "Not beating: no recent success from %s (the container turns unhealthy)",
                ", ".join(stale),
            )
        else:
            beat()
        await asyncio.sleep(interval)


def check(path: Path | None = None, max_age: float = MAX_AGE_SECONDS) -> tuple[bool, str]:
    target = path or HEARTBEAT_PATH
    try:
        age = time.time() - target.stat().st_mtime
    except FileNotFoundError:
        return False, f"no heartbeat file at {target}"
    if age > max_age:
        return False, f"last heartbeat {age:.0f} s ago (limit {max_age:.0f} s)"
    return True, f"last heartbeat {age:.0f} s ago"


def main() -> int:
    healthy, message = check()
    print(("healthy: " if healthy else "unhealthy: ") + message)
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
