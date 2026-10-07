"""Rate limit and turn limit per user (decision D9: the same limits for everyone, values
in `.env`). One local engine serves every user, so one user sending a flood must not starve
the others.

- RATE_LIMIT_MESSAGES_PER_MINUTE: messages a user may send in any 60 seconds (0 = no limit).
- RATE_LIMIT_CONCURRENT_TURNS: turns of one user running at the same time (0 = no limit).

Held in memory: a restart clears the counters. A refused message is not counted.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import asynccontextmanager

from app.config import get_settings

WINDOW_SECONDS = 60
MAX_TRACKED_USERS = 10_000


def _now() -> float:
    return time.monotonic()


class UserLimiter:
    def __init__(self) -> None:
        self._sent: dict[int, deque[float]] = {}
        self._running: dict[int, int] = {}

    def refusal(self, user_id: int) -> str | None:
        """Why this user's message must wait ("rate" or "busy"), or None (then it counts)."""
        settings = get_settings()
        per_minute = settings.rate_limit_messages_per_minute
        concurrent = settings.rate_limit_concurrent_turns
        now = _now()
        sent = self._sent.setdefault(user_id, deque())
        while sent and now - sent[0] >= WINDOW_SECONDS:
            sent.popleft()
        if per_minute and len(sent) >= per_minute:
            return "rate"
        if concurrent and self._running.get(user_id, 0) >= concurrent:
            return "busy"
        sent.append(now)
        if len(self._sent) > MAX_TRACKED_USERS:
            del self._sent[next(iter(self._sent))]
        return None

    @asynccontextmanager
    async def turn(self, user_id: int):
        self._running[user_id] = self._running.get(user_id, 0) + 1
        try:
            yield
        finally:
            self._running[user_id] -= 1
            if not self._running[user_id]:
                del self._running[user_id]

    def clear(self) -> None:
        self._sent.clear()
        self._running.clear()


limiter = UserLimiter()

MESSAGES = {
    "rate": "You are sending messages faster than this assistant can take them. "
    "Please wait a minute and send it again.",
    "busy": "Your previous message is still being answered. Please wait for the answer, "
    "then send this one again.",
}
