"""Normalized event schema every channel adapter must produce.

The Auth Node and app/graph.py only ever see this shape — neither
imports anything Telegram/Email/Matrix-specific, per
.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.db.models import Channel


class ReplyProgress:
    """What a channel shows while a turn runs: `start()` when the turn begins (a
    "typing" sign), `update(text)` with the visible text of the answer so far, `stop()` when the
    turn ends, before the final reply is delivered. Every method is best effort: it never raises,
    and the final reply goes through `NormalizedEvent.reply` as before."""

    async def start(self) -> None:
        return None

    async def update(self, text: str) -> None:
        return None

    async def stop(self) -> None:
        return None


@dataclass
class NormalizedEvent:
    user_id: str
    channel: Channel
    text: str
    # How to deliver the agent's reply back through the channel this
    # event arrived on — e.g. a Telegram adapter's closure over
    # bot.send_message(chat_id, ...). Only app.channels.dispatch calls
    # this; app/graph.py and the Auth Node never see it.
    reply: Callable[[str], Awaitable[None]]
    # How to ask this user a yes/no question and wait for the answer: a tool
    # whose policy is "confirm" asks here. `confirm(question, timeout_seconds)` returns
    # True, False, or None when nobody answered in time. None: this channel cannot ask,
    # and such a tool is refused.
    confirm: Callable[[str, float], Awaitable[bool | None]] | None = None
    # A model chosen by the sender for this one message (`/model <name> <message>`).
    # None: the agent's own model, then the routing rules, then the default.
    model: str | None = None
    # How to send this user a file (`/export`): `send_file(file_name, data)`. None: this
    # channel sends text only.
    send_file: Callable[[str, bytes], Awaitable[None]] | None = None
    # How to send a question with answer buttons (agent builder):
    # `reply_choices(text, choices, nonce)`; a button sends its choice back as a message from
    # this user, and only while the question with that nonce is the one waiting. None: the
    # user types the answer.
    reply_choices: Callable[[str, list[str], str], Awaitable[None]] | None = None
    # What this channel shows while the turn runs: typing, then the answer as it is
    # written. None: the channel shows nothing until the reply (email).
    progress: ReplyProgress | None = None
