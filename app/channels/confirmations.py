"""Questions waiting for a user's yes or no: a tool whose policy is "confirm"
asks in the channel the message came from, and the turn waits for the answer.

One pending question per (channel, user). Each carries a random nonce: a Telegram
button names it, so a button left over from an earlier question cannot answer a later
one. A typed answer ("yes", "no") from the same user on the same channel answers the
current question. Anything else is not an answer and is handled as a message.
"""

import asyncio
import contextlib
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.db.models import Channel

YES_WORDS = frozenset({"yes", "y", "oui", "o", "ok"})
NO_WORDS = frozenset({"no", "n", "non"})


@dataclass
class Pending:
    nonce: str
    future: asyncio.Future


_pending: dict[tuple[Channel, str], Pending] = {}


def parse_answer(text: str) -> bool | None:
    word = text.strip().strip(".!").lower()
    if word in YES_WORDS:
        return True
    if word in NO_WORDS:
        return False
    return None


async def ask(
    channel: Channel,
    user_id: str,
    send: Callable[[str], Awaitable[None]],
    timeout: float,
) -> bool | None:
    """Register a question, let `send(nonce)` deliver it, wait for the answer.
    True yes, False no, None no answer within `timeout` seconds. A second question
    for the same user while one waits is refused (False): one turn at a time asks.
    """
    key = (channel, user_id)
    if key in _pending:
        return False
    pending = Pending(secrets.token_hex(4), asyncio.get_running_loop().create_future())
    _pending[key] = pending
    try:
        await send(pending.nonce)
        with contextlib.suppress(TimeoutError):
            return await asyncio.wait_for(asyncio.shield(pending.future), timeout)
        return None
    finally:
        if _pending.get(key) is pending:
            del _pending[key]


def _resolve(key: tuple[Channel, str], nonce: str | None, value: bool) -> bool:
    pending = _pending.get(key)
    if pending is None or pending.future.done():
        return False
    if nonce is not None and nonce != pending.nonce:
        return False
    pending.future.set_result(value)
    return True


def answer_text(channel: Channel, user_id: str, text: str) -> bool:
    """True when `text` answered a waiting question (the caller then does not treat
    it as a message)."""
    value = parse_answer(text)
    if value is None:
        return False
    return _resolve((channel, user_id), None, value)


def answer_nonce(channel: Channel, user_id: str, nonce: str, value: bool) -> bool:
    """A button press: it answers only the question it was sent with."""
    return _resolve((channel, user_id), nonce, value)


def is_waiting(channel: Channel, user_id: str) -> bool:
    return (channel, user_id) in _pending
