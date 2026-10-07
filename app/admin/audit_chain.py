"""The tamper-evident chain of the admin events (S4).

Each event stores `prev_hash` (the hash of the event before it) and `hash`, the SHA-256 of
`prev_hash` and of the event's own content: id, time, actor, action, target and the details in
clear (decrypted, so a rotation of ENCRYPTION_KEY, which re-encrypts the details, does not
break the chain). Editing a stored event changes its content: its hash no longer matches.
Deleting one breaks the link of the next. `verify` walks the table in id order and reports
each event where this happens.

The hash is sealed in the transaction that writes the event, after the row is flushed: on
SQLite the flush takes the database's write lock, so no other writer can slip an event in
between the one read as "previous" and this one.

Limits: the chain has no secret, so whoever can write
the database can also recompute every hash after an edit; and deleting the newest events
leaves a shorter chain that still verifies. `verify` prints the last hash: kept somewhere else
(a note, a backup), it shows both.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AdminEvent

GENESIS = "0" * 64  # the "previous hash" of the first event ever written


def _utc_text(value: datetime) -> str:
    """One text for a time, whether SQLite gave it back naive (stored in UTC) or not."""
    aware = value if value.tzinfo else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).replace(tzinfo=None).isoformat(timespec="microseconds")


def content(
    event_id: int,
    created_at: datetime,
    actor: str,
    action: str,
    target_type: str,
    target_id: int | None,
    details: str | None,
) -> str:
    return json.dumps(
        [event_id, _utc_text(created_at), actor, action, target_type, target_id, details],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def link(prev_hash: str, text: str) -> str:
    return hashlib.sha256(f"{prev_hash}\n{text}".encode()).hexdigest()


def _event_content(event: AdminEvent) -> str:
    return content(
        event.id, event.created_at, event.actor, event.action, event.target_type,
        event.target_id, event.details,
    )  # fmt: skip


async def seal(session: AsyncSession, event: AdminEvent) -> None:
    """Chain `event`, already flushed (it has its id), to the event written before it."""
    previous = await session.scalar(
        select(AdminEvent.hash)
        .where(AdminEvent.id < event.id, AdminEvent.hash.is_not(None))
        .order_by(AdminEvent.id.desc())
        .limit(1)
    )
    event.prev_hash = previous or GENESIS
    event.hash = link(event.prev_hash, _event_content(event))
    await session.flush()


async def verify(session: AsyncSession) -> dict:
    """Every event in id order: its hash against its content, its `prev_hash` against the
    event before it. The first event's `prev_hash` is taken as it is (its anchor)."""
    problems: list[dict] = []
    count = 0
    first_id = last_id = None
    previous_hash: str | None = None
    rows = await session.stream_scalars(select(AdminEvent).order_by(AdminEvent.id))
    async for event in rows:
        count += 1
        first_id = event.id if first_id is None else first_id
        last_id = event.id
        if event.hash is None or event.prev_hash is None:
            problems.append({"id": event.id, "problem": "not chained (no hash)"})
        else:
            if previous_hash is not None and event.prev_hash != previous_hash:
                problems.append(
                    {"id": event.id, "problem": "the link to the event before is broken "
                     "(an event was removed, added or its hash rewritten)"}
                )  # fmt: skip
            if link(event.prev_hash, _event_content(event)) != event.hash:
                problems.append({"id": event.id, "problem": "its content was changed"})
        previous_hash = event.hash
    return {
        "events": count,
        "first_id": first_id,
        "last_id": last_id,
        "last_hash": previous_hash,
        "problems": problems,
        "intact": not problems,
    }
