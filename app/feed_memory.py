"""What a scheduled task already delivered from its feeds, so a recurring digest does
not repeat itself.

In a task's turn (`app.tasks.execute` sets `current_task`), every result of a built-in "feeds"
server passes through `filter_result` in the MCP executor (app/mcp/catalogue.py): the items the
task already delivered are removed before the model sees them, and the ones left are noted.
Only when the run's reply was delivered are those noted ids remembered (`remember`), so a failed
or undelivered run offers the same items again next time. The filter is the application's: the
model cannot ask for a delivered item back.
"""

from __future__ import annotations

import contextvars

from sqlalchemy import delete, func, select

from app import feeds
from app.db.models import TaskFeedItem
from app.db.session import session_scope

FEEDS_BUILTIN = "feeds"
NOTHING_NEW = (
    "Nothing new in this feed since this task's last run. Do not read it again: say in one "
    "line that there is nothing new."
)
MAX_REMEMBERED = 2000

# {"task_id": int, "seen": set | None (loaded on first use), "shown": set}
current_task: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "feed_task", default=None
)


def start(task_id: int) -> contextvars.Token:
    return current_task.set({"task_id": task_id, "seen": None, "shown": set()})


async def filter_result(builtin_id: str | None, text: str) -> str:
    """The result without the items this task already delivered; unchanged outside a task's
    turn and for any other server."""
    state = current_task.get()
    if state is None or builtin_id != FEEDS_BUILTIN or "\n\n### " not in text:
        return text
    if state["seen"] is None:
        async with session_scope() as session:
            rows = await session.execute(
                select(TaskFeedItem.item_id).where(TaskFeedItem.task_id == state["task_id"])
            )
            state["seen"] = set(rows.scalars())
    head, blocks = feeds.split_blocks(text)
    kept = [(i, b) for i, b in blocks if i is None or i not in state["seen"]]
    state["shown"].update(i for i, _b in kept if i is not None)
    dropped = len(blocks) - len(kept)
    lines = [
        f"Items: {len(kept)}" if line.startswith("Items: ") else line
        for line in head.split("\n")
    ]
    if dropped:
        lines.append(f"({dropped} items this task already delivered were left out)")
    if dropped and not kept:
        # Without this, the owner's model asked the same feed 4 times with other arguments and
        # the run ended on the tool round limit (measured on the real engine).
        lines.append(NOTHING_NEW)
    return "\n\n".join(["\n".join(lines), *[b for _i, b in kept]])


async def remember(task_id: int, item_ids) -> int:
    """Remember the delivered items of one run; keeps the newest MAX_REMEMBERED per task.
    Returns how many were added."""
    ids = sorted(set(item_ids))
    if not ids:
        return 0
    async with session_scope() as session:
        known = set(
            (
                await session.execute(
                    select(TaskFeedItem.item_id).where(
                        TaskFeedItem.task_id == task_id, TaskFeedItem.item_id.in_(ids)
                    )
                )
            ).scalars()
        )
        new = [i for i in ids if i not in known]
        session.add_all(TaskFeedItem(task_id=task_id, item_id=i) for i in new)
        await session.flush()
        count = (
            await session.execute(
                select(func.count()).select_from(TaskFeedItem).where(
                    TaskFeedItem.task_id == task_id
                )
            )
        ).scalar_one()
        if count > MAX_REMEMBERED:
            oldest = (
                await session.execute(
                    select(TaskFeedItem.id).where(TaskFeedItem.task_id == task_id)
                    .order_by(TaskFeedItem.id).limit(count - MAX_REMEMBERED)
                )
            ).scalars().all()  # fmt: skip
            await session.execute(delete(TaskFeedItem).where(TaskFeedItem.id.in_(oldest)))
        await session.commit()
    return len(new)


async def forget(session, task_ids) -> None:
    """Delete what these tasks remembered (before the tasks themselves: foreign key)."""
    ids = list(task_ids)
    if ids:
        await session.execute(delete(TaskFeedItem).where(TaskFeedItem.task_id.in_(ids)))
