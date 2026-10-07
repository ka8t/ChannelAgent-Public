"""Administration of the agents' memory: browse, add, edit and delete the entries
of one agent of one user. The same functions as the agent's own memory tools
(app/memory.py), plus the ownership check and an admin event for every change. The
content is never copied into an admin event, only the entry's id.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app import memory
from app.admin.service import (
    AgentNotFoundError,
    InvalidInputError,
    NotFoundError,
    get_agent,
    record_admin_event,
)
from app.db.models import MemoryEntry


class MemoryEntryNotFoundError(NotFoundError):
    pass


async def _agent_of(session: AsyncSession, user_id: int, agent_id: int) -> None:
    agent = await get_agent(session, agent_id)
    if agent.user_id != user_id:
        raise AgentNotFoundError(f"No agent {agent_id} for user {user_id}")


async def _call(coroutine):
    try:
        return await coroutine
    except memory.MemoryNotFound as exc:
        raise MemoryEntryNotFoundError(str(exc)) from exc
    except memory.MemoryRefused as exc:
        raise InvalidInputError(str(exc)) from exc


async def list_entries(
    session: AsyncSession, user_id: int, agent_id: int, *, query: str | None = None
) -> list[MemoryEntry]:
    await _agent_of(session, user_id, agent_id)
    if query:
        return await memory.search_entries(
            session, user_id, agent_id, query, limit=memory.MAX_ENTRIES_PER_AGENT
        )
    return await memory.list_entries(session, user_id, agent_id)


async def add_entry(
    session: AsyncSession, user_id: int, agent_id: int, title, content, *, actor: str
) -> MemoryEntry:
    await _agent_of(session, user_id, agent_id)
    entry = await _call(memory.add_entry(session, user_id, agent_id, title, content))
    await _event(session, "memory.add", entry.id, user_id, agent_id, actor)
    return entry


async def edit_entry(
    session: AsyncSession,
    user_id: int,
    agent_id: int,
    entry_id: int,
    *,
    title=None,
    content=None,
    actor: str,
) -> MemoryEntry:
    await _agent_of(session, user_id, agent_id)
    entry = await _call(
        memory.edit_entry(session, user_id, agent_id, entry_id, title=title, content=content)
    )
    fields = [name for name, value in (("title", title), ("content", content)) if value is not None]
    await _event(session, "memory.edit", entry.id, user_id, agent_id, actor, fields=fields)
    return entry


async def delete_entry(
    session: AsyncSession, user_id: int, agent_id: int, entry_id: int, *, actor: str
) -> None:
    await _agent_of(session, user_id, agent_id)
    await _call(memory.delete_entry(session, user_id, agent_id, entry_id))
    await _event(session, "memory.delete", entry_id, user_id, agent_id, actor)


async def _event(session, action, entry_id, user_id, agent_id, actor, **extra) -> None:
    await record_admin_event(
        session,
        actor=actor,
        action=action,
        target_type="memory_entry",
        target_id=entry_id,
        details={"user_id": user_id, "agent_id": agent_id, **extra},
    )
