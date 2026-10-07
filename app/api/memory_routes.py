"""An agent's memory: browse, add, edit and delete the entries of one agent of one
user. Memory is text written from conversations, so every route needs the admin scope,
like the logs; each change is an admin event (without the text).
"""

from datetime import datetime

from fastapi import APIRouter, Depends, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app import memory
from app.admin import memory as service
from app.api.actor import current_actor
from app.api.deps import get_db_session
from app.api.errors import error_responses
from app.api.routes import LIMIT, OFFSET, _page
from app.api.scopes import Scope, require
from app.db.models import MemoryEntry

router = APIRouter()
BASE = "/users/{user_id}/agents/{agent_id}/memory"


class MemoryEntryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=memory.MAX_TITLE)
    content: str = Field(min_length=1, max_length=memory.MAX_CONTENT)


class MemoryEntryUpdate(BaseModel):
    """A field left out keeps its value."""

    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, min_length=1, max_length=memory.MAX_TITLE)
    content: str | None = Field(default=None, min_length=1, max_length=memory.MAX_CONTENT)


class MemoryEntryOut(BaseModel):
    id: int
    user_id: int
    agent_id: int
    title: str
    content: str
    created_at: datetime
    updated_at: datetime


def _out(entry: MemoryEntry) -> MemoryEntryOut:
    return MemoryEntryOut(
        id=entry.id,
        user_id=entry.user_id,
        agent_id=entry.agent_id,
        title=entry.title,
        content=entry.content,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


@router.get(
    BASE,
    dependencies=[require(Scope.ADMIN)],
    response_model=list[MemoryEntryOut],
    tags=["memory"],
    responses=error_responses(404),
)
async def list_memory(
    user_id: int,
    agent_id: int,
    response: Response,
    query: str | None = Query(default=None, max_length=500, description="Words to look for."),
    limit: int = LIMIT,
    offset: int = OFFSET,
    session: AsyncSession = Depends(get_db_session),
) -> list[MemoryEntryOut]:
    """An agent's memory entries, newest first, or the ones matching `query`."""
    entries = await service.list_entries(session, user_id, agent_id, query=query)
    return [_out(e) for e in _page(response, entries, limit, offset)]


@router.post(
    BASE,
    dependencies=[require(Scope.ADMIN)],
    response_model=MemoryEntryOut,
    status_code=status.HTTP_201_CREATED,
    tags=["memory"],
    responses=error_responses(404, 409),
)
async def add_memory(
    user_id: int,
    agent_id: int,
    body: MemoryEntryIn,
    session: AsyncSession = Depends(get_db_session),
) -> MemoryEntryOut:
    """Add an entry to an agent's memory."""
    entry = await service.add_entry(
        session, user_id, agent_id, body.title, body.content, actor=current_actor()
    )
    await session.commit()
    return _out(entry)


@router.patch(
    BASE + "/{entry_id}",
    dependencies=[require(Scope.ADMIN)],
    response_model=MemoryEntryOut,
    tags=["memory"],
    responses=error_responses(404, 409),
)
async def edit_memory(
    user_id: int,
    agent_id: int,
    entry_id: int,
    body: MemoryEntryUpdate,
    session: AsyncSession = Depends(get_db_session),
) -> MemoryEntryOut:
    """Change the title or the content of one memory entry."""
    entry = await service.edit_entry(
        session,
        user_id,
        agent_id,
        entry_id,
        title=body.title,
        content=body.content,
        actor=current_actor(),
    )
    await session.commit()
    return _out(entry)


@router.delete(
    BASE + "/{entry_id}",
    dependencies=[require(Scope.ADMIN)],
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["memory"],
    responses=error_responses(404, 409),
)
async def delete_memory(
    user_id: int,
    agent_id: int,
    entry_id: int,
    session: AsyncSession = Depends(get_db_session),
) -> None:
    """Delete one memory entry."""
    await service.delete_entry(session, user_id, agent_id, entry_id, actor=current_actor())
    await session.commit()
