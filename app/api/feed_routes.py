"""The feeds listed for the agent builder: read scope to read, admin scope to change."""

from datetime import datetime

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin import feed_sources as service
from app.api.actor import current_actor
from app.api.deps import get_db_session
from app.api.errors import error_responses
from app.api.scopes import Scope, require

router = APIRouter()


class FeedSourceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    url: str = Field(min_length=9, max_length=2000, description="the feed's https address")
    topics: str = Field(default="", max_length=200, description="what it covers, in words")


class FeedSourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    url: str
    topics: str
    created_at: datetime


@router.get(
    "/feeds",
    dependencies=[require(Scope.READ)],
    response_model=list[FeedSourceOut],
    tags=["feeds"],
    responses=error_responses(),
)
async def list_feeds(session: AsyncSession = Depends(get_db_session)) -> list[FeedSourceOut]:
    """The feeds the agent builder may propose, by name."""
    return [FeedSourceOut.model_validate(s) for s in await service.list_sources(session)]


@router.post(
    "/feeds",
    dependencies=[require(Scope.ADMIN)],
    response_model=FeedSourceOut,
    status_code=status.HTTP_201_CREATED,
    tags=["feeds"],
    responses=error_responses(409),
)
async def create_feed(
    body: FeedSourceIn, session: AsyncSession = Depends(get_db_session)
) -> FeedSourceOut:
    """List a feed for the agent builder (an https address; 409 when the name is taken)."""
    source = await service.create_source(session, **body.model_dump(), actor=current_actor())
    await session.commit()
    return FeedSourceOut.model_validate(source)


@router.delete(
    "/feeds/{feed_id}",
    dependencies=[require(Scope.ADMIN)],
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["feeds"],
    responses=error_responses(404, 409),
)
async def delete_feed(feed_id: int, session: AsyncSession = Depends(get_db_session)) -> None:
    """Remove a feed from the builder's list."""
    await service.delete_source(session, feed_id, actor=current_actor())
    await session.commit()
