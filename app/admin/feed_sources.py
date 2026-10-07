"""The feeds an administrator lists for the agent builder: name, https address,
topics. Reading needs the read scope, a change the admin scope and is an admin event. The list
is data in the database, never a list of third-party sites in the code."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.service import ConflictError, InvalidInputError, NotFoundError, record_admin_event
from app.db.models import FeedSource

MAX_SOURCES = 200
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,99}")


def _clean(name, url, topics) -> dict:
    if not isinstance(name, str) or not _NAME.fullmatch(name.strip()):
        raise InvalidInputError("A feed name is letters, digits, spaces and . _ -, up to 100")
    if not isinstance(url, str) or len(url) > 2000:
        raise InvalidInputError("A feed address is an https URL of at most 2000 characters")
    parts = urlsplit(url.strip())
    if parts.scheme != "https" or not parts.hostname:
        raise InvalidInputError("A feed address is an https URL")
    if not isinstance(topics, str) or len(topics) > 200:
        raise InvalidInputError("topics is a text of at most 200 characters")
    return {"name": name.strip(), "url": url.strip(), "topics": " ".join(topics.split())}


async def list_sources(session: AsyncSession) -> list[FeedSource]:
    return list((await session.execute(select(FeedSource).order_by(FeedSource.name))).scalars())


async def create_source(session: AsyncSession, *, name, url, topics="", actor: str) -> FeedSource:
    fields = _clean(name, url, topics)
    taken = select(FeedSource.id).where(FeedSource.name == fields["name"])
    if (await session.execute(taken)).first():
        raise ConflictError(f"A feed named {fields['name']!r} already exists")
    if len(await list_sources(session)) >= MAX_SOURCES:
        raise ConflictError(f"At most {MAX_SOURCES} feeds are listed")
    source = FeedSource(**fields)
    session.add(source)
    await session.flush()
    await record_admin_event(
        session, actor=actor, action="feed.create", target_type="feed", target_id=source.id,
        details={"name": source.name, "host": urlsplit(source.url).hostname},
    )  # fmt: skip
    return source


async def delete_source(session: AsyncSession, source_id: int, *, actor: str) -> None:
    source = await session.get(FeedSource, source_id)
    if source is None:
        raise NotFoundError(f"No feed {source_id}")
    await session.delete(source)
    await record_admin_event(
        session, actor=actor, action="feed.delete", target_type="feed", target_id=source_id,
        details={"name": source.name},
    )  # fmt: skip
    await session.flush()
