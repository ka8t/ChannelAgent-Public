"""The audit timeline: who asked what and who did what, from five sources merged
(`app/admin/audit_trail.py`). Admin scope: it can include conversation text."""

from datetime import datetime

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin import audit_chain, audit_trail
from app.admin.service import LOG_SEARCH_MAX_LIMIT
from app.api.actor import current_actor
from app.api.deps import get_db_session
from app.api.errors import error_responses
from app.api.scopes import Scope, require
from app.db.models import Channel

router = APIRouter()


class AuditRowOut(BaseModel):
    at: datetime
    kind: str
    who: str
    what: str
    status: str | None
    detail: str | None
    text: str | None


@router.get(
    "/audit/timeline",
    dependencies=[require(Scope.ADMIN)],
    response_model=list[AuditRowOut],
    tags=["audit"],
    responses=error_responses(),
)
async def audit_timeline(
    response: Response,
    user_id: int | None = Query(default=None, description="Messages and tool calls of a user"),
    actor: str | None = Query(default=None, description="Admin events and API calls of an actor"),
    channel: Channel | None = None,
    kinds: list[str] | None = Query(
        default=None, description=f"Among: {', '.join(audit_trail.KINDS)}"
    ),
    since: datetime | None = Query(default=None, description="Inclusive. Naive = UTC."),
    until: datetime | None = Query(default=None, description="Exclusive. Naive = UTC."),
    with_text: bool = Query(default=False, description="Include message text (audited)"),
    limit: int = Query(default=100, ge=1, le=LOG_SEARCH_MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> list[dict]:
    """Who asked what and who did what, newest first: messages, messages of unknown senders,
    admin events, tool calls and Admin API calls. The total is in `X-Total-Count`."""
    rows, total = await audit_trail.timeline(
        session, user_id=user_id, actor=actor, channel=channel, kinds=kinds, since=since,
        until=until, limit=limit, offset=offset, with_text=with_text, reader=current_actor(),
    )  # fmt: skip
    await session.commit()
    response.headers["X-Total-Count"] = str(total)
    return rows


class AuditProblemOut(BaseModel):
    id: int
    problem: str


class AuditVerifyOut(BaseModel):
    events: int
    first_id: int | None
    last_id: int | None
    last_hash: str | None
    problems: list[AuditProblemOut]
    intact: bool


@router.get(
    "/audit/verify",
    dependencies=[require(Scope.ADMIN)],
    response_model=AuditVerifyOut,
    tags=["audit"],
    responses=error_responses(),
)
async def verify_audit(session: AsyncSession = Depends(get_db_session)) -> dict:
    """Check the chain of the admin events: each changed or missing event is reported."""
    return await audit_chain.verify(session)
