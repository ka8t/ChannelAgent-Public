"""Guardrail & Auth Node: the single place that turns a
normalized event's (channel, user_id) into an authorization decision.

This is the direct replacement for the legacy static
TELEGRAM_ALLOWED_USERS / EMAIL_ALLOWED_USERS .env check — every
channel adapter must call authorize() before routing a message to
app/graph.py. An unrecognized identity is denied, not allowed: this
module fails closed by construction (no code path returns "allowed"
without an explicit CHAT or ADMIN permission row).
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Channel, ChannelIdentity, Permission, PermissionKind, User
from app.security.hashing import channel_identifier_key


@dataclass(frozen=True)
class AuthDecision:
    allowed: bool
    user: User | None = None
    is_admin: bool = False
    identity: ChannelIdentity | None = None


async def authorize(session: AsyncSession, channel: Channel, user_id: str) -> AuthDecision:
    key = channel_identifier_key(channel, user_id)

    stmt = (
        select(ChannelIdentity)
        .where(ChannelIdentity.channel == channel, ChannelIdentity.external_id == key)
        .options(selectinload(ChannelIdentity.user), selectinload(ChannelIdentity.permissions))
    )
    identity = (await session.execute(stmt)).scalar_one_or_none()

    if identity is None or not identity.user.is_active:
        return AuthDecision(allowed=False)

    kinds = {p.kind for p in identity.permissions}
    allowed = PermissionKind.CHAT in kinds or PermissionKind.ADMIN in kinds
    is_admin = PermissionKind.ADMIN in kinds

    return AuthDecision(allowed=allowed, user=identity.user, is_admin=is_admin, identity=identity)


async def grant_permission(
    session: AsyncSession, channel_identity: ChannelIdentity, kind: PermissionKind
) -> Permission:
    """Idempotent grant — re-granting an already-held permission is a no-op.

    Queries for the existing row explicitly rather than reading
    channel_identity.permissions: that relationship may not be loaded,
    and triggering a lazy load here would hit AsyncSession's "implicit
    IO" guard (SQLAlchemy's async ORM does not allow lazy-loading
    outside of an explicit await).
    """
    stmt = select(Permission).where(
        Permission.channel_identity_id == channel_identity.id, Permission.kind == kind
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()
    if existing is not None:
        return existing
    permission = Permission(channel_identity_id=channel_identity.id, kind=kind)
    session.add(permission)
    await session.flush()
    return permission


async def revoke_permission(
    session: AsyncSession, channel_identity: ChannelIdentity, kind: PermissionKind
) -> bool:
    """Returns False if the permission wasn't held (nothing to revoke),
    True if a row was actually deleted — lets the Admin API tell
    a caller "already didn't have it" apart from a real change."""
    stmt = select(Permission).where(
        Permission.channel_identity_id == channel_identity.id, Permission.kind == kind
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()
    if existing is None:
        return False
    await session.delete(existing)
    await session.flush()
    return True
