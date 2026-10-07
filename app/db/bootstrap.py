"""One-time admin bootstrap from the legacy TELEGRAM_ALLOWED_USERS.

Without this, the very first deploy has an empty database and no way
to grant the first admin through the (DB-gated) Admin API — a
bootstrapping deadlock. Runs only while the users table is empty; once
seeded, the database is authoritative and .env is never consulted
again for authorization.
"""

import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Channel, ChannelIdentity, PermissionKind, User
from app.security.auth import grant_permission

logger = logging.getLogger("channelagent")


async def bootstrap_admin_from_env(session: AsyncSession) -> None:
    existing = (await session.execute(select(func.count()).select_from(User))).scalar_one()
    if existing > 0:
        return

    raw = get_settings().telegram_allowed_users
    if not raw:
        logger.warning(
            "Database is empty and TELEGRAM_ALLOWED_USERS is not set in .env — "
            "there is no admin user and no way to create one through the Admin API yet. "
            "Set TELEGRAM_ALLOWED_USERS and restart, or create the first admin manually."
        )
        return

    # Accepts a comma-separated list, though the current .env only has one —
    # TELEGRAM_ALLOWED_USERS's plural name implies more than one is a real case.
    telegram_ids = [uid.strip() for uid in raw.split(",") if uid.strip()]

    for telegram_id in telegram_ids:
        user = User(display_name=f"Bootstrap admin ({telegram_id})")
        session.add(user)
        await session.flush()
        identity = ChannelIdentity(
            user_id=user.id, channel=Channel.TELEGRAM, external_id=telegram_id
        )
        session.add(identity)
        await session.flush()
        await grant_permission(session, identity, PermissionKind.ADMIN)
        logger.info(
            "Bootstrapped admin user for Telegram id %s from TELEGRAM_ALLOWED_USERS", telegram_id
        )

    await session.commit()
    logger.info(
        "TELEGRAM_ALLOWED_USERS will not be consulted again — the database is now authoritative."
    )
