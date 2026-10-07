"""Proactive messages to administrators.

A channel adapter registers a sender for its channel when it starts and
removes it when it stops. Code that needs to tell the administrators
something, such as a new access request, calls notify_admins() without
knowing which channel each administrator uses. A channel with no sender
(email is deliberately silent, an adapter that is not running) is skipped.
"""

import logging
from collections.abc import Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    AccessRequest,
    Channel,
    ChannelIdentity,
    Permission,
    PermissionKind,
    User,
)

logger = logging.getLogger("channelagent")

# (external identity id, text) -> delivered
Sender = Callable[[str, str], Awaitable[None]]

_senders: dict[Channel, Sender] = {}

MAX_QUOTED_MESSAGE = 200


def register_sender(channel: Channel, sender: Sender) -> None:
    _senders[channel] = sender


def unregister_sender(channel: Channel) -> None:
    _senders.pop(channel, None)


def get_sender(channel: Channel) -> Sender | None:
    return _senders.get(channel)


def describe_request(request: AccessRequest) -> str:
    quoted = request.first_message_text.strip().replace("\n", " ")
    if len(quoted) > MAX_QUOTED_MESSAGE:
        quoted = quoted[:MAX_QUOTED_MESSAGE] + "..."
    return (
        f"New access request #{request.id} from {request.channel.value}/{request.external_id}: "
        f'"{quoted}". Approve or deny it from the admin console or the Admin API.'
    )


async def notify_admins(session: AsyncSession, text: str) -> int:
    """Send `text` to every active administrator reachable through a
    registered sender. Returns how many were reached. A failing send is
    logged and skipped: telling the admins must never break the caller.
    """
    stmt = (
        select(ChannelIdentity)
        .join(Permission, Permission.channel_identity_id == ChannelIdentity.id)
        .join(User, User.id == ChannelIdentity.user_id)
        .where(Permission.kind == PermissionKind.ADMIN, User.is_active.is_(True))
        .order_by(ChannelIdentity.id)
    )
    reached = 0
    for identity in (await session.execute(stmt)).scalars().all():
        sender = _senders.get(identity.channel)
        if sender is None:
            continue
        try:
            await sender(identity.external_id, text)
            reached += 1
        except Exception:
            logger.exception(
                "Notifying admin %s/%s failed", identity.channel.value, identity.external_id
            )
    return reached
