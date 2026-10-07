"""One-way hashing for identifiers that must be deterministic but must
not expose the underlying plaintext — currently only email addresses.

Not encryption: this is intentionally irreversible. It's used where a
raw value must never be reconstructed from storage (a DB lookup key or
a LangGraph thread_id), unlike app.security.encryption, which is used
where the plaintext does need to be recovered later (e.g. to send an
SMTP reply).
"""

import hashlib

from app.db.models import Channel


def hash_email(address: str) -> str:
    """Deterministic, non-reversible key for an email address.

    Used as ChannelIdentity.external_id for the email channel (see
    app/db/models.py) and as the email_{hash} LangGraph thread_id
    — never the raw address itself.
    """
    normalized = address.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def channel_identifier_key(channel: Channel, raw_id: str) -> str:
    """The deterministic, storage/lookup-safe form of a channel's raw id.

    Shared by app/security/auth.py (DB lookup key) and app/graph.py
    (thread_id) so the two can never drift into using different keys
    for the same identity.
    """
    if channel is Channel.EMAIL:
        return hash_email(raw_id)
    return raw_id
