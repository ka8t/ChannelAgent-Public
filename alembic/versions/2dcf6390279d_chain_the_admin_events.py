"""chain the admin events

Two columns, `prev_hash` and `hash` (app/admin/audit_chain.py), and the events already stored
chained in id order, so the chain covers the whole history from this migration on. The details
are hashed in clear: they are decrypted here with ENCRYPTION_KEY, as the application reads them
(a value that cannot be decrypted is hashed as the marker the application shows for it).

Revision ID: 2dcf6390279d
Revises: 4770d34ea658
Create Date: 2026-10-06

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "2dcf6390279d"
down_revision: str | Sequence[str] | None = "4770d34ea658"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    from app.admin import audit_chain
    from app.db.types import UNDECRYPTABLE_MARKER
    from app.security.encryption import decrypt_value

    with op.batch_alter_table("admin_events") as batch:
        batch.add_column(sa.Column("prev_hash", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("hash", sa.String(length=64), nullable=True))
    events = sa.table(
        "admin_events",
        sa.column("id", sa.Integer()),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("actor", sa.String()),
        sa.column("action", sa.String()),
        sa.column("target_type", sa.String()),
        sa.column("target_id", sa.Integer()),
        sa.column("details", sa.Text()),
        sa.column("prev_hash", sa.String()),
        sa.column("hash", sa.String()),
    )
    bind = op.get_bind()
    previous = audit_chain.GENESIS
    rows = bind.execute(sa.select(events).order_by(events.c.id)).mappings().all()
    for row in rows:
        details = row["details"]
        if details is not None:
            try:
                details = decrypt_value(details)
            except ValueError:
                details = UNDECRYPTABLE_MARKER
        text = audit_chain.content(
            row["id"], row["created_at"], row["actor"], row["action"], row["target_type"],
            row["target_id"], details,
        )  # fmt: skip
        current = audit_chain.link(previous, text)
        bind.execute(
            events.update()
            .where(events.c.id == row["id"])
            .values(prev_hash=previous, hash=current)
        )
        previous = current


def downgrade() -> None:
    with op.batch_alter_table("admin_events") as batch:
        batch.drop_column("hash")
        batch.drop_column("prev_hash")
