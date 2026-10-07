"""add the limited action status

A message refused by the per-user rate limit or turn limit is logged with status "limited",
apart from "denied" (no permission). The status column grows from 6 to 7 characters; the
downgrade turns "limited" rows into "denied" first, so every row stays valid.

Revision ID: d3d0e7313cdc
Revises: 6c872782bce9
Create Date: 2026-09-27

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d3d0e7313cdc"
down_revision: str | Sequence[str] | None = "6c872782bce9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD = sa.Enum("ok", "failed", "denied", name="actionstatus", native_enum=False)
NEW = sa.Enum("ok", "failed", "denied", "limited", name="actionstatus", native_enum=False)


def upgrade() -> None:
    with op.batch_alter_table("action_logs") as batch:
        batch.alter_column("status", existing_type=sa.String(length=6), type_=NEW)


def downgrade() -> None:
    op.execute("UPDATE action_logs SET status = 'denied' WHERE status = 'limited'")
    with op.batch_alter_table("action_logs") as batch:
        batch.alter_column("status", existing_type=NEW, type_=OLD)
