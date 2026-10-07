"""add active_agent_id to channel_identities

Revision ID: 85b89421b227
Revises: 5527b11034f7
Create Date: 2026-09-20 18:37:51.155658

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '85b89421b227'
down_revision: str | Sequence[str] | None = '5527b11034f7'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # SQLite cannot ADD CONSTRAINT, so the column and its named foreign key
    # go through batch mode (copy-and-move). Existing identities get NULL,
    # which means "the user's default agent".
    with op.batch_alter_table("channel_identities") as batch_op:
        batch_op.add_column(sa.Column("active_agent_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_channel_identities_active_agent", "agents", ["active_agent_id"], ["id"]
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("channel_identities") as batch_op:
        batch_op.drop_constraint("fk_channel_identities_active_agent", type_="foreignkey")
        batch_op.drop_column("active_agent_id")
