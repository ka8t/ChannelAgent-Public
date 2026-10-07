"""add resolved_by to access_requests

Revision ID: 5527b11034f7
Revises: bcf3aa387f5e
Create Date: 2026-09-20 13:53:45.134234

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '5527b11034f7'
down_revision: str | Sequence[str] | None = 'bcf3aa387f5e'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable: rows resolved before this column existed simply have no author.
    op.add_column("access_requests", sa.Column("resolved_by", sa.String(length=32), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # batch mode: SQLite's copy-and-move strategy, works on every version.
    with op.batch_alter_table("access_requests") as batch_op:
        batch_op.drop_column("resolved_by")
