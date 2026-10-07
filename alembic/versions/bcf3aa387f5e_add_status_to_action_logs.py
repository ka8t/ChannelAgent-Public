"""add status to action_logs

Revision ID: bcf3aa387f5e
Revises: 417b3835e5c8
Create Date: 2026-09-20 13:37:03.955692

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'bcf3aa387f5e'
down_revision: str | Sequence[str] | None = '417b3835e5c8'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Existing rows are ok: SQLite fills the server default when adding a
    # NOT NULL column, so no data step is needed.
    op.add_column(
        "action_logs",
        sa.Column(
            "status",
            sa.Enum("ok", "failed", "denied", name="actionstatus", native_enum=False),
            server_default="ok",
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    # batch mode: SQLite's copy-and-move strategy, works on every version.
    with op.batch_alter_table("action_logs") as batch_op:
        batch_op.drop_column("status")
