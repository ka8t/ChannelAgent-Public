"""add backup schedule

Revision ID: 4b2d9ca7be51
Revises: 0d6a60180644
Create Date: 2026-09-27 11:50:16.091718

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '4b2d9ca7be51'
down_revision: str | Sequence[str] | None = '0d6a60180644'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: the scheduled backup's settings and last run, one row
    created on first use (daily, keep 7)."""
    op.create_table(
        "backup_schedule",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("1"), nullable=False),
        sa.Column("interval_minutes", sa.Integer(), server_default=sa.text("1440"), nullable=False),
        sa.Column("keep", sa.Integer(), server_default=sa.text("7"), nullable=False),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.String(length=500), nullable=True),
        sa.Column("last_files", sa.JSON(), server_default=sa.text("'[]'"), nullable=False),
        sa.Column("failing", sa.Boolean(), server_default=sa.text("0"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("backup_schedule")
