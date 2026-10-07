"""add the user's tools suspension (MCP guard)

Revision ID: 6e4a8f0b2c75
Revises: 5d2c7e9a1b63
Create Date: 2026-09-29 19:30:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6e4a8f0b2c75"
down_revision: str | Sequence[str] | None = "5d2c7e9a1b63"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: a user's tools can be suspended by the MCP guard."""
    with op.batch_alter_table("users") as batch:
        batch.add_column(
            sa.Column("tools_suspended", sa.Boolean(), server_default=sa.text("0"), nullable=False)
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("users") as batch:
        batch.drop_column("tools_suspended")
