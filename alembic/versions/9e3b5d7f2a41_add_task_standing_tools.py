"""add the standing approval of tools to scheduled tasks

The MCP tools a task's turns may run without asking, each bound to the hash of the definition
approved when the user agreed. Every existing task gets none, so nothing changes for it.

Revision ID: 9e3b5d7f2a41
Revises: 6b2d8f4a1c93
Create Date: 2026-09-28 21:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9e3b5d7f2a41"
down_revision: str | Sequence[str] | None = "6b2d8f4a1c93"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("scheduled_tasks", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("standing_tools", sa.JSON(), server_default=sa.text("'{}'"), nullable=False)
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("scheduled_tasks", schema=None) as batch_op:
        batch_op.drop_column("standing_tools")
