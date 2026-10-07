"""add the tool budget to the routing config

At most max_tools MCP tools per turn (0 = no cap, 5 by default), the tool rules that choose
them first, and the model of tool turns.

Revision ID: 5e1a7c3b9d20
Revises: 2bf6f9f84070
Create Date: 2026-09-28 09:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5e1a7c3b9d20"
down_revision: str | Sequence[str] | None = "2bf6f9f84070"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("routing_config", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("max_tools", sa.Integer(), server_default=sa.text("5"), nullable=False)
        )
        batch_op.add_column(
            sa.Column("tool_rules", sa.JSON(), server_default=sa.text("'[]'"), nullable=False)
        )
        batch_op.add_column(sa.Column("tool_model", sa.String(length=200), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("routing_config", schema=None) as batch_op:
        batch_op.drop_column("tool_model")
        batch_op.drop_column("tool_rules")
        batch_op.drop_column("max_tools")
