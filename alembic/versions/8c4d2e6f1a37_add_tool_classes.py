"""add the administrator's tool class overrides to MCP servers

Per tool, the private / untrusted / outbound classes an administrator set over the ones derived
from the tool's annotations and the server's egress label.

Revision ID: 8c4d2e6f1a37
Revises: 5e1a7c3b9d20
Create Date: 2026-09-28 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8c4d2e6f1a37"
down_revision: str | Sequence[str] | None = "5e1a7c3b9d20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("mcp_servers", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("tool_classes", sa.JSON(), server_default=sa.text("'{}'"), nullable=False)
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("mcp_servers", schema=None) as batch_op:
        batch_op.drop_column("tool_classes")
