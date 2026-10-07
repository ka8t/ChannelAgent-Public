"""add configuration columns to agents

Revision ID: b11c1796e218
Revises: 9dbfacca49a4
Create Date: 2026-09-21 21:05:19.103423

"""
from collections.abc import Sequence

import sqlalchemy as sa

import app.db.types
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b11c1796e218'
down_revision: str | Sequence[str] | None = '9dbfacca49a4'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: the per-agent configuration.

    Existing agents keep working: the prompt and the model are null (the engine's
    defaults), the memory is off and no tool is allowed. SQLite fills the server
    default of a NOT NULL column it adds, so no data step is needed.
    """
    with op.batch_alter_table("agents") as batch_op:
        batch_op.add_column(
            sa.Column("system_prompt", app.db.types.EncryptedString(), nullable=True)
        )
        batch_op.add_column(sa.Column("model", sa.String(length=200), nullable=True))
        batch_op.add_column(
            sa.Column("memory_mode", sa.String(length=16), server_default="off", nullable=False)
        )
        batch_op.add_column(
            sa.Column("tools", sa.JSON(), server_default=sa.text("'[]'"), nullable=False)
        )


def downgrade() -> None:
    """Downgrade schema."""
    # batch mode: SQLite's copy-and-move strategy, works on every version.
    with op.batch_alter_table("agents") as batch_op:
        batch_op.drop_column("tools")
        batch_op.drop_column("memory_mode")
        batch_op.drop_column("model")
        batch_op.drop_column("system_prompt")
