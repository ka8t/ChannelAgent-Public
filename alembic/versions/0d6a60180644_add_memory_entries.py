"""add memory entries

Revision ID: 0d6a60180644
Revises: b004e2f34811
Create Date: 2026-09-27 09:25:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

import app.db.types
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0d6a60180644'
down_revision: str | Sequence[str] | None = 'b004e2f34811'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: persistent memory per (user, agent), title and content
    encrypted. Starts empty: nothing changes until an agent's memory mode is not "off".
    """
    op.create_table(
        "memory_entries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("agent_id", sa.Integer(), nullable=False),
        sa.Column("title", app.db.types.EncryptedString(), nullable=False),
        sa.Column("content", app.db.types.EncryptedString(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], name="fk_memory_entries_agent"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_memory_entries_agent_id"), "memory_entries", ["agent_id"], unique=False
    )
    op.create_index(op.f("ix_memory_entries_user_id"), "memory_entries", ["user_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_memory_entries_user_id"), table_name="memory_entries")
    op.drop_index(op.f("ix_memory_entries_agent_id"), table_name="memory_entries")
    op.drop_table("memory_entries")
