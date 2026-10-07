"""add scheduled tasks, their global switch and the users' timezone

Revision ID: 3a7f2c9e5b14
Revises: 8c4d2e6f1a37
Create Date: 2026-09-28 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

import app.db.types
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3a7f2c9e5b14"
down_revision: str | Sequence[str] | None = "8c4d2e6f1a37"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema. Starts empty and unpaused: nothing runs until a task exists."""
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(sa.Column("timezone", sa.String(length=64), nullable=True))
    op.create_table(
        "scheduled_tasks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("agent_id", sa.Integer(), nullable=False),
        sa.Column("channel_identity_id", sa.Integer(), nullable=False),
        sa.Column("prompt", app.db.types.EncryptedString(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("expr", sa.String(length=100), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("1"), nullable=False),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_status", sa.String(length=16), nullable=True),
        sa.Column("last_error", sa.String(length=200), nullable=True),
        sa.Column("run_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], name="fk_scheduled_tasks_agent"),
        sa.ForeignKeyConstraint(
            ["channel_identity_id"],
            ["channel_identities.id"],
            name="fk_scheduled_tasks_identity",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_scheduled_tasks_next_run_at"), "scheduled_tasks", ["next_run_at"], unique=False
    )
    op.create_index(
        op.f("ix_scheduled_tasks_user_id"), "scheduled_tasks", ["user_id"], unique=False
    )
    op.create_table(
        "task_config",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("paused", sa.Boolean(), server_default=sa.text("0"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("task_config")
    op.drop_index(op.f("ix_scheduled_tasks_user_id"), table_name="scheduled_tasks")
    op.drop_index(op.f("ix_scheduled_tasks_next_run_at"), table_name="scheduled_tasks")
    op.drop_table("scheduled_tasks")
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_column("timezone")
