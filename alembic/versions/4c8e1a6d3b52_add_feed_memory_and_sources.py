"""add the feed items a task delivered and the feeds listed for the builder

Revision ID: 4c8e1a6d3b52
Revises: 9e3b5d7f2a41
Create Date: 2026-09-28 22:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4c8e1a6d3b52"
down_revision: str | Sequence[str] | None = "9e3b5d7f2a41"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "feed_sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("url", sa.String(length=2000), nullable=False),
        sa.Column("topics", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "task_feed_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column("item_id", sa.String(length=16), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["task_id"], ["scheduled_tasks.id"], name="fk_task_feed_items_task"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("task_id", "item_id", name="uq_task_feed_item"),
    )
    with op.batch_alter_table("task_feed_items", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_task_feed_items_task_id"), ["task_id"], unique=False
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("task_feed_items", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_task_feed_items_task_id"))
    op.drop_table("task_feed_items")
    op.drop_table("feed_sources")
