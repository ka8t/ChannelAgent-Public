"""add agents table and FK on action_logs

Revision ID: 417b3835e5c8
Revises: 620c8b04570e
Create Date: 2026-09-18 13:54:57.545341

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '417b3835e5c8'
down_revision: Union[str, Sequence[str], None] = '620c8b04570e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Hand-edited: autogenerate emits plain op.alter_column/create_foreign_key
    # for the action_logs changes below, which SQLite's dialect rejects
    # outright ("No support for ALTER of constraints in SQLite dialect")
    # — confirmed by actually running this migration. batch_alter_table
    # does the copy-and-move SQLite needs instead.
    op.create_table(
        "agents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "name", name="uq_agent_user_name"),
    )
    with op.batch_alter_table("action_logs") as batch_op:
        batch_op.alter_column("agent_id", existing_type=sa.INTEGER(), nullable=False)
        batch_op.create_foreign_key("fk_action_logs_agent_id", "agents", ["agent_id"], ["id"])


def downgrade() -> None:
    with op.batch_alter_table("action_logs") as batch_op:
        batch_op.drop_constraint("fk_action_logs_agent_id", type_="foreignkey")
        batch_op.alter_column("agent_id", existing_type=sa.INTEGER(), nullable=True)
    op.drop_table("agents")
