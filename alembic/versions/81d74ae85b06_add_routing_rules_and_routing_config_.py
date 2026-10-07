"""add routing rules and routing config tables

Revision ID: 81d74ae85b06
Revises: b11c1796e218
Create Date: 2026-09-22 18:08:43.478688

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '81d74ae85b06'
down_revision: str | Sequence[str] | None = 'b11c1796e218'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: the model-routing table. Empty until an admin sets a
    default model or a rule, so app.graph.run_turn keeps its current behavior
    (no model chosen beyond the agent's own) until someone configures one.
    """
    op.create_table(
        "routing_config",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("default_model", sa.String(length=200), nullable=True),
        sa.Column("model_ctx_sizes", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "routing_rules",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("match_type", sa.String(length=32), nullable=False),
        sa.Column("match_value", sa.String(length=200), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_routing_rules_position"), "routing_rules", ["position"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_routing_rules_position"), table_name="routing_rules")
    op.drop_table("routing_rules")
    op.drop_table("routing_config")
