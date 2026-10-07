"""add agent templates and their versions

The agent builder starts from a template an administrator writes as data: what it is for, the
builder's guidance, instructions added to the agent, a fixed memory mode, the tools it attaches,
whether it needs a schedule, and the only questions it may ask. Every change is a version.

Revision ID: 7f1b2c3d4e5a
Revises: 6e4a8f0b2c75
Create Date: 2026-10-04

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "7f1b2c3d4e5a"
down_revision: str | Sequence[str] | None = "6e4a8f0b2c75"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_templates",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=40), nullable=False),
        sa.Column("description", sa.String(length=200), nullable=False),
        sa.Column("guidance", sa.String(length=4000), nullable=False),
        sa.Column("agent_instructions", sa.String(length=2000), nullable=False),
        sa.Column("memory_mode", sa.String(length=10), nullable=True),
        sa.Column("tools", sa.JSON(), server_default=sa.text("'[]'"), nullable=False),
        sa.Column("needs_schedule", sa.Boolean(), server_default=sa.text("0"), nullable=False),
        sa.Column("questions", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("1"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "agent_template_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("template_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["template_id"], ["agent_templates.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_agent_template_versions_template_id", "agent_template_versions", ["template_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_agent_template_versions_template_id", table_name="agent_template_versions")
    op.drop_table("agent_template_versions")
    op.drop_table("agent_templates")
