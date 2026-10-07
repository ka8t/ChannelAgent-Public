"""add mcp grants, policies, pins and call audit

Revision ID: b004e2f34811
Revises: 9be5dd27d58e
Create Date: 2026-09-27 09:47:06.735756

"""
from collections.abc import Sequence

import sqlalchemy as sa

import app.db.types
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b004e2f34811'
down_revision: str | Sequence[str] | None = '9be5dd27d58e'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: grants (default deny), per-tool policies, pinned tool
    definitions, the shared-credentials flag, the confirmation timeout, and who asked
    and what was decided on each call. An existing server starts with no approved
    definition, so its tools are offered again only after an administrator approves
    them.
    """
    op.create_table(
        "mcp_grants",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("agent_id", sa.Integer(), nullable=True),
        sa.Column("server_name", sa.String(length=100), nullable=False),
        sa.Column("tool_name", sa.String(length=200), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], name="fk_mcp_grants_agent"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_mcp_grants_user_id"), "mcp_grants", ["user_id"], unique=False)
    with op.batch_alter_table("mcp_calls") as batch:
        batch.add_column(sa.Column("user_id", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("decision", sa.String(length=16), nullable=True))
        batch.add_column(sa.Column("arguments", app.db.types.EncryptedString(), nullable=True))
        batch.create_index(batch.f("ix_mcp_calls_user_id"), ["user_id"], unique=False)
    with op.batch_alter_table("mcp_servers") as batch:
        batch.add_column(
            sa.Column("tool_policies", sa.JSON(), server_default=sa.text("'{}'"), nullable=False)
        )
        batch.add_column(
            sa.Column(
                "approved_definitions", sa.JSON(), server_default=sa.text("'{}'"), nullable=False
            )
        )
        batch.add_column(
            sa.Column(
                "shared_credentials", sa.Boolean(), server_default=sa.text("0"), nullable=False
            )
        )
        batch.add_column(
            sa.Column(
                "confirm_timeout_seconds",
                sa.Integer(),
                server_default=sa.text("120"),
                nullable=False,
            )
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("mcp_servers") as batch:
        batch.drop_column("confirm_timeout_seconds")
        batch.drop_column("shared_credentials")
        batch.drop_column("approved_definitions")
        batch.drop_column("tool_policies")
    with op.batch_alter_table("mcp_calls") as batch:
        batch.drop_index(batch.f("ix_mcp_calls_user_id"))
        batch.drop_column("arguments")
        batch.drop_column("decision")
        batch.drop_column("user_id")
    op.drop_index(op.f("ix_mcp_grants_user_id"), table_name="mcp_grants")
    op.drop_table("mcp_grants")
