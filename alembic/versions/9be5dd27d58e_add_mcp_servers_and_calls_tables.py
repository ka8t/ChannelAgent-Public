"""add mcp servers and calls tables

Revision ID: 9be5dd27d58e
Revises: 81d74ae85b06
Create Date: 2026-09-22 19:21:52.444300

"""
from collections.abc import Sequence

import sqlalchemy as sa

import app.db.types
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9be5dd27d58e'
down_revision: str | Sequence[str] | None = '81d74ae85b06'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: the MCP server registry and its own minimal call
    audit. Both start empty — nothing changes for an existing deployment until an
    administrator declares a server.
    """
    op.create_table(
        "mcp_calls",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("agent_id", sa.Integer(), nullable=True),
        sa.Column("server_name", sa.String(length=100), nullable=False),
        sa.Column("tool_name", sa.String(length=200), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("result_bytes", sa.Integer(), nullable=False),
        sa.Column("detail", app.db.types.EncryptedString(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_mcp_calls_created_at"), "mcp_calls", ["created_at"], unique=False)
    op.create_index(op.f("ix_mcp_calls_server_name"), "mcp_calls", ["server_name"], unique=False)
    op.create_table(
        "mcp_servers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column(
            "protocol",
            sa.Enum("stdio", "http", name="mcptransport", native_enum=False),
            nullable=False,
        ),
        sa.Column("builtin_id", sa.String(length=100), nullable=True),
        sa.Column("url", sa.String(length=500), nullable=True),
        sa.Column("env_vars", app.db.types.EncryptedString(), nullable=True),
        sa.Column(
            "egress",
            sa.Enum("local", "lan", "internet", name="mcpegress", native_enum=False),
            nullable=False,
        ),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("timeout_seconds", sa.Integer(), nullable=False),
        sa.Column("concurrency_limit", sa.Integer(), nullable=False),
        sa.Column("result_max_bytes", sa.Integer(), nullable=False),
        sa.Column("disabled_tools", sa.JSON(), server_default=sa.text("'[]'"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("mcp_servers")
    op.drop_index(op.f("ix_mcp_calls_server_name"), table_name="mcp_calls")
    op.drop_index(op.f("ix_mcp_calls_created_at"), table_name="mcp_calls")
    op.drop_table("mcp_calls")
