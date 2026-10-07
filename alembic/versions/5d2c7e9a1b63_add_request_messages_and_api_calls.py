"""add request messages and the api call trail

Revision ID: 5d2c7e9a1b63
Revises: 4c8e1a6d3b52
Create Date: 2026-09-29 18:30:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

import app.db.types
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5d2c7e9a1b63"
down_revision: str | Sequence[str] | None = "4c8e1a6d3b52"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: every message of a sender who is not a user yet, the Admin API
    call trail, and its retention period."""
    op.create_table(
        "request_messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("text", app.db.types.EncryptedString(), nullable=False),
        sa.ForeignKeyConstraint(
            ["request_id"], ["access_requests.id"], name="fk_request_messages_request",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_request_messages_request_id"), "request_messages", ["request_id"], unique=False
    )
    op.create_index(
        op.f("ix_request_messages_created_at"), "request_messages", ["created_at"], unique=False
    )
    op.create_table(
        "api_calls",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("method", sa.String(length=8), nullable=False),
        sa.Column("path", sa.String(length=300), nullable=False),
        sa.Column("status", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_api_calls_created_at"), "api_calls", ["created_at"], unique=False)
    op.create_index(op.f("ix_api_calls_actor"), "api_calls", ["actor"], unique=False)
    with op.batch_alter_table("retention_config") as batch:
        batch.add_column(sa.Column("api_calls_days", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("retention_config") as batch:
        batch.drop_column("api_calls_days")
    op.drop_index(op.f("ix_api_calls_actor"), table_name="api_calls")
    op.drop_index(op.f("ix_api_calls_created_at"), table_name="api_calls")
    op.drop_table("api_calls")
    op.drop_index(op.f("ix_request_messages_created_at"), table_name="request_messages")
    op.drop_index(op.f("ix_request_messages_request_id"), table_name="request_messages")
    op.drop_table("request_messages")
