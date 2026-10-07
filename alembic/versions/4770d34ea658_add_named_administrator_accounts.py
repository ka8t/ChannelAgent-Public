"""add named administrator accounts and their api tokens

Named administrators replace the one shared API key: an account has a scope, a
scrypt password hash and an encrypted TOTP secret; a token is stored as its SHA-256 only.

Revision ID: 4770d34ea658
Revises: 8b3d4e5f6a7c
Create Date: 2026-10-06

"""

from collections.abc import Sequence

import sqlalchemy as sa

import app.db.types
from alembic import op

revision: str = "4770d34ea658"
down_revision: str | Sequence[str] | None = "8b3d4e5f6a7c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "admin_accounts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=24), nullable=False),
        sa.Column("scope", sa.String(length=8), nullable=False),
        sa.Column("password_hash", sa.String(length=200), nullable=False),
        sa.Column("totp_secret", app.db.types.EncryptedString(), nullable=True),
        sa.Column("totp_last_step", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("disabled_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "api_tokens",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("account_id", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=32), nullable=False),
        sa.Column("scope", sa.String(length=8), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["account_id"], ["admin_accounts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("ix_api_tokens_account_id", "api_tokens", ["account_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_api_tokens_account_id", table_name="api_tokens")
    op.drop_table("api_tokens")
    op.drop_table("admin_accounts")
