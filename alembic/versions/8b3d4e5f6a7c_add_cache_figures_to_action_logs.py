"""add the engine's cache figures to action_logs

Per reply, the prompt tokens the engine took from its cache (llama-server `timings.cache_n`)
and its time reading the rest (`timings.prompt_ms`), for the reuse rate of GET /telemetry.

Revision ID: 8b3d4e5f6a7c
Revises: 7f1b2c3d4e5a
Create Date: 2026-10-04

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "8b3d4e5f6a7c"
down_revision: str | Sequence[str] | None = "7f1b2c3d4e5a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("action_logs") as batch:
        batch.add_column(sa.Column("cached_tokens", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("prefill_ms", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("action_logs") as batch:
        batch.drop_column("prefill_ms")
        batch.drop_column("cached_tokens")
