"""add turn telemetry to action_logs

The model, latency and token counts of a turn, on its reply row. Nullable: older rows and
rows that are not a turn's reply have none.

Revision ID: 6c872782bce9
Revises: 4b2d9ca7be51
Create Date: 2026-09-27 19:45:04.979803

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "6c872782bce9"
down_revision: str | Sequence[str] | None = "4b2d9ca7be51"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("action_logs") as batch:
        batch.add_column(sa.Column("model", sa.String(length=200), nullable=True))
        batch.add_column(sa.Column("latency_ms", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("prompt_tokens", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("completion_tokens", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("action_logs") as batch:
        batch.drop_column("completion_tokens")
        batch.drop_column("prompt_tokens")
        batch.drop_column("latency_ms")
        batch.drop_column("model")
