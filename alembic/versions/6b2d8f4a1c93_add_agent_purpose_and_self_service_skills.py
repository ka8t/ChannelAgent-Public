"""add an agent's purpose and the self-service flag of skills

`agents.purpose` is one sentence written by the user through the agent builder, encrypted like
the other free text. `skills.self_service` marks the skills a user may attach to an agent they
create themselves; every existing skill stays administrator-only.

Revision ID: 6b2d8f4a1c93
Revises: 3a7f2c9e5b14
Create Date: 2026-09-28 16:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

import app.db.types
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6b2d8f4a1c93"
down_revision: str | Sequence[str] | None = "3a7f2c9e5b14"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("agents", schema=None) as batch_op:
        batch_op.add_column(sa.Column("purpose", app.db.types.EncryptedString(), nullable=True))
    with op.batch_alter_table("skills", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("self_service", sa.Boolean(), server_default=sa.text("0"), nullable=False)
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("skills", schema=None) as batch_op:
        batch_op.drop_column("self_service")
    with op.batch_alter_table("agents", schema=None) as batch_op:
        batch_op.drop_column("purpose")
