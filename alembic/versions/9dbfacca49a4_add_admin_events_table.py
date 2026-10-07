"""add admin_events table

Revision ID: 9dbfacca49a4
Revises: 85b89421b227
Create Date: 2026-09-20 19:52:48.484869

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import app.db.types


# revision identifiers, used by Alembic.
revision: str = '9dbfacca49a4'
down_revision: Union[str, Sequence[str], None] = '85b89421b227'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('admin_events',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('actor', sa.String(length=32), nullable=False),
    sa.Column('action', sa.String(length=64), nullable=False),
    sa.Column('target_type', sa.String(length=32), nullable=False),
    sa.Column('target_id', sa.Integer(), nullable=True),
    sa.Column('details', app.db.types.EncryptedString(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_admin_events_action'), 'admin_events', ['action'], unique=False)
    op.create_index(op.f('ix_admin_events_actor'), 'admin_events', ['actor'], unique=False)
    op.create_index(op.f('ix_admin_events_created_at'), 'admin_events', ['created_at'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_admin_events_created_at'), table_name='admin_events')
    op.drop_index(op.f('ix_admin_events_actor'), table_name='admin_events')
    op.drop_index(op.f('ix_admin_events_action'), table_name='admin_events')
    op.drop_table('admin_events')
