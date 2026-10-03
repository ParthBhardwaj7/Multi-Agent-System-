"""add messages.seq for stable ordering

Revision ID: e2d9d763748e
Revises: e47f1b464bc6
Create Date: 2026-10-03 20:48:26.451428

Hand-adjusted after autogeneration:
- demo tables (documents/faqs) are application-managed, NOT schema-managed — never dropped
- seq needs a server_default to backfill existing rows safely

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e2d9d763748e"
down_revision: str | None = "e47f1b464bc6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("seq", sa.Integer(), nullable=False, server_default="0"))
    op.drop_index(op.f("ix_messages_conversation_created"), table_name="messages")
    op.create_index(
        "ix_messages_conversation_seq", "messages", ["conversation_id", "seq"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_messages_conversation_seq", table_name="messages")
    op.create_index(
        op.f("ix_messages_conversation_created"),
        "messages",
        ["conversation_id", "created_at"],
        unique=False,
    )
    op.drop_column("messages", "seq")
