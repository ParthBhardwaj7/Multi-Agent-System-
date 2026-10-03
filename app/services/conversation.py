"""Conversation management service (list/get/messages/delete) with ownership
enforced at every step."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.db.repository import ConversationRepository, MessageRepository
from app.schemas.conversations import (
    ConversationMessages,
    ConversationOut,
    DeleteResult,
    MessageOut,
)
from app.security.auth import Principal


class ConversationService:
    def __init__(self, session: AsyncSession) -> None:
        self.conversations = ConversationRepository(session)
        self.messages = MessageRepository(session)
        self.session = session

    async def list_for_user(
        self, principal: Principal, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ConversationOut], int]:
        items, total = await self.conversations.list_for_user(
            principal.user_id, limit=limit, offset=offset
        )
        return [ConversationOut.model_validate(item) for item in items], total

    async def get(self, principal: Principal, conversation_id: str) -> ConversationOut:
        conversation = await self.conversations.get_owned(conversation_id, principal.user_id)
        if conversation is None:
            raise NotFoundError("conversation not found")
        return ConversationOut.model_validate(conversation)

    async def get_messages(
        self, principal: Principal, conversation_id: str
    ) -> ConversationMessages:
        conversation = await self.conversations.get_owned(conversation_id, principal.user_id)
        if conversation is None:
            raise NotFoundError("conversation not found")
        rows = await self.messages.list_for_conversation(conversation.id)
        # Built explicitly: `row.metadata` on the ORM model is SQLAlchemy's
        # MetaData object, not our column (stored as `message_metadata`).
        return ConversationMessages(
            conversation_id=conversation.id,
            items=[
                MessageOut(
                    id=row.id,
                    role=row.role,
                    content=row.content,
                    created_at=row.created_at,
                    metadata=row.message_metadata or {},
                )
                for row in rows
            ],
        )

    async def delete(self, principal: Principal, conversation_id: str) -> DeleteResult:
        deleted = await self.conversations.delete(conversation_id, principal.user_id)
        await self.session.commit()
        return DeleteResult(deleted=deleted, conversation_id=conversation_id)
