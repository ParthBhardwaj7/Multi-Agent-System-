"""Persistent conversation memory backed by PostgreSQL repositories, plus the
short-term window policy (bounded recent history, never unbounded)."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.db.repository import ConversationRepository, MessageRepository
from app.graph.state import MessageState


class ConversationMemory:
    """Implements both ShortTermMemory and PersistentMemory."""

    def __init__(self, session: AsyncSession) -> None:
        self.conversations = ConversationRepository(session)
        self.messages = MessageRepository(session)

    async def load_history(self, conversation_id: str, user_id: str) -> list[MessageState]:
        rows = await self.messages.list_for_conversation(conversation_id)
        return [
            {
                "role": row.role,
                "content": row.content,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "metadata": row.message_metadata or {},
            }
            for row in rows
        ]

    async def window(self, conversation_id: str, user_id: str) -> list[MessageState]:
        history = await self.load_history(conversation_id, user_id)
        window = max(2, get_settings().memory_short_term_window)
        return history[-window:]

    async def save_turn(
        self,
        conversation_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
        metadata: dict[str, Any],
    ) -> None:
        """Persist one turn. The caller owns the transaction/commit and the
        optimistic-conversion conflict handling."""
        user_msg = await self.messages.make_message(
            conversation_id=conversation_id,
            role="user",
            content=user_message,
            metadata={"request_id": metadata.get("request_id", "")},
        )
        assistant_msg = await self.messages.make_message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_message,
            metadata={
                "intent": metadata.get("intent"),
                "confidence": metadata.get("confidence"),
                "selected_agent": metadata.get("selected_agent"),
                "routed_to_specialist": metadata.get("routed_to_specialist", False),
            },
        )
        await self.conversations.append_turn(
            conversation_id,
            expected_version=int(metadata["expected_version"]),
            user_message=user_msg,
            assistant_message=assistant_msg,
        )
