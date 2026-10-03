"""Repository layer: all direct database access lives here.

Concurrency: conversations carry a `version` counter; appends update
`message_count`/`updated_at` with `WHERE version = :expected AND id = :id` and
bump the version — a concurrent writer affects 0 rows, which callers translate
into a retry/409 (docs/OBSERVABILITY.md §concurrency)."""

import hashlib
import hmac
from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AuthorizationError, ConflictError, NotFoundError
from app.db.models import (
    AgentRun,
    Conversation,
    Memory,
    Message,
    User,
    utcnow,
)


def hash_api_key(api_key: str) -> str:
    """API keys are high-entropy random secrets; SHA-256 gives a deterministic
    lookup key. PBKDF2 would be pointless overhead and prevents lookup."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_api_key(api_key), stored_hash)


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, *, role: str = "user", display_name: str = "", api_key: str) -> User:
        user = User(role=role, display_name=display_name, api_key_hash=hash_api_key(api_key))
        self.session.add(user)
        await self.session.flush()
        return user

    async def get_by_api_key(self, api_key: str) -> User | None:
        result = await self.session.execute(
            select(User).where(User.api_key_hash == hash_api_key(api_key), User.is_active.is_(True))
        )
        return result.scalar_one_or_none()

    async def get(self, user_id: str) -> User | None:
        return (
            await self.session.execute(select(User).where(User.id == user_id))
        ).scalar_one_or_none()


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, *, user_id: str, title: str = "New conversation") -> Conversation:
        conversation = Conversation(user_id=user_id, title=title[:256])
        self.session.add(conversation)
        await self.session.flush()
        return conversation

    async def get(self, conversation_id: str) -> Conversation | None:
        return (
            await self.session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
        ).scalar_one_or_none()

    async def get_owned(self, conversation_id: str, user_id: str) -> Conversation:
        """Load a conversation and enforce ownership — 404 for unknown,
        403 for someone else's conversation."""
        conversation = await self.get(conversation_id)
        if conversation is None:
            raise NotFoundError("conversation not found")
        if conversation.user_id != user_id:
            raise AuthorizationError("not your conversation")
        return conversation

    async def append_turn(
        self,
        conversation_id: str,
        *,
        expected_version: int,
        user_message: Message,
        assistant_message: Message,
    ) -> Conversation:
        """Atomically append a user+assistant message pair with optimistic
        concurrency: fails with ConflictError if version changed."""
        # Lock the row to serialize writers on the same conversation.
        conversation = (
            await self.session.execute(
                select(Conversation).where(Conversation.id == conversation_id).with_for_update()
            )
        ).scalar_one_or_none()
        if conversation is None:
            raise NotFoundError("conversation not found")
        if conversation.version != expected_version:
            raise ConflictError("conversation was modified concurrently")

        self.session.add(user_message)
        self.session.add(assistant_message)
        user_message.seq = conversation.message_count
        assistant_message.seq = conversation.message_count + 1
        conversation.version += 1
        conversation.message_count += 2
        conversation.updated_at = utcnow()
        await self.session.flush()
        return conversation

    async def list_for_user(
        self, user_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[Conversation], int]:
        items = (
            (
                await self.session.execute(
                    select(Conversation)
                    .where(Conversation.user_id == user_id)
                    .order_by(Conversation.updated_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
            )
            .scalars()
            .all()
        )
        total = (
            await self.session.execute(
                select(func.count())
                .select_from(Conversation)
                .where(Conversation.user_id == user_id)
            )
        ).scalar_one()
        return list(items), int(total)

    async def delete(self, conversation_id: str, user_id: str) -> bool:
        conversation = await self.get_owned(conversation_id, user_id)
        await self.session.execute(
            delete(Message).where(Message.conversation_id == conversation.id)
        )
        await self.session.execute(
            delete(AgentRun).where(AgentRun.conversation_id == conversation.id)
        )
        await self.session.delete(conversation)
        await self.session.flush()
        return True


class MessageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def make_message(
        self, *, conversation_id: str, role: str, content: str, metadata: dict | None = None
    ) -> Message:
        return Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            message_metadata=metadata or {},
        )

    async def list_for_conversation(
        self, conversation_id: str, *, limit: int = 200, before: datetime | None = None
    ) -> list[Message]:
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.seq.asc(), Message.created_at.asc(), Message.id.asc())
            .limit(limit)
        )
        if before is not None:
            query = query.where(Message.created_at < before)
        return list((await self.session.execute(query)).scalars().all())


class AgentRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record(
        self, *, conversation_id: str, message_id: str | None, runs: list[dict]
    ) -> None:
        for run in runs:
            self.session.add(
                AgentRun(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    agent=run.get("agent", "unknown"),
                    status=run.get("status", "success"),
                    intent=run.get("intent"),
                    confidence=run.get("confidence"),
                    latency_ms=float(run.get("latency_ms", 0.0)),
                    prompt_version=run.get("prompt_version"),
                    error_type=run.get("error_type"),
                )
            )
        await self.session.flush()


class MemoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        kind: str,
        content: str,
        embedding: list[float] | None,
        embedding_model: str | None,
    ) -> Memory:
        memory = Memory(
            user_id=user_id,
            conversation_id=conversation_id,
            kind=kind,
            content=content,
            embedding=embedding,
            embedding_model=embedding_model,
        )
        self.session.add(memory)
        await self.session.flush()
        return memory

    async def list_for_user(self, user_id: str, *, limit: int = 500) -> list[Memory]:
        result = await self.session.execute(
            select(Memory)
            .where(Memory.user_id == user_id)
            .order_by(Memory.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_for_user(self, user_id: str) -> int:
        return int(
            (
                await self.session.execute(
                    select(func.count()).select_from(Memory).where(Memory.user_id == user_id)
                )
            ).scalar_one()
        )
