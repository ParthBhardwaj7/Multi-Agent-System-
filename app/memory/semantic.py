"""Semantic long-term memory: stores user facts with embeddings and retrieves
them by cosine similarity.

Storage: `memories` table with JSONB embeddings. Similarity is computed over a
bounded per-user candidate set in Python — documented trade-off
(docs/DATABASE.md): for the current scale this avoids a pgvector dependency;
the interface is designed so swapping in pgvector later changes only this file.

Write policy (docs/ARCHITECTURE.md §memory):
- explicit "remember ..." requests (deterministic rule) — always stored
- LLM-proposed candidates (config-gated extraction) — stored when valid
- routine chatter — never stored"""

import math
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.db.repository import MemoryRepository
from app.llm.client import LLMProvider
from app.memory.interface import RetrievedMemory, extract_explicit_memories
from app.observability.logging import get_logger
from app.schemas.agents import MemoryCandidate
from app.tools.policies import contains_prompt_injection, sanitize_for_tool

logger = get_logger(__name__)


class SemanticMemoryStore:
    def __init__(self, session: AsyncSession, provider: LLMProvider) -> None:
        self.repo = MemoryRepository(session)
        self.provider = provider

    async def recall(
        self, user_id: str, query: str, top_k: int | None = None
    ) -> list[RetrievedMemory]:
        settings = get_settings()
        top_k = top_k or settings.memory_semantic_top_k
        memories = await self.repo.list_for_user(user_id, limit=500)
        if not memories:
            return []

        try:
            query_vectors = await self.provider.embed([query])
        except Exception as exc:  # noqa: BLE001 - memory recall must never break chat
            logger.warning("semantic_memory_embed_failed", error_type=type(exc).__name__)
            return []
        query_vec = query_vectors[0]

        scored: list[RetrievedMemory] = []
        for memory in memories:
            if not memory.embedding:
                continue
            score = _cosine(query_vec, memory.embedding)
            if score >= settings.memory_semantic_min_score:
                scored.append(
                    RetrievedMemory(content=memory.content, kind=memory.kind, score=score)
                )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    async def remember(
        self, user_id: str, conversation_id: str, items: list[dict[str, Any]]
    ) -> int:
        """Store validated memory candidates with embeddings. Returns count written."""
        valid: list[MemoryCandidate] = []
        for item in items:
            try:
                candidate = MemoryCandidate.model_validate(item)
            except Exception:  # noqa: BLE001 - invalid candidates are dropped, never crash
                logger.debug("semantic_memory_candidate_invalid")
                continue
            content = sanitize_for_tool(candidate.content, max_length=512)
            if len(content) < 3 or contains_prompt_injection(content):
                continue
            valid.append(candidate)
        if not valid:
            return 0

        try:
            embeddings = await self.provider.embed([c.content for c in valid])
        except Exception as exc:  # noqa: BLE001
            logger.warning("semantic_memory_store_embed_failed", error_type=type(exc).__name__)
            return 0

        model_name = get_settings().openai_embedding_model
        written = 0
        for candidate, embedding in zip(valid, embeddings, strict=False):
            await self.repo.add(
                user_id=user_id,
                conversation_id=conversation_id,
                kind=candidate.kind,
                content=candidate.content,
                embedding=embedding,
                embedding_model=model_name,
            )
            written += 1
        return written

    async def remember_explicit(self, user_id: str, conversation_id: str, user_message: str) -> int:
        """Deterministic policy: store explicitly requested memories."""
        items = extract_explicit_memories(user_message)
        if not items:
            return 0
        return await self.remember(user_id, conversation_id, items)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
