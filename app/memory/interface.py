"""Memory abstraction: the application never talks to a storage engine
directly, only to these interfaces.

Layers:
- Short-term: bounded recent-message window for the current turn (context).
- Persistent: full conversation history in PostgreSQL.
- Semantic: long-term user facts with embedding retrieval.

Write policy for long-term memory (docs/ARCHITECTURE.md §memory):
- Explicit "remember ..." requests from the user are always stored.
- An LLM extraction pass (config-gated) proposes durable facts from turns;
  candidates shorter than 3 chars or older than the limit are dropped.
- Routine chatter is NEVER stored as permanent memory."""

import re
from dataclasses import dataclass
from typing import Any, Protocol

from app.graph.state import MessageState


@dataclass(frozen=True)
class RetrievedMemory:
    content: str
    kind: str
    score: float


class ShortTermMemory(Protocol):
    async def window(self, conversation_id: str, user_id: str) -> list[MessageState]: ...


class PersistentMemory(Protocol):
    async def load_history(self, conversation_id: str, user_id: str) -> list[MessageState]: ...

    async def save_turn(
        self,
        conversation_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
        metadata: dict[str, Any],
    ) -> None: ...


class SemanticMemory(Protocol):
    async def recall(self, user_id: str, query: str, top_k: int) -> list[RetrievedMemory]: ...

    async def remember(
        self, user_id: str, conversation_id: str, items: list[dict[str, Any]]
    ) -> int: ...


_REMEMBER_PATTERNS = (
    re.compile(r"\bremember (that\s+)?(?P<content>.+)", re.IGNORECASE),
    re.compile(r"\bdon'?t forget (that\s+)?(?P<content>.+)", re.IGNORECASE),
    re.compile(r"\bfor future reference,?\s*(?P<content>.+)", re.IGNORECASE),
)


def extract_explicit_memories(user_message: str) -> list[dict[str, Any]]:
    """Deterministic rule: catch explicit memory requests ("remember that ...")."""
    found: list[dict[str, Any]] = []
    for pattern in _REMEMBER_PATTERNS:
        match = pattern.search(user_message)
        if match:
            content = match.group("content").strip().rstrip(".!?")
            if 3 <= len(content) <= 512:
                lowered = content.lower()
                kind = (
                    "preference"
                    if any(
                        w in lowered
                        for w in ("i prefer", "i like", "i always", "my name", "i hate")
                    )
                    else "fact"
                )
                found.append({"content": content, "kind": kind})
    return found
