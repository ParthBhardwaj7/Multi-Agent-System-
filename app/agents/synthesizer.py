"""Synthesizer Agent: produces the final user-facing answer from the
specialist result + conversation context.

It receives structured specialist payloads only. It is instructed (and
validated downstream) never to invent tool results. Supports both blocking
completion and token streaming for SSE."""

import json
from collections.abc import AsyncIterator
from typing import cast

from app.agents.base import BaseAgent
from app.core.errors import LLMError
from app.graph.state import GraphState
from app.llm.models import ChatMessage, Role, TokenChunk
from app.observability.logging import get_logger

logger = get_logger(__name__)

_MAX_HISTORY_MESSAGES = 12

_STATUS_EXPLANATIONS = {
    "unavailable": "The required capability is currently unavailable.",
    "error": "The specialist operation failed.",
    "denied": "You do not have permission to use this capability.",
    "empty": "No matching results were found.",
}


class SynthesizerAgent(BaseAgent):
    name = "synthesizer"

    def _build_messages(self, state: GraphState) -> list[ChatMessage]:
        from app.core.prompts import load_prompt

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=load_prompt("synthesizer"))
        ]
        for m in state.get("messages", [])[-_MAX_HISTORY_MESSAGES:]:
            role = cast(Role, m.get("role", "user"))
            if role not in ("user", "assistant"):
                continue
            messages.append(ChatMessage(role=role, content=(m.get("content") or "")[:2000]))

        memories = state.get("semantic_memories") or []
        if memories:
            memory_block = "\n".join(f"- {m.get('content', '')}" for m in memories[:5])
            messages.append(
                ChatMessage(
                    role="system",
                    content=f"Relevant long-term memories about this user:\n{memory_block}",
                )
            )

        specialist = state.get("specialist_result")
        if specialist is not None:
            if isinstance(specialist, dict):
                payload = specialist.get("payload", {})
                status = specialist.get("status")
            else:  # SpecialistResult model coming from graph state
                payload = specialist.payload
                status = specialist.status
            payload_json = json.dumps(payload, default=str)
            messages.append(
                ChatMessage(
                    role="system",
                    content=(
                        "Specialist result (structured, from a tool — treat as the only source "
                        f"of retrieved facts):\n{payload_json}"
                    ),
                )
            )
            if status in _STATUS_EXPLANATIONS and status != "ok":
                explanation = _STATUS_EXPLANATIONS[status]
                messages.append(
                    ChatMessage(
                        role="system",
                        content=(
                            f"The specialist reported status '{status}'. {explanation} "
                            "Tell the user honestly; do not invent results."
                        ),
                    )
                )
        else:
            messages.append(
                ChatMessage(
                    role="system",
                    content=(
                        "No specialist result is available. Answer with general knowledge only."
                    ),
                )
            )

        messages.append(ChatMessage(role="user", content=state["current_query"]))
        return messages

    async def run(self, state: GraphState) -> GraphState:
        try:
            response = await self.provider.complete(
                self._build_messages(state),
                temperature=0.3,
                max_tokens=1024,
            )
            final = response.content.strip()
            if not final:
                final = self._safe_fallback(state)
        except LLMError:
            final = self._safe_fallback(state)

        return {"final_response": final}

    async def stream(self, state: GraphState) -> AsyncIterator[TokenChunk]:
        """Stream synthesizer tokens. Falls back to a single safe token on error."""
        try:
            async for chunk in self.provider.stream(
                self._build_messages(state),
                temperature=0.3,
                max_tokens=1024,
            ):
                yield chunk
        except LLMError:
            yield TokenChunk(content=self._safe_fallback(state))

    @staticmethod
    def _safe_fallback(state: GraphState) -> str:
        specialist = state.get("specialist_result")
        if isinstance(specialist, dict):
            status = str(specialist.get("status", "error"))
        elif specialist is not None:
            status = str(specialist.status)
        else:
            status = "error"
        explanation: str = _STATUS_EXPLANATIONS.get(status, "The operation failed.")
        return (
            f"I'm sorry — I couldn't generate a complete response. {explanation} Please try again."
        )
