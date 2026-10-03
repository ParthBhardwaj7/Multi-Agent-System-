"""General Agent: fallback for conversation, ambiguous or unsupported requests.

It explicitly signals when a specialized capability was unavailable so the
synthesizer never presents a hallucinated specialist result."""

from typing import Literal, cast

from app.agents.base import BaseAgent
from app.core.errors import LLMError
from app.graph.state import GraphState
from app.llm.models import ChatMessage, Role
from app.observability.logging import get_logger
from app.schemas.agents import GeneralResult, SpecialistResult

logger = get_logger(__name__)

_MAX_HISTORY_MESSAGES = 10


class GeneralAgent(BaseAgent):
    name = "general"

    def _build_messages(self, state: GraphState) -> list[ChatMessage]:
        from app.core.prompts import load_prompt

        messages: list[ChatMessage] = [ChatMessage(role="system", content=load_prompt("general"))]
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

        query = state["current_query"]
        fallback_reason = state.get("fallback_reason")
        if fallback_reason in ("specialist_failure", "router_failure"):
            messages.append(
                ChatMessage(
                    role="system",
                    content=(
                        "A specialized capability (search or database) was attempted for this "
                        "request but failed or is unavailable. If answering requires it, tell the "
                        "user that capability is currently unavailable. Do not fabricate results."
                    ),
                )
            )
        messages.append(ChatMessage(role="user", content=query))
        return messages

    async def run(self, state: GraphState) -> GraphState:
        try:
            response = await self.provider.complete(
                self._build_messages(state),
                temperature=0.4,
                max_tokens=768,
            )
            answer = response.content.strip()
            status = "ok"
        except LLMError:
            answer = (
                "I'm sorry — the assistant backend is temporarily unavailable. "
                "Please try again in a moment."
            )
            status = "unavailable"

        result = GeneralResult(
            status=cast(Literal["ok", "unavailable"], status),
            answer=answer,
            capability_unavailable=(status != "ok"),
        )
        logger.info("general_agent_completed", status=status)
        specialist = SpecialistResult(
            agent=self.name,
            intent="general",
            status=result.status,
            payload=result.model_dump(),
            prompt_version=self.prompt_version(),
        )
        return {"specialist_result": specialist}
