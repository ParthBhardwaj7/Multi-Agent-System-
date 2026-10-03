"""Router Agent: classifies the query into an intent with confidence.

It uses strict structured output (JSON schema validated with Pydantic) and
NEVER produces the user-facing answer. Routing decisions are made downstream
by the deterministic policy layer."""

from app.agents.base import BaseAgent
from app.core.errors import LLMError
from app.graph.state import GraphState
from app.llm.models import ChatMessage
from app.llm.structured import structured_complete
from app.observability import metrics
from app.observability.logging import get_logger
from app.schemas.agents import RouterOutput
from app.schemas.chat import IntentName

logger = get_logger(__name__)

ROUTER_SCHEMA_NAME = "RouterOutput"


class RouterAgent(BaseAgent):
    name = "router"

    def _build_messages(self, state: GraphState) -> list[ChatMessage]:
        from app.core.prompts import load_prompt

        history_lines: list[str] = []
        for message in state.get("messages", [])[-8:]:
            role = message.get("role", "user")
            content = (message.get("content", "") or "")[:500]
            history_lines.append(f"{role}: {content}")

        context_block = "\n".join(history_lines) if history_lines else "(no prior context)"
        user_block = (
            f"Conversation so far:\n{context_block}\n\n"
            f"Latest user message:\n{state['current_query']}\n\n"
            "Classify the latest user message into exactly one intent."
        )
        return [
            ChatMessage(role="system", content=load_prompt("router")),
            ChatMessage(role="user", content=user_block),
        ]

    async def run(self, state: GraphState) -> GraphState:
        try:
            output: RouterOutput = await structured_complete(
                self.provider,
                RouterOutput,
                self._build_messages(state),
                temperature=0.0,
                max_tokens=64,
            )
        except LLMError:
            # Fail-safe: classify as general with zero confidence so the policy
            # layer deterministically falls back. Never guess an intent.
            metrics.router_fallback_total.labels(reason="router_failure").inc()
            return {
                "intent": "general",
                "confidence": 0.0,
                "fallback_reason": "router_failure",
            }

        metrics.router_confidence.observe(output.confidence)
        metrics.router_intents_total.labels(intent=output.intent).inc()
        logger.info(
            "router_completed",
            intent=output.intent,
            confidence=output.confidence,
        )
        return {
            "intent": output.intent,
            "confidence": output.confidence,
        }

    @staticmethod
    def from_output(intent: str | None, confidence: float) -> IntentName | None:
        if intent in ("search", "database", "general"):
            return intent  # type: ignore[return-value]
        return None
