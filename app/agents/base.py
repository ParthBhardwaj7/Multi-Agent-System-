"""Agent base class and execution context."""

import abc
import time
from dataclasses import dataclass, field
from typing import Any

from app.graph.state import GraphState
from app.llm.client import LLMProvider
from app.observability import metrics
from app.observability.tracing import get_tracer, record_error, safe_set_attributes
from app.schemas.agents import AgentRunRecord


@dataclass
class AgentContext:
    """Runtime dependencies shared by all agents (injected at startup)."""

    provider: LLMProvider
    tools: dict[str, Any] = field(default_factory=dict)  # named tools: "search", "database"


class BaseAgent(abc.ABC):
    """A single-purpose agent node in the orchestration graph."""

    name: str = "base"

    def __init__(self, context: AgentContext) -> None:
        self.context = context
        self.provider = context.provider

    @abc.abstractmethod
    async def run(self, state: GraphState) -> GraphState:
        """Execute the agent; returns a partial GraphState update."""

    async def timed_run(self, state: GraphState) -> tuple[GraphState, AgentRunRecord]:
        """Wraps run() with agent metrics, tracing and an AgentRunRecord."""
        span = get_tracer().start_span(f"agent.{self.name}")
        safe_set_attributes(
            span,
            {"agent.name": self.name, "conversation_id": state.get("conversation_id", "")},
        )
        start = time.monotonic()
        metrics.agent_requests_total.labels(agent=self.name, status="started").inc()
        status = "success"
        error_type: str | None = None
        try:
            update = await self.run(state)
            span.set_attribute("agent.status", "success")
            metrics.agent_requests_total.labels(agent=self.name, status="success").inc()
        except Exception as exc:  # noqa: BLE001 - accounted, then re-raised
            status = "error"
            error_type = type(exc).__name__
            span.set_attribute("agent.status", "error")
            metrics.agent_requests_total.labels(agent=self.name, status="error").inc()
            metrics.agent_errors_total.labels(agent=self.name, error_type=error_type).inc()
            record_error(span, exc)
            raise
        finally:
            latency_ms = round((time.monotonic() - start) * 1000, 1)
            metrics.agent_request_duration_seconds.labels(agent=self.name).observe(
                latency_ms / 1000
            )
            span.end()
        record = AgentRunRecord(
            agent=self.name,
            status=status,
            intent=state.get("intent"),
            confidence=state.get("confidence"),
            latency_ms=latency_ms,
            prompt_version=self.prompt_version(),
            error_type=error_type,
        )
        return update, record

    def prompt_version(self) -> str | None:
        try:
            from app.core.prompts import prompt_version

            return prompt_version(self.name)
        except (KeyError, FileNotFoundError):
            return None
