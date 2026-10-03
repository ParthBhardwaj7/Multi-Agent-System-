"""LangGraph orchestration graph.

Topology: START -> router -> policy -> specialist -> synthesizer -> END

- `router` classifies intent (structured output, validated).
- `policy` deterministically applies the confidence threshold + authorization.
- `specialist` resolves the selected agent from the AgentRegistry and runs it.
- `synthesizer` produces the final answer; supports token streaming into a
  sink injected via `config["configurable"]["token_sink"]` for SSE.

The compiled graph accepts a LangGraph checkpointer for durable state.
"""

import time
from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.registry import AgentRegistry
from app.agents.synthesizer import SynthesizerAgent
from app.graph.routing import policy_node
from app.graph.state import GraphState
from app.observability import metrics
from app.observability.logging import get_logger
from app.schemas.agents import AgentRunRecord

logger = get_logger(__name__)

TokenSink = Callable[[str], Awaitable[None]]


def _router_node(registry: AgentRegistry) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    router = registry.resolve("router")

    async def node(state: GraphState) -> dict[str, Any]:
        update, record = await router.timed_run(state)
        return {**update, "agent_runs": [record]}

    return node


def _specialist_node(registry: AgentRegistry) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    async def node(state: GraphState) -> dict[str, Any]:
        requested = state.get("selected_agent", registry.fallback_name)
        try:
            agent = registry.resolve(requested)
        except KeyError:
            logger.warning("specialist_not_registered", requested=requested)
            agent = registry.fallback

        try:
            update, record = await agent.timed_run(state)
        except Exception as exc:
            # Specialist failure: deterministically fall back to the fallback agent.
            logger.warning("specialist_failed", agent=requested, error_type=type(exc).__name__)
            metrics.router_fallback_total.labels(reason="specialist_failure").inc()
            fallback = registry.fallback
            failure_record = AgentRunRecord(
                agent=requested,
                status="error",
                intent=state.get("intent"),
                confidence=state.get("confidence"),
                error_type=type(exc).__name__,
            )
            fb_state: GraphState = {**state, "fallback_reason": "specialist_failure"}
            fb_update, fb_record = await fallback.timed_run(fb_state)
            return {
                **fb_update,
                "selected_agent": fallback.name,
                "routed_to_specialist": False,
                "fallback_reason": "specialist_failure",
                "agent_runs": [failure_record, fb_record],
            }

        return {**update, "agent_runs": [record]}

    return node


def _synthesizer_node(registry: AgentRegistry) -> Callable[..., Awaitable[dict[str, Any]]]:
    synthesizer = cast(SynthesizerAgent, registry.resolve("synthesizer"))

    async def node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
        sink: TokenSink | None = (config.get("configurable") or {}).get("token_sink")
        start = time.monotonic()

        if sink is None:
            update, record = await synthesizer.timed_run(state)
            return {**update, "agent_runs": [record]}

        # Streaming mode: push token chunks to the sink, aggregate final text.
        metrics.agent_requests_total.labels(agent="synthesizer", status="started").inc()
        status, error_type = "success", None
        aggregated: list[str] = []
        try:
            async for chunk in synthesizer.stream(state):
                aggregated.append(chunk.content)
                await sink(chunk.content)
            metrics.agent_requests_total.labels(agent="synthesizer", status="success").inc()
        except Exception as exc:  # noqa: BLE001 - accounted, converted to safe fallback
            status, error_type = "error", type(exc).__name__
            metrics.agent_requests_total.labels(agent="synthesizer", status="error").inc()
            metrics.agent_errors_total.labels(agent="synthesizer", error_type=error_type).inc()
            fallback_text = "I'm sorry — response generation failed. Please try again."
            aggregated = [fallback_text]
            await sink(fallback_text)
        finally:
            latency_ms = round((time.monotonic() - start) * 1000, 1)
            metrics.agent_request_duration_seconds.labels(agent="synthesizer").observe(
                latency_ms / 1000
            )
        record = AgentRunRecord(
            agent="synthesizer",
            status=status,
            intent=state.get("intent"),
            confidence=state.get("confidence"),
            latency_ms=latency_ms,
            prompt_version=synthesizer.prompt_version(),
            error_type=error_type,
        )
        return {"final_response": "".join(aggregated), "agent_runs": [record]}

    return node


def build_graph(registry: AgentRegistry, checkpointer: Any | None = None) -> CompiledStateGraph:
    """Build and compile the orchestration graph.

    `checkpointer` (optional) enables durable per-conversation state; the
    production wiring attaches the Postgres checkpoint saver."""
    builder = StateGraph(GraphState)
    builder.add_node("router", _router_node(registry))  # type: ignore[call-overload]
    builder.add_node("policy", policy_node)  # type: ignore[arg-type,call-overload]
    builder.add_node("specialist", _specialist_node(registry))  # type: ignore[arg-type,call-overload]
    builder.add_node("synthesizer", _synthesizer_node(registry))  # type: ignore[arg-type,call-overload]

    builder.add_edge(START, "router")
    builder.add_edge("router", "policy")
    builder.add_edge("policy", "specialist")
    builder.add_edge("specialist", "synthesizer")
    builder.add_edge("synthesizer", END)
    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()
