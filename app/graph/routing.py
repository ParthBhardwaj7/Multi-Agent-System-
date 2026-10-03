"""Deterministic routing policy node for the LangGraph orchestration.

The Router LLM proposes (intent, confidence); THIS code decides which agent
runs. It also applies tool-authorization pre-checks so unauthorized database
requests are re-routed before any LLM/agent work happens."""

from typing import Any

from app.config.settings import get_settings
from app.graph.state import GraphState
from app.observability import metrics
from app.observability.logging import get_logger
from app.tools.policies import RoutingPolicy

logger = get_logger(__name__)


async def policy_node(state: GraphState) -> dict[str, Any]:
    decision = RoutingPolicy().decide(
        state.get("intent"),
        float(state.get("confidence", 0.0)),
    )
    fallback_reason = state.get("fallback_reason")
    if decision.reason != "high_confidence" and fallback_reason is None:
        fallback_reason = decision.reason

    # Specialist requested but not authorized for this role -> deterministic re-route.
    if (
        decision.selected_agent == "database"
        and state.get("user_role") not in get_settings().db_agent_roles
    ):
        logger.info("policy_reroute_unauthorized", from_agent="database")
        metrics.router_fallback_total.labels(reason="tool_authorization").inc()
        return {
            "selected_agent": "general",
            "routed_to_specialist": False,
            "fallback_reason": "tool_authorization",
        }

    if not decision.routed_to_specialist and state.get("intent") in ("search", "database"):
        metrics.router_fallback_total.labels(reason=decision.reason).inc()

    logger.info(
        "policy_decided",
        selected_agent=decision.selected_agent,
        reason=decision.reason,
        confidence=state.get("confidence"),
    )
    return {
        "selected_agent": decision.selected_agent,
        "routed_to_specialist": decision.routed_to_specialist,
        "fallback_reason": fallback_reason,
    }
