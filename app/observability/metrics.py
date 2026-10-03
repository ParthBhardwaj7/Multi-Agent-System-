"""Prometheus metrics.

Naming follows the convention required in docs/OBSERVABILITY.md. Labels are
intentionally low-cardinality: no user_id, conversation_id, or raw queries.
"""

from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST as OPENMETRICS_TYPE

from app.config.settings import get_settings

REGISTRY = CollectorRegistry(auto_describe=True)

# --- HTTP ---
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=("method", "endpoint", "status"),
    registry=REGISTRY,
)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=("method", "endpoint"),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=REGISTRY,
)
http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "Active HTTP requests",
    labelnames=("method",),
    registry=REGISTRY,
)

# --- Agents ---
agent_requests_total = Counter(
    "agent_requests_total",
    "Agent invocations",
    labelnames=("agent", "status"),
    registry=REGISTRY,
)
agent_request_duration_seconds = Histogram(
    "agent_request_duration_seconds",
    "Agent execution latency in seconds",
    labelnames=("agent",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)
agent_errors_total = Counter(
    "agent_errors_total",
    "Agent failures by error type",
    labelnames=("agent", "error_type"),
    registry=REGISTRY,
)

# --- Routing ---
router_confidence = Histogram(
    "router_confidence",
    "Router confidence distribution",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
    registry=REGISTRY,
)
router_intents_total = Counter(
    "router_intents_total",
    "Router intent classification results",
    labelnames=("intent",),
    registry=REGISTRY,
)
router_fallback_total = Counter(
    "router_fallback_total",
    "Requests falling back from specialist to general agent",
    labelnames=("reason",),  # low_confidence | policy | specialist_failure
    registry=REGISTRY,
)

# --- LLM ---
llm_tokens_total = Counter(
    "llm_tokens_total",
    "LLM token usage",
    labelnames=("model", "kind"),  # kind: input | output
    registry=REGISTRY,
)
llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM call latency in seconds",
    labelnames=("model", "operation"),  # operation: complete | stream | embed | structured
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=REGISTRY,
)
llm_request_errors_total = Counter(
    "llm_request_errors_total",
    "LLM call failures by error type",
    labelnames=("model", "error_type"),
    registry=REGISTRY,
)
llm_cost_usd_total = Counter(
    "llm_cost_usd_total",
    "Estimated cumulative LLM cost in USD",
    labelnames=("model",),
    registry=REGISTRY,
)

# --- Tools ---
tool_requests_total = Counter(
    "tool_requests_total",
    "Tool invocations",
    labelnames=("tool", "status"),
    registry=REGISTRY,
)
tool_request_duration_seconds = Histogram(
    "tool_request_duration_seconds",
    "Tool execution latency in seconds",
    labelnames=("tool",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)
tool_errors_total = Counter(
    "tool_errors_total",
    "Tool failures by error type",
    labelnames=("tool", "error_type"),
    registry=REGISTRY,
)

# --- Domain ---
chat_turns_total = Counter(
    "chat_turns_total",
    "Completed chat turns",
    labelnames=("status",),  # success | error
    registry=REGISTRY,
)
conversation_lock_conflicts_total = Counter(
    "conversation_lock_conflicts_total",
    "Requests rejected because another turn is active on the conversation",
    registry=REGISTRY,
)
rate_limit_rejections_total = Counter(
    "rate_limit_rejections_total",
    "Requests rejected by the rate limiter",
    labelnames=("scope",),  # user | ip
    registry=REGISTRY,
)
sse_active_streams = Gauge(
    "sse_active_streams",
    "Currently open SSE streams",
    registry=REGISTRY,
)


def render_metrics() -> tuple[bytes, str]:
    """Expose metrics in Prometheus text format."""
    settings = get_settings()
    if settings.metrics_enabled:
        return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
    return b"", CONTENT_TYPE_LATEST


def observe_llm_usage(model: str, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
    if input_tokens:
        llm_tokens_total.labels(model=model, kind="input").inc(input_tokens)
    if output_tokens:
        llm_tokens_total.labels(model=model, kind="output").inc(output_tokens)
    if cost_usd > 0:
        llm_cost_usd_total.labels(model=model).inc(cost_usd)


def as_dict() -> dict[str, Any]:
    return {"registry": "multi-agent-orchestrator", "openmetrics_content_type": OPENMETRICS_TYPE}
