# Observability

Three signals — structured logs, Prometheus metrics, OpenTelemetry traces —
plus health/readiness probes. Correlated end-to-end by `request_id` (client
may supply `X-Request-ID`; otherwise generated) and `conversation_id`.

## Logs (structlog → JSON on stdout)

Fields on every event: `timestamp` (ISO-8601 UTC), `level`, `logger`,
`event`, plus contextvars-injected `request_id`, `conversation_id`, `user_id`
when set. Domain events include:

| Event | Emitted by |
|---|---|
| `http_request` | middleware (method, path, status, latency_ms) |
| `router_completed` | router (intent, confidence) |
| `policy_decided` | policy node (selected_agent, reason) |
| `search_agent_completed` / `database_agent_completed` / `general_agent_completed` | specialists (status, counts) |
| `llm_complete_failed` / `search_tool_failed` / `database_tool_failed` | failure paths (error_type) |
| `sse_client_disconnected` | stream cleanup |
| `app_error` / `unhandled_error` | exception handlers |

A secret-redaction processor drops/redacts `api_key`, `authorization`,
`password`, `secret`, `token`, `jwt` keys. `APP_ENV=local` renders
human-readable console logs; all other environments render JSON.

## Metrics (`GET /metrics`, Prometheus text format)

Low-cardinality labels only — never `user_id`, `conversation_id`, or raw
queries.

| Group | Metrics |
|---|---|
| HTTP | `http_requests_total{method,endpoint,status}`, `http_request_duration_seconds`, `http_requests_in_progress` |
| Agents | `agent_requests_total{agent,status}`, `agent_request_duration_seconds{agent}`, `agent_errors_total{agent,error_type}` |
| Routing | `router_confidence` (histogram), `router_intents_total{intent}`, `router_fallback_total{reason}` |
| LLM | `llm_tokens_total{model,kind}`, `llm_request_duration_seconds{model,operation}`, `llm_request_errors_total{model,error_type}`, `llm_cost_usd_total{model}` |
| Tools | `tool_requests_total{tool,status}`, `tool_request_duration_seconds{tool}`, `tool_errors_total{tool,error_type}` |
| Domain | `chat_turns_total{status}`, `conversation_lock_conflicts_total`, `rate_limit_rejections_total{scope}`, `sse_active_streams` |

`endpoint` is templated (`/v1/conversations/{conversation_id}`) to keep label
cardinality bounded.

## Tracing (OpenTelemetry)

Enabled by `OTEL_TRACES_ENABLED=true`; exports via OTLP/HTTP to
`OTEL_EXPORTER_OTLP_ENDPOINT` (falls back to console exporter without an
endpoint). Resource: `service.name`, `deployment.environment`. Spans:

```
llm.openai.complete / stream / embed     (model, token counts)
agent.router / agent.search / agent.database / agent.general / agent.synthesizer
tool.search.request                      (result count, status)
tool.database.execute                    (table, row count, status)
```

Attributes carry identifiers, statuses, and counts — never message content or
credentials. When disabled, tracer calls are no-ops.

## Dashboards & Alerts

Docker compose ships Prometheus (scrapes `api:8000`, 15s) and Grafana
(datasource auto-provisioned at `docker/grafana/provisioning`; admin password
via `GRAFANA_ADMIN_PASSWORD`).

Suggested starter alerts:

| Alert | Condition |
|---|---|
| High error rate | `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05` |
| Latency SLO burn | `http_request_duration_seconds` p95 > 30s for 10m |
| LLM failures | `rate(llm_request_errors_total[5m]) > 0.1/s` |
| Rate-limit pressure | `rate(rate_limit_rejections_total[10m])` spike |
| Lock contention | `rate(conversation_lock_conflicts_total[10m]) > 0.5/s` |
| Cost guardrail | `increase(llm_cost_usd_total[1h]) > threshold` |

## Health & Readiness

- `GET /health` — process liveness only (no dependency calls).
- `GET /ready` — checks PostgreSQL (`SELECT 1`), Redis (`PING`), and LLM
  provider configuration; `status: ready` requires Postgres+Redis. Used by
  Docker/compose and load balancers.
