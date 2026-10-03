# High-Level Design (HLD)

## 1. Purpose

A multi-agent AI orchestration platform exposing a conversational API. Requests
are classified by a router, executed by a specialist agent under a deterministic
policy, and answered by a synthesizer — with durable conversations, layered
memory, and production operations (authN/authZ, rate limiting, metrics, traces,
streaming).

## 2. Components

| Component | Responsibility | Technology |
|---|---|---|
| API Gateway (app) | Auth, validation, rate limiting, SSE/JSON endpoints | FastAPI, uvicorn |
| Chat Service | Per-turn orchestration: state prep, graph run, persistence, memory writes | Python service layer |
| Orchestration graph | Router → Policy → Specialist → Synthesizer | LangGraph (compiled, checkpointer) |
| Agent Registry | intent → agent resolution; extensibility | in-process registry |
| LLM abstraction | All model access behind `LLMProvider` | OpenAI-compatible SDK |
| Relational store | Users, conversations, messages, memories, agent runs, graph checkpoints | PostgreSQL 16 + SQLAlchemy 2 async |
| Ephemeral store | Rate limiting, conversation locks, cache | Redis 7 |
| Observability | JSON logs, Prometheus metrics, OpenTelemetry traces | structlog, prometheus-client, OTel |
| Packaging / CI | Image, stack, pipeline | Docker, GitHub Actions |

## 3. Data Flow (one chat turn)

1. Client calls `POST /v1/chat` (JWT or API key) — request id assigned, rate
   limits enforced (user + IP), body validated (Pydantic).
2. Conversation resolved or created; ownership enforced (403 on foreign IDs).
3. Memory: short-term window (last N messages) + semantic recall (top-k
   similar long-term memories) loaded from PostgreSQL.
4. LangGraph executes: **router** (structured, schema-validated intent +
   confidence) → **policy** (deterministic threshold/authorization decision) →
   **specialist** (search | database | general, resolved via registry) →
   **synthesizer** (grounded final answer; token streaming when SSE).
5. Each super-step checkpoint is written to PostgreSQL (thread = conversation).
6. The turn is persisted (user + assistant messages, agent run records) under a
   Redis conversation lock with optimistic versioning; long-term memory rules
   run (explicit "remember…" always; extraction config-gated).
7. Response returns as JSON or an SSE event stream
   (`router_started → router_completed → agent_started → agent_completed →
   token* → done`).

## 4. Deployment Architecture

```text
            ┌────────────┐
            │   Client    │
            └──────┬─────┘
                   │ HTTPS
            ┌──────▼─────┐
            │ LB / Gateway│  (Cloudflare / ALB / API Gateway)
            └──────┬─────┘
                   │
        ┌──────────▼──────────┐        ┌───────────────┐
        │  API instances (n)   │───────▶│ Prometheus     │
        │  (stateless FastAPI) │        └───────┬───────┘
        └───┬─────────┬────────┘                ▼
            │         │                  ┌──────────────┐
   ┌────────▼──┐   ┌──▼──────┐           │   Grafana     │
   │ PostgreSQL│   │  Redis  │           └──────────────┘
   │ (managed) │   │ (managed)│
   └───────────┘   └─────────┘
            │
            ▼
        OpenAI (or compatible gateway)
```

- API instances are **stateless**; all state lives in PostgreSQL/Redis, so
  scale-out is horizontal.
- Cloud-agnostic: core code has no cloud-specific APIs.

## 5. Scalability

| Dimension | Approach |
|---|---|
| API throughput | Stateless replicas behind LB; uvicorn workers per instance |
| DB reads | Connection pooling per process; separate read-only pool for the AI DB tool |
| DB writes | Single-row transactions; versioned appends; indexes on hot paths |
| Checkpoints | Dedicated connection pool (separate from request pool) to prevent starvation |
| Streaming | SSE is per-connection; scale by adding replicas (sticky not required) |
| Rate limiting | Redis sliding window, atomic Lua — shared across replicas |

## 6. Reliability

- Every external dependency: timeout + bounded exponential-backoff retries
  (transient only) + structured errors + metrics + traces.
- Failure degradation paths: router failure → general agent; specialist failure
  → fallback agent with honest messaging; Redis outage → documented
  fail-open/fail-closed per feature; search unconfigured → honest
  "unavailable" response (never fabricated).
- Client disconnect during SSE cancels the graph task and frees resources.
- Optimistic versioning makes concurrent conversation writes safe (409 on
  conflict) even if the Redis lock is unavailable.

## 7. Security Summary

Authentication (JWT/API key), per-conversation authorization, whitelist+parameterized
read-only database tool, prompt-injection defense-in-depth, rate limits,
bounded request bodies, secret redaction in logs. Details: `SECURITY.md`.
