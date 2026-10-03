# Multi-Agent Orchestrator

A production-grade multi-agent AI orchestration platform: FastAPI + LangGraph +
PostgreSQL + Redis, with full observability, security hardening, evaluation and
load testing, Docker packaging, and CI/CD.

```
Client → FastAPI (auth/validation/rate-limit)
       → Chat Service (conversation state, memory)
       → LangGraph: Router → Policy → Specialist (search | database | general) → Synthesizer
       → PostgreSQL (persistence, checkpoints, memory, agent runs) + Redis (limits/locks/cache)
       → SSE stream / JSON response
```

## Features

- **Orchestration** — LangGraph state machine; router classifies intent with
  validated structured output; a deterministic policy layer (configurable
  confidence threshold, tool-authorization re-routing) decides; specialists
  return structured results; the synthesizer writes the final answer.
- **Agent Registry** — new agents (billing, RAG, analytics…) register without
  touching the orchestrator.
- **Secure database tool** — the LLM proposes a *query plan*; the app validates
  it against whitelists and compiles parameterized SQL; execution runs on a
  dedicated READ-ONLY Postgres role with statement timeouts.
- **Search tool** — timeout/retry/error-mapped HTTP client; unavailable when
  unconfigured; never fabricates results.
- **Memory** — short-term window + persistent history + semantic long-term
  memory (explicit "remember…" rules, config-gated extraction).
- **Persistence** — PostgreSQL via Alembic migrations; LangGraph state
  checkpointed per conversation thread (custom async Postgres checkpointer).
- **SSE streaming** — token-level streaming with lifecycle events and safe
  error handling; client disconnects cancel the turn.
- **Security** — JWT + API-key authN, per-conversation ownership authZ,
  Redis sliding-window rate limits (per-user/per-IP, fail-open/closed policy),
  distributed conversation locks + optimistic versioning, prompt-injection
  defense-in-depth, secret redaction in logs, request-size guards.
- **Observability** — JSON structured logs with correlation IDs, Prometheus
  metrics (HTTP/agents/LLM/tools/routing/costs), OpenTelemetry traces
  (OTLP), health/readiness endpoints, Grafana + Prometheus in compose.
- **Evaluation** — 450-case golden dataset; automated runner measuring routing
  accuracy, fallback accuracy, tool selection, adversarial flagging, latency
  and token usage (heuristic baseline in CI; real LLM with `--router llm`).
- **Load testing** — Locust scenarios (10/50/100 users) with recorded results
  in [docs/performance/RESULTS.md](docs/performance/RESULTS.md).
- **CI/CD** — GitHub Actions: lint (ruff), types (mypy), unit/integration/e2e/
  security tests on real Postgres+Redis services, bandit, pip-audit,
  evaluation, Docker build.

## Tech Stack

Python 3.11 · FastAPI · LangGraph · SQLAlchemy 2 (async) · asyncpg · Alembic ·
Redis · OpenAI SDK (abstracted behind `LLMProvider`) · Pydantic v2 ·
prometheus-client · OpenTelemetry · structlog · pytest · Locust · Docker ·
GitHub Actions.

## Project Structure

```
app/
├── api/            # routes (chat, conversations, auth, health), middleware, DI container
├── agents/         # base, router, search, database, general, synthesizer, registry
├── config/         # pydantic-settings
├── core/           # errors, retry, prompts, redis utils
├── db/             # engine, models, repositories
├── graph/          # LangGraph state, routing policy node, graph assembly
├── llm/            # provider abstraction, OpenAI impl, structured output, fake (dev/test)
├── memory/         # interface, conversation, semantic, checkpoint saver
├── observability/  # logging, metrics, tracing
├── schemas/        # API + agent-facing Pydantic models
├── security/       # auth, authorization, rate limiting, locks
├── services/       # chat orchestration, conversations
└── main.py
prompts/            # versioned prompt templates (git = prompt versioning)
tests/              # unit | integration | e2e | security | load | evaluation
migrations/         # Alembic
docker/             # Dockerfile, compose, prometheus, grafana provisioning
scripts/            # seed, eval dataset + runner, read-only role SQL
docs/               # HLD, LLD, ARCHITECTURE, API, DATABASE, SECURITY,
                    # OBSERVABILITY, TESTING, EVALUATION, DEPLOYMENT, DEVELOPMENT
```

## Local Setup (Docker)

```bash
cp .env.example .env         # then fill in OPENAI_API_KEY and JWT_SECRET
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml exec api python -m alembic -c migrations/alembic.ini upgrade head
docker compose -f docker/docker-compose.yml exec api python scripts/seed_local.py
```

Services: API http://localhost:8000 (docs at `/docs`) · Prometheus :9090 ·
Grafana :3001 (admin/admin) · Postgres :5434 · Redis :6390.

> No OpenAI key yet? Set `LLM_FAKE_MODE=true` in `.env` to explore the full
> system with a deterministic fake model (refused in `APP_ENV=production`).

## Local Setup (bare metal)

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # POSIX: python3 -m venv ...
cp .env.example .env                              # set DATABASE_URL/REDIS_URL etc.
python -m alembic -c migrations/alembic.ini upgrade head
python scripts/seed_local.py
python -m uvicorn app.main:app --reload
```

## Environment Variables

See [.env.example](.env.example) for the annotated list. Key groups:
application (`APP_ENV`, `LOG_LEVEL`), OpenAI (`OPENAI_API_KEY`, `OPENAI_MODEL`),
databases (`DATABASE_URL`, `DATABASE_RO_URL`, `REDIS_URL`), security
(`JWT_SECRET`, `RATE_LIMIT_*`, `RATE_LIMIT_FAIL_OPEN`), routing
(`ROUTER_CONFIDENCE_THRESHOLD`), memory, search tool, DB-agent policy, cost
(`LLM_PRICING`), observability (`OTEL_*`, `METRICS_ENABLED`).

## Database Migrations

```bash
python -m alembic -c migrations/alembic.ini upgrade head     # apply
python -m alembic -c migrations/alembic.ini downgrade -1     # roll back one
python -m alembic -c migrations/alembic.ini revision --autogenerate -m "..."  # create
```

Tables: `users`, `conversations`, `messages`, `memories`, `agent_runs`,
`checkpoints`, `checkpoint_writes` (+ demo `documents`/`faqs` for the database
agent, created by the seed script, not by migrations). See docs/DATABASE.md.

## Running Tests

```bash
pytest tests/unit                                  # fast, no services
pytest tests/integration tests/e2e tests/security # needs Postgres :5434 + Redis :6390
pytest tests/unit tests/integration tests/e2e tests/security --cov=app
```

## Running Evaluation

```bash
python scripts/generate_eval_dataset.py                  # (re)build the golden dataset
python scripts/run_evaluation.py --router heuristic      # CI mode, no API key
python scripts/run_evaluation.py --router llm            # real model (needs OPENAI_API_KEY)
```

Machine-readable reports land in `docs/evaluation/`.

## Running Load Tests

```bash
# 1. start the stack (compose or local uvicorn with LLM_FAKE_MODE=true)
# 2. seed a load-test user
LOADTEST_API_KEY=<key> locust -f tests/load/locustfile.py \
    --host http://localhost:8000 --headless -u 50 -r 5 -t 60s \
    --csv docs/performance/run50
```

Recorded results and interpretation: [docs/performance/RESULTS.md](docs/performance/RESULTS.md).

## API Examples

```bash
# Exchange an API key for a JWT
curl -s -X POST localhost:8000/v1/auth/token \
  -H 'Content-Type: application/json' -d '{"api_key": "mao_..."}'

# Chat (blocking)
curl -s -X POST localhost:8000/v1/chat \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"message": "What is the latest news about AI?"}'

# Chat (SSE streaming)
curl -N -X POST localhost:8000/v1/chat/stream \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"message": "What does our return policy say?"}'

# Conversations
curl -s localhost:8000/v1/conversations -H "Authorization: Bearer $TOKEN"
curl -s localhost:8000/v1/conversations/<id>/messages -H "Authorization: Bearer $TOKEN"
```

Full reference: [docs/API.md](docs/API.md) and `/docs` (OpenAPI UI).

## Observability

- `GET /metrics` — Prometheus (HTTP, agent, routing, LLM token/cost, tool metrics)
- `GET /health` — liveness (no external dependencies)
- `GET /ready` — Postgres/Redis/provider readiness
- JSON logs with `request_id`/`conversation_id` correlation; OpenTelemetry
  spans per request/agent/tool when `OTEL_TRACES_ENABLED=true`.
- Grafana + Prometheus ship in docker compose. See docs/OBSERVABILITY.md.

## Deployment

Cloud-agnostic by design (stateless API instances; Postgres/Redis managed
services). Reference architectures and operational runbooks for Cloud Run,
Kubernetes and Docker Compose — including scaling, secrets, monitoring and
rollback — are in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Security

Authentication (JWT/API keys), per-conversation authorization, SQL safety via
query-plan whitelisting + parameterization + read-only role, prompt-injection
defense-in-depth, rate limiting with documented fail-open/closed policy,
secret-redaction in logs, and request-size guards are documented in
[docs/SECURITY.md](docs/SECURITY.md).

## Known Limitations

- Semantic memory uses JSONB-stored embeddings with in-Python cosine ranking
  (bounded per-user scan) — swap to pgvector for large-scale retrieval.
- The checkpointer serializes full graph state per step; very long
  conversations increase checkpoint size.
- SSE streaming keeps a worker per active stream; multi-worker production
  deployments need a scale-out transport (e.g., Redis pub/sub fan-out) for
  very large concurrent stream counts.
- Prompt-injection detection is heuristic defense-in-depth, not a guarantee;
  primary safety comes from structural controls (plan whitelists, read-only
  role, no raw SQL from the LLM).
- Search results require configuring an external search API (`SEARCH_API_*`);
  otherwise the search agent honestly reports unavailability.

## Future Improvements

- pgvector-backed semantic memory with ANN indexes.
- Parallel specialist execution for multi-intent queries.
- Response caching keyed on user+query+context hash (semantically safe cases only).
- Human-in-the-loop approval step for sensitive database plans.
- Dashboard-as-code (Grafana provisioning of agent/routing dashboards).
- Canary + blue/green deployment automation in CI/CD.
