# Testing

Run everything from the repository root. Unit tests need no services;
integration/e2e/security tests need PostgreSQL (`:5434`) and Redis (`:6390`) —
`make infra-up` or the compose stack provides both.

```bash
pytest tests/unit                                       # fast
pytest tests/integration tests/e2e tests/security       # real Postgres/Redis
pytest tests/unit tests/integration tests/e2e tests/security --cov=app
```

## Layers

### Unit (`tests/unit/`, 72 tests)
No network, no database (fakeredis for Redis behaviour):
- Routing policy: thresholds, boundary inclusivity, invalid/absent intent,
  custom threshold/fallback.
- Content policies: prompt-injection patterns, secret-material detection,
  sanitization.
- Agent registry: registration, resolution, fallback, duplicate rejection,
  extensibility (new agent without orchestrator changes).
- LLM: structured-output validation + repair-then-raise, retry policy
  (transient retried, permanent not, exhaustion re-raises), cost estimation.
- Database tool: SQL compilation (parameterized values, identifier
  whitelisting, row-limit clamping, rejected tables/columns/filters),
  authorization verdicts.
- Rate limiter (limits, isolation, fail-open/fail-closed), conversation locks
  (acquire/release, conflict, fail-closed), JWT lifecycle (expiry, tamper),
  API-key hashing, memory write rules.
- Graph with fake provider end-to-end: search flow, low-confidence fallback,
  token streaming via sink, router-failure degradation; SSE event mapping;
  schema validation; checkpoint-saver interface conformance.

### Integration (`tests/integration/`)
Real FastAPI app + real PostgreSQL + (fakeredis) + deterministic fake LLM:
- health/ready/metrics endpoints,
- token exchange and `/me`,
- full chat turn: routed intent, persisted user/assistant messages,
  `agent_runs` audit rows, usage reporting,
- checkpoint thread written per conversation,
- cross-session semantic memory ("remember pineapple" recalled later),
- conversation list/get/delete lifecycle.

### E2E (`tests/e2e/`)
Complete flows through every agent: database-agent answers grounded in seeded
data, database authorization re-routing for role `user`, honest
"unavailable" behaviour when search is unconfigured, general conversation,
and multi-turn context continuation.

### Security (`tests/security/`)
Authentication (missing/invalid/expired/tampered credentials), authorization
(cross-user read/append/delete), input validation (oversized message, extra
fields), rate limiting (429 + headers), concurrency (parallel turns on one
conversation → 200+409), Redis outage (fail-closed 429), LLM failure
degradation, read-only role enforcement at the database level, SQL-injection
parameterization.

### Evaluation (`scripts/run_evaluation.py`)
450-case golden dataset; see docs/EVALUATION.md.

### Load (`tests/load/locustfile.py`)
Locust scenarios; see docs/performance/RESULTS.md for measured results.

## Test Doubles Policy

- `FakeLLMProvider` (`app/llm/fake.py`) is an explicitly labelled,
  schema-aware deterministic double used by tests, local dev
  (`LLM_FAKE_MODE=true`), load testing, and CI evaluation. Production startup
  **refuses** fake mode (`Settings.validate_runtime_security`).
- Redis: fakeredis (with Lua support) for unit tests; real Redis for
  integration/e2e.
- No other fakes: the database tool, checkpoint saver, repositories, and
  agents run against real PostgreSQL in integration tests.

## Conventions

- pytest-asyncio auto mode; markers `unit`, `integration`, `e2e`, `security`,
  `evaluation` registered in `pyproject.toml`.
- `tests/conftest.py` sets environment before app imports (settings cache),
  truncates tables, and seeds demo data per test session.
- Coverage config targets `app/` (see `[tool.coverage.run]`).
