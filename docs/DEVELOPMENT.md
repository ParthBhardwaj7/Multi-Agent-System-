# Development Guide

## Prerequisites

- Python 3.11 (3.13 works; 3.11 is the CI target)
- Docker Desktop (for Postgres/Redis; also builds the production image)
- Git

## First-Time Setup

```bash
# Windows
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"

# POSIX
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

cp .env.example .env        # then edit values
make infra-up               # postgres :5434 + redis :6390 (or use compose)
make migrate                # alembic upgrade head
make seed                   # demo tables + local admin user
make dev                    # uvicorn with reload → http://localhost:8000/docs
```

> No OpenAI key? Keep `LLM_FAKE_MODE=true` for a fully working local system
> (deterministic fake model; refused when `APP_ENV=production`).

## Daily Workflow

| Task | Command |
|---|---|
| Run API (reload) | `make dev` |
| Lint + format check | `make lint` |
| Auto-format | `make format` |
| Type check | `make typecheck` |
| Unit tests | `make test-unit` |
| Integration/e2e/security tests | `make test-integration` |
| Everything | `make test` |
| Evaluation (heuristic) | `make eval` |
| New migration | `python -m alembic -c migrations/alembic.ini revision --autogenerate -m "…"` then review |
| Apply migrations | `make migrate` |
| Full stack | `make docker-up` / `make docker-down` |
| Security scans | `make security-scan` (bandit + pip-audit) |
| Load test | start API, then `make load-test LOADTEST_API_KEY=…` |

## Code Organization Rules

- **Layering**: `api/ → services/ → (agents/, graph/, memory/, tools/) →
  (db/, llm/, core/)`. Nothing below may import upward; routes never touch
  repositories directly.
- **LLM access** only through `LLMProvider` — never instantiate OpenAI
  clients in agents.
- **SQL** only in `app/db/` and the `DatabaseTool` compiler. The LLM never
  writes SQL.
- **Errors**: raise `AppError` subclasses with correct `retryable`; handlers
  in `main.py` own HTTP mapping.
- **Prompts** live in `prompts/*.txt` (git is the version record);
  `prompt_version()` content-hashes them into `agent_runs`.
- **Config** via `Settings` only — no `os.environ` reads scattered around.

## Adding a New Agent

1. Create `app/agents/billing.py`:

   ```python
   class BillingAgent(BaseAgent):
       name = "billing"
       async def run(self, state: GraphState) -> GraphState:
           ...
           return {"specialist_result": SpecialistResult(agent=self.name, intent="general", ...)}
   ```

2. Register in `app/api/dependencies.get_agent_registry`:
   `registry.register(BillingAgent(context), intents=["billing"])`.
3. Extend `IntentName` in `app/schemas/chat.py` and the router prompt's
   intent list (`prompts/router.txt`).
4. The policy node, graph topology, and persistence need **no changes**.
5. Add evaluation cases (search/database-style slices) and rerun `make eval`.

## Adding a Migration

```bash
python -m alembic -c migrations/alembic.ini revision --autogenerate -m "add x"
```

Review the generated file: demo tables (`documents`, `faqs`) are
application-managed — never drop them from a schema migration; add
`server_default` for NOT NULL columns on populated tables.

## Prompt Changes

1. Edit `prompts/<agent>.txt`.
2. `pytest tests/unit` (prompt content is loaded, not hardcoded in tests).
3. `make eval` and compare `docs/evaluation/results-*.json` routing metrics.
4. Commit — the content hash recorded in `agent_runs` changes automatically.

## Debugging Tips

- `APP_ENV=local LOG_LEVEL=DEBUG` gives readable logs; set
  `OTEL_TRACES_ENABLED=true` to print spans to console when no collector is
  configured.
- `GET /ready` shows per-dependency status; `docker compose logs api` for
  container logs.
- For DB inspection: `docker exec -it mao-postgres psql -U orchestrator`.

## CI

`.github/workflows/ci.yml` runs: ruff, mypy, unit tests, integration/e2e/
security tests (service containers), bandit + pip-audit, heuristic
evaluation, and a Docker build. Keep it green — all gates are required
before merge.
