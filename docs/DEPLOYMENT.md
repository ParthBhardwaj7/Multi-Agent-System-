# Deployment

The application is cloud-agnostic: API instances are stateless; all state
lives in PostgreSQL and Redis. Core code contains no cloud-specific APIs.

## Reference Topology

```
Cloudflare / Load Balancer → API Gateway → Cloud Run (or K8s) instances
                                              ├─ PostgreSQL (managed, e.g. Cloud SQL / RDS)
                                              ├─ Redis (managed, e.g. Memorystore / ElastiCache)
                                              └─ OpenAI (or LLM gateway)
```

## Environment & Secrets

Copy `.env.example` and provide at minimum:

| Variable | Notes |
|---|---|
| `APP_ENV=production` | activates startup security validation |
| `OPENAI_API_KEY` | required (fake mode is refused in production) |
| `JWT_SECRET` | ≥32 random chars (`secrets.token_urlsafe(48)`) |
| `DATABASE_URL` / `DATABASE_RO_URL` | app role + read-only AI-tool role |
| `REDIS_URL` | managed Redis |
| `SEARCH_API_URL` / `SEARCH_API_KEY` | external search (else search reports unavailable) |
| `OTEL_*`, `CORS_ORIGINS`, rate-limit & routing knobs | per environment |

Inject secrets via the platform secret manager (Cloud Run secrets, K8s
Secrets/External Secrets); never bake them into images.

## Docker (image)

`docker/Dockerfile` — multi-stage, `python:3.11-slim`, non-root `appuser`,
no secrets, healthcheck on `/health`, SIGTERM graceful shutdown
(`--timeout-graceful-shutdown 20`), 2 uvicorn workers.

```bash
docker build -f docker/Dockerfile -t multi-agent-orchestrator:1.0.0 .
```

## Docker Compose (single host)

`docker/docker-compose.yml` — api, postgres (with read-only-role init
script), redis, prometheus, grafana.

```bash
cp .env.example .env     # fill values
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml exec api \
    python -m alembic -c migrations/alembic.ini upgrade head
docker compose -f docker/docker-compose.yml exec api python scripts/seed_local.py
```

## Cloud Run

```bash
gcloud run deploy orchestrator \
  --image multi-agent-orchestrator:1.0.0 \
  --min-instances 1 --max-instances 10 \
  --cpu 1 --memory 1Gi --concurrency 40 \
  --set-secrets OPENAI_API_KEY=openai:latest,JWT_SECRET=jwt:latest \
  --set-env-vars APP_ENV=production,\
DATABASE_URL=postgresql+asyncpg://…,DATABASE_RO_URL=postgresql+asyncpg://…,\
REDIS_URL=redis://…
```

- Migrations run as a one-off job (Cloud Run Job or CI step) before rollout —
  never at instance startup.
- Use a Serverless VPC connector (or direct VPC) for Postgres/Redis access.
- SSE: streaming responses work on Cloud Run; ensure the LB/gateway disables
  response buffering (`X-Accel-Buffering: no` header is already sent).

## Kubernetes (sketch)

- Deployment: 2+ replicas, readinessProbe `/ready`, livenessProbe `/health`,
  resources 500m/512Mi requests.
- HPA on CPU or RPS; `sse_active_streams` gauge informs stream capacity.
- Run migrations as a pre-upgrade Job (initContainer or Helm hook).
- Ingress must support streaming (nginx: `proxy_buffering off` for the
  stream route) and pass `X-Request-ID` through.

## Scaling Notes

- Instances scale horizontally; Postgres connection budget =
  (instances × pool) + (instances × checkpointer pool). Defaults: 20+10
  request, 5+10 checkpointer per instance — size `max_connections`
  accordingly or use PgBouncer.
- Redis: single logical store for limits/locks; use managed Redis with
  persistence disabled (no durable state) — conversations never depend on
  Redis for durability.
- Measured single-instance capacity and methodology:
  docs/performance/RESULTS.md (~6–7 chat RPS per dev-laptop worker with
  latency-mode backpressure; production numbers depend on LLM latency).

## Monitoring & Rollback

- Prometheus scrapes `/metrics`; alert set suggested in docs/OBSERVABILITY.md.
- Deploy tag-per-release; rollback = redeploy previous image tag (schema
  migrations are forward-compatible: additive first, destructive later).
- `alembic downgrade -1` is available but treat rollbacks as image-level;
  avoid destructive downgrades on live data.

## Checklist Before Go-Live

- [ ] `APP_ENV=production` + strong `JWT_SECRET` (startup enforces)
- [ ] Migrations applied by job, read-only role provisioned
- [ ] Managed Postgres/Redis reachable from VPC; backups enabled
- [ ] Secrets in secret manager; `.env` absent from images
- [ ] Prometheus + alerting wired; `OTEL_TRACES_ENABLED=true` with collector
- [ ] Rate limits tuned to expected traffic and LLM budget
- [ ] Load test executed against a staging replica
