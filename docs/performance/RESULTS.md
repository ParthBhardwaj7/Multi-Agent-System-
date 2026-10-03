# Load Test Results — Multi-Agent Orchestrator

**Measured, not estimated.** Numbers below come from actual Locust runs against a
locally running API (`uvicorn`, 1 worker) on the development machine
(Windows 11, Docker Desktop Postgres 16 + Redis 7), with `LLM_FAKE_MODE=true`
(deterministic fake LLM — measures orchestration/persistence overhead, NOT
OpenAI latency). Raw Locust CSVs are committed alongside this file.

Environment: `APP_ENV=local`, rate limits raised for the run, otherwise stock
configuration (DB pool 20+10, dedicated checkpointer pool 5+10).

## Results

| Scenario | Users | Duration | Chat reqs | Chat RPS | Errors | Chat p50 | Chat p95 | Chat p99 |
|---|---|---|---|---|---|---|---|---|
| Light | 10 | 25s | 114 | 4.62 | **0%** | 700ms | 2.2s | 2.4s |
| Medium | 50 | 40s | 267 | 6.74 | **0%** | 5.2s | 7.7s | 8.5s |
| Heavy | 100 | 40s | 253 | 6.40 | **0%** | 9.6s | 18s | 20s |

Supporting endpoints during the 100-user run: `/health` p50 55ms (0 errors),
`GET /v1/conversations` p50 4.8s (0 errors), `POST /v1/auth/token` p50 1.1s
(0 errors).

## Interpretation

- **Zero failed requests at every tier.** Backpressure manifests as latency, not
  errors: the connection pool and per-conversation locks queue work instead of
  failing it.
- **Saturation point of this single-worker dev setup is ~6-7 chat RPS.** Each
  turn performs: router + specialist + synthesizer LLM calls (faked, ~0ms),
  4-5 checkpoint writes, memory window load, semantic recall, and a locked,
  version-checked persist. Beyond saturation, latency grows roughly linearly
  with concurrent users.
- Latency at 100 users is dominated by queueing on the single uvicorn worker
  and the 20-connection request pool — horizontal scaling (multiple API
  replicas, documented in docs/DEPLOYMENT.md) is the intended remedy, not
  bigger pools.
- An earlier 50-user run with a shared pool for request sessions and checkpoint
  writes deadlocked the server (pool starvation). The checkpointer now uses a
  dedicated pool; that failure mode is fixed and covered by these runs.

## Caveats

- Fake LLM mode removes OpenAI latency (typically 0.5-3s per call in
  production); real-world p50 will be higher and dominated by model latency.
- Single dev machine, single worker, Dockerized databases. Production numbers
  require the documented Cloud Run/Kubernetes deployment with per-instance
  workers configured.
- Rate limiting was disabled for these runs (limits raised) to measure
  orchestration capacity rather than the limiter. A separate run with stock
  limits (30 req/min/user) confirmed correct 429 behavior with `Retry-After`
  headers at >30 RPS for a single user.

Reproduce:

```bash
make load-test   # starts services, seeds a load-test user, runs locust
```
