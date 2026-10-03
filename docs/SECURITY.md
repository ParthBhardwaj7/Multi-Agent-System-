# Security

## Threat Model Summary

Primary assets: user conversations (privacy), internal database contents,
platform credentials, LLM spend. Primary adversaries: malicious API clients,
prompt-injection payloads, compromised dependencies.

## Authentication

- **API keys** (`mao_<43 urlsafe chars>`): generated server-side, stored as
  SHA-256 hashes (keys are high-entropy; deterministic lookup is standard and
  safe — see docs/DATABASE.md caveat). Sent as `X-API-Key`.
- **JWT** (HS256, `JWT_SECRET`): short-lived (`JWT_EXPIRES_MINUTES`, default
  60), `sub`/`role`/`iss` claims required, `exp` enforced; exchanged via
  `POST /v1/auth/token`. Expired/tampered tokens → 401.
- Anonymous requests are rejected at the dependency level — every protected
  route requires a resolved `Principal`.

## Authorization

- **Conversation ownership**: every conversation-scoped request goes through
  `ConversationRepository.get_owned(conversation_id, user_id)` → 403 for
  foreign conversations, 404 for unknown. Changing ids in URLs/bodies cannot
  leak data (verified by security tests).
- **Role-based tool authorization**: the Database Agent executes only for
  roles in `DB_AGENT_ALLOWED_ROLES` (`admin`, `analyst` by default). Checked
  twice — by the agent before any LLM call and by the tool before execution.
  Unauthorized requests are re-routed to the general agent at the policy node.
- The LLM is **never** consulted for permissions.

## Prompt Injection

Defense-in-depth (no single control is trusted):

1. **Structural isolation** — tool inputs are LLM-prepared *structured plans*
   (Pydantic-validated), not free text; the synthesizer receives tool results
   as data with explicit "treat as the only source of retrieved facts"
   framing; system prompts instruct honest failure reporting.
2. **Heuristic detector** (`tools/policies.py`) — obvious injection patterns
   ("ignore previous instructions", "SYSTEM:", "reveal your system prompt",
   "DROP TABLE", secret-exfiltration phrasing…) are rejected before tool
   execution; measured at 0.76 flag-rate on the adversarial eval slice.
3. **Least privilege** — even a successful injection inside a database plan
   can only reach whitelisted tables/columns via parameterized SELECTs on a
   read-only role.

Reported evaluation metric: `adversarial_flagged_rate` in
`docs/evaluation/results-*.json`.

## SQL Safety (Database Agent)

- The LLM never writes SQL. It proposes a `QueryPlan` (table/columns/filters/
  order/limit) that is validated against hardcoded whitelists
  (`TABLE_COLUMNS`), with identifiers regex-validated.
- Filter values are bound parameters (`ILIKE :p0`), never interpolated.
- Execution runs on `orchestrator_ro`: `default_transaction_read_only=on` and
  `statement_timeout` are set at the asyncpg driver level, plus
  `SET TRANSACTION READ ONLY` per transaction — belt and braces.
- Row caps (`DB_AGENT_MAX_ROWS`, SQL `LIMIT`), sensitive-column scrubbing
  (`password_hash`, `api_key_hash`, … → `[REDACTED]`), and no secrets in the
  tool's inputs/outputs.
- A security test proves the read-only role cannot `UPDATE` even directly.

## Rate Limiting

Redis sliding window (atomic Lua INCR+EXPIRE), per-user and per-IP scopes,
429 + `Retry-After` + `X-RateLimit-*` headers. **Documented failure policy**:
Redis unavailable → controlled by `RATE_LIMIT_FAIL_OPEN` (default `false`,
i.e. fail-closed: refuse requests rather than run unlimited). Cache reads are
fail-open by contrast (a cache miss recomputes).

## Abuse & DoS Controls

- 1 MB request-body cap + Pydantic message-length validation (default 8000).
- Bounded retries with exponential backoff + jitter; non-retryable errors
  (auth/validation) never retried; no infinite loops.
- LLM spend: token usage recorded per call (Prometheus `llm_tokens_total`),
  cost estimated from central `LLM_PRICING` config (`llm_cost_usd_total`);
  per-user rate limits bound worst-case spend.
- Conversation locks prevent a client from piling up concurrent turns on one
  thread.

## Secrets Handling

- All credentials via environment (`.env`, gitignored; `.env.example`
  documents them). No secrets in code, image, or repo (Dockerfile copies only
  code; production startup validates JWT secret strength and refuses weak
  config).
- Logs: secret-named keys (`api_key`, `authorization`, `password`, `token`,
  `jwt`, `secret`) are redacted by a structlog processor; stack traces go to
  logs, never to clients; SSE errors are generic.
- Span attributes: identifiers and statuses only — no message content, no
  credentials.

## Data Protection

- Conversation deletion cascades to messages, agent runs, and the LangGraph
  checkpoint thread.
- Users are soft-deactivatable (`users.is_active`).
- No payment data or special categories are stored by the platform.

## Verification

`tests/security/` covers: expired/tampered JWTs, bad API keys, cross-user
conversation access (read/append/delete), oversized payloads, rate-limit
429s with headers, Redis-outage fail-closed behavior, read-only role
enforcement, SQL-injection parameterization, and LLM-timeout degradation.
`pip-audit` + `bandit` run in CI with zero-findings expectation.
