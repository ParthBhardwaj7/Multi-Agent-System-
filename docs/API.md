# API Reference

Interactive OpenAPI docs: `/docs` (Swagger UI) and `/redoc` when the API runs.
Base URL: `http://localhost:8000`. All errors share one envelope:

```json
{ "error": { "code": "not_found", "message": "Resource not found.",
             "request_id": "b3f2…", "details": null } }
```

| Status | Codes |
|---|---|
| 401 | `authentication_error` |
| 403 | `authorization_error`, `tool_authorization_error` |
| 404 | `not_found` |
| 409 | `conflict` (concurrent turn on the same conversation) |
| 413 | `request_too_large` |
| 422 | `validation_error` |
| 429 | `rate_limit_exceeded` (+ `Retry-After`) |
| 502/503 | `llm_error`, `tool_error`, `database_error` |

Every response carries `X-Request-ID`; rate-limited responses carry
`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.

## Authentication

Two accepted schemes (never anonymous):

```bash
# 1. API key header (machine clients)
curl -H "X-API-Key: mao_…" …

# 2. JWT bearer (browser clients) — exchange the key once:
curl -s -X POST localhost:8000/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"api_key": "mao_..."}'
# → {"access_token": "eyJ...", "token_type": "bearer", "expires_at": "...",
#    "user_id": "...", "role": "user"}
```

`GET /v1/auth/me` validates credentials and returns the principal.

## Chat

### `POST /v1/chat` (blocking)

Request:

```json
{ "message": "What does our internal document say about the return policy?",
  "conversation_id": null }
```

- `message`: 1–`MAX_MESSAGE_LENGTH` (default 8000) chars, not blank.
- `conversation_id`: optional; omit to start a new conversation.

Response 200:

```json
{ "conversation_id": "7130131c-…",
  "response": "Items can be returned within 30 days…",
  "intent": "database",
  "confidence": 0.9,
  "selected_agent": "database",
  "routed_to_specialist": true,
  "usage": {"input_tokens": 560, "output_tokens": 30} }
```

### `POST /v1/chat/stream` (SSE)

Same request body. Response is `text/event-stream` with ordered events:

```text
event: router_started      data: {"conversation_id": "..."}
event: router_completed    data: {"intent": "search", "confidence": 0.9}
event: agent_started       data: {"agent": "search"}
event: agent_completed     data: {"agent": "search", "status": "error"}
event: token               data: {"content": "You"}
event: token               data: {"content": " have"}
event: done                data: {"conversation_id": "...", "intent": "search",
                                  "confidence": 0.9, "selected_agent": "search",
                                  "latency_ms": 1234.5}
```

On failure an `error` event (generic message + request id) is followed by
`done`. Client disconnects cancel the turn server-side.

## Conversations

| Endpoint | Description |
|---|---|
| `GET /v1/conversations?limit=50&offset=0` | list own conversations (`{items, total}`) |
| `GET /v1/conversations/{id}` | conversation metadata (403 for other users' IDs) |
| `GET /v1/conversations/{id}/messages` | full message history (ordered) |
| `DELETE /v1/conversations/{id}` | delete conversation + messages + checkpoint thread |

Ownership is enforced on every route: another user's conversation id yields
**403**, an unknown id **404**.

## Operations

| Endpoint | Description |
|---|---|
| `GET /health` | liveness; no external dependencies |
| `GET /ready` | `postgres` + `redis` + `llm_provider` checks; `status: ready/not_ready` |
| `GET /metrics` | Prometheus exposition (unauthenticated; keep internal) |

## Rate Limits

Sliding window per minute, both scopes enforced per request:
`RATE_LIMIT_PER_MINUTE` (per user, default 30) and
`RATE_LIMIT_IP_PER_MINUTE` (per client IP, default 120). Exceeding either
returns 429 with `Retry-After`. If Redis is unavailable the limiter follows
`RATE_LIMIT_FAIL_OPEN` (default `false` = refuse with 429).

## Concurrency

Two simultaneous turns on one conversation: exactly one succeeds; the other
gets `409 conflict` (Redis conversation lock; PostgreSQL version check as the
durable backstop). Safe to retry the request afterward.
