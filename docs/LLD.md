# Low-Level Design (LLD)

## 1. Module Map

```
app/
├── main.py                 FastAPI app factory, lifespan, exception handlers
├── api/
│   ├── dependencies.py     DI container (providers, registry, graph, tools) + auth/rate-limit deps
│   ├── middleware.py       request id, metrics, body-size guard, access logs
│   └── routes/             chat.py (JSON + SSE), conversations.py, auth.py, health.py
├── agents/
│   ├── base.py             BaseAgent (run + timed_run w/ metrics + AgentRunRecord)
│   ├── router.py           RouterAgent — structured intent classification
│   ├── search.py           SearchAgent — query prep → SearchTool → SearchResult
│   ├── database.py         DatabaseAgent — authz check → QueryPlan → DatabaseTool
│   ├── general.py          GeneralAgent — fallback LLM conversation
│   ├── synthesizer.py      SynthesizerAgent — final answer, blocking + streaming
│   └── registry.py         AgentRegistry (name/intent → agent)
├── graph/
│   ├── state.py            GraphState TypedDict (reducers: operator.add)
│   ├── routing.py          policy_node — deterministic decision + authz re-route
│   └── graph.py            node wiring; build_graph(registry, checkpointer)
├── core/
│   ├── errors.py           AppError taxonomy (code/status/retryable)
│   ├── retry.py            with_retries (bounded exponential backoff + jitter)
│   ├── prompts.py          prompt loader + content-hash versions
│   └── redis_utils.py      client, ping, fail-open JSON cache
├── db/
│   ├── engine.py           engines/session factories (request pool vs checkpoint pool)
│   ├── models.py           ORM: users, conversations, messages, memories, agent_runs, checkpoints*
│   └── repository.py       data access; optimistic-concurrency append_turn
├── llm/
│   ├── client.py           LLMProvider protocol; OpenAIProvider; usage contextvar; cost model
│   ├── structured.py       structured_complete: JSON schema + Pydantic validation + 1 repair
│   └── fake.py             FakeLLMProvider (dev/load-test/eval; forbidden in production)
├── memory/
│   ├── interface.py        protocols + deterministic explicit-memory rules
│   ├── conversation.py     short-term window + persistent history (ConversationMemory)
│   ├── semantic.py         embeddings store + cosine recall + write policy
│   └── checkpoint.py       PostgresCheckpointSaver (BaseCheckpointSaver, async-only)
├── observability/          logging.py (structlog), metrics.py (prometheus), tracing.py (OTel)
├── schemas/                chat.py, agents.py, conversations.py, common.py (Pydantic v2)
├── security/               auth.py (JWT/API keys), authorization.py, rate_limit.py, locks.py
├── services/               chat.py (ChatService), conversation.py (ConversationService)
├── tools/
│   ├── policies.py         RoutingPolicy, injection/secret detectors, sanitizers
│   ├── search.py           SearchTool (httpx, retries, structured results)
│   └── database.py         DatabaseTool (plan validation, SQL compile, read-only exec)
└── config/settings.py      pydantic-settings, runtime security validation
```

## 2. Key Interfaces

```python
class LLMProvider(Protocol):
    async def complete(messages, *, model=None, temperature=0.2, max_tokens=1024,
                       response_schema=None) -> LLMResponse
    def stream(messages, *, ...) -> AsyncIterator[TokenChunk]
    async def embed(texts) -> list[list[float]]

class BaseAgent(abc.ABC):
    name: str
    async def run(self, state: GraphState) -> GraphState
    async def timed_run(self, state) -> tuple[GraphState, AgentRunRecord]

class AgentRegistry:
    register(agent, intents) / resolve(name) / resolve_intent(intent) / fallback

class MemoryLayer(Protocol):        # semantic
    async def recall(user_id, query, top_k) -> list[RetrievedMemory]
    async def remember(user_id, conversation_id, items) -> int
```

## 3. Graph

- State: `GraphState` (TypedDict). Reducers: `errors`, `agent_runs`,
  `streamed_tokens` append (`operator.add`).
- Nodes: `router` → `policy` → `specialist` → `synthesizer` (edges fixed).
- `policy` node: applies `RoutingPolicy` (threshold from
  `ROUTER_CONFIDENCE_THRESHOLD`); re-routes database requests from
  unauthorized roles; records `router_fallback_total{reason}`.
- `specialist` node: resolves `selected_agent` from the registry; on agent
  exception falls back to the fallback agent and records both run records.
- `synthesizer` node: blocking `complete()` or, when
  `config.configurable.token_sink` is set, streams chunks to the sink and
  aggregates `final_response`.
- Compiled with `PostgresCheckpointSaver`; `thread_id = conversation_id`.

## 4. Concurrency Control (one turn per conversation)

1. **Redis lock** (`lock:conversation:{id}`, SET NX PX 120s, compare-and-delete
   release) — fail-closed: if Redis is down → 409.
2. **Optimistic versioning** — `conversations.version` is checked inside
   `append_turn` (row locked with `SELECT … FOR UPDATE`); version mismatch →
   `ConflictError` (409). The lock is an optimization; the version check is the
   durable guarantee.

## 5. Database Tool Security Flow

```
user role ─▶ check_authorized(roles allowlist)      # app code, pre-LLM
LLM ─▶ QueryPlan (Pydantic: table/columns/filters/limit)
        ─▶ validate_plan: table+columns+order_by whitelists (TABLE_COLUMNS),
           filter values checked for injection/secret patterns
        ─▶ compile_plan: parameterized SELECT (bound values, quoted identifiers)
        ─▶ execute: read-only connection (default_transaction_read_only=on,
           statement_timeout), row limit, sensitive-column scrub
```

The LLM never sees or produces SQL; write verbs are impossible (whitelist of
SELECT construction only + read-only role enforced by PostgreSQL itself).

## 6. Error Model

`AppError` subclasses carry `status_code`, `code`, `retryable`. Exception
handlers in `main.py` render `{error: {code, message, request_id, details?}}`
with correct status codes (401/403/404/409/413/422/429/502/503). Unhandled
exceptions → 500 with no internal detail. SSE errors emit a generic `error`
event; details stay in logs.

## 7. Observability Contracts

- **Logs** (structlog JSON): ts, level, logger, event, request_id,
  conversation_id, user_id, latency, status, error_type. Secret-named keys are
  redacted at the processor level.
- **Metrics** (Prometheus): `http_*`, `agent_*`, `router_*` (confidence
  histogram, intent distribution, fallback total), `llm_*` (tokens, latency,
  errors, cost USD), `tool_*`, `chat_turns_total`,
  `conversation_lock_conflicts_total`, `rate_limit_rejections_total`,
  `sse_active_streams`. Labels are low-cardinality only (no user/query IDs).
- **Traces** (OTel): spans `llm.openai.{complete,stream,embed}`, `agent.*`,
  `tool.search.request`, `tool.database.execute`; safe attributes only.
