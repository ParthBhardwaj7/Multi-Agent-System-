"""Process-wide dependency wiring (singleton container).

Constructed lazily on first request; swapped wholesale in tests via the
`set_*` overrides. The LLM provider selection is explicit: an OpenAI-compatible
provider when a key is configured, otherwise a startup FAILURE in production
and a clear error in development (never a silent fake)."""

from collections.abc import AsyncIterator
from functools import lru_cache

from langgraph.graph.state import CompiledStateGraph

from app.agents.base import AgentContext
from app.agents.database import DatabaseAgent
from app.agents.general import GeneralAgent
from app.agents.registry import AgentRegistry
from app.agents.router import RouterAgent
from app.agents.search import SearchAgent
from app.agents.synthesizer import SynthesizerAgent
from app.config.settings import get_settings
from app.core.errors import LLMError
from app.db.engine import create_db_engine, create_session_factory, get_session_factory
from app.db.repository import hash_api_key
from app.graph.graph import build_graph
from app.llm.client import LLMProvider, OpenAIProvider
from app.memory.checkpoint import PostgresCheckpointSaver
from app.security.auth import Principal
from app.tools.database import DatabaseTool
from app.tools.search import SearchTool


@lru_cache
def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    if settings.llm_fake_mode:
        if settings.is_production:
            raise LLMError("fake LLM mode is forbidden in production", retryable=False)
        from app.llm.fake import FakeLLMProvider

        return FakeLLMProvider()
    if settings.openai_api_key:
        return OpenAIProvider()
    raise LLMError(
        "no LLM provider configured: set OPENAI_API_KEY (or run tests/evaluation in fake mode)",
        retryable=False,
    )


@lru_cache
def get_agent_registry() -> AgentRegistry:
    """Build the agent registry: intents -> agents. New agents register here."""
    context = AgentContext(
        provider=get_llm_provider(),
        tools={"search": get_search_tool(), "database": get_database_tool()},
    )
    registry = AgentRegistry(fallback_agent=get_settings().router_fallback_agent)
    registry.register(RouterAgent(context), intents=[])
    registry.register(SearchAgent(context), intents=["search"])
    registry.register(DatabaseAgent(context), intents=["database"])
    registry.register(GeneralAgent(context), intents=["general"])
    registry.register(SynthesizerAgent(context), intents=[])
    return registry


@lru_cache
def get_checkpoint_saver() -> PostgresCheckpointSaver:
    """The checkpointer gets its OWN connection pool: checkpoint writes happen
    while request sessions are held, so sharing one pool deadlocks under load
    (pool starvation). Small dedicated pool keeps the two concerns independent."""
    settings = get_settings()
    engine = create_db_engine(
        settings.database_url,
        pool_size=5,
        max_overflow=10,
    )
    factory = create_session_factory(engine)
    _CHECKPOINT_ENGINES.append(engine)
    return PostgresCheckpointSaver(factory)


_CHECKPOINT_ENGINES: list = []


@lru_cache
def get_orchestration_graph() -> CompiledStateGraph:
    return build_graph(get_agent_registry(), checkpointer=get_checkpoint_saver())


@lru_cache
def get_search_tool() -> SearchTool:
    return SearchTool()


@lru_cache
def get_database_tool() -> DatabaseTool:
    return DatabaseTool()


def reset_container() -> None:
    """Clear cached singletons (tests / config changes)."""
    get_llm_provider.cache_clear()
    get_agent_registry.cache_clear()
    get_checkpoint_saver.cache_clear()
    get_orchestration_graph.cache_clear()
    get_search_tool.cache_clear()
    get_database_tool.cache_clear()


async def shutdown_container() -> None:
    """Release engines/clients at process shutdown."""
    import contextlib

    reset_container()
    for engine in list(_CHECKPOINT_ENGINES):
        with contextlib.suppress(Exception):
            await engine.dispose()
    _CHECKPOINT_ENGINES.clear()
    for closer in (_close_db_engine, _close_database_tool, _close_search_tool, _close_redis):
        with contextlib.suppress(Exception):
            await closer()


async def _close_db_engine() -> None:
    from app.db.engine import _session_factory  # noqa: PLC0415

    if _session_factory is not None:
        engine = _session_factory.kw["bind"]
        await engine.dispose()


async def _close_database_tool() -> None:
    if get_database_tool.cache_info().currsize:  # type: ignore[attr-defined]
        await get_database_tool().aclose()


async def _close_search_tool() -> None:
    if get_search_tool.cache_info().currsize:  # type: ignore[attr-defined]
        await get_search_tool().aclose()


async def _close_redis() -> None:
    from app.core.redis_utils import get_redis, set_redis

    client = get_redis()
    set_redis(None)
    await client.aclose()


__all__ = [
    "get_llm_provider",
    "get_agent_registry",
    "get_checkpoint_saver",
    "get_orchestration_graph",
    "get_search_tool",
    "get_database_tool",
    "reset_container",
    "shutdown_container",
    "hash_api_key",
    "create_db_engine",
    "create_session_factory",
    "get_current_user",
    "enforce_rate_limits",
    "pop_rate_limit_headers",
    "SessionDep",
    "CurrentUser",
]


# ---------------------------------------------------------------------- #
# Per-request dependencies                                                #
# ---------------------------------------------------------------------- #

from typing import Annotated  # noqa: E402

from fastapi import Depends, Header, Request  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from app.core.errors import AuthenticationError, RateLimitError  # noqa: E402
from app.security.rate_limit import RATE_LIMITER, rate_limit_headers  # noqa: E402


async def _db_session() -> AsyncIterator[AsyncSession]:
    factory = get_session_factory()
    async with factory() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(_db_session)]


async def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Principal:
    """Resolve the caller: Bearer JWT first, then X-API-Key. Never anonymous."""
    from app.security.auth import decode_access_token

    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            principal = decode_access_token(token)
            return principal

    if x_api_key:
        from app.db.repository import UserRepository

        async with get_session_factory()() as session:
            user = await UserRepository(session).get_by_api_key(x_api_key)
        if user is None:
            raise AuthenticationError("invalid API key")
        return Principal(user_id=user.id, role=user.role)

    raise AuthenticationError("missing credentials (Authorization: Bearer <jwt> or X-API-Key)")


CurrentUser = Annotated[Principal, Depends(get_current_user)]


async def enforce_rate_limits(request: Request, current_user: CurrentUser) -> None:
    """Per-user and per-IP sliding-window limits; headers attached to responses."""
    settings = get_settings()
    client_ip = _client_ip(request)

    try:
        limit, remaining, reset = await RATE_LIMITER.enforce_or_raise(
            scope="user", identifier=current_user.user_id, limit=settings.rate_limit_per_minute
        )
        _attach_headers(request, rate_limit_headers(limit, remaining, reset))
    except RateLimitError as exc:
        _attach_headers(request, rate_limit_headers(exc.retry_after, 0, exc.retry_after))
        raise

    ip_limit, ip_remaining, ip_reset = await RATE_LIMITER.enforce_or_raise(
        scope="ip", identifier=client_ip, limit=settings.rate_limit_ip_per_minute
    )
    _attach_headers(request, rate_limit_headers(ip_limit, ip_remaining, ip_reset))


RateLimitedDeps = Annotated[None, Depends(enforce_rate_limits)]


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _attach_headers(request: Request, headers: dict[str, str]) -> None:
    existing = getattr(request.state, "rate_limit_headers", {}) or {}
    existing.update(headers)
    request.state.rate_limit_headers = existing


def pop_rate_limit_headers(request: Request) -> dict[str, str]:
    return getattr(request.state, "rate_limit_headers", {}) or {}
