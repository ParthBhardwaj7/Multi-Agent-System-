"""Redis client management.

Failure policies (documented in docs/ARCHITECTURE.md §redis):
- cache: fail-open (a cache miss just recomputes)
- rate limiting: configurable fail-open/fail-closed (RATE_LIMIT_FAIL_OPEN)
- conversation lock: fail-closed for correctness — if Redis is unavailable,
  optimistic versioning in PostgreSQL still protects consistency; a lock
  acquisition failure yields 409 rather than overlapping writes."""

from typing import Any

import redis.asyncio as redis

from app.config.settings import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)

_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        settings = get_settings()
        _client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=1.5,
            socket_timeout=1.5,
            health_check_interval=30,
        )
    return _client


def set_redis(client: redis.Redis | None) -> None:
    """Swap the process-wide client (tests)."""
    global _client
    _client = client


async def ping() -> tuple[bool, float]:
    """Returns (ok, latency_ms). Never raises."""
    import time

    start = time.monotonic()
    try:
        await get_redis().ping()
        return True, round((time.monotonic() - start) * 1000, 1)
    except Exception as exc:  # noqa: BLE001 - readiness reports, doesn't crash
        logger.warning("redis_ping_failed", error_type=type(exc).__name__)
        return False, round((time.monotonic() - start) * 1000, 1)


async def cache_get_json(key: str) -> Any | None:
    """Fail-open cache read: returns None on any error."""
    try:
        raw = await get_redis().get(key)
        if raw is None:
            return None
        import json

        return json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.debug("cache_get_failed", error_type=type(exc).__name__)
        return None


async def cache_set_json(key: str, value: Any, ttl_seconds: int) -> bool:
    """Fail-open cache write."""
    try:
        import json

        await get_redis().set(key, json.dumps(value, default=str), ex=ttl_seconds)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("cache_set_failed", error_type=type(exc).__name__)
        return False
