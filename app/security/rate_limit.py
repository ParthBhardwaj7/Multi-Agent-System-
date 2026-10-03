"""Redis-backed sliding-window rate limiting (per-user and per-IP).

Uses an atomic Lua script (INCR + expire inside Redis) so the counter and TTL
are consistent under concurrency. On Redis failure the configured fail-open /
fail-closed policy applies (docs/SECURITY.md §rate-limiting)."""

from app.config.settings import get_settings
from app.core.errors import RateLimitError
from app.core.redis_utils import get_redis
from app.observability import metrics
from app.observability.logging import get_logger

logger = get_logger(__name__)

# KEYS[1]=counter key; ARGV: window_seconds, limit
_LUA = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
end
local ttl = redis.call('TTL', KEYS[1])
return {current, ttl}
"""


class RateLimiter:
    def __init__(self, fail_open: bool | None = None) -> None:
        settings = get_settings()
        self.fail_open = settings.rate_limit_fail_open if fail_open is None else fail_open

    async def check(
        self,
        *,
        scope: str,
        identifier: str,
        limit: int | None = None,
        window_seconds: int = 60,
    ) -> tuple[bool, int, int, int]:
        """Returns (allowed, limit, remaining, retry_after_seconds)."""
        limit = limit or get_settings().rate_limit_per_minute
        key = f"rl:{scope}:{identifier}:{window_seconds}"
        try:
            redis_client = get_redis()
            current, ttl = await redis_client.eval(_LUA, 1, key, window_seconds, limit)
            current = int(current)
            ttl = max(1, int(ttl))
            remaining = max(0, limit - current)
            return (current <= limit, limit, remaining, ttl if current > limit else 0)
        except Exception as exc:  # noqa: BLE001 - policy-controlled degradation
            logger.warning("rate_limiter_unavailable", error_type=type(exc).__name__)
            if self.fail_open:
                return (True, limit, limit, 0)
            raise RateLimitError("rate limiter unavailable", retry_after=window_seconds) from None

    async def enforce_or_raise(
        self,
        *,
        scope: str,
        identifier: str,
        limit: int | None = None,
        window_seconds: int = 60,
    ) -> tuple[int, int, int]:
        allowed, limit, remaining, retry_after = await self.check(
            scope=scope, identifier=identifier, limit=limit, window_seconds=window_seconds
        )
        if not allowed:
            metrics.rate_limit_rejections_total.labels(scope=scope).inc()
            raise RateLimitError(retry_after=max(1, retry_after))
        return limit, remaining, retry_after


RATE_LIMITER = RateLimiter()


def rate_limit_headers(limit: int, remaining: int, reset_seconds: int) -> dict[str, str]:
    return {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(max(0, reset_seconds)),
    }
