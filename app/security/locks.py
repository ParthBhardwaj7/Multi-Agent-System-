"""Distributed conversation lock (Redis SET NX PX + compare-and-delete).

A turn on a conversation must not interleave with another concurrent turn.
Primary durability guarantee is PostgreSQL optimistic versioning; this lock
prevents wasted concurrent work and returns 409 early.

Acquisition policy on Redis failure: fail-closed for correctness — we surface
409 instead of risking overlapping writes. PostgreSQL versioning remains the
authoritative backstop."""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.core.errors import ConflictError
from app.core.redis_utils import get_redis
from app.observability import metrics
from app.observability.logging import get_logger

logger = get_logger(__name__)

_LOCK_TTL_MS = 120_000  # hard upper bound for one turn


@asynccontextmanager
async def conversation_lock(conversation_id: str, *, wait_ms: int = 0) -> AsyncIterator[bool]:
    """Acquire the conversation lock or raise ConflictError(409)."""
    key = f"lock:conversation:{conversation_id}"
    token = uuid.uuid4().hex
    acquired = False
    try:
        client = get_redis()
        if wait_ms > 0:
            elapsed = 0
            step = 25
            while elapsed <= wait_ms and not acquired:
                acquired = bool(await client.set(key, token, nx=True, px=_LOCK_TTL_MS))
                if not acquired:
                    import asyncio

                    await asyncio.sleep(step / 1000)
                    elapsed += step
        else:
            acquired = bool(await client.set(key, token, nx=True, px=_LOCK_TTL_MS))
    except Exception as exc:  # noqa: BLE001
        logger.warning("conversation_lock_unavailable", error_type=type(exc).__name__)
        raise ConflictError("conversation lock unavailable; try again") from None

    if not acquired:
        metrics.conversation_lock_conflicts_total.inc()
        raise ConflictError("another turn is in progress on this conversation")

    try:
        yield True
    finally:
        try:
            # Compare-and-delete: only remove our own lock (Lua for atomicity).
            await get_redis().eval(
                """
                if redis.call('GET', KEYS[1]) == ARGV[1] then
                  return redis.call('DEL', KEYS[1])
                else
                  return 0
                end
                """,
                1,
                key,
                token,
            )
        except Exception as exc:  # noqa: BLE001 - TTL is the safety net
            logger.warning("conversation_lock_release_failed", error_type=type(exc).__name__)
