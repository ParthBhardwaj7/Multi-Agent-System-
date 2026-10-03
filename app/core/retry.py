"""Retry policies for external dependencies.

Transient failures (timeouts, 429/5xx, connection errors) are retried with
exponential backoff and jitter, bounded by a maximum attempt count.
Permanent failures (auth/validation) are never retried.

Callers must raise `AppError` subclasses (with `retryable` set) or standard
timeout/connection exceptions; `with_retries` re-raises the last original
exception when attempts are exhausted.
"""

from collections.abc import Awaitable, Callable
from typing import TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.errors import AppError

T = TypeVar("T")

# httpx exception class names treated as transient.
_TRANSIENT_TRANSPORT_ERRORS = {
    "ReadTimeout",
    "ConnectTimeout",
    "ConnectError",
    "WriteTimeout",
    "PoolTimeout",
    "RemoteProtocolError",
}


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, AppError):
        return exc.retryable
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return type(exc).__name__ in _TRANSIENT_TRANSPORT_ERRORS


async def with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    initial_wait: float = 0.25,
    max_wait: float = 4.0,
) -> T:
    """Run an async operation with bounded exponential backoff + jitter.

    Non-retryable exceptions propagate immediately; retryable ones retry up to
    `max_attempts` total executions, then the last exception is re-raised.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=initial_wait, max=max_wait),
        retry=retry_if_exception(is_retryable),
        reraise=True,
    ):
        with attempt:
            return await operation()
    raise AssertionError("unreachable")  # pragma: no cover
