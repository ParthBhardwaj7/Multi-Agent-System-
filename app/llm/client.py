"""LLM provider abstraction.

The application depends on the `LLMProvider` protocol only; concrete
providers (OpenAI today, others later) are injected via the service layer.
All providers: enforce timeouts, record token usage and estimated cost in
Prometheus metrics, and map provider errors into the AppError taxonomy.
"""

import contextvars
import time
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

import openai
from openai import AsyncOpenAI

from app.config.settings import get_settings
from app.core.errors import (
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from app.core.retry import with_retries
from app.llm.models import ChatMessage, LLMResponse, StreamResult, TokenChunk, Usage
from app.observability import metrics
from app.observability.logging import get_logger
from app.observability.tracing import get_tracer, record_error, safe_set_attributes

logger = get_logger(__name__)
tracer = get_tracer()

# Per-turn usage sink: when set, every LLM call appends its usage here so the
# chat service can report aggregate token usage for the turn.
usage_sink: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "llm_usage_sink", default=None
)


def record_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    sink = usage_sink.get()
    if sink is not None:
        sink.append(
            {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )


_ERROR_TYPE_BY_STATUS = {
    401: LLMError,  # auth failure -> not retryable
    403: LLMError,
    404: LLMError,
    422: LLMResponseError,
    429: LLMRateLimitError,
}


def _map_openai_error(exc: Exception) -> LLMError:
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeoutError()
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimitError()
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return LLMError("LLM authentication failed", retryable=False)
    if isinstance(exc, openai.BadRequestError):
        return LLMError("LLM rejected the request", retryable=False)
    if isinstance(exc, openai.APIStatusError):
        cls = _ERROR_TYPE_BY_STATUS.get(exc.status_code, LLMError)
        retryable = exc.status_code in {429, 500, 502, 503, 504}
        return cls(retryable=retryable)
    if isinstance(exc, openai.APIConnectionError):
        return LLMError("LLM connection failure", retryable=True)
    return LLMError("LLM request failed", retryable=False)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimated USD cost using configured per-model pricing (USD per 1M tokens)."""
    pricing = get_settings().pricing.get(model)
    if not pricing:
        return 0.0
    input_price, output_price = pricing
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


@runtime_checkable
class LLMProvider(Protocol):
    """The only LLM interface the rest of the application may use."""

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse: ...

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[TokenChunk]: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIProvider:
    """OpenAI implementation of LLMProvider (also works with compatible gateways)."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url or settings.openai_base_url or None,
            timeout=settings.openai_timeout_seconds,
            max_retries=0,  # we own retry policy
        )
        self._default_model = settings.openai_model

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        resolved_model = model or self._default_model
        span = tracer.start_span("llm.openai.complete")
        safe_set_attributes(
            span,
            {
                "llm.model": resolved_model,
                "llm.operation": "complete",
                "llm.max_tokens": max_tokens,
            },
        )
        start = time.monotonic()
        try:
            kwargs: dict[str, Any] = {
                "model": resolved_model,
                "messages": [m.model_dump() for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_schema is not None:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "strict": True,
                        "schema": response_schema,
                    },
                }

            def _call() -> Any:
                return self._client.chat.completions.create(**kwargs)

            response = await with_retries(_call, max_attempts=get_settings().openai_max_retries + 1)
            elapsed = time.monotonic() - start
            metrics.llm_request_duration_seconds.labels(
                model=resolved_model, operation="complete"
            ).observe(elapsed)

            choice = response.choices[0]
            usage = getattr(response, "usage", None)
            usage_model = Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
            cost = estimate_cost(
                resolved_model, usage_model.input_tokens, usage_model.output_tokens
            )
            metrics.observe_llm_usage(
                resolved_model, usage_model.input_tokens, usage_model.output_tokens, cost
            )

            content = choice.message.content or ""
            span.set_attribute("llm.input_tokens", usage_model.input_tokens)
            span.set_attribute("llm.output_tokens", usage_model.output_tokens)
            span.set_status(trace_ok())
            record_usage(resolved_model, usage_model.input_tokens, usage_model.output_tokens)
            return LLMResponse(
                content=content,
                model=resolved_model,
                usage=usage_model,
                finish_reason=choice.finish_reason,
            )
        except Exception as exc:
            err = _map_openai_error(exc)
            record_error(span, exc)
            metrics.llm_request_errors_total.labels(model=resolved_model, error_type=err.code).inc()
            logger.warning(
                "llm_complete_failed",
                model=resolved_model,
                error_type=err.code,
                latency_ms=round((time.monotonic() - start) * 1000, 1),
            )
            raise err from exc
        finally:
            span.end()

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[TokenChunk]:
        """Yield token chunks; yields an aggregated StreamResult via collect helper."""
        resolved_model = model or self._default_model
        span = tracer.start_span("llm.openai.stream")
        safe_set_attributes(span, {"llm.model": resolved_model, "llm.operation": "stream"})
        start = time.monotonic()

        async def _gen() -> AsyncIterator[TokenChunk]:
            nonlocal start
            aggregated = StreamResult(content="", model=resolved_model)
            try:
                stream = await self._client.chat.completions.create(
                    model=resolved_model,
                    messages=[m.model_dump() for m in messages],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},  # type: ignore[call-overload]
                )
                async for event in stream:
                    if event.choices:
                        delta = event.choices[0].delta
                        piece = delta.content if delta and delta.content else ""
                        if piece:
                            aggregated.content += piece
                            yield TokenChunk(content=piece)
                        if event.choices[0].finish_reason:
                            aggregated.finish_reason = event.choices[0].finish_reason
                    if getattr(event, "usage", None):
                        aggregated.usage = Usage(
                            input_tokens=event.usage.prompt_tokens or 0,
                            output_tokens=event.usage.completion_tokens or 0,
                        )
                elapsed = time.monotonic() - start
                metrics.llm_request_duration_seconds.labels(
                    model=resolved_model, operation="stream"
                ).observe(elapsed)
                cost = estimate_cost(
                    resolved_model, aggregated.usage.input_tokens, aggregated.usage.output_tokens
                )
                metrics.observe_llm_usage(
                    resolved_model,
                    aggregated.usage.input_tokens,
                    aggregated.usage.output_tokens,
                    cost,
                )
                record_usage(
                    resolved_model, aggregated.usage.input_tokens, aggregated.usage.output_tokens
                )
                span.set_status(trace_ok())
            except Exception as exc:
                err = _map_openai_error(exc)
                record_error(span, exc)
                metrics.llm_request_errors_total.labels(
                    model=resolved_model, error_type=err.code
                ).inc()
                raise err from exc
            finally:
                span.end()

        async for chunk in _gen():
            yield chunk

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        span = tracer.start_span("llm.openai.embed")
        safe_set_attributes(span, {"llm.operation": "embed", "llm.batch_size": len(texts)})
        start = time.monotonic()
        try:
            response = await self._client.embeddings.create(
                model=get_settings().openai_embedding_model,
                input=texts,
            )
            metrics.llm_request_duration_seconds.labels(
                model=get_settings().openai_embedding_model, operation="embed"
            ).observe(time.monotonic() - start)
            usage = getattr(response, "usage", None)
            tokens = getattr(usage, "total_tokens", 0) or 0
            metrics.observe_llm_usage(get_settings().openai_embedding_model, tokens, 0, 0.0)
            span.set_status(trace_ok())
            return [item.embedding for item in response.data]
        except Exception as exc:
            err = _map_openai_error(exc)
            record_error(span, exc)
            metrics.llm_request_errors_total.labels(
                model=get_settings().openai_embedding_model, error_type=err.code
            ).inc()
            raise err from exc
        finally:
            span.end()


def trace_ok() -> Any:
    from opentelemetry.trace import Status, StatusCode

    return Status(StatusCode.OK)
