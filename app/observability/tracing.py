"""OpenTelemetry tracing setup.

Tracing is disabled unless OTEL_TRACES_ENABLED=true (and optionally an OTLP
endpoint is configured). All tracer calls are safe no-ops when disabled, so
application code does not branch on configuration.
"""

import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Span, Status, StatusCode

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

_TRACER: trace.Tracer | None = None


def setup_tracing() -> None:
    """Initialize the global tracer provider. Called once at app startup."""
    global _TRACER
    settings = get_settings()

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "deployment.environment": settings.app_env,
        }
    )

    if not settings.otel_traces_enabled:
        # No provider set: `trace.get_tracer` returns a no-op implementation.
        _TRACER = trace.get_tracer(settings.otel_service_name)
        logger.info("OpenTelemetry tracing disabled")
        return

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=resource)

    if settings.otel_exporter_otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor as BatchProcessor

        exporter = OTLPSpanExporter(endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces")
        provider.add_span_processor(BatchProcessor(exporter))
        logger.info(
            "OpenTelemetry OTLP exporter configured",
            extra={"endpoint": settings.otel_exporter_otlp_endpoint},
        )
    else:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.warning("OTLP endpoint not set; traces will print to console")

    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer(settings.otel_service_name)


def get_tracer() -> trace.Tracer:
    global _TRACER
    if _TRACER is None:
        _TRACER = trace.get_tracer(get_settings().otel_service_name)
    return _TRACER


def safe_set_attributes(span: Span, attributes: dict[str, Any]) -> None:
    """Set span attributes, skipping None values and enforcing safe types.

    Callers must never pass secrets, raw user content, or credentials here.
    """
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, bool | int | float | str):
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, str(value))


def record_error(span: Span, error: BaseException) -> None:
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, str(type(error).__name__)))
