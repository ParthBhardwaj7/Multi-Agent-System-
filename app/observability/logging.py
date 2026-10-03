"""Structured JSON logging with correlation IDs (structlog)."""

import contextvars
import logging
import sys
from typing import Any

import structlog

# Correlation context, bound per request.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
conversation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "conversation_id", default=""
)
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="")


def _inject_context(
    _logger: Any, _method: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    for key, var in (
        ("request_id", request_id_var),
        ("conversation_id", conversation_id_var),
        ("user_id", user_id_var),
    ):
        value = var.get()
        if value:
            event_dict.setdefault(key, value)
    return event_dict


def _drop_secret_keys(
    _logger: Any, _method: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    """Defense-in-depth: never emit obviously sensitive keys in logs."""
    banned = {"api_key", "authorization", "password", "secret", "token", "jwt"}
    for key in list(event_dict):
        if key.lower() in banned:
            event_dict[key] = "[REDACTED]"
    return event_dict


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure structlog + stdlib logging once at startup.

    json_output=False renders human-readable console logs (local development).
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _inject_context,
        _drop_secret_keys,
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Uvicorn/lognoise through the same pipeline.
    for noisy in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logging.getLogger(noisy).handlers.clear()
        logging.getLogger(noisy).propagate = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
