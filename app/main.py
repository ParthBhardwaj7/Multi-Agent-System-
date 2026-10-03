"""Application entrypoint: FastAPI app factory, middleware, error handlers,
lifespan (logging, tracing, config validation)."""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.middleware import RequestContextMiddleware
from app.api.routes import auth, chat, conversations, health
from app.config.settings import get_settings
from app.core.errors import AppError, RateLimitError
from app.observability.logging import configure_logging, get_logger, request_id_var
from app.observability.tracing import setup_tracing

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.app_env != "local")
    setup_tracing()

    problems = settings.validate_runtime_security()
    for problem in problems:
        logger.critical("security_configuration_problem", detail=problem)
    if problems and settings.is_production:
        raise RuntimeError(f"Refusing to start in production: {problems}")

    logger.info(
        "startup_complete",
        app_env=settings.app_env,
        model=settings.openai_model,
        metrics_enabled=settings.metrics_enabled,
        tracing_enabled=settings.otel_traces_enabled,
    )
    yield
    from app.api.dependencies import shutdown_container

    await shutdown_container()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Multi-Agent Orchestrator",
        version="1.0.0",
        description="Production-grade multi-agent AI orchestration platform.",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "X-API-Key", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(conversations.router)

    _register_exception_handlers(app)
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        request_id = (
            request_id_var.get("") or getattr(request.state, "request_id", "") or uuid.uuid4().hex
        )
        headers = {}
        if isinstance(exc, RateLimitError):
            headers["Retry-After"] = str(exc.retry_after)
        headers.update(getattr(request.state, "rate_limit_headers", {}) or {})
        logger.warning(
            "app_error",
            code=exc.code,
            status=exc.status_code,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": request_id,
                    "details": exc.details or None,
                }
            },
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = (
            request_id_var.get("") or getattr(request.state, "request_id", "") or uuid.uuid4().hex
        )
        # exc.errors() can contain non-serializable exception objects — sanitize.
        fields = []
        for error in exc.errors()[:10]:
            fields.append(
                {
                    "loc": [str(part) for part in error.get("loc", [])],
                    "msg": str(error.get("msg", "")),
                    "type": str(error.get("type", "")),
                }
            )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed.",
                    "request_id": request_id,
                    "details": {"fields": fields},
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = (
            request_id_var.get("") or getattr(request.state, "request_id", "") or uuid.uuid4().hex
        )
        # Full detail in logs, nothing sensitive to the client.
        logger.exception(
            "unhandled_error",
            path=request.url.path,
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An internal error occurred.",
                    "request_id": request_id,
                }
            },
        )


app = create_app()
