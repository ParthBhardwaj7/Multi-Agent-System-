"""Health, readiness and metrics endpoints."""

import time

from fastapi import APIRouter, Response

from app.config.settings import get_settings
from app.core.redis_utils import ping as redis_ping
from app.schemas.common import HealthResponse, ReadinessCheck, ReadinessResponse

router = APIRouter(tags=["operations"])

APP_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Process liveness — deliberately NOT dependent on external services."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.app_env,
        version=APP_VERSION,
    )


@router.get("/ready", response_model=ReadinessResponse)
async def ready() -> ReadinessResponse:
    """Dependency readiness per the documented policy:
    - PostgreSQL: required -> not ready if unreachable
    - Redis: required for rate limiting/locks -> not ready if unreachable
    """
    checks: list[ReadinessCheck] = []

    ok, latency = await _check_database()
    checks.append(
        ReadinessCheck(name="postgres", status="ok" if ok else "unavailable", latency_ms=latency)
    )

    redis_ok, redis_latency = await redis_ping()
    checks.append(
        ReadinessCheck(
            name="redis", status="ok" if redis_ok else "unavailable", latency_ms=redis_latency
        )
    )

    settings = get_settings()
    llm_configured = bool(settings.openai_api_key) or settings.llm_fake_mode
    checks.append(
        ReadinessCheck(
            name="llm_provider",
            status="ok" if llm_configured else "unconfigured",
            detail=None if llm_configured else "OPENAI_API_KEY not set",
        )
    )

    overall = "ready" if (ok and redis_ok) else "not_ready"
    return ReadinessResponse(status=overall, checks=checks)


async def _check_database() -> tuple[bool, float]:
    from sqlalchemy import text

    from app.db.engine import get_session_factory

    start = time.monotonic()
    try:
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        return True, round((time.monotonic() - start) * 1000, 1)
    except Exception:  # noqa: BLE001 - readiness must report, not crash
        return False, round((time.monotonic() - start) * 1000, 1)


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    from app.observability.metrics import render_metrics

    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)
