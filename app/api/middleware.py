"""HTTP middleware: request IDs, metrics, access logs, body-size guard."""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.observability import metrics
from app.observability.logging import get_logger, request_id_var

logger = get_logger("http")


# Template the path so labels stay low-cardinality: /v1/conversations/{id}
def _endpoint_template(path: str) -> str:
    for prefix in ("/v1/conversations/",):
        if path.startswith(prefix) and len(path) > len(prefix):
            return "/v1/conversations/{conversation_id}"
    return path


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, records metrics/logs, guards request size."""

    MAX_BODY_BYTES = 1_000_000  # 1 MB hard cap on request bodies

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request_id_var.set(request_id)
        request.state.request_id = request_id

        content_length = request.headers.get("content-length")
        if (
            content_length
            and content_length.isdigit()
            and int(content_length) > self.MAX_BODY_BYTES
        ):
            metrics.http_requests_total.labels(
                method=request.method, endpoint="request_too_large", status=413
            ).inc()
            return Response(
                content=(
                    '{"error":{"code":"request_too_large","message":"Request body too large."}}'
                ),
                status_code=413,
                media_type="application/json",
                headers={"X-Request-ID": request_id},
            )

        endpoint = _endpoint_template(request.url.path)
        metrics.http_requests_in_progress.labels(method=request.method).inc()
        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            status = 500
            metrics.http_requests_total.labels(
                method=request.method, endpoint=endpoint, status=status
            ).inc()
            metrics.http_request_duration_seconds.labels(
                method=request.method, endpoint=endpoint
            ).observe(time.monotonic() - start)
            logger.exception(
                "request_failed",
                extra={"http_method": request.method, "path": request.url.path},
            )
            raise
        finally:
            metrics.http_requests_in_progress.labels(method=request.method).dec()

        status = response.status_code
        latency = time.monotonic() - start
        metrics.http_requests_total.labels(
            method=request.method, endpoint=endpoint, status=status
        ).inc()
        metrics.http_request_duration_seconds.labels(
            method=request.method, endpoint=endpoint
        ).observe(latency)
        response.headers["X-Request-ID"] = request_id
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=status,
            latency_ms=round(latency * 1000, 1),
        )
        return response
