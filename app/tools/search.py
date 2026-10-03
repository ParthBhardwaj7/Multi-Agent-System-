"""External search tool.

Clean tool interface: query in -> validated structured result out. Has its own
timeout, bounded retries, and error mapping. When the tool is not configured
or fails, it returns `SearchResult(status=...)` — it NEVER fabricates results.
"""

import time
from typing import Any, Literal

import httpx

from app.config.settings import get_settings
from app.core.errors import ToolError
from app.core.retry import with_retries
from app.observability import metrics
from app.observability.logging import get_logger
from app.observability.tracing import get_tracer, record_error, safe_set_attributes
from app.schemas.agents import SearchCitation, SearchResult

logger = get_logger(__name__)
tracer = get_tracer()

_ALLOWED_URL_SCHEMES = ("https",)


class SearchTool:
    """HTTP search client for a configurable search API (Tavily-compatible by default).

    Response contract expected: {"results": [{"title", "url", "content"}...]}.
    Any other shape is treated as an error, not silently accepted.
    """

    def __init__(self, api_url: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self.api_url = api_url or settings.search_api_url
        self.api_key = api_key if api_key is not None else settings.search_api_key
        self.timeout = settings.search_timeout_seconds
        self.max_retries = settings.search_max_retries
        self._client: httpx.AsyncClient | None = None

    @property
    def available(self) -> bool:
        return bool(self.api_key) and self.api_url.startswith(_ALLOWED_URL_SCHEMES)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                follow_redirects=False,  # never follow redirects to arbitrary hosts
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def search(self, query: str, max_results: int = 5) -> SearchResult:
        """Execute a web search. Returns a structured, validated result."""
        span = tracer.start_span("tool.search.request")
        safe_set_attributes(span, {"tool.name": "search"})
        start = time.monotonic()
        metrics.tool_requests_total.labels(tool="search", status="started").inc()

        if not self.available:
            metrics.tool_errors_total.labels(tool="search", error_type="unavailable").inc()
            span.set_attribute("tool.status", "unavailable")
            span.end()
            return SearchResult(status="unavailable", query=query, error_type="not_configured")

        client = await self._get_client()
        payload = {"query": query, "max_results": max_results, "search_depth": "basic"}

        async def _call() -> dict[str, Any]:
            response = await client.post(self.api_url, json=payload)
            if response.status_code == 429 or response.status_code >= 500:
                raise ToolError(f"search transient http {response.status_code}", retryable=True)
            if response.status_code >= 400:
                raise ToolError(f"search client error http {response.status_code}", retryable=False)
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("results", []), list):
                raise ToolError("search returned malformed payload", retryable=False)
            return data

        try:
            data = await with_retries(_call, max_attempts=self.max_retries + 1)
            results = []
            for item in data.get("results", [])[:max_results]:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url", ""))
                if not url.startswith(_ALLOWED_URL_SCHEMES):
                    continue
                results.append(
                    SearchCitation(
                        title=str(item.get("title", ""))[:300],
                        url=url,
                        snippet=str(item.get("content", ""))[:1000],
                    )
                )
            status: Literal["ok", "empty"] = "ok" if results else "empty"
            metrics.tool_requests_total.labels(tool="search", status=status).inc()
            metrics.tool_request_duration_seconds.labels(tool="search").observe(
                time.monotonic() - start
            )
            span.set_attribute("tool.status", status)
            span.set_attribute("tool.result_count", len(results))
            return SearchResult(status=status, query=query, citations=results)
        except Exception as exc:
            record_error(span, exc)
            metrics.tool_errors_total.labels(tool="search", error_type="error").inc()
            metrics.tool_request_duration_seconds.labels(tool="search").observe(
                time.monotonic() - start
            )
            logger.warning("search_tool_failed", error_type=type(exc).__name__)
            error_type = (
                "unavailable" if isinstance(exc, ToolError) and not exc.retryable else "error"
            )
            return SearchResult(status="error", query=query, error_type=error_type)
        finally:
            span.end()
