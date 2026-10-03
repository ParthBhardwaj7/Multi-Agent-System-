"""Secure database tool used by the Database Agent.

Security model (docs/SECURITY.md):
- The LLM proposes a *QueryPlan* (tables/columns/filters). It never writes SQL.
- The tool validates the plan against a strict whitelist and compiles a
  parameterized SELECT with SQLAlchemy — values are always bound parameters.
- Execution happens on a dedicated READ-ONLY database user whose connections
  set `default_transaction_read_only=on` and `statement_timeout` at the
  driver level, so even a tool bug cannot mutate data or hang the pool.
- Row limits are enforced in SQL and after fetch.
- Authorization is checked here (role allowlist), not by the LLM.
"""

import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config.settings import get_settings
from app.core.errors import ToolValidationError
from app.observability import metrics
from app.observability.logging import get_logger
from app.observability.tracing import get_tracer, record_error, safe_set_attributes
from app.schemas.agents import DatabaseResult, QueryPlan
from app.tools.policies import contains_prompt_injection, contains_secret_material

logger = get_logger(__name__)
tracer = get_tracer()

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


@dataclass(frozen=True)
class AuthorizationVerdict:
    allowed: bool
    reason: str = ""


# Column whitelist per table. Anything not listed here is inaccessible.
TABLE_COLUMNS: dict[str, set[str]] = {
    "documents": {"id", "title", "content", "category", "updated_at"},
    "faqs": {"id", "question", "answer", "category", "updated_at"},
}

# Columns whose raw values must never be returned to the model or user.
SENSITIVE_COLUMNS = {"password_hash", "api_key_hash", "secret", "token"}


def validate_identifier(name: str) -> str:
    if not _IDENTIFIER.match(name):
        raise ToolValidationError(f"invalid identifier: {name!r}")
    return name


class DatabaseTool:
    """Executes validated query plans against the internal database, read-only."""

    def __init__(self, engine: AsyncEngine | None = None, engine_url: str | None = None) -> None:
        settings = get_settings()
        if engine is not None:
            self._engine = engine
            self._owns_engine = False
        else:
            self._engine = create_async_engine(
                engine_url or settings.database_ro_url,
                pool_size=3,
                max_overflow=2,
                pool_pre_ping=True,
                connect_args=self._connect_args(settings.db_agent_query_timeout_ms),
            )
            self._owns_engine = True
        self.max_rows = settings.db_agent_max_rows
        self.allowed_tables = set(settings.db_agent_tables)
        self.query_timeout_ms = settings.db_agent_query_timeout_ms

    @staticmethod
    def _connect_args(timeout_ms: int) -> dict[str, Any]:
        # asyncpg server settings: hard enforcement, independent of application code.
        return {
            "server_settings": {
                "default_transaction_read_only": "on",
                "statement_timeout": str(timeout_ms),
            }
        }

    async def aclose(self) -> None:
        if self._owns_engine:
            await self._engine.dispose()

    def check_authorized(self, user_role: str) -> AuthorizationVerdict:
        """Role-based authorization for the database tool. Application code —
        never the LLM — decides whether this tool may run."""
        settings = get_settings()
        if user_role in settings.db_agent_roles:
            return AuthorizationVerdict(allowed=True)
        return AuthorizationVerdict(
            allowed=False,
            reason="database tool not permitted for this role",
        )

    # ------------------------------------------------------------------ #
    # Plan validation                                                     #
    # ------------------------------------------------------------------ #

    def validate_plan(self, plan: QueryPlan) -> None:
        """Raise ToolValidationError unless the plan is fully within policy."""
        if not plan.answerable:
            return
        if plan.table not in self.allowed_tables:
            raise ToolValidationError(f"table {plan.table!r} is not allowed")
        allowed_columns = TABLE_COLUMNS.get(plan.table, set())
        if not plan.columns:
            raise ToolValidationError("at least one column must be selected")
        for column in plan.columns:
            if column not in allowed_columns:
                raise ToolValidationError(f"column {column!r} is not allowed on {plan.table!r}")
        if plan.order_by is not None and plan.order_by not in allowed_columns:
            raise ToolValidationError(f"order_by column {plan.order_by!r} is not allowed")
        for condition in plan.filters:
            if condition.column not in allowed_columns:
                raise ToolValidationError(
                    f"filter column {condition.column!r} is not allowed on {plan.table!r}"
                )

        # Defense-in-depth: strip injection-style payloads from filter values.
        for condition in plan.filters:
            value_str = str(condition.value)
            if contains_prompt_injection(value_str):
                raise ToolValidationError("filter value rejected by content policy")
            if contains_secret_material(value_str):
                raise ToolValidationError("filter value rejected by secret-content policy")

    # ------------------------------------------------------------------ #
    # Compilation                                                         #
    # ------------------------------------------------------------------ #

    def compile_plan(self, plan: QueryPlan) -> tuple[str, dict[str, Any]]:
        """Compile a validated plan into SQL + bound parameters.

        Identifiers are whitelisted and quote-escaped; values are bound, never
        interpolated into the SQL text.
        """
        self.validate_plan(plan)
        if not plan.answerable:
            raise ToolValidationError("plan is not answerable; nothing to compile")

        table = validate_identifier(plan.table)
        columns = [validate_identifier(c) for c in plan.columns]
        params: dict[str, Any] = {}

        where_clauses: list[str] = []
        for i, condition in enumerate(plan.filters):
            column = validate_identifier(condition.column)
            pname = f"p{i}"
            params[pname] = condition.value
            if condition.op == "eq":
                where_clauses.append(f"{column} = :{pname}")
            elif condition.op == "contains":
                where_clauses.append(f"{column} ILIKE :{pname}")
                params[pname] = f"%{condition.value}%"
            elif condition.op == "gte":
                where_clauses.append(f"{column} >= :{pname}")
            elif condition.op == "lte":
                where_clauses.append(f"{column} <= :{pname}")
            elif condition.op == "gt":
                where_clauses.append(f"{column} > :{pname}")
            elif condition.op == "lt":
                where_clauses.append(f"{column} < :{pname}")

        limit = min(plan.limit, self.max_rows, 100)
        # B608/S608: identifiers are whitelisted+validated, values are bound params.
        sql = f"SELECT {', '.join(columns)} FROM {table}"  # noqa: S608  # nosec B608
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        if plan.order_by:
            order_column = validate_identifier(plan.order_by)
            sql += f" ORDER BY {order_column} {'ASC' if plan.order_dir == 'asc' else 'DESC'}"
        sql += f" LIMIT {int(limit)}"

        return sql, params

    # ------------------------------------------------------------------ #
    # Execution                                                           #
    # ------------------------------------------------------------------ #

    async def execute_plan(self, plan: QueryPlan, *, user_role: str) -> DatabaseResult:
        """Validate, authorize, compile and run the plan. Never raises for
        policy failures — returns a structured `DatabaseResult` instead."""
        span = tracer.start_span("tool.database.execute")
        safe_set_attributes(span, {"tool.name": "database", "db.table": plan.table})
        start = time.monotonic()
        metrics.tool_requests_total.labels(tool="database", status="started").inc()

        settings = get_settings()
        if user_role not in settings.db_agent_roles:
            metrics.tool_errors_total.labels(tool="database", error_type="denied").inc()
            span.set_attribute("tool.status", "denied")
            span.end()
            return DatabaseResult(status="denied", reason="database tool not permitted for role")

        if not plan.answerable:
            span.set_attribute("tool.status", "unanswerable")
            span.end()
            return DatabaseResult(
                status="empty", reason=plan.reason or "not answerable from database"
            )

        try:
            self.validate_plan(plan)
        except ToolValidationError as exc:
            metrics.tool_errors_total.labels(tool="database", error_type="invalid_plan").inc()
            span.set_attribute("tool.status", "invalid_plan")
            span.end()
            return DatabaseResult(status="error", reason=str(exc), error_type="invalid_plan")

        sql, params = self.compile_plan(plan)

        try:
            async with self._engine.connect() as conn, conn.begin():
                # Belt-and-braces: force read-only + timeout for this transaction.
                await conn.execute(text("SET TRANSACTION READ ONLY"))
                await conn.execute(
                    text(f"SET LOCAL statement_timeout = {int(self.query_timeout_ms)}")  # noqa: S608
                )
                result = await conn.execute(text(sql), params)
                rows = [dict(row._mapping) for row in result.fetchmany(self.max_rows + 1)]
            truncated = len(rows) > self.max_rows
            rows = rows[: self.max_rows]
            rows = self._scrub_rows(plan.columns, rows)

            status: Literal["ok", "empty"] = "ok" if rows else "empty"
            metrics.tool_requests_total.labels(tool="database", status=status).inc()
            metrics.tool_request_duration_seconds.labels(tool="database").observe(
                time.monotonic() - start
            )
            span.set_attribute("tool.status", status)
            span.set_attribute("tool.row_count", len(rows))
            return DatabaseResult(
                status=status,
                table=plan.table,
                columns=list(plan.columns),
                rows=rows,
                row_count=len(rows),
                reason="results truncated to row limit" if truncated else "",
            )
        except ToolValidationError:
            raise
        except Exception as exc:
            record_error(span, exc)
            metrics.tool_errors_total.labels(tool="database", error_type="error").inc()
            metrics.tool_request_duration_seconds.labels(tool="database").observe(
                time.monotonic() - start
            )
            logger.warning("database_tool_failed", error_type=type(exc).__name__)
            return DatabaseResult(status="error", error_type=type(exc).__name__)
        finally:
            span.end()

    @staticmethod
    def _scrub_rows(columns: list[str], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove sensitive columns and serialize non-JSON-safe values."""
        clean: list[dict[str, Any]] = []
        for row in rows:
            item: dict[str, Any] = {}
            for key, value in row.items():
                if key.lower() in SENSITIVE_COLUMNS:
                    item[key] = "[REDACTED]"
                    continue
                if hasattr(value, "isoformat"):  # datetime/date
                    item[key] = value.isoformat()
                elif isinstance(value, (bytes, memoryview)):
                    item[key] = "[binary]"
                else:
                    item[key] = value
            clean.append(item)
        return clean
