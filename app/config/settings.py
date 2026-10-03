"""Application settings loaded from environment variables (.env supported)."""

import json
from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    app_env: Literal["local", "development", "staging", "production"] = "local"
    app_name: str = "multi-agent-orchestrator"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"  # nosec B104 - containers must bind all interfaces
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000"
    max_message_length: int = 8000

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_base_url: str | None = None
    openai_timeout_seconds: float = 30.0
    openai_max_retries: int = 2
    # DEV/LOAD-TEST ONLY: use the deterministic FakeLLMProvider instead of OpenAI.
    # Refused in production (see validate_runtime_security).
    llm_fake_mode: bool = False

    # --- PostgreSQL ---
    database_url: str = "postgresql+asyncpg://orchestrator:orchestrator@localhost:5433/orchestrator"
    database_ro_url: str = (
        "postgresql+asyncpg://orchestrator_ro:orchestrator_ro@localhost:5433/orchestrator"
    )
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_statement_timeout_ms: int = 10000

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Security ---
    jwt_secret: str = "dev-only-secret-change-me"  # noqa: S105 - dev default, prod-validated
    jwt_algorithm: str = "HS256"
    jwt_expires_minutes: int = 60
    bootstrap_admin_api_key: str = ""
    rate_limit_fail_open: bool = False

    # --- Routing policy ---
    router_confidence_threshold: float = 0.80
    router_fallback_agent: str = "general"

    # --- Memory ---
    memory_short_term_window: int = 20
    memory_semantic_top_k: int = 5
    memory_semantic_min_score: float = 0.30
    memory_extraction_enabled: bool = True

    # --- Search tool ---
    search_api_url: str = "https://api.tavily.com/search"
    search_api_key: str = ""
    search_timeout_seconds: float = 8.0
    search_max_retries: int = 2

    # --- Database agent policy ---
    db_agent_allowed_tables: str = "documents,faqs"
    db_agent_max_rows: int = 50
    db_agent_query_timeout_ms: int = 5000
    db_agent_allowed_roles: str = "admin,analyst"

    # --- Rate limiting ---
    rate_limit_per_minute: int = 30
    rate_limit_burst: int = 10
    rate_limit_ip_per_minute: int = 120

    # --- Cost control ---
    llm_pricing: str = '{"gpt-4o-mini": [0.15, 0.60]}'

    # --- Observability ---
    otel_service_name: str = "multi-agent-orchestrator"
    otel_exporter_otlp_endpoint: str = ""
    otel_traces_enabled: bool = False
    metrics_enabled: bool = True

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def db_agent_tables(self) -> list[str]:
        return [t.strip() for t in self.db_agent_allowed_tables.split(",") if t.strip()]

    @property
    def db_agent_roles(self) -> list[str]:
        return [r.strip() for r in self.db_agent_allowed_roles.split(",") if r.strip()]

    @property
    def pricing(self) -> dict[str, tuple[float, float]]:
        """Model -> (input USD/1M tokens, output USD/1M tokens)."""
        try:
            raw = json.loads(self.llm_pricing)
            return {
                model: (float(prices[0]), float(prices[1]) if len(prices) > 1 else 0.0)
                for model, prices in raw.items()
            }
        except (json.JSONDecodeError, IndexError, TypeError):
            return {}

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def validate_runtime_security(self) -> list[str]:
        """Return a list of security configuration problems (checked at startup)."""
        problems: list[str] = []
        if self.is_production:
            if len(self.jwt_secret) < 32 or self.jwt_secret.startswith("dev-only"):
                problems.append("JWT_SECRET must be a strong random value in production")
            if not self.openai_api_key:
                problems.append("OPENAI_API_KEY is required in production")
            if self.llm_fake_mode:
                problems.append("LLM_FAKE_MODE=true is forbidden in production")
        return problems


@lru_cache
def get_settings() -> Settings:
    return Settings()
