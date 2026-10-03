"""Auth API schemas and shared response models."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class TokenRequest(BaseModel):
    api_key: str = Field(min_length=8, max_length=256)

    @field_validator("api_key")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("api_key must not be blank")
        return v


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 - RFC 6750 token type, not a secret
    expires_at: datetime
    user_id: str
    role: str


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: dict | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str
    version: str


class ReadinessCheck(BaseModel):
    name: str
    status: str
    latency_ms: float | None = None
    detail: str | None = None


class ReadinessResponse(BaseModel):
    status: str
    checks: list[ReadinessCheck]


class APIKeyCreated(BaseModel):
    user_id: str
    api_key: str  # returned exactly once at creation
    role: str
