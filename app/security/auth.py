"""Authentication: JWT issuance/validation and API-key verification.

- API keys: high-entropy random strings, stored as SHA-256 hashes.
- JWTs: HS256, short expiry, carry user_id + role claims.
- The request path accepts either `Authorization: Bearer <jwt>` (normal) or
  `X-API-Key: <key>` (machine clients). Application code resolves the principal;
  the LLM and request bodies are never trusted for identity."""

import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt as pyjwt

from app.config.settings import get_settings
from app.core.errors import AuthenticationError
from app.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: str


def generate_api_key() -> str:
    return f"mao_{secrets.token_urlsafe(32)}"


def create_access_token(user_id: str, role: str) -> tuple[str, datetime]:
    settings = get_settings()
    now = datetime.now(UTC)
    expires = now + timedelta(minutes=settings.jwt_expires_minutes)
    payload = {
        "sub": user_id,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
        "jti": uuid.uuid4().hex,
        "iss": settings.app_name,
    }
    token = pyjwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, expires


def decode_access_token(token: str) -> Principal:
    settings = get_settings()
    try:
        payload: dict[str, Any] = pyjwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            issuer=settings.app_name,
            options={"require": ["exp", "sub", "role"]},
        )
    except pyjwt.ExpiredSignatureError as exc:
        raise AuthenticationError("token expired") from exc
    except pyjwt.InvalidTokenError as exc:
        raise AuthenticationError("invalid token") from exc
    return Principal(user_id=str(payload["sub"]), role=str(payload["role"]))
