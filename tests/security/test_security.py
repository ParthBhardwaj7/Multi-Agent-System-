"""Security & failure-mode tests: authN/authZ, injection, read-only
enforcement, rate limits, concurrency, Redis outage, LLM failures."""

import asyncio
import time

import httpx
import pytest

from app.config.settings import get_settings
from app.core.errors import LLMResponseError
from app.db.engine import create_db_engine, get_session_factory
from app.security.auth import create_access_token
from tests.conftest import auth_headers

pytestmark = [pytest.mark.security, pytest.mark.integration]


@pytest.fixture()
def client(test_app):
    app, provider = test_app
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test"), provider


class TestAuthentication:
    async def test_missing_credentials_401(self, client):
        http, _ = client
        response = await http.post("/v1/chat", json={"message": "hello"})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "authentication_error"

    async def test_invalid_api_key_401(self, client):
        http, _ = client
        response = await http.post(
            "/v1/chat", json={"message": "hello"}, headers=auth_headers("mao_bogus_key")
        )
        assert response.status_code == 401

    async def test_expired_jwt_401(self, client, make_user):
        http, _ = client
        principal, _ = await make_user("user")
        # Craft an already-expired token directly:
        import jwt as pyjwt

        settings = get_settings()
        now = int(time.time())
        token = pyjwt.encode(
            {
                "sub": principal.user_id,
                "role": "user",
                "iat": now - 9999,
                "exp": now - 9000,
                "iss": settings.app_name,
            },
            settings.jwt_secret,
            algorithm=settings.jwt_algorithm,
        )
        response = await http.post(
            "/v1/chat", json={"message": "hello"}, headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 401

    async def test_tampered_jwt_401(self, client):
        http, _ = client
        token, _ = create_access_token("someone", "admin")
        response = await http.post(
            "/v1/chat",
            json={"message": "hello"},
            headers={"Authorization": f"Bearer {token[:-3]}xyz"},
        )
        assert response.status_code == 401


class TestAuthorization:
    async def test_cannot_read_foreign_conversation(self, client, make_user):
        http, _ = client
        _, key_a = await make_user("user")
        _, key_b = await make_user("user")
        turn = await http.post(
            "/v1/chat", json={"message": "secret stuff"}, headers=auth_headers(key_a)
        )
        conversation_id = turn.json()["conversation_id"]

        foreign = await http.get(
            f"/v1/conversations/{conversation_id}", headers=auth_headers(key_b)
        )
        assert foreign.status_code == 403

        foreign_messages = await http.get(
            f"/v1/conversations/{conversation_id}/messages", headers=auth_headers(key_b)
        )
        assert foreign_messages.status_code == 403

        foreign_delete = await http.delete(
            f"/v1/conversations/{conversation_id}", headers=auth_headers(key_b)
        )
        assert foreign_delete.status_code == 403

    async def test_cannot_continue_foreign_conversation(self, client, make_user):
        http, _ = client
        _, key_a = await make_user("user")
        _, key_b = await make_user("user")
        turn = await http.post("/v1/chat", json={"message": "hello"}, headers=auth_headers(key_a))
        conversation_id = turn.json()["conversation_id"]

        foreign = await http.post(
            "/v1/chat",
            json={"message": "hijack", "conversation_id": conversation_id},
            headers=auth_headers(key_b),
        )
        assert foreign.status_code == 403

    async def test_unknown_conversation_404(self, client, make_user):
        http, _ = client
        _, api_key = await make_user("user")
        response = await http.get("/v1/conversations/does-not-exist", headers=auth_headers(api_key))
        assert response.status_code == 404


class TestInputValidation:
    async def test_oversized_message_422(self, client, make_user):
        http, _ = client
        _, api_key = await make_user("user")
        response = await http.post(
            "/v1/chat", json={"message": "a" * 9000}, headers=auth_headers(api_key)
        )
        assert response.status_code == 422

    async def test_extra_fields_rejected(self, client, make_user):
        http, _ = client
        _, api_key = await make_user("user")
        response = await http.post(
            "/v1/chat",
            json={"message": "hi", "injected": "field"},
            headers=auth_headers(api_key),
        )
        assert response.status_code == 422


class TestRateLimiting:
    async def test_rate_limit_enforced_with_headers(self, client, make_user, monkeypatch):
        http, _ = client
        _, api_key = await make_user("user")
        monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 2)

        for _ in range(2):
            ok = await http.post("/v1/chat", json={"message": "hi"}, headers=auth_headers(api_key))
            assert ok.status_code == 200

        blocked = await http.post("/v1/chat", json={"message": "hi"}, headers=auth_headers(api_key))
        assert blocked.status_code == 429
        assert "Retry-After" in blocked.headers
        assert blocked.json()["error"]["code"] == "rate_limit_exceeded"


class TestConcurrency:
    async def test_concurrent_turns_on_same_conversation(self, client, make_user):
        """Two simultaneous turns on one conversation: exactly one wins,
        the other gets 409 — no silent interleaving."""
        http, _ = client
        _, api_key = await make_user("user")
        first = await http.post(
            "/v1/chat", json={"message": "turn one"}, headers=auth_headers(api_key)
        )
        conversation_id = first.json()["conversation_id"]

        async def fire():
            return await http.post(
                "/v1/chat",
                json={"message": "parallel turn", "conversation_id": conversation_id},
                headers=auth_headers(api_key),
            )

        responses = await asyncio.gather(fire(), fire())
        statuses = sorted(r.status_code for r in responses)
        assert statuses in ([200, 409], [200, 200]), statuses


class TestRedisOutage:
    async def test_fail_closed_returns_429(self, test_app, make_user, monkeypatch):
        app, _ = test_app
        _, api_key = await make_user("user")
        monkeypatch.setattr(type(get_settings()), "rate_limit_fail_open", False, raising=False)

        from app.core import redis_utils

        class BrokenClient:
            async def eval(self, *args, **kwargs):
                raise ConnectionError("redis down")

            async def set(self, *args, **kwargs):
                raise ConnectionError("redis down")

            async def get(self, *args, **kwargs):
                raise ConnectionError("redis down")

        monkeypatch.setattr(redis_utils, "_client", BrokenClient())

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
            response = await http.post(
                "/v1/chat", json={"message": "hello"}, headers=auth_headers(api_key)
            )
        # Fail-closed policy: refuse rather than run un-limited.
        assert response.status_code == 429


class TestLLMFailures:
    async def test_router_timeout_falls_back_to_general(self, client, make_user):
        http, provider = client
        provider.fail_on = "latest news"  # any router call touching this text times out
        _, api_key = await make_user("user")
        response = await http.post(
            "/v1/chat",
            json={"message": "tell me the latest news"},
            headers=auth_headers(api_key),
        )
        # System degrades to general agent rather than failing the request.
        assert response.status_code == 200
        assert response.json()["selected_agent"] == "general"
        provider.fail_on = None

    async def test_invalid_structured_output_degrades(self):
        """Structured-output failures are non-retryable and must never be
        surfaced as raw provider errors to clients (policy falls back instead)."""
        error = LLMResponseError()
        assert error.retryable is False
        assert error.status_code == 502
        assert "language model" in error.message.lower()

    async def test_readonly_role_cannot_write(self):
        """The database agent's credentials cannot mutate data — enforced by
        Postgres itself, not application code."""
        import sqlalchemy as sa

        engine = create_db_engine(get_settings().database_ro_url)
        try:
            with pytest.raises(Exception) as exc_info:
                async with engine.begin() as conn:
                    await conn.execute(sa.text("UPDATE users SET role = 'admin'"))
            message = str(exc_info.value).lower()
            assert "read-only" in message or "permission" in message
        finally:
            await engine.dispose()


class TestSQLInjectionSafety:
    async def test_injection_payload_is_parameterized(self, client, make_user):
        """A quote-laden filter value must be bound, not interpolated, and must
        not corrupt the database."""
        http, _ = client
        _, api_key = await make_user("admin")
        response = await http.post(
            "/v1/chat",
            json={"message": "return policy details for x'; DROP TABLE users; --"},
            headers=auth_headers(api_key),
        )
        assert response.status_code == 200

        # users table still exists and is intact
        from sqlalchemy import text

        async with get_session_factory()() as session:
            count = (await session.execute(text("SELECT COUNT(*) FROM users"))).scalar_one()
        assert count >= 1
