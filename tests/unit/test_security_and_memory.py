"""Unit tests: rate limiter, conversation locks (fakeredis), auth, cost, memory rules."""

import time

import jwt as pyjwt
import pytest

from app.config.settings import get_settings
from app.core.errors import ConflictError, RateLimitError
from app.memory.interface import extract_explicit_memories
from app.security.auth import (
    Principal,
    create_access_token,
    decode_access_token,
    generate_api_key,
)


class TestRateLimiter:
    async def test_allows_within_limit_and_blocks_beyond(self, fake_redis, monkeypatch):
        from app.security.rate_limit import RateLimiter

        limiter = RateLimiter(fail_open=False)
        monkeypatch.setattr("app.config.settings.get_settings", get_settings)

        for i in range(3):
            allowed, limit, remaining, _ = await limiter.check(
                scope="user", identifier="u1", limit=3, window_seconds=60
            )
            assert allowed, f"request {i + 1} should pass"

        allowed, limit, remaining, retry_after = await limiter.check(
            scope="user", identifier="u1", limit=3, window_seconds=60
        )
        assert allowed is False
        assert retry_after >= 1

    async def test_isolated_identifiers(self, fake_redis):
        from app.security.rate_limit import RateLimiter

        limiter = RateLimiter()
        await limiter.check(scope="ip", identifier="1.1.1.1", limit=1)
        allowed, *_ = await limiter.check(scope="ip", identifier="2.2.2.2", limit=1)
        assert allowed is True

    async def test_fail_closed_on_redis_outage(self, monkeypatch):
        from app.security import rate_limit as rl

        class BrokenClient:
            async def eval(self, *args, **kwargs):
                raise ConnectionError("redis down")

        monkeypatch.setattr(rl, "get_redis", lambda: BrokenClient())
        limiter = rl.RateLimiter(fail_open=False)
        with pytest.raises(RateLimitError):
            await limiter.check(scope="user", identifier="u9")

    async def test_fail_open_on_redis_outage(self, monkeypatch):
        from app.security import rate_limit as rl

        class BrokenClient:
            async def eval(self, *args, **kwargs):
                raise ConnectionError("redis down")

        monkeypatch.setattr(rl, "get_redis", lambda: BrokenClient())
        limiter = rl.RateLimiter(fail_open=True)
        allowed, *_ = await limiter.check(scope="user", identifier="u9")
        assert allowed is True


class TestConversationLock:
    async def test_acquire_and_release(self, fake_redis):
        from app.security.locks import conversation_lock

        async with conversation_lock("c1"):
            # second acquisition while held must conflict
            from app.core.redis_utils import get_redis

            token = await get_redis().get("lock:conversation:c1")
            assert token is not None
        from app.core.redis_utils import get_redis

        assert await get_redis().get("lock:conversation:c1") is None

    async def test_conflict_when_held(self, fake_redis):
        from app.security.locks import conversation_lock

        async with conversation_lock("c2"):
            with pytest.raises(ConflictError):
                async with conversation_lock("c2"):
                    pass

    async def test_fail_closed_when_redis_down(self, monkeypatch):
        from app.security import locks as lock_module
        from app.security.locks import conversation_lock

        class BrokenClient:
            async def set(self, *args, **kwargs):
                raise ConnectionError("redis down")

        monkeypatch.setattr(lock_module, "get_redis", lambda: BrokenClient())
        with pytest.raises(ConflictError):
            async with conversation_lock("c3"):
                pass


class TestAuth:
    def test_token_roundtrip(self):
        token, expires = create_access_token("user-123", "admin")
        principal = decode_access_token(token)
        assert principal == Principal(user_id="user-123", role="admin")

    def test_expired_token_rejected(self):
        settings = get_settings()
        now = int(time.time())
        token = pyjwt.encode(
            {
                "sub": "u1",
                "role": "user",
                "iat": now - 9999,
                "exp": now - 9000,
                "iss": settings.app_name,
            },
            settings.jwt_secret,
            algorithm=settings.jwt_algorithm,
        )
        from app.core.errors import AuthenticationError

        with pytest.raises(AuthenticationError):
            decode_access_token(token)

    def test_tampered_token_rejected(self):
        token, _ = create_access_token("u1", "user")
        tampered = token[:-4] + "AAAA"
        from app.core.errors import AuthenticationError

        with pytest.raises(AuthenticationError):
            decode_access_token(tampered)

    def test_api_keys_unique_and_hashable(self):
        key1, key2 = generate_api_key(), generate_api_key()
        assert key1 != key2
        from app.db.repository import hash_api_key, verify_api_key

        assert verify_api_key(key1, hash_api_key(key1))
        assert not verify_api_key(key2, hash_api_key(key1))


class TestCostEstimation:
    def test_pricing_config_applied(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(
            type(settings), "pricing", property(lambda self: {"gpt-4o-mini": (0.15, 0.60)})
        )
        from app.llm.client import estimate_cost

        # 1M input + 0 output at $0.15/1M = $0.15
        assert estimate_cost("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
        # 1M output at $0.60/1M = $0.60
        assert estimate_cost("gpt-4o-mini", 0, 1_000_000) == pytest.approx(0.60)

    def test_unknown_model_zero_cost(self):
        from app.llm.client import estimate_cost

        assert estimate_cost("unknown-model", 1000, 1000) == 0.0


class TestMemoryRules:
    def test_explicit_remember_fact(self):
        items = extract_explicit_memories("Remember that our quarterly review is on Fridays")
        assert len(items) == 1
        assert "quarterly review" in items[0]["content"]
        assert items[0]["kind"] == "fact"

    def test_explicit_remember_preference(self):
        items = extract_explicit_memories("Please remember that I prefer short answers")
        assert items[0]["kind"] == "preference"

    def test_do_not_forget(self):
        items = extract_explicit_memories("Don't forget that the launch is in May")
        assert len(items) == 1

    def test_routine_chatter_not_memorized(self):
        assert extract_explicit_memories("What's the weather today?") == []

    def test_too_short_ignored(self):
        assert extract_explicit_memories("remember it") == []
