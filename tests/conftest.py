"""Shared pytest fixtures.

Environment is configured BEFORE app imports (settings are cached at first
use). Unit tests run without any external service; integration/e2e/security
tests require the dev containers: Postgres on :5434 and Redis on :6379
(`make infra-up` / docker compose)."""

import os
import sys
from pathlib import Path

os.environ["APP_ENV"] = "local"
os.environ["LLM_FAKE_MODE"] = "true"
os.environ["JWT_SECRET"] = "test-secret-" + "x" * 40
os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://orchestrator:orchestrator@localhost:5434/orchestrator",
)
os.environ["DATABASE_RO_URL"] = os.environ.get(
    "TEST_DATABASE_RO_URL",
    "postgresql+asyncpg://orchestrator_ro:orchestrator_ro@localhost:5434/orchestrator",
)
os.environ["REDIS_URL"] = os.environ.get("TEST_REDIS_URL", "redis://localhost:6390/15")
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["MEMORY_EXTRACTION_ENABLED"] = "false"
os.environ["RATE_LIMIT_PER_MINUTE"] = "10000"  # per-test override
os.environ["BOOTSTRAP_ADMIN_API_KEY"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402

from app.config.settings import get_settings  # noqa: E402


@pytest.fixture()
def fake_provider():
    from app.llm.fake import FakeLLMProvider

    return FakeLLMProvider()


@pytest.fixture()
def fake_redis(monkeypatch):
    """Swap the process-wide Redis client for fakeredis (async)."""
    import fakeredis.aioredis

    from app.core import redis_utils

    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_utils, "_client", client)
    yield client


def _reset_singletons():
    from app.api import dependencies
    from app.db.engine import set_session_factory

    dependencies.reset_container()
    set_session_factory(None)
    get_settings.cache_clear()


_DEMO_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL,
        category TEXT NOT NULL DEFAULT 'general', updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS faqs (
        id SERIAL PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL,
        category TEXT NOT NULL DEFAULT 'general', updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
    "TRUNCATE documents, faqs",
    """INSERT INTO documents (title, content, category) VALUES
        ('Return Policy', 'Items can be returned within 30 days of purchase with the original receipt.', 'policies'),
        ('Shipping Guide', 'Standard shipping takes 3-5 business days. Express arrives in 1-2 business days.', 'policies'),
        ('Platform Overview', 'The platform orchestrates multiple AI agents behind one API.', 'product')""",
    """INSERT INTO faqs (question, answer, category) VALUES
        ('How do I reset my password?', 'Click Forgot password on the sign-in page and follow the email instructions.', 'account')""",
    """DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'orchestrator_ro') THEN
            CREATE ROLE orchestrator_ro LOGIN PASSWORD 'orchestrator_ro';
        END IF;
    END
    $$""",
    "GRANT USAGE ON SCHEMA public TO orchestrator_ro",
    "GRANT SELECT ON ALL TABLES IN SCHEMA public TO orchestrator_ro",
]


@pytest_asyncio.fixture()
async def test_app(monkeypatch, fake_redis):
    """Full app wired to the real test Postgres + fakeredis + fake LLM.

    Yields (app, provider). Used by integration/e2e/security tests.
    """
    from sqlalchemy import text

    from app.api import dependencies
    from app.db.engine import create_db_engine, create_session_factory, set_session_factory
    from app.main import create_app

    _reset_singletons()

    engine = create_db_engine(get_settings().database_url)
    factory = create_session_factory(engine)
    set_session_factory(factory)

    provider = dependencies.get_llm_provider()  # LLM_FAKE_MODE=true -> FakeLLMProvider

    async with factory() as session:
        await session.execute(
            text(
                "TRUNCATE agent_runs, messages, conversations, memories, users, checkpoints,"
                " checkpoint_writes CASCADE"
            )
        )
        for statement in _DEMO_STATEMENTS:
            await session.execute(text(statement))
        await session.commit()

    app = create_app()
    async with app.router.lifespan_context(app):
        yield app, provider

    await engine.dispose()
    _reset_singletons()


@pytest_asyncio.fixture()
async def make_user(test_app):
    """Factory fixture: `user, api_key = await make_user(role='user')`."""
    from app.db.engine import get_session_factory
    from app.db.repository import UserRepository
    from app.security.auth import Principal, generate_api_key

    async def create(role: str = "user") -> tuple[Principal, str]:
        api_key = generate_api_key()
        async with get_session_factory()() as session:
            user = await UserRepository(session).create(
                role=role, display_name=f"{role}-test", api_key=api_key
            )
            await session.commit()
            return Principal(user_id=user.id, role=user.role), api_key

    return create


def auth_headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key}
