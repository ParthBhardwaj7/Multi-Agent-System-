"""Seed local development data: bootstrap admin user and demo database-agent
tables (documents, faqs) with a few sample rows.

Usage:
    .venv/Scripts/python.exe scripts/seed_local.py            # uses .env
Env: DATABASE_URL, BOOTSTRAP_ADMIN_API_KEY"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.config.settings import get_settings  # noqa: E402
from app.db.engine import create_db_engine, create_session_factory  # noqa: E402
from app.db.repository import UserRepository  # noqa: E402

DEMO_STATEMENTS: list[str] = [
    """CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        category TEXT NOT NULL DEFAULT 'general',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )""",
    """CREATE TABLE IF NOT EXISTS faqs (
        id SERIAL PRIMARY KEY,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        category TEXT NOT NULL DEFAULT 'general',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )""",
    "TRUNCATE documents, faqs",
    """INSERT INTO documents (title, content, category) VALUES
        ('Return Policy',
         'Items can be returned within 30 days of purchase with the original receipt. '
         'Refunds are processed within 5 business days.',
         'policies'),
        ('Shipping Guide',
         'Standard shipping takes 3-5 business days. Express shipping arrives in '
         '1-2 business days and is free above $75.',
         'policies'),
        ('Platform Overview',
         'The platform orchestrates multiple AI agents: a router classifies requests, '
         'specialists handle search and database lookups, and a synthesizer writes the '
         'final answer.',
         'product')""",
    """INSERT INTO faqs (question, answer, category) VALUES
        ('How do I reset my password?',
         'Click "Forgot password" on the sign-in page and follow the email instructions.',
         'account'),
        ('What models power the assistant?',
         'The platform uses configurable OpenAI models; see the OPENAI_MODEL setting.',
         'product')""",
]


async def seed() -> None:
    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    factory = create_session_factory(engine)

    async with factory() as session:
        for statement in DEMO_STATEMENTS:
            await session.execute(text(statement))
        await session.commit()

        api_key = settings.bootstrap_admin_api_key
        if api_key:
            from sqlalchemy import select

            from app.db.models import User

            existing = (
                (await session.execute(select(User).where(User.role == "admin"))).scalars().first()
            )
            if existing is None:
                await UserRepository(session).create(
                    role="admin", display_name="Local Admin", api_key=api_key
                )
                await session.commit()
                print(f"Seeded admin user (API key from BOOTSTRAP_ADMIN_API_KEY): {api_key[:8]}...")
            else:
                print("Admin user already exists; skipping.")

    await engine.dispose()
    print("Seed complete: demo tables documents/faqs created.")


if __name__ == "__main__":
    asyncio.run(seed())
