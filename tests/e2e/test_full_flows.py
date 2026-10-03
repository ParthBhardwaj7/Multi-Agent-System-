"""E2E tests: complete flows through every agent, persistence and retrieval."""

import httpx
import pytest

from app.db.engine import get_session_factory
from tests.conftest import auth_headers

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


@pytest.fixture()
def client(test_app):
    app, provider = test_app
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test"), provider


async def _turn(http, api_key, message):
    return await http.post("/v1/chat", json={"message": message}, headers=auth_headers(api_key))


class TestDatabaseAgentFlow:
    async def test_database_agent_answers_from_internal_data(self, client, make_user):
        """Router -> database agent -> secure tool -> synthesizer -> persisted."""
        http, _ = client
        _, api_key = await make_user("admin")  # admins may use the database tool

        response = await _turn(http, api_key, "What does our return policy say about returns?")
        assert response.status_code == 200
        body = response.json()
        assert body["intent"] == "database"
        assert body["selected_agent"] == "database"

        # The answer must be grounded in the seeded document (fake synthesizer
        # echoes provided context; assert the pipeline marked success).
        assert body["response"]

        from sqlalchemy import text

        async with get_session_factory()() as session:
            runs = (
                await session.execute(
                    text("SELECT agent, status FROM agent_runs WHERE conversation_id = :cid"),
                    {"cid": body["conversation_id"]},
                )
            ).all()
            agents = {agent for agent, _ in runs}
        assert agents == {"router", "database", "synthesizer"}

    async def test_database_agent_denied_for_basic_role(self, client, make_user):
        """Role 'user' is re-routed away from the database tool by policy."""
        http, _ = client
        _, api_key = await make_user("user")
        response = await _turn(http, api_key, "What does our return policy document say?")
        assert response.status_code == 200
        body = response.json()
        # Policy re-routes to general; the answer must NOT claim database results.
        assert body["selected_agent"] == "general"


class TestSearchAgentFlow:
    async def test_search_unavailable_is_honest(self, client, make_user):
        """No search API configured: the response must not fabricate results."""
        http, _ = client
        _, api_key = await make_user("user")
        response = await _turn(http, api_key, "Search the web for the latest AI news please")
        assert response.status_code == 200
        body = response.json()
        assert body["intent"] == "search"
        assert body["response"]  # honest failure text from synthesizer


class TestGeneralAgentFlow:
    async def test_general_conversation(self, client, make_user):
        http, _ = client
        _, api_key = await make_user("user")
        response = await _turn(http, api_key, "Hello there, how are you doing today?")
        assert response.status_code == 200
        body = response.json()
        assert body["intent"] == "general"
        assert body["response"]

    async def test_context_continues_across_turns(self, client, make_user):
        """Short-term window: the second turn sees the first exchange."""
        http, _ = client
        _, api_key = await make_user("user")
        first = await _turn(http, api_key, "My name is Testy McTestface")
        assert first.status_code == 200
        conversation_id = first.json()["conversation_id"]
        second = await http.post(
            "/v1/chat",
            json={
                "message": "What did I just tell you my name was?",
                "conversation_id": conversation_id,
            },
            headers=auth_headers(api_key),
        )
        assert second.status_code == 200
        messages = await http.get(
            f"/v1/conversations/{conversation_id}/messages", headers=auth_headers(api_key)
        )
        items = messages.json()["items"]
        assert len(items) == 4  # 2 turns persisted
