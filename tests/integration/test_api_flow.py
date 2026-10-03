"""Integration tests: FastAPI + LangGraph + PostgreSQL + Redis (fakeredis for
transport control) with the deterministic fake LLM. Requires the dev Postgres
container on :5434."""

import json

import httpx
import pytest

from app.db.engine import get_session_factory
from tests.conftest import auth_headers

pytestmark = pytest.mark.integration


@pytest.fixture()
def client(test_app):
    app, provider = test_app
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test"), provider


class TestHealthAndReadiness:
    async def test_health(self, client):
        http, _ = client
        response = await http.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"

    async def test_ready_reports_dependencies(self, client):
        http, _ = client
        response = await http.get("/ready")
        assert response.status_code == 200
        body = response.json()
        checks = {c["name"]: c["status"] for c in body["checks"]}
        assert checks["postgres"] == "ok"
        assert checks["redis"] == "ok"
        assert body["status"] == "ready"

    async def test_metrics_exposed(self, client):
        http, _ = client
        response = await http.get("/metrics")
        assert response.status_code == 200
        assert "http_requests_total" in response.text


class TestAuthEndpoints:
    async def test_token_exchange(self, client, make_user):
        http, _ = client
        _, api_key = await make_user("user")
        response = await http.post("/v1/auth/token", json={"api_key": api_key})
        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["role"] == "user"

    async def test_token_rejects_bad_key(self, client):
        http, _ = client
        response = await http.post("/v1/auth/token", json={"api_key": "mao_not_a_real_key_123"})
        assert response.status_code == 401

    async def test_me_with_bearer(self, client, make_user):
        http, _ = client
        principal, api_key = await make_user("admin")
        token_response = await http.post("/v1/auth/token", json={"api_key": api_key})
        token = token_response.json()["access_token"]
        response = await http.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["user_id"] == principal.user_id


class TestChatTurnFlow:
    async def test_full_turn_persists(self, client, make_user):
        http, _ = client
        _, api_key = await make_user("user")
        response = await http.post(
            "/v1/chat",
            json={"message": "What is the latest news about AI?"},
            headers=auth_headers(api_key),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["intent"] == "search"
        assert body["selected_agent"] == "search"
        assert body["routed_to_specialist"] is True
        assert body["response"]
        assert body["usage"]["input_tokens"] > 0

        conversation_id = body["conversation_id"]
        messages_response = await http.get(
            f"/v1/conversations/{conversation_id}/messages", headers=auth_headers(api_key)
        )
        assert messages_response.status_code == 200
        messages = messages_response.json()["items"]
        assert [m["role"] for m in messages] == ["user", "assistant"]

        # Agent runs were persisted for observability.
        from sqlalchemy import text

        async with get_session_factory()() as session:
            runs = (
                (
                    await session.execute(
                        text("SELECT agent FROM agent_runs WHERE conversation_id = :cid"),
                        {"cid": conversation_id},
                    )
                )
                .scalars()
                .all()
            )
        assert set(runs) == {"router", "search", "synthesizer"}

    async def test_second_turn_reuses_conversation(self, client, make_user):
        http, _ = client
        _, api_key = await make_user("user")
        first = await http.post(
            "/v1/chat", json={"message": "hello there"}, headers=auth_headers(api_key)
        )
        conversation_id = first.json()["conversation_id"]
        second = await http.post(
            "/v1/chat",
            json={"message": "and now tell me a joke", "conversation_id": conversation_id},
            headers=auth_headers(api_key),
        )
        assert second.status_code == 200
        assert second.json()["conversation_id"] == conversation_id

    async def test_checkpoint_thread_written(self, client, make_user):
        """LangGraph state is checkpointed per conversation thread."""
        http, _ = client
        _, api_key = await make_user("user")
        turn = await http.post("/v1/chat", json={"message": "hello"}, headers=auth_headers(api_key))
        conversation_id = turn.json()["conversation_id"]

        from sqlalchemy import text

        async with get_session_factory()() as session:
            count = (
                await session.execute(
                    text("SELECT COUNT(*) FROM checkpoints WHERE thread_id = :tid"),
                    {"tid": conversation_id},
                )
            ).scalar_one()
        assert count >= 1

    async def test_cross_session_semantic_memory(self, client, make_user):
        """A fact remembered in one conversation is recalled in a later one."""
        http, _ = client
        _, api_key = await make_user("user")
        first = await http.post(
            "/v1/chat",
            json={"message": "Remember that my favorite fruit is pineapple"},
            headers=auth_headers(api_key),
        )
        assert first.status_code == 200

        second = await http.post(
            "/v1/chat",
            json={"message": "What is my favorite fruit?"},
            headers=auth_headers(api_key),
        )
        assert second.status_code == 200

        from sqlalchemy import text

        async with get_session_factory()() as session:
            memories = (await session.execute(text("SELECT content FROM memories"))).scalars().all()
        assert any("pineapple" in content for content in memories)


class TestConversationRoutes:
    async def test_list_get_delete(self, client, make_user):
        http, _ = client
        principal, api_key = await make_user("user")
        turn = await http.post("/v1/chat", json={"message": "hi"}, headers=auth_headers(api_key))
        conversation_id = turn.json()["conversation_id"]

        listing = await http.get("/v1/conversations", headers=auth_headers(api_key))
        assert listing.status_code == 200
        assert listing.json()["total"] == 1

        detail = await http.get(
            f"/v1/conversations/{conversation_id}", headers=auth_headers(api_key)
        )
        assert detail.status_code == 200
        assert detail.json()["message_count"] == 2

        deleted = await http.delete(
            f"/v1/conversations/{conversation_id}", headers=auth_headers(api_key)
        )
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True

        gone = await http.get(f"/v1/conversations/{conversation_id}", headers=auth_headers(api_key))
        assert gone.status_code == 404


class TestSSEStreaming:
    async def test_stream_event_sequence(self, client, make_user):
        http, _ = client
        _, api_key = await make_user("user")
        events: list[tuple[str, dict]] = []

        async with http.stream(
            "POST",
            "/v1/chat/stream",
            json={"message": "What is the latest news about AI?"},
            headers=auth_headers(api_key),
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            event_name = None
            data_buffer: list[str] = []
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event_name = line[len("event: ") :]
                elif line.startswith("data: "):
                    data_buffer.append(line[len("data: ") :])
                elif line == "" and event_name is not None:
                    events.append((event_name, json.loads("\n".join(data_buffer) or "{}")))
                    event_name, data_buffer = None, []

        names = [name for name, _ in events]
        assert "router_started" in names
        assert "router_completed" in names
        assert "agent_started" in names
        assert "agent_completed" in names
        assert "token" in names
        assert names[-1] == "done"
        # Done event carries operational metadata (no internal detail).
        assert events[-1][1]["selected_agent"] in ("search", "database", "general")
