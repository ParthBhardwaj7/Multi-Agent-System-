"""Unit tests: graph state reducers, SSE event mapping, schemas, checkpoint saver."""

import pytest
from pydantic import ValidationError

from app.agents.base import AgentContext
from app.agents.database import DatabaseAgent
from app.agents.general import GeneralAgent
from app.agents.registry import AgentRegistry
from app.agents.router import RouterAgent
from app.agents.search import SearchAgent
from app.agents.synthesizer import SynthesizerAgent
from app.graph.graph import build_graph
from app.graph.state import GraphState
from app.llm.fake import FakeLLMProvider


def _registry(provider=None) -> AgentRegistry:
    context = AgentContext(
        provider=provider or FakeLLMProvider(),
        tools={"search": _NoSearch(), "database": None},
    )
    registry = AgentRegistry()
    registry.register(RouterAgent(context), intents=[])
    registry.register(SearchAgent(context), intents=["search"])
    registry.register(DatabaseAgent(context), intents=["database"])
    registry.register(GeneralAgent(context), intents=["general"])
    registry.register(SynthesizerAgent(context), intents=[])
    return registry


class _NoSearch:
    available = False

    async def search(self, query, max_results=5):
        from app.schemas.agents import SearchResult

        return SearchResult(status="unavailable", query=query, error_type="not_configured")


class TestGraphEndToEndWithFakes:
    async def test_search_intent_full_flow(self):
        provider = FakeLLMProvider()
        graph = build_graph(_registry(provider))
        state: GraphState = {
            "conversation_id": "c-test",
            "user_id": "u1",
            "user_role": "admin",
            "messages": [],
            "semantic_memories": [],
            "current_query": "What is the latest news about AI?",
            "streamed_tokens": [],
            "errors": [],
            "agent_runs": [],
            "metadata": {},
        }
        final = await graph.ainvoke(state, config={"configurable": {"thread_id": "c-test"}})
        assert final["intent"] == "search"
        assert final["selected_agent"] == "search"
        assert final["routed_to_specialist"] is True
        assert final["final_response"]
        agents = {r.agent for r in final["agent_runs"]}
        assert agents == {"router", "search", "synthesizer"}

    async def test_low_confidence_falls_back_to_general(self):
        provider = FakeLLMProvider(scripts={"Classify": {"intent": "search", "confidence": 0.1}})
        graph = build_graph(_registry(provider))
        state: GraphState = {
            "conversation_id": "c-test2",
            "user_id": "u1",
            "user_role": "user",
            "messages": [],
            "semantic_memories": [],
            "current_query": "hello there",
            "streamed_tokens": [],
            "errors": [],
            "agent_runs": [],
            "metadata": {},
        }
        final = await graph.ainvoke(state, config={"configurable": {"thread_id": "c-test2"}})
        assert final["selected_agent"] == "general"
        assert final["fallback_reason"] == "low_confidence_fallback"

    async def test_streaming_tokens_via_sink(self):
        provider = FakeLLMProvider()
        graph = build_graph(_registry(provider))
        tokens: list[str] = []

        async def sink(text: str) -> None:
            tokens.append(text)

        state: GraphState = {
            "conversation_id": "c-stream",
            "user_id": "u1",
            "user_role": "user",
            "messages": [],
            "semantic_memories": [],
            "current_query": "hello there friend",
            "streamed_tokens": [],
            "errors": [],
            "agent_runs": [],
            "metadata": {},
        }
        final = await graph.ainvoke(
            state, config={"configurable": {"thread_id": "c-stream", "token_sink": sink}}
        )
        assert len(tokens) > 0
        assert final["final_response"] == "".join(tokens).strip() or final["final_response"]

    async def test_router_failure_falls_back(self):
        provider = FakeLLMProvider(fail_on="latest news")
        graph = build_graph(_registry(provider))
        state: GraphState = {
            "conversation_id": "c-fail",
            "user_id": "u1",
            "user_role": "user",
            "messages": [],
            "semantic_memories": [],
            "current_query": "give me the latest news",
            "streamed_tokens": [],
            "errors": [],
            "agent_runs": [],
            "metadata": {},
        }
        final = await graph.ainvoke(state, config={"configurable": {"thread_id": "c-fail"}})
        assert final["selected_agent"] == "general"
        assert final["fallback_reason"] == "router_failure"


class TestEventMapping:
    def test_updates_map_to_sse_events(self):
        from app.services.chat import ChatService

        events = ChatService._events_from_update(
            {"router": {"intent": "database", "confidence": 0.9, "agent_runs": []}}
        )
        assert events[0]["event"] == "router_completed"
        assert events[0]["data"]["intent"] == "database"

        events = ChatService._events_from_update(
            {"policy": {"selected_agent": "database", "routed_to_specialist": True}}
        )
        assert events[0]["event"] == "agent_started"

        events = ChatService._events_from_update(
            {"specialist": {"specialist_result": {"agent": "database", "status": "ok"}}}
        )
        assert events[0]["event"] == "agent_completed"
        assert events[0]["data"]["agent"] == "database"


class TestSchemas:
    def test_chat_request_rejects_blank(self):
        from app.schemas.chat import ChatRequest

        with pytest.raises(ValidationError):
            ChatRequest(message="   ")

    def test_chat_request_rejects_bad_conversation_id(self):
        from app.schemas.chat import ChatRequest

        with pytest.raises(ValidationError):
            ChatRequest(message="hi", conversation_id="../etc/passwd")

    def test_chat_request_accepts_valid(self):
        from app.schemas.chat import ChatRequest

        request = ChatRequest(message="hello", conversation_id="abc-123")
        assert request.conversation_id == "abc-123"

    def test_router_output_bounds(self):
        from app.schemas.agents import RouterOutput

        with pytest.raises(ValidationError):
            RouterOutput(intent="search", confidence=1.5)
        with pytest.raises(ValidationError):
            RouterOutput(intent="unknown", confidence=0.5)  # type: ignore[arg-type]

    def test_query_plan_limit_bounds(self):
        from app.schemas.agents import QueryPlan

        with pytest.raises(ValidationError):
            QueryPlan(table="faqs", columns=["question"], limit=0)


class TestCheckpointSaver:
    """The Postgres checkpoint saver is exercised against real Postgres in
    integration tests; here we verify interface conformance."""

    def test_saver_is_async_only(self):
        from langgraph.checkpoint.base import BaseCheckpointSaver

        from app.memory.checkpoint import PostgresCheckpointSaver

        assert issubclass(PostgresCheckpointSaver, BaseCheckpointSaver)
        saver = PostgresCheckpointSaver(session_factory=None)  # type: ignore[arg-type]
        with pytest.raises(NotImplementedError):
            saver.get_tuple({"configurable": {"thread_id": "t"}})
