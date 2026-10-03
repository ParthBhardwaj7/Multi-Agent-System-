"""Unit tests: agent registry, LLM structured output, retry policy."""

import pytest

from app.agents.base import AgentContext
from app.agents.general import GeneralAgent
from app.agents.registry import AgentRegistry
from app.agents.router import RouterAgent
from app.agents.search import SearchAgent
from app.core.errors import LLMResponseError, LLMTimeoutError
from app.core.retry import with_retries
from app.llm.fake import FakeLLMProvider
from app.llm.models import ChatMessage


class TestAgentRegistry:
    def _registry(self) -> AgentRegistry:
        provider = FakeLLMProvider()
        context = AgentContext(provider=provider)
        registry = AgentRegistry(fallback_agent="general")
        registry.register(RouterAgent(context), intents=[])
        registry.register(SearchAgent(context), intents=["search"])
        registry.register(GeneralAgent(context), intents=["general"])
        return registry

    def test_register_and_resolve(self):
        registry = self._registry()
        assert registry.resolve("search").name == "search"
        assert registry.resolve_intent("search").name == "search"
        assert registry.resolve_intent("general").name == "general"

    def test_fallback(self):
        registry = self._registry()
        assert registry.fallback.name == "general"
        assert registry.fallback_name == "general"

    def test_unknown_intent_raises(self):
        registry = self._registry()
        with pytest.raises(KeyError):
            registry.resolve_intent("database")  # type: ignore[arg-type]

    def test_duplicate_registration_rejected(self):
        context = AgentContext(provider=FakeLLMProvider())
        registry = AgentRegistry()
        registry.register(GeneralAgent(context), intents=["general"])
        with pytest.raises(ValueError):
            registry.register(GeneralAgent(context), intents=["general"])

    def test_extensibility_new_agent_without_orchestrator_changes(self):
        """Future agents register themselves; the graph never changes."""
        from app.agents.base import BaseAgent

        class BillingAgent(BaseAgent):
            name = "billing"

            async def run(self, state):
                return {}

        context = AgentContext(provider=FakeLLMProvider())
        registry = self._registry()
        registry.register(BillingAgent(context), intents=["database"])  # rebind for demo
        assert registry.resolve("billing").name == "billing"


class TestStructuredOutput:
    async def test_valid_structured_response(self, fake_provider):
        fake_provider.scripts["Classify"] = {"intent": "database", "confidence": 0.94}
        from app.llm.structured import structured_complete
        from app.schemas.agents import RouterOutput

        output = await structured_complete(
            fake_provider,
            RouterOutput,
            [ChatMessage(role="user", content="Please Classify: where are my documents")],
        )
        assert output.intent == "database"
        assert output.confidence == pytest.approx(0.94)

    async def test_invalid_json_repairs_then_raises(self):
        # Empty-string marker matches EVERY user message, including the repair
        # prompt, so both attempts return garbage and the helper must raise.
        provider = FakeLLMProvider(scripts={"": "not json at all"})
        from app.llm.structured import structured_complete
        from app.schemas.agents import RouterOutput

        with pytest.raises(LLMResponseError):
            await structured_complete(
                provider,
                RouterOutput,
                [ChatMessage(role="user", content="classify this")],
            )


class TestRetryPolicy:
    async def test_retries_transient_then_succeeds(self):
        attempts = {"n": 0}

        async def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise LLMTimeoutError()
            return "ok"

        result = await with_retries(flaky, max_attempts=3, initial_wait=0.01)
        assert result == "ok"
        assert attempts["n"] == 3

    async def test_no_retry_on_permanent_failure(self):
        attempts = {"n": 0}

        async def permanent():
            attempts["n"] += 1
            from app.core.errors import LLMError

            raise LLMError(retryable=False)

        from app.core.errors import LLMError

        with pytest.raises(LLMError):
            await with_retries(permanent, max_attempts=5, initial_wait=0.01)
        assert attempts["n"] == 1

    async def test_exhaustion_raises_last_error(self):
        async def always_timeout():
            raise LLMTimeoutError()

        with pytest.raises(LLMTimeoutError):
            await with_retries(always_timeout, max_attempts=2, initial_wait=0.01)
