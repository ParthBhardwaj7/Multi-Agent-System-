"""Agent Registry: maps intents to agent instances.

The orchestrator never writes `if intent == "search"` chains — new agents
(analytics, billing, RAG, ...) register themselves here without changes to
the graph."""

from app.agents.base import BaseAgent
from app.schemas.chat import IntentName


class AgentRegistry:
    def __init__(self, fallback_agent: str = "general") -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._intent_map: dict[IntentName, BaseAgent] = {}
        self._fallback = fallback_agent

    def register(self, agent: BaseAgent, intents: list[IntentName]) -> None:
        """Register an agent instance as the handler for the given intents."""
        if agent.name in self._agents:
            raise ValueError(f"agent {agent.name!r} already registered")
        self._agents[agent.name] = agent
        for intent in intents:
            self._intent_map[intent] = agent

    def resolve(self, name: str) -> BaseAgent:
        agent = self._agents.get(name)
        if agent is None:
            raise KeyError(f"no agent registered under {name!r}")
        return agent

    def resolve_intent(self, intent: IntentName) -> BaseAgent:
        agent = self._intent_map.get(intent)
        if agent is None:
            raise KeyError(f"no agent registered for intent {intent!r}")
        return agent

    @property
    def fallback(self) -> BaseAgent:
        return self.resolve(self._fallback)

    @property
    def fallback_name(self) -> str:
        return self._fallback

    @property
    def registered_names(self) -> list[str]:
        return sorted(self._agents)
