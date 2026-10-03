"""Strongly-typed LangGraph state.

The whole orchestrator passes this TypedDict around — never bare dicts. Fields
use `Annotated[..., reducer]` where concurrent writes must merge (LangGraph
semantics)."""

import operator
from typing import Annotated, Any, TypedDict

from app.schemas.agents import AgentRunRecord, SpecialistResult
from app.schemas.chat import IntentName


class MessageState(TypedDict, total=False):
    """One turn of conversation as stored/persisted."""

    role: str  # "user" | "assistant" | "system"
    content: str
    created_at: str | None
    metadata: dict[str, Any]


class GraphState(TypedDict, total=False):
    """The single source of truth flowing through the LangGraph orchestration."""

    # --- Identity / conversation ---
    conversation_id: str
    user_id: str
    user_role: str
    request_id: str

    # --- Conversation context (loaded from memory before the graph starts) ---
    messages: list[MessageState]  # short-term window (bounded)
    semantic_memories: list[dict[str, Any]]  # retrieved long-term memories

    # --- Current turn ---
    current_query: str

    # --- Routing ---
    intent: IntentName | None
    confidence: float
    selected_agent: str
    routed_to_specialist: bool
    fallback_reason: str | None

    # --- Specialist output ---
    specialist_result: SpecialistResult | None

    # --- Final answer ---
    final_response: str
    # Token deltas emitted by the synthesizer during streaming (reducer appends).
    streamed_tokens: Annotated[list[str], operator.add]

    # --- Errors / operational metadata ---
    errors: Annotated[list[dict[str, Any]], operator.add]  # structured error entries
    metadata: dict[str, Any]  # timings, usage, prompt versions
    agent_runs: Annotated[list[AgentRunRecord], operator.add]
