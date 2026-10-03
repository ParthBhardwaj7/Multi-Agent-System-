"""Chat orchestration service.

One turn = load memory -> run LangGraph (router -> policy -> specialist ->
synthesizer) -> persist turn -> update memory. Supports blocking and SSE
streaming execution. Holds no per-request state; safe for concurrent use."""

import asyncio
import contextlib
import dataclasses
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.core.errors import ConflictError, NotFoundError
from app.db.repository import ConversationRepository
from app.graph.state import GraphState
from app.llm.client import usage_sink
from app.memory.conversation import ConversationMemory
from app.memory.semantic import SemanticMemoryStore
from app.observability import metrics
from app.observability.logging import conversation_id_var, get_logger
from app.schemas.chat import ChatRequest
from app.schemas.common import TokenResponse  # noqa: F401  (re-export convenience)
from app.security.auth import Principal
from app.security.locks import conversation_lock

logger = get_logger(__name__)


@dataclass
class TurnResult:
    conversation_id: str
    response: str
    intent: str
    confidence: float
    selected_agent: str
    routed_to_specialist: bool
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0


class ChatService:
    def __init__(
        self,
        graph: CompiledStateGraph,
        session: AsyncSession,
        *,
        memory_extraction_enabled: bool | None = None,
    ) -> None:
        self.graph = graph
        self.session = session
        settings = get_settings()
        self.memory_extraction_enabled = (
            settings.memory_extraction_enabled
            if memory_extraction_enabled is None
            else memory_extraction_enabled
        )
        self.conversations = ConversationRepository(session)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    async def run_turn(self, principal: Principal, request: ChatRequest) -> TurnResult:
        """Blocking execution of one chat turn."""
        start = time.monotonic()
        conversation = await self._resolve_conversation(principal, request)
        base_state, expected_version = await self._prepare_state(principal, conversation, request)

        final_state = await self._execute_graph(base_state, conversation.id, token_sink=None)
        result = self._result_from_state(conversation.id, final_state, start)
        await self._persist_and_learn(
            principal, conversation, request, final_state, expected_version
        )
        metrics.chat_turns_total.labels(status="success").inc()
        return result

    async def stream_turn(
        self,
        principal: Principal,
        request: ChatRequest,
        *,
        poll_disconnect: Callable[[], Awaitable[bool]],
    ) -> AsyncIterator[dict[str, Any]]:
        """SSE execution. Yields ChatEvent-shaped dicts:
        {event, data} — the transport layer formats them.

        Event order: router_started, router_completed, agent_started,
        agent_completed, token*, done (or error + done). Client disconnects
        are detected while waiting and cancel the graph task."""

        start = time.monotonic()
        conversation = await self._resolve_conversation(principal, request)
        base_state, expected_version = await self._prepare_state(principal, conversation, request)

        queue: asyncio.Queue[Any] = asyncio.Queue()
        done_sentinel = object()

        async def token_sink(text: str) -> None:
            await queue.put({"event": "token", "data": {"content": text}})

        async def producer() -> None:
            try:
                graph_config = cast(RunnableConfig, self._graph_config(conversation.id, token_sink))
                async for update in self.graph.astream(
                    base_state, config=graph_config, stream_mode="updates"
                ):
                    for event in self._events_from_update(update):
                        await queue.put(event)
                await queue.put(done_sentinel)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced as a safe SSE error
                logger.warning("stream_turn_failed", error_type=type(exc).__name__)
                await queue.put({"event": "error", "data": {"message": "stream failed"}})
                await queue.put(done_sentinel)

        task = asyncio.create_task(producer())
        consumed_done = False
        producer_failed = False
        try:
            while not consumed_done:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.5)
                except TimeoutError:
                    if await poll_disconnect():
                        task.cancel()
                        logger.info("sse_client_disconnected", conversation_id=conversation.id)
                        return
                    continue
                if item is done_sentinel:
                    consumed_done = True
                    break
                if item.get("event") == "error":
                    producer_failed = True
                yield item

            if producer_failed:
                metrics.chat_turns_total.labels(status="error").inc()
                yield {"event": "done", "data": {"error": True}}
                return

            final_state = await self._final_state(conversation.id, token_sink=None)
            result = self._result_from_state(conversation.id, final_state, start)
            await self._persist_and_learn(
                principal, conversation, request, final_state, expected_version
            )
            metrics.chat_turns_total.labels(status="success").inc()
            yield {
                "event": "done",
                "data": {
                    "conversation_id": result.conversation_id,
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "selected_agent": result.selected_agent,
                    "latency_ms": result.latency_ms,
                },
            }
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    async def _resolve_conversation(self, principal: Principal, request: ChatRequest) -> Any:
        if request.conversation_id is None:
            conversation = await self.conversations.create(user_id=principal.user_id)
            await self.session.commit()
            return conversation
        conversation = await self.conversations.get_owned(
            request.conversation_id, principal.user_id
        )
        if conversation is None:
            raise NotFoundError("conversation not found")
        return conversation

    async def _prepare_state(
        self, principal: Principal, conversation: Any, request: ChatRequest
    ) -> tuple[GraphState, int]:
        conversation_id_var.set(conversation.id)

        memory = ConversationMemory(self.session)
        semantic = SemanticMemoryStore(self.session, self._provider())
        window = await memory.window(conversation.id, principal.user_id)
        try:
            memories = await semantic.recall(principal.user_id, request.message)
        except Exception as exc:  # noqa: BLE001 - recall failure must not break chat
            logger.warning("semantic_recall_failed", error_type=type(exc).__name__)
            memories = []

        state: GraphState = {
            "conversation_id": conversation.id,
            "user_id": principal.user_id,
            "user_role": principal.role,
            "messages": window,
            "semantic_memories": [dataclasses.asdict(m) for m in memories],
            "current_query": request.message,
            "streamed_tokens": [],
            "errors": [],
            "agent_runs": [],
            "metadata": {},
        }
        return state, conversation.version

    def _provider(self) -> Any:
        from app.api.dependencies import get_llm_provider

        return get_llm_provider()

    def _graph_config(
        self, thread_id: str, token_sink: Callable[[str], Awaitable[None]] | None
    ) -> dict[str, Any]:
        configurable: dict[str, Any] = {"thread_id": thread_id}
        if token_sink is not None:
            configurable["token_sink"] = token_sink
        return {"configurable": configurable}

    async def _execute_graph(
        self, state: GraphState, thread_id: str, token_sink: Callable[[str], Awaitable[None]] | None
    ) -> GraphState:
        sink_token = usage_sink.set([])
        try:
            graph_config = cast(RunnableConfig, self._graph_config(thread_id, token_sink))
            result_state: Any = await self.graph.ainvoke(state, config=graph_config)
            final_state: GraphState = result_state
            usage = usage_sink.get() or []
            final_state.setdefault("metadata", {})
            final_state["metadata"]["usage"] = _aggregate_usage(usage)
            return final_state
        finally:
            usage_sink.reset(sink_token)

    async def _final_state(
        self, thread_id: str, token_sink: Callable[[str], Awaitable[None]] | None
    ) -> GraphState:
        graph_config = cast(RunnableConfig, self._graph_config(thread_id, token_sink))
        snapshot = await self.graph.aget_state(graph_config)
        if snapshot and snapshot.values:
            return cast(GraphState, dict(snapshot.values))
        return {}

    @staticmethod
    def _events_from_update(update: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ANN201
        """Map LangGraph node updates to SSE events."""
        events: list[dict[str, Any]] = []
        for node, payload in update.items():
            if node == "router":
                events.append(
                    {
                        "event": "router_completed",
                        "data": {
                            "intent": payload.get("intent"),
                            "confidence": payload.get("confidence"),
                        },
                    }
                )
            elif node == "policy":
                events.append(
                    {
                        "event": "agent_started",
                        "data": {"agent": payload.get("selected_agent")},
                    }
                )
            elif node == "specialist":
                specialist = payload.get("specialist_result")
                if isinstance(specialist, dict):
                    agent_name, status = specialist.get("agent"), specialist.get("status")
                elif specialist is not None:
                    agent_name, status = specialist.agent, specialist.status
                else:
                    agent_name, status = None, "missing"
                events.append(
                    {
                        "event": "agent_completed",
                        "data": {"agent": agent_name, "status": status},
                    }
                )
        return events

    @staticmethod
    def _result_from_state(conversation_id: str, state: GraphState, start: float) -> TurnResult:
        return TurnResult(
            conversation_id=conversation_id,
            response=str(state.get("final_response", "")),
            intent=str(state.get("intent", "general")),
            confidence=float(state.get("confidence", 0.0)),
            selected_agent=str(state.get("selected_agent", "general")),
            routed_to_specialist=bool(state.get("routed_to_specialist", False)),
            usage=state.get("metadata", {}).get("usage", {}),
            latency_ms=round((time.monotonic() - start) * 1000, 1),
        )

    async def _persist_and_learn(
        self,
        principal: Principal,
        conversation: Any,
        request: ChatRequest,
        final_state: GraphState,
        expected_version: int,
    ) -> None:
        """Persist the turn + agent runs, then update long-term memory.

        Runs under the conversation lock; optimistic versioning is the durable
        backstop. Concurrency conflict -> ConflictError (409) after the response
        was produced: the client may retry the persistence-safe way.
        """
        response_text = str(final_state.get("final_response", ""))
        if not response_text:
            logger.warning("empty_final_response", conversation_id=conversation.id)

        async with conversation_lock(conversation.id):
            from app.db.repository import AgentRunRepository, MessageRepository

            messages = MessageRepository(self.session)
            user_msg = await messages.make_message(
                conversation_id=conversation.id,
                role="user",
                content=request.message,
                metadata={"request_id": final_state.get("request_id", "")},
            )
            assistant_msg = await messages.make_message(
                conversation_id=conversation.id,
                role="assistant",
                content=response_text,
                metadata={
                    "intent": final_state.get("intent"),
                    "confidence": final_state.get("confidence"),
                    "selected_agent": final_state.get("selected_agent"),
                    "routed_to_specialist": final_state.get("routed_to_specialist", False),
                },
            )
            try:
                await self.conversations.append_turn(
                    conversation.id,
                    expected_version=expected_version,
                    user_message=user_msg,
                    assistant_message=assistant_msg,
                )
                await AgentRunRepository(self.session).record(
                    conversation_id=conversation.id,
                    message_id=assistant_msg.id,
                    runs=[
                        run.model_dump() if hasattr(run, "model_dump") else dict(run)
                        for run in final_state.get("agent_runs", [])
                    ],
                )
                if conversation.title == "New conversation":
                    conversation.title = request.message[:120]
                await self.session.commit()
            except ConflictError:
                await self.session.rollback()
                raise

        await self._update_semantic_memory(principal.user_id, conversation.id, request.message)

    async def _update_semantic_memory(
        self, user_id: str, conversation_id: str, message: str
    ) -> None:
        semantic = SemanticMemoryStore(self.session, self._provider())
        try:
            stored = await semantic.remember_explicit(user_id, conversation_id, message)
            if stored:
                await self.session.commit()
        except Exception as exc:  # noqa: BLE001 - memory write failure never breaks chat
            await self.session.rollback()
            logger.warning("semantic_memory_write_failed", error_type=type(exc).__name__)


def _aggregate_usage(entries: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "input_tokens": sum(int(e.get("input_tokens", 0)) for e in entries),
        "output_tokens": sum(int(e.get("output_tokens", 0)) for e in entries),
    }
