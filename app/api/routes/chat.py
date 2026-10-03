"""Chat routes: blocking and SSE streaming conversation turns."""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langgraph.graph.state import CompiledStateGraph

from app.api.dependencies import CurrentUser, RateLimitedDeps, SessionDep, pop_rate_limit_headers
from app.observability.logging import request_id_var
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import ChatService

router = APIRouter(prefix="/v1/chat", tags=["chat"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    session: SessionDep,
    current_user: CurrentUser,
    _: RateLimitedDeps,
) -> ChatResponse:
    service = ChatService(graph=_graph(), session=session)
    result = await service.run_turn(current_user, payload)
    response = ChatResponse(
        conversation_id=result.conversation_id,
        response=result.response,
        intent=result.intent,  # type: ignore[arg-type]
        confidence=result.confidence,
        selected_agent=result.selected_agent,
        routed_to_specialist=result.routed_to_specialist,
        usage=result.usage,
    )
    return response


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    session: SessionDep,
    current_user: CurrentUser,
    _: RateLimitedDeps,
) -> StreamingResponse:
    """Server-Sent Events stream of the orchestration turn.

    Events: router_started, router_completed, agent_started, agent_completed,
    token, error, done. Internal errors never leak details into the stream.
    """
    service = ChatService(graph=_graph(), session=session)

    async def event_generator() -> AsyncIterator[str]:
        yield _sse("router_started", {"conversation_id": payload.conversation_id or ""})

        async def poll_disconnect() -> bool:
            return await request.is_disconnected()

        # stream_turn emits router/agent/token/error events and terminates with
        # a final done event (carrying turn metadata) — forward verbatim.
        try:
            async for event in service.stream_turn(
                current_user, payload, poll_disconnect=poll_disconnect
            ):
                yield _sse(event["event"], event["data"])
        except Exception:  # noqa: BLE001 - never leak internals to the stream
            request_id = request_id_var.get("")
            yield _sse(
                "error", {"message": "The conversation turn failed.", "request_id": request_id}
            )
            yield _sse("done", {"error": True})

    headers = dict(_SSE_HEADERS)
    headers.update(pop_rate_limit_headers(request))
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )


def _graph() -> CompiledStateGraph:
    from app.api.dependencies import get_orchestration_graph

    return get_orchestration_graph()
