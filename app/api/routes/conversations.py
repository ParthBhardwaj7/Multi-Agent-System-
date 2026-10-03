"""Conversation management routes — all enforce ownership."""

from fastapi import APIRouter, Query, Request

from app.api.dependencies import CurrentUser, SessionDep
from app.memory.checkpoint import PostgresCheckpointSaver
from app.observability.logging import get_logger
from app.schemas.conversations import (
    ConversationMessages,
    ConversationOut,
    DeleteResult,
)
from app.services.conversation import ConversationService

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])

logger = get_logger(__name__)


@router.get("", response_model=None)
async def list_conversations(
    current_user: CurrentUser,
    session: SessionDep,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    service = ConversationService(session)
    items, total = await service.list_for_user(current_user, limit=limit, offset=offset)
    return {"items": [item.model_dump(mode="json") for item in items], "total": total}


@router.get("/{conversation_id}", response_model=ConversationOut)
async def get_conversation(
    conversation_id: str, current_user: CurrentUser, session: SessionDep
) -> ConversationOut:
    service = ConversationService(session)
    return await service.get(current_user, conversation_id)


@router.get("/{conversation_id}/messages", response_model=ConversationMessages)
async def get_messages(
    conversation_id: str, current_user: CurrentUser, session: SessionDep
) -> ConversationMessages:
    service = ConversationService(session)
    return await service.get_messages(current_user, conversation_id)


@router.delete("/{conversation_id}", response_model=DeleteResult)
async def delete_conversation(
    conversation_id: str,
    request: Request,
    current_user: CurrentUser,
    session: SessionDep,
) -> DeleteResult:
    service = ConversationService(session)
    result = await service.delete(current_user, conversation_id)
    # Best-effort: clear the LangGraph checkpoint thread for this conversation.
    try:
        saver = _checkpoint_saver()
        await saver.adelete_thread(conversation_id)
    except Exception:  # noqa: BLE001 - checkpoint cleanup is best-effort
        logger.debug("checkpoint_thread_cleanup_failed", conversation_id=conversation_id)
    return result


def _checkpoint_saver() -> PostgresCheckpointSaver:
    from app.api.dependencies import get_checkpoint_saver

    return get_checkpoint_saver()
