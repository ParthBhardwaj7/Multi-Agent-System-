"""Authorization: conversation ownership enforcement and role helpers.

Every conversation-scoped request MUST go through `require_conversation_owner`
— changing the conversation_id in a URL or body must never leak another
user's data. The LLM is never involved in permissions."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.core.errors import AuthorizationError
from app.db.models import Conversation
from app.db.repository import ConversationRepository
from app.security.auth import Principal


def require_role(principal: Principal, allowed: list[str]) -> None:
    if principal.role not in allowed:
        raise AuthorizationError(f"role '{principal.role}' is not permitted for this action")


async def require_conversation_owner(
    session: AsyncSession, principal: Principal, conversation_id: str
) -> Conversation:
    return await ConversationRepository(session).get_owned(conversation_id, principal.user_id)


def database_agent_roles() -> list[str]:
    return get_settings().db_agent_roles
