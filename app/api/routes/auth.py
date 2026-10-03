"""Authentication routes: exchange an API key for a short-lived JWT."""

from fastapi import APIRouter

from app.api.dependencies import CurrentUser, SessionDep
from app.core.errors import AuthenticationError
from app.db.repository import UserRepository
from app.schemas.common import TokenRequest, TokenResponse
from app.security.auth import create_access_token

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/token", response_model=TokenResponse)
async def issue_token(payload: TokenRequest, session: SessionDep) -> TokenResponse:
    """Exchange a valid API key for a JWT (recommended for browser clients)."""
    user = await UserRepository(session).get_by_api_key(payload.api_key)
    if user is None:
        raise AuthenticationError("invalid API key")
    token, expires = create_access_token(user.id, user.role)
    return TokenResponse(
        access_token=token,
        expires_at=expires,
        user_id=user.id,
        role=user.role,
    )


@router.get("/me")
async def whoami(current_user: CurrentUser) -> dict:
    """Validate credentials; return the resolved principal."""
    return {"user_id": current_user.user_id, "role": current_user.role}
