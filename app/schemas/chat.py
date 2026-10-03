"""Chat API request/response schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

IntentName = Literal["search", "database", "general"]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, description="The user's message.")
    conversation_id: str | None = Field(
        default=None,
        description="Existing conversation ID. Omit to start a new conversation.",
    )

    @field_validator("message")
    @classmethod
    def _message_not_blank(cls, v: str) -> str:
        from app.config.settings import get_settings

        stripped = v.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        max_length = get_settings().max_message_length
        if len(stripped) > max_length:
            raise ValueError(f"message exceeds maximum length of {max_length} characters")
        return stripped

    @field_validator("conversation_id")
    @classmethod
    def _conversation_id_shape(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v or len(v) > 64 or any(not (c.isalnum() or c == "-") for c in v):
            raise ValueError("conversation_id must be 1-64 chars of [a-zA-Z0-9-]")
        return v


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    intent: IntentName
    confidence: float
    selected_agent: str
    routed_to_specialist: bool
    usage: dict[str, int] = Field(default_factory=dict)


class ChatEvent(BaseModel):
    """A single SSE event payload."""

    event: str
    data: dict
