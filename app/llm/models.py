"""LLM request/response models shared by all providers."""

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    content: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str | None = None
    raw: dict[str, Any] | None = Field(default=None, exclude=True)  # never serialized/logs


class TokenChunk(BaseModel):
    """A single streamed delta."""

    content: str


class StreamResult(BaseModel):
    """Aggregated result of a streamed completion."""

    content: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str | None = None
