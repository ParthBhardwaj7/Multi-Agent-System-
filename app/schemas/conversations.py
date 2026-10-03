"""Conversation API schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    created_at: datetime
    metadata: dict = Field(default_factory=dict)


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    version: int
    message_count: int


class ConversationList(BaseModel):
    items: list[ConversationOut]
    total: int


class ConversationMessages(BaseModel):
    conversation_id: str
    items: list[MessageOut]


class DeleteResult(BaseModel):
    deleted: bool
    conversation_id: str
