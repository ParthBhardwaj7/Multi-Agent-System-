"""Agent-facing structured schemas (LLM outputs validated with Pydantic)."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.chat import IntentName


class RouterOutput(BaseModel):
    """Strict structured output of the Router Agent."""

    model_config = ConfigDict(extra="forbid")

    intent: IntentName
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence")
    @classmethod
    def _round_confidence(cls, v: float) -> float:
        return round(v, 4)


class SearchQueryPrep(BaseModel):
    """Search agent's prepared query for the external search tool."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=400)

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class SearchCitation(BaseModel):
    title: str
    url: str
    snippet: str = ""


class SearchResult(BaseModel):
    """Structured result handed to the synthesizer by the Search Agent."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "empty", "unavailable", "error"] = "ok"
    query: str = ""
    citations: list[SearchCitation] = Field(default_factory=list, max_length=20)
    error_type: str | None = None


class FilterCondition(BaseModel):
    """One filter in a database query plan."""

    model_config = ConfigDict(extra="forbid")

    column: str
    op: Literal["eq", "contains", "gte", "lte", "gt", "lt"]
    value: str | int | float | bool


class QueryPlan(BaseModel):
    """Structured, LLM-proposed query plan. The DB agent validates and compiles
    this into a parameterized SELECT — the LLM never writes SQL."""

    model_config = ConfigDict(extra="forbid")

    answerable: bool = True
    reason: str = ""
    table: str = ""
    columns: list[str] = Field(default_factory=list, max_length=10)
    filters: list[FilterCondition] = Field(default_factory=list, max_length=6)
    order_by: str | None = None
    order_dir: Literal["asc", "desc"] = "desc"
    limit: int = Field(default=10, ge=1, le=100)


class DatabaseResult(BaseModel):
    """Structured result handed to the synthesizer by the Database Agent."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "empty", "unavailable", "denied", "error"] = "ok"
    table: str = ""
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    row_count: int = 0
    reason: str = ""
    error_type: str | None = None


class GeneralResult(BaseModel):
    """Structured result handed to the synthesizer by the General Agent."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "unavailable"] = "ok"
    answer: str
    capability_unavailable: bool = False


class SpecialistResult(BaseModel):
    """Wrapper stored in graph state for whichever specialist ran."""

    agent: str
    intent: IntentName
    status: str
    payload: dict[str, Any]
    latency_ms: float = 0.0
    prompt_version: str | None = None


class AgentRunRecord(BaseModel):
    """Operational metadata persisted per graph node execution."""

    agent: str
    status: str
    intent: IntentName | None = None
    confidence: float | None = None
    latency_ms: float = 0.0
    prompt_version: str | None = None
    error_type: str | None = None
    started_at: datetime | None = None


class MemoryCandidate(BaseModel):
    """A fact proposed for long-term semantic memory."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=3, max_length=512)
    kind: Literal["preference", "fact", "context"] = "fact"
