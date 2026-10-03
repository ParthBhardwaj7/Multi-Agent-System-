"""Structured output helper: force JSON from the LLM and validate with Pydantic.

The router and memory extractor use this. Validation failures trigger one
repair attempt (a follow-up turn asking for valid JSON); if that also fails
we raise `LLMResponseError` — callers must degrade gracefully, never parse
free-form text.
"""

import json
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from app.core.errors import LLMResponseError
from app.llm.client import LLMProvider
from app.llm.models import ChatMessage
from app.observability.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def pydantic_to_json_schema(model: type[BaseModel]) -> dict:
    """Best-effort strict JSON schema from a Pydantic model (OpenAI structured outputs)."""
    schema = model.model_json_schema()
    # OpenAI strict mode requires every property in `required` and no $defs refs at top level.
    schema.pop("$defs", None)
    schema.pop("title", None)
    properties = schema.get("properties", {})
    schema["required"] = list(properties.keys())
    schema["additionalProperties"] = False
    for prop in properties.values():
        if prop.get("type") == "array" and "items" in prop and "$defs" in str(prop):
            prop.pop("items", None)
    return schema


async def structured_complete(
    provider: LLMProvider,
    schema_model: type[T],
    messages: list[ChatMessage],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> T:
    """Request a response conforming to `schema_model`; validate strictly."""
    schema = pydantic_to_json_schema(schema_model)
    attempt_messages = list(messages)

    for attempt in (1, 2):
        response = await provider.complete(
            attempt_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=schema,
        )
        try:
            return schema_model.model_validate_json(response.content)
        except PydanticValidationError:
            if attempt == 2:
                logger.warning("structured_output_validation_failed", schema=schema_model.__name__)
                raise LLMResponseError() from None
            attempt_messages = [
                *messages,
                ChatMessage(role="assistant", content=response.content),
                ChatMessage(
                    role="user",
                    content=(
                        "Your previous response was not valid JSON for the required schema. "
                        f"Schema: {json.dumps(schema)}. Respond again with ONLY valid JSON."
                    ),
                ),
            ]

    raise LLMResponseError()  # pragma: no cover
