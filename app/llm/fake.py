"""Deterministic FakeLLMProvider used by tests, load testing, evaluation
heuristic mode, and local development without API keys (LLM_FAKE_MODE=true).

This is an explicitly-labelled test double: production startup REFUSES
fake mode (Settings.validate_runtime_security). It is schema-aware so the
whole orchestration (router -> specialist -> synthesizer) can run end-to-end
without network access."""

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from app.llm.client import record_usage
from app.llm.models import ChatMessage, LLMResponse, TokenChunk, Usage
from app.observability import metrics

ScriptFn = Callable[[list[ChatMessage]], dict[str, Any] | str]

_GENERALY = (
    "hello",
    "hi there",
    "how are you",
    "joke",
    "thank",
    "good morning",
    "your name",
    "who are you",
    "tell me about yourself",
)
_SEARCHY = ("news", "latest", "today", "weather", "who won", "price of", "search", "web", "current")
_DATABASEY = (
    "document",
    "faq",
    "policy",
    "return policy",
    "shipping",
    "password reset",
    "in our",
    "internal",
    "knowledge base",
    "platform overview",
)


def _last_user_text(messages: list[ChatMessage]) -> str:
    user_messages = [m for m in messages if m.role == "user"]
    return user_messages[-1].content if user_messages else ""


def _classify(text: str) -> tuple[str, float]:
    lowered = text.lower()
    if any(k in lowered for k in _GENERALY) and not any(k in lowered for k in _DATABASEY):
        return "general", 0.9
    if any(k in lowered for k in _DATABASEY) and not any(k in lowered for k in _SEARCHY):
        return "database", 0.9
    if any(k in lowered for k in _SEARCHY):
        return "search", 0.9
    return "general", 0.9


class FakeLLMProvider:
    """Schema-aware deterministic provider with optional per-marker scripts."""

    def __init__(
        self,
        scripts: dict[str, ScriptFn | dict[str, Any] | str] | None = None,
        default_script: ScriptFn | dict[str, Any] | str = "This is a fake response.",
        model: str = "fake-model",
        embedding_dim: int = 16,
        fail_on: str | None = None,
    ) -> None:
        self.scripts = scripts or {}
        self.default_script = default_script
        self.model = model
        self.embedding_dim = embedding_dim
        self.calls: list[list[ChatMessage]] = []
        self.fail_on = fail_on  # substring that triggers an LLMError (failure testing)

    # ------------------------------------------------------------------ #

    def _resolve(self, messages: list[ChatMessage], schema_hint: str | None) -> str:
        self.calls.append(messages)
        text = _last_user_text(messages)
        if self.fail_on and self.fail_on in text:
            from app.core.errors import LLMTimeoutError

            raise LLMTimeoutError()

        for marker, script in self.scripts.items():
            if marker in text:
                if callable(script):
                    script = script(messages)
                return script if isinstance(script, str) else json.dumps(script)

        if schema_hint is not None:
            return json.dumps(self._schema_response(schema_hint, messages, text))
        if callable(self.default_script):
            resolved = self.default_script(messages)
        else:
            resolved = self.default_script
        return resolved if isinstance(resolved, str) else json.dumps(resolved)

    def _schema_response(
        self, schema_hint: str, messages: list[ChatMessage], text: str
    ) -> dict[str, Any]:
        if "RouterOutput" in schema_hint:
            # Classify ONLY the actual user query, not the prompt boilerplate.
            marker = "Latest user message:"
            query = text.split(marker)[-1].strip() if marker in text else text
            query = query.split("Classify the latest")[0].strip()
            intent, confidence = _classify(query)
            return {"intent": intent, "confidence": confidence}
        if "SearchQueryPrep" in schema_hint:
            return {"query": text[-min(len(text), 200) :].strip() or "latest news"}
        if "QueryPlan" in schema_hint:
            lowered = text.lower()
            table = "faqs" if ("faq" in lowered or "how do" in lowered) else "documents"
            keywords = [w for w in ("return", "policy", "shipping", "password") if w in lowered]
            contains = keywords[0] if keywords else "platform"
            return {
                "answerable": True,
                "table": table,
                "columns": ["id", "title", "content"]
                if table == "documents"
                else ["id", "question", "answer"],
                "filters": [
                    {
                        "column": "content" if table == "documents" else "question",
                        "op": "contains",
                        "value": contains,
                    }
                ],
                "order_by": None,
                "order_dir": "desc",
                "limit": 5,
            }
        return {"status": "ok"}

    # ------------------------------------------------------------------ #

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        schema_hint = None
        if response_schema:
            schema_hint = json.dumps(response_schema)
            # Name hint: strict schemas put "title" absent, so detect by properties.
            props = list(response_schema.get("properties", {}).keys())
            if props[:2] == ["intent", "confidence"]:
                schema_hint = "RouterOutput"
            elif props[:1] == ["query"]:
                schema_hint = "SearchQueryPrep"
            elif props[:1] == ["answerable"]:
                schema_hint = "QueryPlan"
        content = self._resolve(messages, schema_hint)
        input_tokens = sum(len(m.content.split()) for m in messages)
        output_tokens = len(content.split())
        model_name = model or self.model
        metrics.observe_llm_usage(model_name, input_tokens, output_tokens, 0.0)
        record_usage(model_name, input_tokens, output_tokens)
        return LLMResponse(
            content=content,
            model=model_name,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[TokenChunk]:
        content = self._resolve(messages, None)
        for word in content.split(" "):
            yield TokenChunk(content=word + " ")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.embedding_dim
            for token in text.lower().split():
                digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
                for i in range(self.embedding_dim):
                    vec[i] += (digest[i % len(digest)] / 255.0) - 0.5
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors
