"""Deterministic routing policy — application code, never the LLM, decides
whether the router's classification is trusted.

Also contains content-policy checks used before tool execution (prompt
injection defense-in-depth) and tool authorization rules."""

import re
from dataclasses import dataclass

from app.config.settings import get_settings
from app.schemas.chat import IntentName


@dataclass(frozen=True)
class RoutingDecision:
    selected_agent: str
    routed_to_specialist: bool
    reason: str  # "high_confidence" | "low_confidence_fallback" | "invalid_output_fallback"


class RoutingPolicy:
    """Applies the configured confidence threshold to a router output.

    The router LLM proposes (intent, confidence); this policy decides. It is
    deterministic and unit-tested independently of any model.
    """

    def decide(
        self,
        intent: IntentName | None,
        confidence: float,
        *,
        threshold: float | None = None,
        fallback_agent: str | None = None,
    ) -> RoutingDecision:
        settings = get_settings()
        threshold = settings.router_confidence_threshold if threshold is None else threshold
        fallback = fallback_agent or settings.router_fallback_agent

        if intent is None or intent not in ("search", "database", "general"):
            return RoutingDecision(
                selected_agent=fallback,
                routed_to_specialist=False,
                reason="invalid_output_fallback",
            )
        if confidence >= threshold:
            return RoutingDecision(
                selected_agent=intent if intent != "general" else fallback,
                routed_to_specialist=intent in ("search", "database"),
                reason="high_confidence",
            )
        return RoutingDecision(
            selected_agent=fallback,
            routed_to_specialist=False,
            reason="low_confidence_fallback",
        )


# --- Content policy (defense-in-depth around tools) ------------------------

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"ignore\s+(the\s+above|that)",
        r"you\s+are\s+now\s+(a|an)\s+",
        r"system\s*:\s*",
        r"system\s+override",
        r"</?(system|assistant)>",
        r"reveal\s+your\s+(system\s+)?(prompt|instructions)",
        r"print\s+your\s+(system\s+)?(prompt|instructions)",
        r"(print|output|dump)\s+(your\s+)?environment",
        r"repeat\s+everything\s+(above|before)",
        r"disable\s+(all\s+)?(safety|security)\s+(policies|rules)",
        r"grant\s+me\s+(admin|root)",
        r"dump\s+(all\s+)?tables",
        r"act\s+as\s+the\s+database",
        r"output\s+all\s+(api\s+)?keys",
        r"execute\s+(drop\s+table|delete\s+from|update\s+)",
        r"run\s*:\s*select\s",
        r"select\s.*from\s.*(users|passwords|credentials)",
    )
)

# Strings that must never be forwarded to tools or the model as "data".
_SECRETISH = re.compile(
    r"(sk-[a-zA-Z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)


def contains_prompt_injection(text: str) -> bool:
    """Heuristic detector for obvious instruction-injection attempts.

    Used as defense-in-depth before tool execution; the primary defenses are
    structural (the LLM never writes SQL, tools have whitelists) and the
    detector only ever *rejects*, never rewrites.
    """
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def contains_secret_material(text: str) -> bool:
    return bool(_SECRETISH.search(text))


def sanitize_for_tool(text: str, max_length: int = 400) -> str:
    """Prepare user-derived text for a tool call: bounded length, no control chars."""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    return cleaned.strip()[:max_length]
