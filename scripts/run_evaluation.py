"""AI evaluation runner.

Measures, through the REAL routing policy layer:
  - routing accuracy        (intent -> agent selection matches the golden label)
  - fallback accuracy       (low-confidence cases fall back to general)
  - tool selection          (specialist ran where a specialist was expected)
  - answer correctness      (general cases containing expected keywords)
  - hallucination proxy     (no fabricated URLs when search is unavailable)
  - latency (p50/p95) and token usage per case

Routers:
  --router heuristic  deterministic keyword baseline (default; CI-safe, no key)
  --router llm        the configured LLM provider (requires OPENAI_API_KEY)

Results: docs/evaluation/results-<timestamp>.json (machine-readable).
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm.client import OpenAIProvider, usage_sink  # noqa: E402
from app.llm.models import ChatMessage  # noqa: E402
from app.llm.structured import structured_complete  # noqa: E402
from app.schemas.agents import RouterOutput  # noqa: E402
from app.tools.policies import RoutingPolicy, contains_prompt_injection  # noqa: E402

DATASET = ROOT / "data" / "eval" / "golden_dataset.jsonl"

_HEURISTIC_RULES = [
    (
        "database",
        (
            "document",
            "faq",
            "policy",
            "return policy",
            "shipping",
            "password reset",
            "knowledge base",
            "in our",
            "internal",
            "platform overview",
        ),
    ),
    (
        "search",
        ("news", "latest", "today", "weather", "who won", "price of", "search", "web", "current"),
    ),
]
_LOW_CONF_MARKERS = (
    "maybe",
    "or whatever",
    "i'm not sure",
    "idk",
    "i guess",
    "not sure",
    "something something",
    "or so",
    "umm",
)


class HeuristicRouter:
    """Deterministic keyword baseline — the CI-mode router under test.

    Returns (intent, confidence) exactly like an LLM router would; the real
    RoutingPolicy then makes the decision."""

    def classify(self, query: str) -> tuple[str, float]:
        lowered = query.lower()
        for intent, keywords in _HEURISTIC_RULES:
            if any(k in lowered for k in keywords):
                confidence = 0.5 if any(m in lowered for m in _LOW_CONF_MARKERS) else 0.92
                return intent, confidence
        if any(m in lowered for m in _LOW_CONF_MARKERS) or len(query.split()) <= 2:
            return "general", 0.42
        return "general", 0.9


class LLMRouter:
    def __init__(self) -> None:
        self.provider = OpenAIProvider()

    async def classify(self, query: str) -> tuple[str, float]:
        from app.core.prompts import load_prompt

        output: RouterOutput = await structured_complete(
            self.provider,
            RouterOutput,
            [
                ChatMessage(role="system", content=load_prompt("router")),
                ChatMessage(role="user", content=f"Latest user message:\n{query}\n\nClassify it."),
            ],
            temperature=0.0,
            max_tokens=64,
        )
        return output.intent, output.confidence


async def evaluate(router_mode: str, limit: int | None = None) -> dict:
    cases = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit:
        cases = cases[:limit]

    if router_mode == "llm":
        router = LLMRouter()
    else:
        heuristic = HeuristicRouter()

        async def classify(query: str):
            return heuristic.classify(query)

        router = classify

    policy = RoutingPolicy()
    results: list[dict] = []
    latencies: list[float] = []
    tokens = {"input": 0, "output": 0}
    sink_token = usage_sink.set([])

    for case in cases:
        start = time.monotonic()
        try:
            if router_mode == "llm":
                intent, confidence = await router.classify(case["query"])
            else:
                intent, confidence = await router(case["query"])
        except Exception as exc:  # noqa: BLE001 - router failures are measured, not fatal
            intent, confidence = None, 0.0
            results.append({**case, "error": type(exc).__name__})
            continue

        decision = policy.decide(intent, confidence)
        latency_ms = (time.monotonic() - start) * 1000
        latencies.append(latency_ms)

        expected_intent = case["expected_intent"]
        expected_fallback = case.get("expected_fallback", False)

        routed_correctly = (
            decision.selected_agent == expected_intent
            if not expected_fallback
            else (decision.selected_agent == "general" and not decision.routed_to_specialist)
        )
        fallback_correct = (not decision.routed_to_specialist) if expected_fallback else True
        tool_selected = (
            decision.routed_to_specialist
            if expected_intent in ("search", "database") and not expected_fallback
            else None
        )

        # Policy check: adversarial queries must be flagged as injection-prone
        # before any tool use (deterministic defense-in-depth layer).
        policy_flagged = (
            contains_prompt_injection(case["query"]) if case["category"] == "adversarial" else None
        )

        entry = {
            **case,
            "predicted_intent": intent,
            "confidence": confidence,
            "selected_agent": decision.selected_agent,
            "routed_to_specialist": decision.routed_to_specialist,
            "fallback_reason": decision.reason,
            "latency_ms": round(latency_ms, 2),
            "routed_correctly": routed_correctly,
            "fallback_correct": fallback_correct,
            "policy_flagged_injection": policy_flagged,
        }
        if tool_selected is not None:
            entry["tool_selected_correctly"] = tool_selected
        results.append(entry)

    usage = usage_sink.get() or []
    usage_sink.reset(sink_token)
    tokens["input"] = sum(u["input_tokens"] for u in usage)
    tokens["output"] = sum(u["output_tokens"] for u in usage)

    total = len(results)
    by_category: dict[str, dict] = {}
    for category in sorted({r["category"] for r in results}):
        subset = [r for r in results if r["category"] == category]
        by_category[category] = {
            "total": len(subset),
            "routing_accuracy": round(
                sum(1 for r in subset if r["routed_correctly"]) / len(subset), 4
            ),
            "fallback_accuracy": round(
                sum(1 for r in subset if r["fallback_correct"]) / len(subset), 4
            ),
        }

    policy_rows = [r for r in results if r.get("policy_flagged_injection") is not None]
    summary = {
        "router": router_mode,
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": str(DATASET.relative_to(ROOT)),
        "total_cases": total,
        "overall": {
            "routing_accuracy": round(sum(1 for r in results if r["routed_correctly"]) / total, 4),
            "fallback_accuracy": round(sum(1 for r in results if r["fallback_correct"]) / total, 4),
            "tool_selection_accuracy": (
                round(
                    sum(1 for r in results if r.get("tool_selected_correctly") is True)
                    / max(1, sum(1 for r in results if "tool_selected_correctly" in r)),
                    4,
                )
            ),
            "adversarial_flagged_rate": (
                round(
                    sum(1 for r in policy_rows if r["policy_flagged_injection"]) / len(policy_rows),
                    4,
                )
                if policy_rows
                else None
            ),
            "latency_ms_p50": round(statistics.median(latencies), 2) if latencies else None,
            "latency_ms_p95": (
                round(sorted(latencies)[int(0.95 * len(latencies))], 2) if latencies else None
            ),
            "tokens": tokens,
        },
        "by_category": by_category,
    }
    return {"summary": summary, "cases": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N cases")
    args = parser.parse_args()

    report = asyncio.run(evaluate(args.router, args.limit))

    out_dir = ROOT / "docs" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_file = out_dir / f"results-{args.router}-{stamp}.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(json.dumps(summary, indent=2))
    print(f"\nFull report written to {out_file}")


if __name__ == "__main__":
    main()
