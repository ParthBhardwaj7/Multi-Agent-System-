# AI Evaluation

## Golden Dataset

`data/eval/golden_dataset.jsonl` — 450 deterministic cases (seeded
generator: `scripts/generate_eval_dataset.py`), versioned in git:

| Category | Cases | Expected behaviour |
|---|---|---|
| search | 100 | route to `search` specialist |
| database | 100 | route to `database` specialist |
| general | 100 | route to `general` |
| ambiguous | 50 | safe general routing |
| low_confidence | 50 | fall back to general (below threshold) |
| adversarial | 50 | flagged by injection policy; safe routing |

Each case: `{id, category, query, expected_intent, expected_fallback,
expected_keywords?}`.

## What Is Measured

Through the **real routing policy layer** (`RoutingPolicy`, thresholds,
authorization re-route):

- **routing accuracy** — final agent selection matches the golden label
- **fallback accuracy** — low-confidence scenarios actually fall back
- **tool selection accuracy** — the specialist runs where expected
- **adversarial flagged rate** — injection-pattern detector coverage on the
  adversarial slice
- **latency** (p50/p95 per case) and **token usage** (via the LLM usage sink)

Answer-correctness checks (keyword grounding, hallucination proxies such as
"no fabricated URLs when search is unavailable") run in the e2e suite; the
eval runner focuses on routing/policy behaviour.

## Routers

- `--router heuristic` (default): deterministic keyword baseline. **CI mode**
  — no API keys required; validates the harness and the policy layer, and
  gives a regression baseline.
- `--router llm`: runs the actual configured OpenAI model through the same
  structured-output router path. Requires `OPENAI_API_KEY`; costs ~450 small
  calls. Use for real routing-accuracy numbers before prompt/model changes.

## Running

```bash
python scripts/generate_eval_dataset.py        # rebuild dataset (deterministic)
python scripts/run_evaluation.py --router heuristic
python scripts/run_evaluation.py --router llm
python scripts/run_evaluation.py --router heuristic --limit 50   # smoke
```

Reports are written to `docs/evaluation/results-<router>-<timestamp>.json`
(machine-readable: summary + per-case rows).

## Baseline (measured, heuristic router, 2026-10-03)

| Metric | Value |
|---|---|
| Routing accuracy (overall) | 0.9089 |
| Fallback accuracy | 1.0 |
| Tool selection accuracy | 0.88 |
| Adversarial flagged rate | 0.76 |
| Per-case latency p50/p95 | sub-millisecond (heuristic) |

By category: database 1.00, low_confidence 1.00, ambiguous 0.94, general
0.93, adversarial 0.86, search 0.76 routing accuracy.

## CI Integration

The `evaluation` job runs the heuristic evaluation on every push/PR (no
secrets needed) and fails the build if the runner errors. Prompt changes are
testable: `prompts/*.txt` carry content-hash versions (`prompt_version`)
recorded per agent run — compare eval reports before/after a prompt edit.

## Known Limitations

- The heuristic baseline's search slice (0.76) reflects keyword-overlap
  weakness of the baseline, not the platform: run `--router llm` for real
  model numbers.
- The injection detector is intentionally conservative (defense-in-depth
  layer, not the primary control); a low flag-rate does not mean the system
  is injection-vulnerable — see docs/SECURITY.md for the layered controls.
- Answer-quality evaluation is keyword/structural; LLM-as-judge scoring is a
  future improvement.
