# Stage 7 — Adversarial review: attack, then fix

Inputs: everything in `outputs/`, plus the evidence base (`analysis/`, `reproduction/`, `litreview/`, `ingest/`).

Run three independent adversarial passes over ALL outputs. For each pass, adopt the persona fully and hunt for real problems — a pass that finds nothing is suspicious, not reassuring. Then fix what you found by editing the output files directly.

## Pass A — The Skeptical Reviewer (technical correctness)
NeurIPS reviewer #2. Check: every equation and technical statement against `ingest/paper.md`; every characterization of related work against `litreview/bibliography.md`; whether stated conclusions actually follow from the reproduction evidence (including `reproduction/analysis-delta.md` if present — outputs contradicting it are high-severity findings); whether the reproduction's comparisons are matched-scale and its verdicts respect seed variance; whether limitations are honestly represented. Flag overclaiming ruthlessly.

Then run the **reviewer-defense drill**: generate the 10 most likely objections a hostile expert reader (or top Reddit/HN commenter) would raise, ranked by probability. For each: can our evidence answer it? Write these with draft answers to `review/objections.md`. Any objection our evidence cannot answer must be acknowledged in the outputs themselves (a preempted objection builds credibility; a dodged one destroys it).

## Pass B — The Fact-Checker (claim-by-claim verification)
Do not take `reproduction/results.json` on faith — the reproduction agent could have erred. **Spot re-verification:** pick the cheapest command from `reproduction/README.md` (ideally the evaluation/plot step, not full training), re-run it, and confirm it regenerates numbers consistent with `results.json`. If it doesn't run or disagrees, that's a high-severity finding — the affected claims get softened or cut, and the discrepancy goes in verdicts.json.

For every quantitative claim in every output (numbers, dates, author names, benchmark names, citations): trace it to its source file. Any claim that cannot be traced gets corrected or deleted. Hunt specifically for the classic failure modes: rounding inconsistencies between outputs, stale numbers from an earlier draft or run, percentage-vs-absolute confusion, percentage-point-vs-percent confusion, and the same quantity stated differently in two places. Verify figure captions match what the figures actually show, and that data provenance (our run vs paper-reported) is labeled on every figure and table.

## Pass C — The Editor & Virality Critic (will anyone read this?)
For each output, judge against its platform: Is the hook the first sentence? Does the X thread's first tweet work with zero context? Is the Reddit post hype-free? Is the blog's TL;DR accurate? Does any piece bury the reproduction result (our differentiator)? Is there AI-slop cadence? Rewrite weak hooks and cut flab — but never at the cost of accuracy fixed in passes A/B.

## Procedure

1. Run all three passes, collecting findings.
2. Apply fixes directly to the files in `outputs/` (edit in place).
3. Re-check that fixes didn't introduce new inconsistencies.
4. **Condense lessons (cross-run learning).** Append to `state/lessons.md` in the repository root (create if absent) 1–4 short bullets about the *process*, not the paper: recurring mistakes upstream stages made, what wasted budget, what a future run should do differently (e.g., "scout overweights papers whose datasets are gated — check access before selecting"). These are injected into every future run — only write lessons general enough to reuse, and delete any existing bullet this run proved wrong. Keep the file under ~30 bullets total: when adding would exceed that, merge or drop the least valuable existing ones (it's a distillation, not a log).

## Required output: `review/verdicts.json`

```json
{
  "passes": {
    "skeptic":     { "findings": [{ "file": "...", "issue": "...", "severity": "high|medium|low", "resolution": "fixed|accepted-risk|removed-claim" }], "verdict": "pass|pass-with-fixes|fail" },
    "factchecker": { "findings": [...], "verdict": "..." },
    "editor":      { "findings": [...], "verdict": "..." }
  },
  "overall": "publish|hold",
  "hold_reason": null,
  "summary": "2-3 sentences on the state of the content suite"
}
```

Set `"overall": "hold"` only for unresolved high-severity accuracy problems. This file gates future auto-publishing — be trustworthy.
