# Harness Improvement Plan — handoff for the next session

Written 2026-07-18 after the first complete run (2026-07-16, verdict: publish, $20.00).
Context: run #1 evidence is in `runs/2026-07-16/`, process lessons in `state/lessons.md`,
system audit in `REVIEW.md`, deferred features in `TODO.md`. Read those before changing anything.

> **STATUS (2026-07-18, implemented):** A1–A3, B4–B7, C8–C10, and D11–D12 are done
> (D13 was guidance-only and honored; E stays deferred). Conventions are documented in
> README "Runtime conventions". Also fixed beyond this list: manifests no longer pin a
> machine-specific absolute `runDir` (reconciled on load), and `--dry-run` was added.
> Items below are kept as written for the rationale record.

## A. Token efficiency (zero quality loss — do these first)

1. **Async experiment handoff (biggest win).** The reproduce agent burned budget polling its
   own 40-min experiment. Change: a stage may write `_waiting.json` (`{"file": "...", "resume_hint": "..."}`)
   and exit; `pipeline.ts` watches for the file's arrival with plain fs polling (zero tokens),
   then re-invokes the stage agent to interpret results. Est. saving $1–2/run.
2. **Ingest on Sonnet 5.** Pure plumbing (downloads, pdftotext, JSON reshaping) — switch
   `config.ts` ingest model to `claude-sonnet-5`. Do NOT demote judgment stages
   (scout/litreview/analyze/write/review) without A/B evidence via lessons.
3. **Write stage reads the distilled layer.** Amend `prompts/06-write.md`: write from
   `analysis/core.md` + `reproduction/REPORT.md` + `litreview/lineage.md`; open the full
   `ingest/paper.md` only for spot checks. Review keeps full-paper access (fact-checking
   needs primary sources — non-negotiable).

## B. Reliability (from run #1 failure modes)

4. **Per-stage wall-clock timeout** (config value + AbortController around `query()`), so a
   hung network call can't stall a stage silently.
5. **Cost accounting in failure paths.** A stage that throws (budget cap, session limit) loses
   its `costUsd` from the manifest — capture cost from the SDK error result if available.
6. **Usage-limit awareness.** Session-limit errors ("resets Xpm") killed 4 legs. On that error,
   pipeline should parse the reset time and either sleep+retry internally or exit with a
   machine-readable marker so the operator/scheduler relaunches at the right moment.
7. **Stale-run hygiene.** `runs/2026-07-17/` holds a half-started scout (stale "running") from
   a killed launch. Add `--abandon <date>` or auto-mark stale "running" stages (no process +
   old timestamp) as failed on next invocation.

## C. Quality improvements

8. **Lessons with numbers.** Pipeline auto-appends per-run stats (stage cost vs cap, retries,
   failure causes) to `state/lessons.md` so learning carries data, not just impressions.
   Cap stays ~30 bullets.
9. **Review proposes prompt diffs.** Let the review stage emit `review/prompt-proposals.md`
   (concrete edit suggestions for `prompts/*.md`) — applied only after human/operator approval.
   This is the next rung of self-improvement.
10. **Novelty check earlier.** LoFiT (uncited prior art) was found in litreview — good, but the
    scout could run the 1-call Semantic Scholar recommendations check on its top-3 candidates
    before selection, avoiding ever picking a pure rediscovery.

## D. Dashboard / database (user found it unintuitive)

11. **Content click-through:** render markdown artifacts (blog, paper, REPORT, objections) in
    the dashboard instead of only listing paths; serve viz PNGs inline.
12. Hide/label stale runs (see #7); add a "what am I looking at" header line per view.
13. Keep the DB as an index (by design, files stay source of truth) — don't duplicate prose
    into it; completeness concerns were mostly the single-run sparsity + the stale 07-17 row.

## E. Deferred (unchanged, build only when asked)

- Publishing via headed Chrome gated on `review/verdicts.json` "publish".
- Engagement feedback loop into `state/history.json` + scout scoring.
- Windows Task Scheduler for the daily trigger.
- Kaggle rung is live (token verified); S2_API_KEY still needs free signup.

## Operating notes for the fresh session

- Pipeline is self-contained: `npm run daily` (today) / `-- --date YYYY-MM-DD` (resume).
- Dashboard: `npm run dash` → localhost:4242. `npm run sync` rebuilds the DB from files.
- Interactive model: pipeline stages run claude-opus-4-8; keep it that way except item 2.
- The scout may skip a weak day (`proceed: false`) — that's a feature, not a failure.
