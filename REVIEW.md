# System Audit — 2026-07-16 (adversarial self-review)

Scope: every component — orchestrator code, agent wrapper, config, all 8 stage prompts,
state files, and the assumptions between them. Findings are ranked; everything marked
FIXED was fixed in this pass.

## Verdict in one line

The architecture is sound and the skeleton is verified by execution; the fixed bugs were
real but contained; the remaining risks are (a) untested middle stages and (b) inherent
trust limits of autonomous agents, both mitigated but not eliminated.

---

## Bugs found and FIXED

1. **Quality-bar bypass on resume (worst finding).** After a `proceed: false` skip day,
   re-running `npm run daily` skipped the completed scout and ran the full pipeline on the
   rejected paper. Startup gate added in `pipeline.ts`; `--from`/`--only` remain as
   deliberate human overrides.
2. **Unpinned core dependency.** `@anthropic-ai/claude-agent-sdk: "latest"` — the harness
   demanded lockfiles from reproductions while being unpinned itself. Pinned to `^0.3.211`.
3. **Prompt-injection surface.** Stage agents read arbitrary web content, paper text, and
   repo code with `bypassPermissions`. Added an explicit data-not-instructions rule to the
   shared stage system prompt. (Mitigation, not immunity — see Residual risks.)
4. **Untrusted code execution.** The reproduce stage clones and runs third-party repos on
   the user's personal machine. Added: inspect install scripts/entry points before running;
   fall back to clean reimplementation on suspicion; prefer Kaggle sandbox for unfamiliar
   codebases.
5. **Unverified reproduction numbers.** The fact-checker previously trusted `results.json`
   as written. Now it must re-run the cheapest command from `reproduction/README.md` and
   confirm the numbers regenerate; disagreement is a high-severity finding.
6. **Unbounded lessons growth.** `state/lessons.md` is injected into every stage of every
   run; without a cap it becomes context bloat. Capped at ~30 distilled bullets.
7. **Budget math ambiguity.** Per-stage caps sum to $46 vs the $40 run ceiling, and the
   ceiling is checked before (not during) a stage, so true worst case is ~$48. Documented
   as intentional back-pressure in `config.ts`.

## Design decisions re-examined and kept (with reasons)

- **`allowedTools` under `bypassPermissions` is documentation, not enforcement** (all tools
  are auto-approved). Kept: every stage needs the same broad research toolset anyway, and
  the alternative (default mode + canUseTool) risks silently blocking benign calls mid-run.
  Now honestly labeled in `agent.ts`.
- **History appended after `write`, before `review`.** A paper that reaches `write` but gets
  held at review is still marked covered. Kept: prevents re-burning budget on a paper that
  already consumed a full pipeline; the held outputs remain on disk for manual salvage.
- **Files as source of truth, no DB dependency.** Confirmed nothing depends on state beyond
  `history.json` + run artifacts; the parked dashboard DB stays a rebuildable index.
- **Fresh agent per stage, disk-only handoff.** Re-verified as the right isolation model:
  a hallucination cannot propagate except through written artifacts, which review audits.
- **Cost figures are API-equivalent estimates.** Auth is the Claude Code login (no API key
  set); "cost" = plan-usage consumption. Budget caps still function as spend governors.

## Residual risks (accepted, monitored)

- **Stages 1–7 have never executed.** Only scout is execution-verified ($0.94, correct
  behavior incl. self-correction). Expect the first full runs to surface prompt/environment
  issues, especially in reproduce (Windows CPU, dependency resolution). The resumability +
  lessons mechanisms exist precisely to absorb this.
- **Agent honesty is policy-enforced, not proven.** Pre-registration, spot re-verification,
  and claim tracing raise the cost of fabrication substantially but cannot make it
  impossible. The human read of `verdicts.json` before posting remains the final gate.
- **Semantic Scholar throttling** until `S2_API_KEY` exists (prompts now use it when set —
  the key itself needs a free signup, user action).
- **Kaggle rung dark** until the API token is configured (CLI already installed); scout's
  compute gate is slightly optimistic until then, reproduce degrades to CPU gracefully.
- **No per-stage wall-clock timeout** — only turn caps and dollar caps. A hung network call
  could stall a stage. In TODO (hardening).
- **Mid-run plan-usage exhaustion** fails a stage; resume works after limits reset.
- **Repo growth**: run artifacts (PDFs, charts) accumulate in git; cloned repos and venvs
  are gitignored. Cleanup policy is in TODO.

## Consistency checks performed

- `stages.ts` required outputs ↔ what each prompt actually instructs to write: consistent.
- `config.ts` StageName ↔ STAGES ↔ manifest keys: type-linked, cannot drift silently.
- Scout prompt ↔ pipeline `proceed` handling ↔ types: consistent (post-fix).
- Back-edge chain: reproduce → `analysis-delta.md` → write precedence rule → review
  high-severity check: closed loop.
- Dead source removed (Papers with Code, sunset 2025-07); successor + HN/Reddit/GitHub
  trending/OpenReview/alphaXiv wired; anti-signal rule present.
- Smoke-test artifacts (`runs/2026-07-16/`): manifest, candidates, selection all
  schema-conformant.

## Executed from TODO this pass (no user input needed)

- OpenReview + alphaXiv added to scout sources.
- `S2_API_KEY` support wired into ingest + litreview prompts (key signup still pending).
- Git repository initialized, initial commit made.
- Dashboard/DB design parked in TODO as agreed.

## Still needs the user

- Kaggle API token (`kaggle.json` → `%USERPROFILE%\.kaggle\`).
- Semantic Scholar API key (free) → `S2_API_KEY` env var.
- Decision to launch the first full run.
