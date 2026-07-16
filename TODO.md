# TODO

Deferred items, in rough priority order. (Deliberately deferred by design — see README "Deliberately deferred".)

## Before/with the first full run
- [ ] **Kaggle API token** — CLI is installed (v2.1.2) but credentials are NOT configured. Create token at kaggle.com/settings → drop `kaggle.json` into `%USERPROFILE%\.kaggle\`. Unlocks the free-GPU rung (~30 h/week) for the reproduce stage.
- [ ] **Semantic Scholar API key** (free, higher rate limits) — request at semanticscholar.org/product/api, set env var `S2_API_KEY`. (Prompt wiring done 2026-07-16 — key signup is the only remaining step.)
- [x] **`git init`** the repo — done 2026-07-16, initial commit made.
- [ ] **First full run** (`npm run daily` resumes from ingest on the 2026-07-16 AHR paper) — expect prompt tuning from what breaks; capture lessons into `state/lessons.md`.

## Source expansion (scout)
- [x] **OpenReview** — wired into scout prompt 2026-07-16 (seasonal; skip fast outside review cycles).
- [x] **alphaXiv** — wired into scout prompt 2026-07-16.
- [ ] Evaluate Semantic Scholar recommendations API for scout candidate generation.

## Infrastructure (build only when asked)
- [ ] **Dashboard + database** (design agreed 2026-07-16): SQLite `state/engine.db` as a rebuildable index over run artifacts (`syncRun()` after each stage; `npm run sync` to backfill); tables: runs, stages, candidates (top-5 shortlist w/ plain-English summaries), papers, outputs, review_findings, reproductions, engagement, lessons. Local web dashboard (`npm run dash`, localhost:4242, no build step): Today (live stage timeline), Shortlist, Runs drill-down, Library, Ops/metrics. Scout prompt change: top-5 candidates get `simple_summary`. Keep local-only, no auth.
- [ ] **Scheduling** — Windows Task Scheduler entry invoking `npm run daily` each morning.
- [ ] **Publishing** — auto-publish via headed Chrome (Claude in Chrome), gated on `review/verdicts.json` `"overall": "publish"`.
- [ ] **Feedback loop** — collect engagement metrics of published posts into `state/history.json` `engagement` field; inject into scout scoring.
- [ ] Convert stage prompts into proper Claude Code skills (`.claude/skills/`) for ad-hoc invocation.

## Hardening (as issues appear)
- [ ] Retry/backoff policy for flaky stage failures (currently: halt + manual `npm run daily` resume).
- [ ] Post-run cleanup of large artifacts (cloned repos, datasets) after N days.
- [ ] Per-stage wall-clock timeout (currently only turn/budget caps).
- [ ] Revisit paid GPU rung (RunPod/Modal spot, ~$0.30/hr) only if the engine earns money.
