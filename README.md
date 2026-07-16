# AI Content Engine

A daily autonomous research pipeline. Each run it scouts the fresh AI/ML literature for the one paper with the best combination of **minimal change · novel perspective · viral potential**, researches it like an actual researcher, and produces a full content suite:

- a conference-grade deep-dive & reproduction report
- insightful visualizations (built from our own reproduction data)
- a replicable codebase verifying the paper's core claim
- long-form blog + platform-native short-form posts (X, LinkedIn, Reddit)

Every output passes a three-persona adversarial review (skeptic, fact-checker, virality editor) before it's considered publishable.

## Architecture

```
harness/
  pipeline.ts   orchestrator: runs stages in order, checkpoints, resumes on failure
  stages.ts     stage definitions: prompt file + required outputs + allowed tools
  agent.ts      wraps the Claude Agent SDK query() for one autonomous stage session
  config.ts     models, effort, budgets, scoring weights, domains
  manifest.ts   run manifest + covered-papers history
prompts/        the 8 stage prompts — the real IP of the system
runs/<date>/    daily artifacts: selection, ingest, litreview, analysis,
                reproduction, viz, outputs, review + manifest.json
state/          history.json — papers already covered (never repeated)
```

### Pipeline stages

| # | Stage      | Produces | Gate |
|---|-----------|----------|------|
| 0 | scout      | `candidates.json`, `selection.json` | rubric-scored pick, deduped vs history |
| 1 | ingest     | PDF, full text, citation graph, cloned repo | `ingest/metadata.json` |
| 2 | litreview  | annotated bibliography + research-lineage narrative | `litreview/*.md` |
| 3 | analyze    | deep dissection: problem, insight, mechanics, evidence, threats | `analysis/core.md` |
| 4 | reproduce  | tiered verification with running code | `reproduction/REPORT.md` |
| 5 | visualize  | headline chart, method diagram, explainer | `viz/index.md` |
| 6 | write      | paper, blog, X thread, LinkedIn, Reddit | `outputs/**` |
| 7 | review     | 3 adversarial passes, fixes applied in place | `review/verdicts.json` |

Each stage is an independent Claude Agent SDK session with its own tool allowlist, effort level, turn cap, and dollar budget. A stage only counts as complete when its required output files exist — re-running `npm run daily` resumes at the first incomplete stage.

## Setup

```sh
npm install
```

Auth: the Agent SDK uses your existing Claude Code credentials automatically (or set `ANTHROPIC_API_KEY`).

Prerequisites for full-strength runs: `git`, `python` (for reproductions and matplotlib charts).

## Usage

```sh
npm run daily                        # run/resume today's pipeline
npm run daily -- --date 2026-07-16   # run/resume a specific date
npm run daily -- --from analyze      # force re-run from a stage onward
npm run daily -- --only reproduce    # run a single stage
```

Budgets: per-stage caps and a whole-run ceiling live in `harness/config.ts` (`maxRunBudgetUsd`).

## Deliberately deferred

- **Scheduling** — invoke `npm run daily` manually for now; Task Scheduler wiring comes later.
- **Publishing** — the pipeline stops at reviewed, ready-to-post files in `runs/<date>/outputs/`. Auto-publishing via headed Chrome, gated on `review/verdicts.json` `"overall": "publish"`, comes later.
- **Feedback loop** — `state/history.json` has an `engagement` field reserved for post-performance metrics that will eventually tune the scout's scoring.
