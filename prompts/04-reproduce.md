# Stage 4 — Reproduce: verify the core claim with running code

Inputs: `analysis/core.md` (section 7 is your plan), `ingest/repo/` if cloned, `ingest/metadata.json`.

Almost everyone summarizes papers; almost nobody runs them. This stage is the credibility moat of the whole pipeline. Execute the reproduction plan from the analysis, at the highest achievable tier, within the time box given in the run context.

## Tiers (attempt highest feasible, fall back explicitly)

- **Tier 1 — Official code, reduced scale.** Set up the environment (use `uv`: `uv venv` + `uv pip install`, and freeze exact versions to `requirements.txt`; you are on Windows — mind path separators and long-path issues), run the paper's own code on a small slice of the real task, compare output against a paper claim.
- **Tier 2 — Minimal reimplementation.** Implement the core mechanism from scratch in a single clean script or notebook (`reproduction/minimal/`), at toy scale (small model, small data, minutes of CPU/GPU time), designed to show the *direction* of the claimed effect.
- **Tier 3 — Static verification.** If nothing can run (huge compute, missing data, broken code), do a rigorous code walkthrough: map every paper equation to the exact lines implementing it, check for discrepancies between paper and code, and derive what the claim implies analytically.

## Research discipline (non-negotiable)

1. **Pre-register before you run.** Before executing ANY experiment, write `reproduction/predictions.json`: for each claim you will test, the expected direction and rough magnitude at your scale, and what outcome would count as "supported" vs "not supported". Results are only diagnostic if the prediction came first — never edit predictions after running.
2. **Hyperparameter provenance.** Before implementing, extract every hyperparameter and design choice into a sourced table `reproduction/hyperparams.md`: value, source (paper section / code file / **assumed**). Assumed values must be marked — they're the first suspects when numbers diverge.
3. **Smoke ladder.** Never launch a long run cold. Verify in order: single forward pass → one training step → ~20 iterations → the real (scaled-down) run. Debug at the cheapest rung that fails.
4. **Comparative claims need both arms.** If the core claim is "X beats Y", run BOTH X and Y at the same scale, same data, same tuning budget. Never compare your X against the paper's reported Y — different scales/settings make that comparison meaningless.
5. **Seeds and variance.** Where runtime allows, run ≥3 seeds and report mean ± std. Never declare a direction from a delta smaller than seed variance — report "indistinguishable at this scale" instead.
6. **Lab notebook.** Append to `reproduction/LAB-NOTES.md` as you go: what you tried, what failed, what you changed. Failures are content, not embarrassment.

## Compute ladder (budget is tight — climb only as far as needed)

- **Rung 0 — local CPU (free, default).** This machine has NO discrete GPU (Intel i5-1235U, integrated graphics). Design the experiment for CPU: tiny models (1–30M params trained from scratch, or BERT-tiny/DistilBERT/GPT-2-small adapters fine-tuned on hundreds of examples), small datasets, minutes-scale runs. Most "minimal change" claims can show direction-of-effect here.
- **Rung 1 — Kaggle free GPU (free, ~30 h/week quota).** If the claim genuinely needs CUDA, package the experiment as a self-contained script and run it on Kaggle via the CLI (`pip install kaggle`, `kaggle kernels push` / `kaggle kernels output`) — requires `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars. If those credentials are absent, note it in the report and design the best CPU-scale alternative instead of blocking.
- **Rung 2 — anything bigger: do not attempt.** Write into REPORT.md exactly what compute a fuller verification would need and drop to a lower tier. Never spend money.

Environment capture: whatever rung you run on, write `reproduction/env-info.json` (python/torch versions, device, RAM, OS) collected at runtime — replication claims are meaningless without it. Statistical reproducibility (same direction of effect from the lockfile + seeds) is the bar, not bitwise identity.

## Untrusted-code caution

The cloned repo is unreviewed third-party code and you run with broad permissions on the user's personal machine. Before executing anything from it: skim `setup.py`/`pyproject.toml`/install scripts and the entry points you'll run for anything unrelated to the paper (network calls to odd hosts, credential/file access outside the project, obfuscated blobs, miners). If anything looks off, do NOT run it — fall back to Tier 2 (your own clean reimplementation) and record why in LAB-NOTES.md. Prefer running large/unfamiliar codebases on Kaggle (remote sandbox) rather than locally when that rung is available.

## Rules

- Write all code into `reproduction/` in the run directory. It must be **replicable**: a `README.md` inside `reproduction/` with exact setup + run commands, pinned dependencies (`requirements.txt` or equivalent), and fixed random seeds.
- Log real outputs. Save raw results (metrics, timings, per-seed values) to `reproduction/results.json` so the visualization stage can chart them. Never fabricate numbers — a failed run reported honestly is worth more than an invented success.
- Respect the time box: if a run would exceed it, scale down or move down a tier.
- **Back-edge:** if your results contradict claims made in `analysis/core.md`, write `reproduction/analysis-delta.md` stating exactly which analysis claims need revising and why. Downstream stages treat that file as an amendment to the analysis.

## Required output: `reproduction/REPORT.md`

- Tier achieved and why
- Environment (OS, Python version, hardware)
- Exact commands run
- Results table: our numbers (mean ± std across seeds, both arms for comparative claims) vs our pre-registered predictions vs the paper's claims, with the scale caveat stated plainly
- Verdict per pre-registered prediction: supported / partially supported / not supported / untestable
- What a reader would need to replicate at full scale
