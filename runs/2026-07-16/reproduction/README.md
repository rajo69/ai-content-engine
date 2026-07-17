# Reproduction — AHR (arXiv:2607.13425)

A from-scratch, Tier-2 minimal reimplementation of **Attention Head Reweighting**, testing the
pre-registered predictions in `predictions.json`. Runs on a laptop CPU in ~40 minutes.

Tier 1 (run the authors' code) is impossible: the official repo is a README-only placeholder
with no code (`ingest/repo-notes.md`). Every line here is ours or a mainstream library.

## What this tests

Not the paper's headline ("AHR beats LoRA"), which its own Appendix B already settles. Instead
**the gradient** (`analysis/core.md` §4): AHR's advantage over *doing nothing* — i.e. over ICL
with the same 10 examples in the prompt — shrinks as the base model gets better at ICL. Two base
models span the competence axis; three arms per model per seed:

| Arm | What it is | Trainable params |
|---|---|---|
| `icl` | frozen model, 10-shot prompt, **no training** | 0 |
| `ahr` | one scalar γ per attention head, IC-FT trained (Eq. 3 + Eq. 5) | 144 (GPT-2) / 336 (Qwen) |
| `lora` | LoRA r=1 α=1, IC-FT trained, identical lr/steps/seeds/prompt | 36,864 / 67,584 |

All three share the same 10-shot prompt at test time, so AHR is measured the way the paper
measures it: ICL **plus** trained head reweighting over the same ten examples.

**Outcome (spoiler, full detail in `REPORT.md`):** the gradient test (**P3**) is **not supported** —
our weak-end model turned out to be *degenerate* at SST-2 ICL (50.83 ± 7.07 — below the eval set's
54.17% majority-class baseline, one-class-dominant in every seed) rather than merely weak, so the
axis we spanned was broken → good, not the paper's weak → good. What did
reproduce: **AHR lost to doing nothing on both models** (−3.33 ± 1.36 on Qwen2.5-0.5B, outside seed
noise), and **LoRA at 10 examples is catastrophic** (−25.83 vs the frozen model), which is where
AHR's entire nominal +22.50 "win" over LoRA comes from.

## Files

| File | What |
|---|---|
| `predictions.json` | Pre-registered predictions — written **before any code ran**, never edited after |
| `hyperparams.md` | Every hyperparameter with its source; **assumed** values flagged |
| `minimal/ahr.py` | The AHR implementation (Eq. 3 via forward pre-hook on the attention output projection) |
| `minimal/gate_p5.py` | P5 correctness gate: γ=0 must reproduce base-model logits |
| `minimal/run_experiment.py` | The experiment |
| `minimal/analyze.py` | Aggregation + verdict scoring (see REPORT.md § *Scoring integrity*) |
| `results.json` | Raw per-seed numbers |
| `summary.json` | Aggregated means/spreads/verdicts — the visualization stage reads this |
| `p5_gate.json` | Gate output |
| `env-info.json` | Runtime environment |
| `LAB-NOTES.md` | What was tried, what failed, what changed |
| `REPORT.md` | **The report** |
| `analysis-delta.md` | Amendments to `analysis/core.md` forced by these results |

## Setup

Python 3.13. The versions in `requirements.txt` are the ones actually executed.

```bash
uv venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

> Note: this session did **not** create a venv — torch/transformers/peft/datasets were already
> installed system-wide and the 90-minute box was better spent on experiments than on a
> reinstall. `requirements.txt` pins what ran, which is what replicability needs.

## Run

Order matters: the gate must pass before any result counts.

```bash
cd minimal

# P5 correctness gate (~1 min). Must print "P5 GATE: PASS".
python gate_p5.py

# Full experiment: 2 models x 3 arms x 3 seeds, ~40 min on 10 CPU cores.
python run_experiment.py \
    --models gpt2 Qwen/Qwen2.5-0.5B \
    --arms icl ahr lora \
    --seeds 0 1 2 \
    --n-eval 120 \
    --out ../results.json

# Aggregate into the results table + verdicts.
python analyze.py
```

`run_experiment.py` rewrites `results.json` after every arm and skips cells already present, so
an interrupted run resumes for free.

## Determinism

- Seeds `0,1,2` control **which 10 SST-2 training examples are sampled and their order** (the
  paper's own definition of a seed: "over choice of examples").
- The 120-example eval subsample is drawn with a fixed independent seed (1234) and is
  **identical across every model, arm and seed**, so all comparisons are paired.
- CPU float32 throughout, no dropout, full-batch training — runs are deterministic given a seed.
- The bar is statistical reproducibility (same direction of effect), not bitwise identity.

## Known deviations from the paper

These are the first suspects if numbers diverge. Full detail in `hyperparams.md`.

1. **Scale.** GPT-2-small (124M) + Qwen2.5-0.5B on CPU, vs the paper's 4 models up to 8B.
   One dataset (SST-2), 120 eval examples, 3 seeds — vs 6 datasets, full test sets, 10 seeds.
2. **No cross-validation.** The paper runs 5-fold CV over 168 configs *per seed*, selecting on
   validation loss. We run one fixed config for every arm. This is a fair like-for-like
   comparison (equal tuning budget: zero) but it is **not** the paper's selection procedure —
   notably it removes the `steps=0` escape hatch that `analysis/core.md` §5/T2 argues is doing
   much of AHR's work. Absolute numbers will not match the paper's, and we say so.
3. **Regularizer form.** Eq. 4's literal `λ(Σ|γ|^p)^(1/p)` is the Lp *norm*, which is
   non-differentiable at the origin — where the paper's own zero-init puts γ. It NaNs on the
   first backward pass. We use the standard convention (`λΣγ²` for p=2). See `LAB-NOTES.md`.
