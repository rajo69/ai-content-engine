# Repo notes — tuomaso/attention-head-reweighting

**Cloned:** yes (`git clone --depth 1`), **but the repo contains no code.**

- URL: https://github.com/tuomaso/attention-head-reweighting
- Cloned at: 2026-07-16
- HEAD: `5464e7c` — "Create README.md" (2026-07-09)
- Files: `README.md` only (1 file, 0 code files)

## Full README contents

> This is the official repository for our COLM 2026 paper "Data-Efficient Adaptation of LLMs via Attention Head Reweighting". Stay tuned for the full code release shortly.

## Structure / entry points / dependencies

None. There is no source, no entry point, no dependency manifest, no license file. The
repo is a placeholder registered a week before the arXiv posting. The paper's footnote 1
states code "will be available" and will be "fully compatible with the `peft` package" —
that release has not landed.

Note: `selection.json` recorded `code_url: null`; the repo does exist and is named in the
paper's footnote, but since it is empty, `null` was operationally accurate.

## What it would take to run the core experiment

**A reimplementation is required — nothing can be run from this repo.** The upside is that
the method is genuinely small; the paper specifies it completely enough to rebuild.

Core method (~30 lines): register one learnable scalar per attention head, `gamma_h`,
initialized at 0, and scale that head's contribution to the residual stream by `(1 + gamma_h)`
(Eq. 3). Concretely, per-head slices of the attention output projection `W_O` are scaled
before being summed into the residual stream. Everything else, including MLPs, stays frozen.
Trainable parameter count = total number of attention heads (512 for Llama-3.2-1B).
Cleanest implementation: a forward hook on each attention module's output projection, or a
per-head scale multiplied into `o_proj` input. At inference gamma can be merged into `W_O`.

Loss: cross-entropy on answer tokens + `lambda * (sum_h |gamma_h|^p)^(1/p)`, p in {1,2} (Eq. 4).

The expensive part is not the method but the **evaluation protocol**:
- 4 models x 6 datasets x 6 training-set sizes x 10 random seeds,
- with per-seed 5-fold cross-validation over a 3 x 7 x 8 = 168-point hyperparameter grid,
- evaluated on the **full test set** each time.
That full grid is far beyond a 90-minute box.

### Suggested 90-minute reproduction scope
- One model: **Llama-3.2-1B** (512 heads, fits a modest GPU; GPT2-XL is the cheaper fallback).
- One dataset: **Web phishing URLs** (`https://doi.org/10.17632/c2gw7fy2j4.3`) — the largest
  claimed AHR margin (+5.8 over best baseline at |D_train| = 10) and short inputs, so it is
  fast. **Jailbreak** (`jackhhao/jailbreak-classification`, on HF, trivial to load) is an
  equally good and more convenient choice.
- |D_train| = 10 only, 3-5 seeds rather than 10 (the paper stresses seed variance is large —
  do not report a single seed).
- Fix hyperparameters instead of running cross-validation (lr 0.003, ~50 steps, small L2).
- Baseline: LoRA r=1, alpha=1 via `peft`.
- Must implement **IC-FT** (Sec 3.1, k=10) — the paper applies it to AHR *and* baselines, and
  the ablation says it is worth ~10 points. Comparing AHR-with-IC-FT against plain LoRA would
  be an unfair comparison and would not reproduce the claim.
- Score only the **first answer token** (paper does this for classification).

### Reproduction risks
- The headline is a **few-point gap with high seed variance** (std up to ~4% in Appendix B).
  Few seeds may not resolve a 2-4 point effect. This is the main threat to a clean result.
- Fixed hyperparameters deviate from the paper's per-seed CV protocol; the paper does not
  report which values CV selected, so exact numbers will not match.
- Gated weights: Llama-3.2 requires HF license acceptance; Qwen3-8B does not.
- Head/layer indices cited in the analysis (L15H14, L15H3, L14H11) are Llama-3.2-1B specific.
