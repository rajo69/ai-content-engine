# Hyperparameter provenance

Every value used in this reproduction, with its source. **Assumed** values are the first suspects when numbers diverge from the paper.

Sources: `paper` = arXiv:2607.13425 (section noted), `analysis` = `analysis/core.md` §7 reproduction plan, `assumed` = our choice, not determined by the paper, **`code` = N/A: the official repo (`tuomaso/attention-head-reweighting`) contains only a README; no code exists to source anything from.**

## Method (AHR)

| Choice | Value | Source |
|---|---|---|
| Per-head scalar `γ_h`, residual contribution scaled by `(1 + γ_h)` | Eq. 3 | paper §3 |
| `γ` initialization | 0 (⟹ model at init is exactly the frozen model) | paper §3 |
| Trainable parameter count | = total attention-head count (144 GPT-2-small, 336 Qwen2.5-0.5B) | paper §3 (Table 1 method) |
| Implementation point | scale per-head slice of the attention output projection's **input** (`c_proj` GPT-2 / `o_proj` Qwen2) | paper §3 ("equivalently a per-head block scaling of `W_O`"); exact hook site **assumed** |
| Everything else frozen (all MLPs, all in-head weights, embeddings) | yes | paper §3 |
| Heads counted | query heads (`config.num_attention_heads`), not KV heads | **assumed** — paper predates GQA models in its own lineup? No: Qwen3-8B uses GQA and the paper reports head counts = query heads. Consistent with paper Table 1. |
| Loss | CE on answer tokens + `λ(Σ|γ_h|^p)^(1/p)` | paper Eq. 4 |
| `p` | 2 (L2) | paper §3 (L2 for main results; Table A.1 shows L1 vs L2 differ <0.6pp) |

## In-Context Finetuning (IC-FT)

| Choice | Value | Source |
|---|---|---|
| IC-FT used for **all** arms (AHR and LoRA) | yes | paper §3.1 — mandatory; worth ~10pp, omitting it for the baseline would manufacture AHR's win |
| `k` (shots) | 10 | paper §3.1 (never varied — paper does no sensitivity analysis on k) |
| Objective | maximize `P(y_i | X_i)` for all i=1..k | paper Eq. 5 |
| Tokens scored | **first answer token only** | paper §3.1 |
| Efficient equivalence | all k terms of Eq. 5 computed in **one** forward pass over the concatenated k-shot sequence, gathering logits at the k positions preceding each answer's first token | **assumed (but exact)** — causal masking makes position `pos(y_i)-1` condition on exactly `{x_1,y_1,…,y_{i-1},x_i}` = `X_i`. Mathematically identical to k separate passes, ~k× cheaper. |
| Test-time prompt contains the 10 in-context examples | yes | analysis §3 **[inference, arithmetically verified]** — Fig. A.3 baseline 70.68 == reconstructed ICL row 70.68. AHR = ICL + trained reweighting over the same 10 examples. |
| Example order within the k-shot sequence | fixed per seed; **train and test prompts use the identical order** | **assumed** — with \|D_train\|=10 and k=10 there is exactly one example set; order is arbitrary and the paper does not specify it. Fixed order keeps the train prompt and test prefix consistent. |

## Training

| Choice | Value | Source |
|---|---|---|
| `|D_train|` | 10 (only) | analysis §7 — the disputed regime |
| Optimizer | AdamW | **assumed** (paper does not state) |
| Learning rate | 3e-3, **same for AHR and LoRA** | paper's grid is {0.03, 0.003, 0.0003}; we take the middle value. **Fixing it (rather than searching) is assumed** — see below. |
| Training steps | 50, **same for AHR and LoRA** | paper's grid is {0, 5, 10, 20, 50, 100, 200}; we take 50. **Assumed.** |
| `λ` (L2 on γ) | 0.01 | paper's grid is {0, 0.1, …, 100.0}; we take a small value. **Assumed.** |
| Batch | full-batch — one 10-shot sequence, 10 scored answer tokens per step | **assumed** (follows from \|D_train\|=k=10) |
| Seeds | 3 | analysis §7 (paper uses 10; 3 is the honest CPU minimum, and we report spread) |
| What the seed controls | which 10 SST-2 train examples are sampled, and their order | paper §4 ("seeds over choice of examples") |

## **The deliberate deviation: no cross-validation**

The paper runs **5-fold CV over 3 lr × 7 steps × 8 λ = 168 configurations per seed**, selecting by validation *loss* on 2 held-out points. We run **one fixed configuration, no search**, for every arm.

This is a real deviation and it cuts both ways — stated plainly because it is the main reason our absolute numbers will not match the paper's:

- **It removes the `steps = 0` escape hatch.** Analysis §5/T2 argues that a large part of AHR's reported margin is that CV can silently fall back to the frozen model (`steps=0` ⟹ zero-init ⟹ ICL) for AHR while failing to do so for LoRA. With no CV, **neither** arm gets that fallback. Our AHR is therefore measured *without* its safety net, and our LoRA *without* the selection failure the analysis blames for its collapse.
- **Consequence for P1/P2:** both predictions restate the paper's numbers, which were produced *with* CV. Our setup is a fair like-for-like AHR-vs-LoRA comparison (equal tuning budget: zero), but it is **not** a reproduction of the paper's selection procedure. A P1/P2 miss is therefore ambiguous — it may indict the paper's CV rather than its method.
- **Consequence for P3/P4 (the ones that matter):** these compare AHR against ICL *across models*, holding the configuration fixed. The gradient is a claim about how the AHR−ICL gap moves with base-model competence, and removing CV applies the same handicap to both models, so the comparison across the axis stays valid.

## Evaluation

| Choice | Value | Source |
|---|---|---|
| Dataset | SST-2 (`stanfordnlp/sst2`) | analysis §7 (ungated; short inputs fit GPT-2's 1024-token window at 10 shots) |
| Test set | fixed 200-example subsample of the validation split, **identical across all models/arms/seeds** | analysis §7 (paper uses full test sets) |
| Metric | accuracy | paper §4 |
| Scoring | one forward pass; compare first-token logits of " Positive" vs " Negative" — no generation | analysis §7 |
| Models | GPT-2-small (124M, 12×12=144 heads); Qwen2.5-0.5B (24×14=336 heads) | analysis §7 — the two ends of the base-competence axis that CPU allows |
| Arms | ICL (no training), AHR (trained), LoRA r=1 α=1 (trained) | analysis §7 — comparative arm mandatory |

## LoRA arm specifics

| Choice | Value | Source |
|---|---|---|
| `r` | 1 | paper §4 (chosen by the authors for "training stability") |
| `α` | 1 | paper §4 |
| Target modules | `["c_attn"]` (GPT-2) / `["q_proj","v_proj"]` (Qwen2) | **assumed** — paper does not name target modules; these are the peft/LoRA-paper defaults for each architecture |
| Dropout | 0 | **assumed** |
| lr / steps | 3e-3 / 50 — **identical to AHR** | **assumed**, and deliberately equal: comparative claims require the same tuning budget for both arms |
