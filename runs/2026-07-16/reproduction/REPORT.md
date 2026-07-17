# Reproduction Report — *Data-Efficient Adaptation of LLMs via Attention Head Reweighting* (AHR)

**Paper:** arXiv:2607.13425 (COLM 2026) · **Run:** 2026-07-16 · **Status:** COMPLETE — all 18 pre-registered cells ran; every number below is from `results.json`.

---

## Headline

**At our scale, AHR never beat doing nothing.** On both models the trained method scored at or
below the frozen model given the same ten examples in its prompt: **−2.22 ± 3.14** points on
GPT-2-small (inside seed noise — no direction declarable) and **−3.33 ± 1.36** on Qwen2.5-0.5B
(outside seed noise — a real decline at this scale).

**Our core pre-registered experiment, P3 — the gradient — is NOT supported.** We predicted AHR's
advantage over ICL would shrink as base-model competence rises, with ≥3.0 points of separation
between our weak and strong models. Observed separation: **1.11 points**, below both the
pre-registered threshold and the pooled seed spread (3.42). We cannot declare the gradient at this
scale, and we say so rather than reaching for the partial-credit clause (see *Scoring integrity*).

**What did reproduce, decisively: LoRA at 10 examples is catastrophic.** On Qwen2.5-0.5B, LoRA
scored **−25.83** points versus the frozen model. AHR beat LoRA by **+22.50** — nominally a
thumping replication of the paper's headline — but our ICL arm shows the entire margin, and more,
is *LoRA collapsing*, not AHR winning. This is the paper's central decomposition (`analysis/core.md`
§4: "91.5% of AHR's margin over LoRA is LoRA failing to beat the frozen model") reproduced
independently at 0.5B scale, in a sharper form than the paper's own numbers.

---

## What we set out to test, and why not the headline

The paper's headline is "AHR beats LoRA at ≤100 examples." Re-running that would burn the time box
confirming something the paper's own Appendix B already establishes and nobody disputes.
`analysis/core.md` §4 identified a different, cheaper, genuinely disputed claim — **the gradient**:
AHR's advantage over doing nothing (ICL with the same 10 examples in the prompt) *shrinks as the
base model gets better at in-context learning*, and is already negative for the paper's two
strongest models. Evaluating ICL costs zero training, which is the only reason this fits in 90
minutes on a laptop CPU.

Pre-registered in `predictions.json` **before any code ran**: P5 (correctness gate), P1/P2
(replicate the paper — these only validate the harness), and **P3/P4 (the gradient — the actual
experiment, and the ones that can embarrass us).** P3 duly embarrassed us.

---

## Tier achieved

**Tier 2 — minimal reimplementation.** Not a choice between tiers so much as a forced move: Tier 1
is impossible because **the official repo contains no code** (`ingest/repo-notes.md` — it is a
README-only placeholder). There is nothing to run. AHR is reimplemented from the paper's Eq. 3 in
~100 lines (`minimal/ahr.py`), gated on an exact-equivalence check against the frozen model (P5).

Because no third-party code is executed, the untrusted-code concern is moot — every line run here
is our own or a mainstream library (`transformers`, `peft`).

## Compute rung

**Rung 0 — local CPU.** No discrete GPU (Intel i5-1235U, integrated graphics;
`torch.cuda.is_available() == False`). Rung 1 (Kaggle GPU) was not needed: the experiment was
designed to fit on CPU at 124M–0.5B scale. Total runtime ≈ 40 min of compute inside the 90-min box.

## Environment

Collected at runtime → `env-info.json`.

| | |
|---|---|
| OS | Windows 11 (10.0.26200) |
| CPU | Intel i5-1235U — 10 physical / 12 logical cores |
| RAM | 16.9 GB |
| GPU | none (CUDA unavailable) |
| Python | 3.13.7 |
| torch | 2.9.1+cpu |
| transformers | 5.3.0 |
| peft | 0.19.1 |
| datasets | 4.8.5 |

Pinned in `requirements.txt`. Packages were already installed system-wide, so no venv was created —
see `LAB-NOTES.md` for why that deviation from the stage's `uv` instruction was taken deliberately.

---

## Exact commands run

```bash
cd minimal
python gate_p5.py                                   # P5 correctness gate — must PASS first
python run_experiment.py --models gpt2 Qwen/Qwen2.5-0.5B \
    --arms icl ahr lora --seeds 0 1 2 --n-eval 120 --out ../results.json
python analyze.py                                   # aggregation + verdicts → ../summary.json
```

## Configuration (and how it departs from the paper)

| | Ours | Paper |
|---|---|---|
| Models | GPT-2-small (124M), Qwen2.5-0.5B | GPT2-XL, Llama-3.2-1B/3B, Qwen3-8B |
| Dataset | SST-2, 120-example fixed eval subsample | 6 datasets, full test sets |
| \|D_train\| | 10 | 10–100 |
| Seeds | 3 | 10 |
| Hyperparameters | **fixed** (lr 3e-3, 50 steps, λ=0.01, p=2) | **168-config CV per seed** |
| IC-FT | k=10, both arms | k=10, all arms |

The eval subsample (rng seed 1234) is **identical across every model, arm and seed**, so all
comparisons are paired. It is 54% positive.

**The no-CV deviation is the most important caveat in this report, and it cuts against our own
LoRA result.** The paper's CV grid contains `steps = 0`, which — since every method is zero-init —
means "fall back to the frozen model." Our LoRA has no such escape hatch, so our −25.83 is
*harsher* than the paper's CV'd LoRA numbers. `analysis/core.md` §T2/F3 predicted precisely this:
that LoRA's catastrophic |D|=10 numbers are substantially a model-selection artifact. Our run is
evidence about **untuned LoRA at 50 steps**, not about the paper's CV'd LoRA. Both arms are equally
untuned, which preserves the fair-comparison discipline, but absolute numbers are not the paper's
and are not claimed to be.

---

## P5 — the correctness gate (passed before anything else was run)

| Model | γ params | max abs dev at γ=0 | gate | hook-is-live check |
|---|---|---|---|---|
| GPT-2-small | 12×12 = **144** | **0.000e+00** | PASS (< 1e-4) | 7.998 |
| Qwen2.5-0.5B | 24×14 = **336** | **0.000e+00** | PASS (< 1e-4) | 8.425 |

γ=0 reproduces the frozen model **bitwise**, not merely within tolerance — so the forward pre-hook
is an exact implementation of Eq. 3, and the paper's zero-init property ("the model at
initialization *is* the frozen model") is confirmed. The last column is the anti-trivial check:
perturbing a single γ to 0.5 moves the logits by ~8, so the gate is not passed by a hook that
silently does nothing. **Verdict: P5 supported.** Everything below is admissible.

## A discrepancy found in the paper itself (unplanned)

Eq. 4's regularizer `λ(Σ_h|γ_h|^p)^(1/p)` is the Lp **norm**. The Lp norm is not differentiable at
the origin — and §3 specifies **γ is zero-initialized**. For p=2 (the paper's main setting),
`d/dγ √(Σγ²)` is `0/0` at γ=0, so the first backward pass from the paper's own stated
initialization emits a NaN gradient that poisons every γ. We hit this exactly: finite CE at step 0,
NaN thereafter.

**Eq. 4 as literally written, at the paper's own stated initialization, does not run.** This cannot
be checked against the authors' implementation because the repo has no code. The paper almost
certainly means squared L2 (`λΣγ²`, the universal convention), which is what we use — but it does
not say so, and a reader implementing Eq. 4 verbatim gets NaN. Minor as a correction; notable as a
sign that the released artifact cannot resolve it.

---

## Results

All accuracies are % on the same 120 SST-2 eval examples. **Mean ± population std across 3 seeds**,
per-seed values shown because the spread is the story on several rows.

### Per-arm accuracy

| Model | 10-shot ICL (no training) | AHR (144/336 params) | LoRA r=1 (36.9k/67.6k params) |
|---|---|---|---|
| GPT-2-small | **50.83 ± 7.07** <br><small>60.83 / 45.83 / 45.83</small> | 48.61 ± 3.93 <br><small>54.17 / 45.83 / 45.83</small> | 48.61 ± 3.93 <br><small>54.17 / 45.83 / 45.83</small> |
| Qwen2.5-0.5B | **92.50 ± 0.68** <br><small>91.67 / 93.33 / 92.50</small> | 89.17 ± 2.04 <br><small>86.67 / 91.67 / 89.17</small> | 66.67 ± 13.81 <br><small>70.00 / 81.67 / 48.33</small> |

**The frozen model is the best arm on both models.** No training run of either method, at either
scale, beat leaving the weights alone.

### Paired deltas (arms paired within seed — the seed picks the 10 examples)

| Model | AHR − ICL | LoRA − ICL | AHR − LoRA |
|---|---|---|---|
| GPT-2-small | **−2.22 ± 3.14** <br><small>−6.67 / 0 / 0</small> | −2.22 ± 3.14 <br><small>−6.67 / 0 / 0</small> | 0.00 ± 0.00 <br><small>0 / 0 / 0</small> |
| Qwen2.5-0.5B | **−3.33 ± 1.36** <br><small>−5.00 / −1.67 / −3.33</small> | −25.83 ± 13.59 <br><small>−21.67 / −11.67 / −44.17</small> | +22.50 ± 13.25 <br><small>+16.67 / +10.00 / +40.83</small> |

### Ours vs pre-registered prediction vs paper

| | Our result | We predicted | Paper's SST-2 claim (\|D\|=10) |
|---|---|---|---|
| AHR − ICL, weak model | −2.22 ± 3.14 | large positive (GPT2-XL basis +14.71) | +14.71 (GPT2-XL, ICL 72.00) |
| AHR − ICL, strong model | **−3.33 ± 1.36** | ≤ +3.0; fit says +0.51 | −0.11 (Llama-3.2-3B, ICL 92.71) |
| Gradient (separation) | **1.11** | ≥ 3.0 | 13.8 across their 4 models |
| AHR − LoRA, weak model | 0.00 ± 0.00 | \|·\| < 3.0 (§7) / ≥3.0 (predictions.json) | −0.15 (GPT2-XL) |
| AHR − LoRA, strong model | +22.50 ± 13.25 | ≥ 3.0 | +4.91 (Qwen3-8B) |
| LoRA − ICL, strong model | −25.83 ± 13.59 | < 0 | LoRA < ICL in 20/24 cells |

**Scale caveat, stated plainly:** 124M and 0.5B parameters on a laptop CPU, one dataset, 120 eval
examples, 3 seeds, no CV. The paper runs up to 8B on 6 datasets with 10 seeds and 168-config CV.
Absolute numbers are not expected to match and do not. Only direction-of-effect is testable here,
and on two of six rows above even the direction is inside our noise.

**One comparison is worth its own line.** Our Qwen2.5-0.5B sits at ICL **92.50**; the paper's
Llama-3.2-3B sits at ICL **92.71** — near-identical base competence on the same dataset. The paper
reports AHR − ICL = **−0.11** there. We measure **−3.33 ± 1.36**. Same sign, larger magnitude,
independently derived, at 1/6th the parameter count. That agreement in sign at matched competence
is the most meaningful correspondence in this report — and it is agreement that **AHR does not
help**.

---

## Verdict per pre-registered prediction

| ID | What it tested | Verdict |
|---|---|---|
| **P5** | γ=0 ≡ frozen model | **supported** (bitwise, 0.000e+00) |
| **P1** | AHR > LoRA | **conflicted — see below** (`predictions.json`: not supported · `core.md` §7: supported) |
| **P2** | LoRA < ICL | **conflicted — see below** (`predictions.json`: supported · `core.md` §7: not supported) |
| **P3** | **the gradient — THE EXPERIMENT** | **NOT SUPPORTED** (separation 1.11 < 3.0 threshold, and < 3.42 seed spread) |
| **P4** | AHR ≤ +3.0 on a modern model | **supported** (−3.33 ± 1.36; also within ±5 of the paper's SST-2 fit, which predicted +0.51) |

### P3 — why it failed, without excusing it

The pre-registered falsification condition fired: the two deltas fall within 3.0 points (1.11), so
P3 is not supported. Full stop. That is the verdict.

The *diagnostic* reason is worth recording, because it changes what a follow-up should do — but it
does not convert the verdict into a pass. **The low end of our axis is degenerate, not weak.**
GPT-2-small's 10-shot ICL is **50.83 ± 7.07** — at coin-flip, and *below the 54.17% majority-class
baseline* of our eval set, i.e. worse than the trivial strategy of always answering "Positive". It
is one-class-dominant in **every** seed: seed 0 answers "Positive" for 91.7% of examples, seeds 1
and 2 answer "Negative" for **100%**. In seeds 1 and 2 *every arm scores identically 45.83* — the
model is pinned to one class and neither AHR nor LoRA moves it at all. A model that cannot do the
task in-context has no ICL machinery for AHR to amplify, so it cannot exhibit "large AHR gain over
weak ICL."

**Which bias, specifically — checked, not assumed.** The collapse direction is not a property of the
model alone; it flips with the prompt, and it tracks the *class balance of the ten demonstrations*
exactly. Seed 0 draws 6 positive / 4 negative demos and collapses to Positive; seeds 1 and 2 draw 4
positive / 6 negative and collapse to Negative — 3 of 3. This is **majority-label bias** (Zhao et
al. 2021, *Calibrate Before Use*), and it is specifically **not recency bias**: the last
demonstration is Positive in all three seeds, yet two of them collapse to Negative. The check costs
no compute (it reads only the sampled demo labels) and is reproducible via the snippet in
`LAB-NOTES.md` session 4. Caveat: n=3 seeds, so this is a consistent signal, not an established law.

So we did not span the paper's axis (**weak → good**, GPT2-XL at 72.00 → Qwen3-8B at 93.29). We
spanned **broken → good** (50.83 → 92.50). Our P0 pre-registration predicted GPT-2-small at 60–80.
The n=20 smoke probe returned **45.0**, and that is the number on which the (wrong) decision to
proceed was taken — recorded in `LAB-NOTES.md` before any full result existed, which is the only
reason it is usable as a diagnosis rather than a story. The full 120-example, 3-seed run puts it at
**50.83 ± 7.07**, still far outside the predicted band on the mean, though seed 0 alone (60.83)
lands inside it. Either way the miss is the direct cause of P3's failure. The prompt was fixed a
priori and **not** retuned after seeing the 45 — retuning it would have been exactly the
researcher-degrees-of-freedom move this stage's discipline exists to prevent.

**What this does and does not license.** It does not license "P3 would have passed with a better
weak model" — that is an untested hypothesis, and the honest statement is that **our two-point test
of the gradient failed, and the gradient remains supported only by the paper's own Appendix B.**
A follow-up needs a model that is *weak but not broken* at SST-2 ICL (GPT-2-large/XL, ~72; or
Qwen2.5-0.5B against a harder dataset where its ICL is mid-range).

### P1/P2 — a scoring conflict between two pre-run documents

`predictions.json` claims to transcribe `analysis/core.md` §7 "verbatim, not revised." It does not,
and the divergence flips both verdicts:

- **P1.** §7 calibrated to SST-2: AHR−LoRA ≥ 3.0 on Qwen, **no margin** (\|·\|<3.0) on GPT-2-small
  — and explicitly warned that a naive "AHR beats LoRA by ≥3 on both" prediction "would be
  falsified on GPT-2-small even by a perfect reproduction." `predictions.json` then registered
  exactly that naive both-models form, using the pooled 6-dataset basis (+6.23) instead of the
  SST-2 column (−0.15). Observed: Qwen +22.50 ✓, GPT-2 0.00. §7's form → **supported**.
  `predictions.json`'s form → **not supported**, falsified in precisely the way §7 predicted a
  perfect reproduction would be.
- **P2.** §7: LoRA < ICL on Qwen **and** LoRA > ICL on GPT-2 (falsified if either reverses).
  `predictions.json`: LoRA < ICL on ≥1 model. Observed: Qwen −25.83 ✓ decisively; GPT-2 −2.22 ± 3.14
  — **below seed spread, so by our own variance rule no direction is declarable there**. §7's form →
  not supported (its GPT-2 half is really *untestable*, not cleanly reversed).
  `predictions.json`'s form → supported.

Both documents pre-date the run, so neither is post-hoc, and I have reported both rather than
picking the flattering one. §7 is the better-calibrated instrument in both cases — it did the
dataset-specific homework that the transcription discarded. The substantive finding underneath is
unaffected and is not in doubt: **LoRA at \|D\|=10 on a competent model is a disaster (−25.83),
and AHR's +22.50 "win" over it is that disaster, not an AHR gain.**

### Scoring integrity

`predictions.json`'s P3 has **overlapping clauses**: `partially_supported_if` ("ordering holds,
separation < 3.0") and `not_supported_if` ("the two deltas are within 3.0 points") both fire at
separation = 1.11. My original scorer checked the partial clause first and returned "partially
supported." That is the flattering reading of a contradictory rule, and I overrode it: **a
pre-registered falsification criterion that fires is a falsification**, or the criterion is
unfalsifiable and worthless. `core.md` §7 — the source — has no partial clause and says plainly
"falsified if the two deltas fall within 3.0 points." The verdict is **not supported**.

Separately, the nominal ordering (−2.22 > −3.33) *does* hold, and it would be easy to sell that as
"the gradient's sign reproduced." It should not be sold: 1.11 points is under the pooled seed spread
of 3.42, so the ordering is not distinguishable from noise. Per the pre-registered variance rule,
the honest statement is **indistinguishable at this scale**.

The predictions file was **not edited** at any point. Only `minimal/analyze.py`'s scoring logic was
corrected, after results existed, and this section exists so a reader can audit that decision.

---

## Other observations from the run

- **LoRA seed 2 collapsed completely**: 48.33% accuracy, predicting "Negative" for 97.5% of
  examples, with final training loss **0.0001**. It drove its 10 training examples to zero loss and
  emerged a constant classifier. This is the paper's "too much capacity at ten examples" thesis
  (`core.md` §1) rendered about as literally as it can be — and the source of LoRA's ±13.8 spread.
- **AHR's γ stayed small and never collapsed**: |γ| mean 0.07–0.11, max 0.17, final CE 0.02–0.19
  (vs LoRA's 1e-4). AHR cannot memorize 10 examples because it cannot write new directions into the
  residual stream — only re-mix existing ones. **The paper's core mechanical claim is visible in our
  run and we confirm it.** AHR is robust where LoRA is not. What our data disputes is only whether
  that robustness buys anything *over not training at all*.
- **AHR is not free at 0.5B**: 130–140 s to train 336 scalars, essentially LoRA's cost (127–145 s),
  because both backpropagate through the full network. Confirms `core.md` §T7 — parameter count is a
  storage metric, not a training-cost metric.

---

## The banked result (holds regardless of what the compute produced)

`analysis/core.md` §4 contains a complete, zero-compute reproduction: a re-analysis of the paper's
**own Appendix B**, which reports the ICL baseline in all 24 model×dataset cells and omits it from
every main-paper table. It is reproducible from `ingest/_appendixB.json` and was validated against
the paper's Fig. A.3 (the reconstructed ICL row matches the paper's stated "baseline (unmodified)"
value of **70.68** to two decimals, confirming both the transcription and that the paper's
"unmodified model" already has the 10-shot prompt).

- At |D|=10, **ICL (72.26) outranks every trained baseline the paper reports** — LoRA 68.27,
  AdaLoRA 71.22, IA3 67.88.
- On the 3 modern models, **91.5% of AHR's margin over LoRA is LoRA failing to beat the frozen
  model.** Only 0.53 points is AHR beating "do nothing" (paired across 18 cells: t = 0.78, p ≈ 0.45).
- **AHR − ICL per model is monotone in base-model ICL competence** (+9.20, +3.42, −0.39, −1.45;
  Kendall τ = −1.00) and negative for both strongest models.
- Win rate collapses from **10/12 cells on the two weaker models to 2/12 on the two stronger**
  (Fisher exact **p = 0.0033**).

**How our compute changes the standing of that argument — precisely.** Our run **independently
confirms the sharp end** (P4: at ICL 92.50, AHR is −3.33 ± 1.36; the paper's own Llama-3.2-3B at
ICL 92.71 is −0.11 — same sign) and **independently confirms the LoRA-is-broken decomposition**
(−25.83). It **fails to confirm the gradient itself** (P3), because our weak-end model was
degenerate rather than weak. So the gradient rests on the paper's four models, as it did before
this run — **we did not add two points to it, and downstream stages must not claim we did.** Written
up as an amendment in `analysis-delta.md`.

---

## What a reader would need to replicate at full scale

- **Our run, as-is:** any CPU, ~40 min, no GPU, no downloads beyond `gpt2` and `Qwen/Qwen2.5-0.5B`
  from HF. `pip install -r requirements.txt`, then the three commands above. Seeds 0/1/2 and eval
  subsample seed 1234 are fixed; the P5 gate is bitwise, so a correct rebuild reproduces it exactly.
  Statistical reproducibility (same direction of effect) is the bar for the accuracy numbers, not
  bitwise identity.
- **The paper's actual protocol:** 4 models (to 8B) × 6 datasets × 6 train sizes × 10 seeds ×
  168-config CV ≈ 10⁵ training runs, each backpropagating through the full network. At Qwen3-8B
  that is a multi-GPU-week job — roughly 4×A100-80GB for several days, well beyond both this
  machine and this pipeline's zero-budget constraint. Not attempted; Rung 2 is explicitly out of
  scope.
- **The decisive follow-up, which is cheap** (`core.md` F1): a single-family competence sweep —
  Qwen2.5-0.5B/1.5B/3B/7B on SST-2 + one harder dataset, |D|=10, ICL vs AHR only, 10 seeds. That
  holds tokenizer and pretraining data constant while varying only competence, which is the one
  axis the claim is about, and it repairs our degenerate-low-end problem. One A100-day. It is the
  experiment the paper should have run, and the one our failed P3 still points at.
</content>
