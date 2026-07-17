# Amendment to `analysis/core.md` — issued by the reproduction stage

**Status:** downstream stages (viz, content) must treat this file as an amendment to
`analysis/core.md`. Where the two conflict, this file wins, because it is backed by measurements in
`results.json`.

**Scope:** this amends §7 (the reproduction plan and its pre-registered predictions) and qualifies
how §4's gradient claim may be described downstream. **It does not retract §4's re-analysis of
Appendix B**, which is arithmetic on the paper's own published tables and is unaffected by anything
that happened on this laptop.

---

## 1. The headline claim downstream MUST NOT make

**Do not say we independently reproduced "the gradient."** We did not. P3 — the pre-registered
core experiment — is **not supported**. The separation between our two models' AHR−ICL deltas is
**1.11 points**, below both the pre-registered 3.0 threshold and the pooled seed spread (3.42).

§7 stated: "A confirmed P3 plus the banked Appendix B re-analysis is a complete argument: the
paper's own data shows the trend across 4 models, and we independently reproduce its sign on 2
more." **The second half of that sentence is now false.** The gradient rests on the paper's four
models exactly as it did before this run. Any downstream claim of "we reproduced it on two more
models" would be fabrication.

## 2. §7 claims that are now falsified by measurement

| §7 claim | Status | Measured |
|---|---|---|
| P0: "GPT-2-small ICL ≈ 60–80" on SST-2 | **wrong** | **50.83 ± 7.07** — at coin-flip, *below* the eval set's 54.17% majority-class baseline; one-class-dominant in all 3 seeds |
| P0: "Qwen2.5-0.5B ≈ 85–92" | **marginal miss (high)** | **92.50 ± 0.68** — 0.5 pt above the predicted band |
| "GPT-2-small would reproduce the GPT2-XL regime (+9.20)" | **wrong** | AHR−ICL = **−2.22 ± 3.14**; all arms identical in 2 of 3 seeds |
| P3: separation ≥ 3.0 points | **falsified** | **1.11** |
| P1 (as written in `predictions.json`): AHR−LoRA ≥ 3.0 on **both** models | **falsified** | GPT-2 **0.00** |
| P2 (as written in §7): LoRA > ICL on GPT-2-small | **not confirmed** | −2.22 ± 3.14 → indistinguishable |

**Root cause of the P3 failure, for the record:** §7 chose GPT-2-small as the weak end assuming it
would land in the paper's *weak-but-functional* ICL regime (GPT2-XL, 72.00 on SST-2). It is instead
**degenerate**: it cannot do SST-2 in-context at all, and neither AHR nor LoRA moves it (seeds 1
and 2: every arm scores exactly 45.83). A model with no functioning ICL machinery has nothing for
AHR to amplify, so the "large gain at the weak end" that P3 needed was unreachable by construction.
We spanned **broken → good** (50.83 → 92.50), not **weak → good**.

The failure mode is **majority-label bias** (Zhao et al. 2021), verified rather than assumed: the
direction of GPT-2-small's collapse tracks the class balance of its ten demonstrations in 3 of 3
seeds (6 pos/4 neg → answers Positive; 4 pos/6 neg → answers Negative). It is **not** recency bias —
the final demonstration is Positive in all three seeds, yet two collapse to Negative.

That is a *diagnosis*, not an excuse, and it must not be laundered into "P3 would have passed with a
better model." It is untested. `analysis/core.md` §7's own model-choice reasoning was wrong, and the
verdict stands as not supported.

## 3. §4 claims that our run independently STRENGTHENS

These downstream may state as reproduced, with the scale caveat (124M–0.5B, SST-2, 3 seeds, no CV):

1. **"AHR does not beat doing nothing on a competent base model" (§4, T1, P4) — confirmed at a new
   point.** At ICL 92.50, AHR is **−3.33 ± 1.36** (magnitude exceeds seed spread, so the direction
   is declarable). The paper's own Llama-3.2-3B sits at near-identical competence (ICL 92.71) and
   reports **−0.11**. Same sign, larger magnitude, independently derived.
2. **"91.5% of AHR's margin over LoRA is LoRA failing to beat the frozen model" (§4) — confirmed,
   in stronger form.** Our Qwen2.5-0.5B: AHR−LoRA = **+22.50**, LoRA−ICL = **−25.83**, AHR−ICL =
   **−3.33**. Here **more than 100%** of AHR's margin is LoRA collapsing; AHR itself is net
   negative. The paper's headline comparison, reproduced, points the opposite way once the omitted
   baseline is restored.
3. **"Capacity is actively harmful at ten examples" (§1, §2) — confirmed mechanically.** LoRA seed 2
   drove training loss to 1e-4 and emerged a constant classifier (97.5% one class, 48.33%). AHR's γ
   stayed at |γ|≈0.07–0.11 and never collapsed. AHR's inexpressiveness is real and protective — the
   paper's central mechanism is sound. Only its *benefit over not training* is in dispute.
4. **"Parameter count is not a training-cost metric" (§T7) — confirmed.** AHR trains 336 scalars in
   130–140 s; LoRA trains 67,584 in 127–145 s. Identical, both backprop through the full network.

## 4. New findings not in `analysis/core.md` at all

1. **Eq. 4 does not run at the paper's own initialization.** `λ(Σ|γ_h|^p)^(1/p)` is the Lp *norm*,
   non-differentiable at the origin; §3 zero-initializes γ. For p=2 the first backward pass is 0/0 →
   NaN, which poisons every γ. We hit it exactly. Unresolvable against the authors' code because the
   repo has no code. This is a **new, concrete, checkable defect** in the paper — good content, and
   §4/§5 did not have it.
2. **`predictions.json` misquoted §7 on P1/P2**, using the pooled 6-dataset basis (+6.23) where §7
   had explicitly calibrated to the SST-2 column (−0.15) and *warned* that the naive form "would be
   falsified on GPT-2-small even by a perfect reproduction." Both scorings are reported in
   `REPORT.md`; neither is quietly preferred. §7 was right and its transcription was wrong.
3. **GPT-2-small at 124M is unusable as a weak-ICL probe on SST-2**: 50.83 ± 7.07, below the eval
   set's 54.17% majority-class baseline, and one-class-dominant in every seed via **majority-label
   bias** — the collapse direction is set by the demonstrations' class balance (3/3 seeds), not by
   the input and not by recency. Useful for anyone planning F1: a weak-ICL probe must be checked for
   degeneracy with a `frac_pred_pos`-style diagnostic *before* it is adopted as an axis endpoint,
   because accuracy alone cannot distinguish a broken model from an honest coin flip.

## 5. Net effect on the story

The reframing in `analysis/core.md` **survives and is better evidenced at the sharp end, but its
independent-replication leg is gone.**

- Still true, still the strongest thing we have: the paper omits ICL from every main table; ICL
  outranks every trained baseline it does report; AHR's margin over LoRA is mostly LoRA being
  broken. **All of this is from the paper's own Appendix B and our run independently agrees with it
  at 0.5B.**
- Now measured, and new: **on a competent small model, AHR is actively worse than doing nothing
  (−3.33 ± 1.36).**
- Now off the table: the claim that we extended the gradient to two fresh models. The gradient is
  the paper's own four points, τ = −1.00 — cite it as *their* data, not ours.

The honest one-line version for downstream: **"The paper's own appendix shows AHR's advantage
vanishing as models improve; we ran it on a modern small model and AHR lost to doing nothing — and
our attempt to trace the full trend failed, because the weak model we picked couldn't do the task
at all."** The failure is part of the story and should be told, not trimmed.
</content>
