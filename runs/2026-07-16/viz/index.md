# Visual assets — *Data-Efficient Adaptation of LLMs via Attention Head Reweighting* (AHR)

**Run:** 2026-07-16 · **Paper:** arXiv:2607.13425 (COLM 2026)
**Status:** ✅ complete — all 8 assets generated and validated against source data. Regenerate everything with `python viz/make_charts.py` (runs clean on matplotlib 3.10 / numpy 2.4; every number is recomputed from `ingest/_appendixB.json` + `reproduction/results.json`, none hand-typed).

## The one thing every asset here exists to show

The paper's own Appendix B contains the baseline its main tables omit: **ICL — the frozen model with the same ten examples in the prompt.** Put it back and AHR's advantage is a *decreasing function of base-model competence*, already negative for the paper's two strongest models. Every chart below is in service of that single claim.

## Provenance — read this before embedding anything

Three different classes of number appear in these figures, and each is marked **on the figure itself**:

| Marker | Meaning |
|---|---|
| **Recomputed (Appendix B)** | The paper's own published numbers, re-aggregated by us from `ingest/_appendixB.json`. Not our measurements — our *arithmetic* on their data. Validated: our reconstructed ICL row reproduces the paper's stated Fig. A.3 value of 70.68 exactly. |
| **Ours (reproduction)** | Measured this run on a laptop CPU, `reproduction/results.json`. Fixed hyperparameters, **no cross-validation**, 120 eval examples, ≤3 seeds. Not comparable to the paper's protocol in absolute terms. |
| **Paper (as reported)** | Quoted directly from a paper table/figure, unmodified. |

**The headline chart rests on recomputed Appendix B numbers, not on our reproduction.** This is deliberate and is stated in the figure's own subtitle. Our CPU reproduction is underpowered (see `reproduction/REPORT.md`) and at GPT-2-small scale it lands at chance with the classifier collapsing to a single class — it cannot carry the claim, and pretending otherwise would be the exact sin the analysis accuses the paper of.

---

## Assets

### 1. `headline.png` — the killer chart
**Caption:** AHR's benefit over doing nothing shrinks as the base model gets better at in-context learning, and has already gone negative by Llama-3.2-3B — a trend visible only in the paper's appendix, never plotted in the paper.
**Insight:** The method's value is inversely proportional to base-model quality. Its addressable market is models too weak to do ICL, which is a shrinking market.
**Data:** Recomputed from Appendix B — 4 models × 6 datasets at |D_train| = 10. Kendall τ = −1.00 (perfectly monotone).
**Embed in:** paper · blog (lede figure) · X thread (post 1) · LinkedIn.

### 2. `winrate.png` — the cliff
**Caption:** AHR beats the frozen model in 10 of 12 dataset cells on the two weaker models, but only 2 of 12 on the two stronger ones (Fisher exact p = 0.0033) — the benefit doesn't decay, it collapses.
**Insight:** The gradient is a cliff crossed between 1B and 3B, not a gentle slope. Makes the τ = −1.00 result robust to "you fit a line through four points."
**Data:** Recomputed from Appendix B, cell-level win counts.
**Embed in:** paper · blog · X thread (post 3).

### 3. `decomposition.png` — where the headline actually comes from
**Caption:** On modern models, 91.5% of AHR's advertised 6.23-point margin over LoRA is LoRA failing to beat a frozen model; only 0.53 points (p ≈ 0.45) is AHR beating "do nothing."
**Insight:** The abstract's headline is mainly a fact about LoRA being broken at ten examples, not a fact about AHR.
**Data:** Recomputed from Appendix B, modern-3 models × 6 datasets at |D| = 10.
**Embed in:** blog · X thread (post 2) · LinkedIn.

### 4. `method.md` (mermaid) + `method.svg` — method diagram
**Caption:** AHR adds exactly one scalar per attention head — a volume knob on machinery the frozen model already computes — and changes nothing else.
**Insight:** Makes the "minimal change" visible: the delta is one multiply per head against an otherwise untouched network, and at γ=0 the model *is* the frozen model (which is why `steps=0` in the hyperparameter grid quietly means "fall back to ICL").
**Embed in:** paper · blog · LinkedIn.

### 5. `explainer.png` — concept explainer
**Caption:** LoRA can write new features into the model; AHR can only turn existing ones up or down — and at ten examples, that inability is the whole advantage.
**Insight:** For a smart non-expert: capacity is a liability under a tiny label budget. Explains *why* 512 scalars beat 106,496 LoRA parameters without any math.
**Embed in:** blog · X thread (post 4) · LinkedIn (lead image).

### 6. `repro_sst2.png` — our reproduction, honestly scoped
**Caption:** Our laptop reproduction on SST-2, plotted against the paper's own SST-2 column on the same axis — underpowered, and at GPT-2-small scale it lands at chance.
**Insight:** Shows what a 90-minute CPU box can and cannot settle, and marks our points as distinct from the paper's. Included for honesty, not for support.
**Data:** Ours (reproduction) + Paper (Appendix B, SST-2 column).
**Embed in:** paper (appendix) · blog (methods section). **Not** for social — it is a caveat figure, not a claim figure.

---

*Design rules applied: one point per chart, stated in the caption; axes labelled with units; no truncated axes; provenance marked on every figure; fonts sized for social.*
