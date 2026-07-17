# Reviewer-defense drill — AHR content suite

The 10 most likely objections a hostile expert reader (NeurIPS reviewer #2, or a top HN/r/MachineLearning commenter) would raise against our content, ranked by probability. For each: the objection, whether our **evidence can answer it**, and the draft answer. Objections we *cannot* fully answer are marked **⚠ PARTIAL** and are acknowledged in the outputs themselves (see "Acknowledged-in-outputs" column at the end).

Grounding: all numbers below are recomputed from `ingest/_appendixB.json` (paper's own data) or measured in `reproduction/results.json` (our run). Both were re-verified this stage — Appendix-B aggregates recomputed from source, and `analyze.py` re-run to regenerate `summary.json` bit-for-bit from `results.json`.

---

### 1. "Excluding ICL from the main table is standard — ICL is capped by the context window and taxed on every query. Not a trick." — **ANSWERABLE**
At |D_train| = 10 the ten demonstrations fit comfortably: ICL is populated in all 24 appendix cells, and the context-window truncation only bites at *larger* |D| (and only for GPT2-XL's 1024-token window). The paper drops ICL from the main table precisely at |D| = 10 — where it wins (ICL 72.26 > AdaLoRA 71.22 > LoRA 68.27 > IA3 67.88). And AHR, as the paper measures it, keeps the *same ten demonstrations in every test prompt* (proven by the exact 70.68 = Fig. A.3 "unmodified" ↔ ICL-row identity), so AHR pays ICL's per-query cost too. The inference-cost argument does not separate AHR from ICL at |D| = 10.

### 2. "AHR still wins on average — 74.95 vs ICL 72.26, +2.7 points across four models. You're cherry-picking." — **ANSWERABLE**
The +2.70 average is carried entirely by GPT2-XL (2019, ICL competence 54.5), where AHR − ICL = +9.20. On the three modern models the margin is **+0.53** (paired across 18 model×dataset cells, t = 0.78, p ≈ 0.45 — indistinguishable from zero); on the two strongest it is **−0.91**. A benefit present on one of four models that reverses sign on the two best is not "on average."

### 3. "Your reproduction is 124M–0.5B on a laptop CPU, one dataset, 120 eval examples, 3 seeds, no CV. It proves nothing about an 8B-scale paper." — **⚠ PARTIAL**
Conceded — loudly, in every output. The reproduction is a *sign-check at the small end*, not a replication. The load-bearing argument (omitted ICL baseline + monotone gradient) is arithmetic on the paper's own Appendix B and needs no GPU. Our run contributes exactly one datum — AHR loses to doing nothing (−3.33 ± 1.36) on a competent 0.5B model whose ICL competence (92.50) matches the paper's Llama-3.2-3B (92.71), which reports −0.11 — plus an independent reproduction of the LoRA-collapse decomposition. We cannot speak to 3B–8B ourselves and never claim to.

### 4. "P3 failed because you picked a degenerate model. That's your error — you have no positive evidence for the gradient." — **⚠ PARTIAL (owned)**
Correct, and we do not launder it. GPT-2-small is degenerate at SST-2 ICL (50.83, *below* the 54.17% majority-class baseline; one-class-dominant in all three seeds via majority-label bias, not recency). We spanned broken→good, not weak→good, so the separation was unreachable by construction (measured 1.11, below both the 3.0 threshold and the 3.42 seed spread). The gradient rests on the paper's four models (τ = −1.00), not on our run. The failure is reported as a failure.

### 5. "τ = −1.00 is a line through four points. That's not a trend." — **ANSWERABLE**
It does not rest on the four-point fit. The independent, cell-level test uses all 24 cells: AHR beats ICL in 10/12 cells on the two weaker models and 2/12 on the two stronger — Fisher exact **p = 0.0033**. The cliff is what makes τ = −1.00 more than curve-fitting, and it is the reason `winrate.png` exists alongside `headline.png`.

### 6. "AHR's edge is on the atypical security tasks (Web, Jailbreak), which are least contaminated. Averaging over SST-2 buries it." — **ANSWERABLE**
Against the frozen model on modern models, AHR − ICL is **+1.24 on Web and +0.32 on Jailbreak** — small — and **negative on ToxiGen**. The advertised "6–7 points on security" is measured against the crippled LoRA/PEFT baselines, not against ICL. The contamination/atypicality argument is the paper's *own* explanation for the Web result, and it predicts exactly the vanishing gradient we chart.

### 7. "Once AHR is trained, drop the demonstrations and keep the merged γ — a permanent zero-context adapter. ICL can't do that. Did you test AHR *without* the in-context examples?" — **⚠ PARTIAL (genuine gap)**
No — and neither did the paper. The paper measures AHR *with* the ten demonstrations still in every test prompt (the 70.68 identity). Whether AHR keeps its accuracy demonstration-free is untested by us and by the authors. This is the single place AHR could hold a real, unmeasured advantage over ICL (amortize the demonstrations into γ once, pay zero context tax thereafter). We flag it as an open question rather than claim AHR buys nothing — and it is the sharpest missing experiment after LoFiT. *This is the strongest pro-AHR objection and is the one now acknowledged explicitly in the outputs.*

### 8. "The Lp-NaN and the uncited LoFiT/Voita/Michel are nitpicks. Obviously they mean squared-L2; prior-art gaps don't invalidate results." — **ANSWERABLE**
Agreed on severity for the NaN — we label it minor (a symptom of there being no released code to check against, not a fatal flaw). But LoFiT is not a nitpick: it trained AHR's exact Step-1 parameterization (a *vector* per head, same `(1+·)` form, same frozen backbone, same L1 penalty) and *discarded* it as a head-selection heuristic. That makes "does scalar-per-head actually beat vector-per-head at |D| = 10?" — the entire data-efficiency thesis — the paper's central untested comparison. An evaluation gap, not pedantry.

### 9. "You didn't run LoFiT either, and you're not citing it in a published paper. Pot, kettle." — **ANSWERABLE**
We are a reproduction flagging a missing baseline, not proposing a method or claiming novelty, so the citation standard differs. We couldn't run LoFiT on a 90-minute CPU box and say so explicitly; it is our #1 compute-bound follow-up (F2).

### 10. "The steps=0 / model-selection critique is speculation — you can't see which hyperparameters CV actually chose." — **⚠ PARTIAL**
True: the selected-hyperparameter table is unpublished — which is itself the ask (T2). But the mechanism is concrete, not speculative: steps=0 is in the grid, every method is zero-init, so steps=0 ≡ fall back to ICL, which explains AHR ≈ max(ICL, trained) and the "+0.53-and-never-much-worse" pattern exactly. LoRA landing 5.7 points below an option that was *in its own search space* is direct evidence CV failed to select it. Our own no-CV run cuts *against* us (our LoRA has no steps=0 escape hatch, so −25.83 is harsher than the paper's), which we flag.

---

## Which objections the evidence cannot fully answer (must be — and are — acknowledged in the outputs)

| # | Objection | Status | Where acknowledged in outputs |
|---|---|---|---|
| 3 | Reproduction scale | ⚠ partial | blog TL;DR + closing caveat, paper §4.1/abstract, x-thread 8, reddit, linkedin (all carry the scale caveat) |
| 4 | Degenerate weak model / P3 failed | ⚠ partial | blog "honest verdict", paper §4.3, x-thread 8, reddit — reported *as a failure* |
| 7 | AHR demonstration-free (untested) | ⚠ partial | **added this stage** to paper §3 (the deployment-cost subtlety paragraph) and blog "What we'd try next" |
| 10 | steps=0 is inference, not the selected-HP table | ⚠ partial | paper §5 T2, analysis T2 ("invisible without the selected-hyperparameter table the paper does not print") |

The remaining six (1, 2, 5, 6, 8, 9) are answerable directly from the evidence base and need no softening in the outputs — the outputs already carry the numbers that answer them.
