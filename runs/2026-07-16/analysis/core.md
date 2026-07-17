# Core Analysis — *Data-Efficient Adaptation of LLMs via Attention Head Reweighting* (AHR)

**Paper:** Oikarinen, Chen, Siska, Weng, Singh & Gao. COLM 2026. arXiv:2607.13425v1.
**Inputs:** `ingest/paper.md` (full, two passes), `litreview/lineage.md`, `ingest/repo-notes.md`, `ingest/_appendixB.json`.
**Status of this document:** all numeric claims below were recomputed from the paper's own Appendix B tables (24 tables, 4 models × 6 datasets) and are reproducible from `_appendixB.json`. Where I infer something the paper does not state, it is marked **[inference]** and, where possible, validated arithmetically against the paper's own figures.

---

## 0. The research question

**Does learning one scalar per attention head actually adapt a model to a task — or does it merely restore in-context-learning ability that a weak model was already failing to express, in which case the method's value should evaporate exactly as base models get good?**

The paper asks "can we beat LoRA with 500× fewer parameters at |D_train| ≤ 30?" and answers yes. That question has a load-bearing gap: LoRA is not the thing to beat. The thing to beat is *doing nothing* — putting the same ten examples in the prompt and running the frozen model. The paper measures that baseline, reports it in Appendix B, and omits it from the headline table. Our question is what happens when you put it back.

The answer, from the paper's own numbers, is section 4's subject and determines everything downstream: **AHR's advantage over doing nothing is monotonically decreasing in base-model competence (Kendall τ = −1.00 across the four models) and is already negative for the two strongest models tested.** On the two weaker models AHR beats a frozen model with the same ten examples in the prompt in 10 of 12 dataset cells; on the two stronger ones, in 2 of 12 (Fisher exact **p = 0.0033**). The paper ran this comparison, printed it in Appendix B, and left it out of every main-paper table.

---

## 1. The problem

The real problem is **cheap task specialization under a label budget so small that the adaptation method's own capacity is the dominant source of error.** At |D_train| = 10, any method with thousands of free parameters can drive training loss to zero through routes that have nothing to do with the task. The failure isn't "not enough signal to learn the task"; it's "far too much freedom for the amount of signal available."

This bites hardest where the paper points: security classification. Phishing URLs, jailbreak prompts, and toxic text mutate faster than annotation cycles, so the labeled set is permanently small and permanently stale. You want the model updated this afternoon, from the twelve examples of the new attack that someone just triaged.

Prior approaches fail in three distinct ways, and it matters that they're distinct (citing `lineage.md`):

- **LoRA (Hu et al., 2022) and descendants — too much capacity, wrong failure mode.** LoRA's low-rank update is *unconstrained within its rank*: it can express arbitrary rank-1 transformations of any layer's weights, including transformations that encode "example 4 had the word 'bank'." AdaLoRA (Zhang et al., 2023a) reallocates the rank budget by importance and LoRA+ (Hayou et al., 2024) fixes learning rates, but both refine *within* LoRA's parameterization rather than questioning its granularity. The paper's data confirms the failure empirically and brutally: at |D_train| = 10, **LoRA is worse than the frozen model in 20 of 24 model×dataset cells**. LoRA at ten examples is not a weak adapter; it is a net-negative one.
- **ICL (Brown et al., 2020) — cheap and strong, but capped and taxed.** Min et al. (2022) showed ICL largely *locates* a task rather than learning one, and it's brittle to formatting. The decisive practical limits are mundane: the context window caps how many examples fit (visible in Appendix B as missing ICL entries for GPT2-XL, whose context is 1024 tokens), and every query pays the demonstrations' inference cost forever.
- **(IA)³ (Liu et al., 2022) — right principle, semantically meaningless unit.** It rescales frozen activations rather than editing weights — exactly AHR's move — but scales per hidden dimension, a unit with no functional referent, so it has no principled story about *which* knobs matter.

The gap: nobody had spent the parameter budget at a unit that *means* something. Heads mean something (Elhage et al., 2021; Olsson et al., 2022).

---

## 2. The key insight

**For a smart undergraduate.** A transformer's attention heads each do a small, fairly specific job, and each one adds its output into a shared running total — the residual stream. Because the additions are independent, "how loud is this head?" is a well-defined question with a one-number answer. Pretrained models already contain heads that notice the things your task cares about; a URL-shaped task doesn't need a *new* URL detector so much as it needs the existing one turned up. So instead of teaching the model anything new, give each head a volume knob, freeze absolutely everything else, and let ten examples do nothing but adjust the mix. There are only a few hundred knobs, so ten examples can't do much damage — and that's the point, not a compromise.

**Technically.** Following Elhage et al.'s decomposition of `W_O` into per-head blocks, the MHA update `z_att^(l+1) = z_r^l + Σ_h h(z_r^l)` becomes

```
z_att^(l+1) = z_r^l + Σ_{h ∈ H_(l+1)} (1 + γ_h) · h(z_r^l)          (Eq. 3)
```

with one scalar `γ_h` per head, zero-initialized, everything else — including all MLPs and all weights *inside* every head — frozen. Trainable parameters = the model's total head count: **512 for Llama-3.2-1B, 1,200 for GPT2-XL** — ~0.0001% of weights, 200–1000× below LoRA. At test time `γ` folds into `W_O`, so inference cost is unchanged.

**Pinning down "the minimal change."** The constraint is sharper than "few parameters," and this is what the paper gets right: AHR's hypothesis class is *closed under the pretrained model's existing computation*. Every reachable model is a re-mix of functions the frozen network already computes. It cannot invent a feature, only re-weight one. LoRA with r=1 has ~200× more parameters but the qualitative difference is categorical, not quantitative: LoRA can write *new* directions into the residual stream; AHR can only change the gain on existing ones. The bet is that at |D_train| = 10, expressiveness is a liability and this specific inexpressiveness is the mechanism.

**What the insight is not (from `lineage.md`).** The parameterization is not new. **LoFiT** (Yin et al., NeurIPS 2024) learns `(1 + A_i^l) ⊙ z^(l,i)` — a *vector* per head — on small labeled data with an L1 penalty and a frozen backbone, then **throws the scalars away**, using them only to rank heads before learning bias vectors on the top-K. Voita et al. (2019) and Michel et al. (2019) had per-head gates in 2019, pointed at pruning. **None of the three is cited.** AHR's genuine delta is *granularity + retention*: one scalar instead of a `d_head` vector, and keep it as the method. That is a real contribution — "the thing you discarded works alone, and better 128× smaller" — but it is narrower than "we propose learning a scalar per head" implies, and it makes LoFiT the natural baseline the paper never ran.

---

## 3. Method mechanics

**Inputs.** A frozen causal LM with parameters θ; a few-shot set `D_train` of (text, label) pairs, |D_train| ∈ {10, 15, 20, 30, 50, 100}; a fixed instruction prompt naming the task and its class names (Table A.3).

**Trainable state.** One vector γ ∈ ℝ^(n_heads), init 0. Note `γ = 0` ⟹ Eq. 3 collapses to Eq. 1 ⟹ **the model at initialization is exactly the frozen model.** This is the zero-init-adapter property LoRA also has, and it has a consequence the paper never draws out (see §5).

**Forward.** Per attention layer, per head, scale that head's residual-stream contribution by `(1 + γ_h)`. Implementation is a scale on the per-head slice of the `o_proj` input, or equivalently a per-head block scaling of `W_O`.

**Objective.**

```
min_γ  Σ_{(x,y) ∈ D_train}  − log P(y | x; γ, θ)  +  λ ( Σ_h |γ_h|^p )^(1/p)      (Eq. 4)
```

cross-entropy on answer tokens plus an `Lp` penalty on γ, `p ∈ {1, 2}`. Since γ is zero-init, the penalty is a pull toward *the unmodified pretrained model* — it is a proximity-to-θ regularizer, not a generic shrinkage.

**In-Context Finetuning (IC-FT), §3.1 — carries far more weight than its half-page suggests.** Rather than one example per input, build `X_i = {x_1, y_1, …, y_(i−1), x_i}` — a k=10-shot prompt — and maximize the probability of *all k* answers inside it:

```
min_γ  − Σ_{i=1..k} log P(y_i | X_i; γ, θ)  +  λ ( Σ_h |γ_h|^p )^(1/p)          (Eq. 5)
```

Only the **first answer token** is scored (subsequent tokens are trivial given the first). IC-FT is applied to **AHR and every baseline**, so it is not a confound between methods — but it is worth ~10 accuracy points for all of them (Fig. 3a), and the paper notes standard FT **never** beats ICL in any setting. IC-FT is doing more work than AHR is.

**[Inference — verified] Test-time prompts contain the 10 in-context examples.** The paper never states this cleanly, and it changes the meaning of every number. I verified it arithmetically: Fig. A.3's "baseline (unmodified)" is reported as **70.68** for Llama-3.2-1B averaged over SST2/AGNews/Emotion/Web. Reconstructing the **ICL row** at |D_train| = 10 from Appendix B for that exact model and those exact four datasets gives **70.68** — an exact match to two decimals. So the "unmodified model" the paper compares against is the model *with a 10-shot prompt*, and AHR is measured as **ICL + trained head reweighting over the same ten examples**. AHR does not replace ICL; it rides on top of it. The `γ` merge is free, but the ten demonstrations are still in every query's context, so **"no additional inference cost" is true of AHR's parameters and false of AHR's deployment.**

**Load-bearing hyperparameters and what's unsaid.**

| Choice | Value | Load it carries |
|---|---|---|
| `k` (IC-FT shots) | 10, always | Worth ~10 points. **No sensitivity analysis whatsoever.** k is never varied. |
| lr | {0.03, 0.003, 0.0003} | Note 0.03 is enormous — viable only because γ is 512 numbers. |
| training steps | **{0, 5, 10, 20, 50, 100, 200}** | **Contains 0. See §5 — this is the single most consequential grid entry in the paper.** |
| λ | {0, 0.1, …, 100.0} | 8 values spanning 3 orders of magnitude. |
| `p` | L2 main results, L1 for §5 analysis | Table A.1: L1 vs L2 differ by <0.6pp, inside noise. Genuinely doesn't matter. |
| selection | 5-fold CV, **minimize val loss**, per seed | **See §5. At |D_train| = 10 this is 8 train / 2 val per fold.** |
| seeds | 10, over choice of examples | Correct and necessary — std reaches ±13.7 (IA3, Web/Llama-3B). |

The paper never reports **which** hyperparameters CV actually selected. That single omitted table would resolve the paper's central ambiguity, and §5 explains why.

---

## 4. Evidence

**Headline claims and their fair-comparison status.**

| Claim | Dataset / setting | Margin as stated | Fair? |
|---|---|---|---|
| "~3pp avg over baselines at ≤100 ex." | 6 datasets × 4 models, |D|=10 | AHR 74.95 vs LoRA 68.27 = **+6.68** | Fair **vs PEFT only**. Omits ICL (72.26), which beats LoRA by +3.98. |
| "6–7pp on Web & Jailbreak at 10 ex." | Web, Jailbreak, |D|=10 | Web +5.84, Jailbreak +6.75 vs best PEFT | **Misleading.** vs ICL: Web +4.86, Jailbreak **+2.34**. On modern models only: Web **+1.24**, Jailbreak **+0.32**. |
| "200–1000× fewer parameters" | Table 1 | 512 vs 106,496 (Llama-1B) | **True and verified** (= head count). The paper's cleanest claim. |
| "IC-FT worth ~10pp" | Fig. 3a | ~+10pp, all methods | Fair, and the largest real effect in the paper — but it's not AHR. |
| "AHR doesn't damage other tasks" | Fig. 4 | AHR slightly *improves* held-out tasks | True, and **evidence against the paper's own framing** (§5). |

**The single most impressive number.** Not an accuracy — the **parameter count, 512**, and specifically that 512 scalars at |D_train| = 10 beat 106,496 LoRA parameters by 6.23 points on modern models. That comparison is fair, replicated across 4 models × 6 datasets × 10 seeds, and the effect is large relative to noise. Something real is here: **capacity is actively harmful at ten examples.** Runner-up, and closer to the truth of the paper: **modifying 1 head out of 512 buys ~2% accuracy, 5 heads ~4%** (Fig. 3b). If that holds, the effective dimensionality of few-shot adaptation is ~5.

**The single most fragile number.** The **"6–7% on security tasks"** headline. It decomposes as follows at |D_train| = 10 (all recomputed from Appendix B):

```
                     AHR     LoRA     ICL    AHR-LoRA   =  (ICL-LoRA)  +  (AHR-ICL)
all 4 models        74.95    68.27   72.26     +6.68         +3.98         +2.70
modern 3 models     78.70    72.47   78.17     +6.23         +5.70         +0.53
```

**On the three modern models, 91.5% of AHR's margin over LoRA is LoRA failing to beat the frozen model. Only 8.5% — 0.53 points, a paired t of 0.78 across the 18 cells (p ≈ 0.45) — is AHR beating "do nothing."** The paper's headline is not mainly a fact about AHR. It is a fact about LoRA being broken at |D_train| = 10, which the paper *itself notes in passing* (Fig. 2b) and then builds its abstract on top of anyway.

Worse for the security framing specifically: on **ToxiGen** at |D|=10, AHR scores 72.78 against ICL's 73.07 — **AHR is worse than doing nothing** on one of the three security-relevant tasks, while the paper advertises +7.94 over the best PEFT baseline there. Both numbers are true. Only one is in the abstract.

**The gradient — the most important table in this analysis.** AHR − ICL at |D_train| = 10, averaged over all 6 datasets, per model. The final column counts the datasets (of 6) where AHR actually beats doing nothing:

| Model | ICL competence | AHR | ICL | LoRA | **AHR − ICL** | AHR − LoRA | **Cells AHR > ICL** |
|---|---|---|---|---|---|---|---|
| GPT2-XL (2019, 1.5B) | 54.52 | 63.73 | 54.52 | 55.68 | **+9.20** | +8.04 | **5 / 6** |
| Llama-3.2-1B | 71.31 | 74.73 | 71.31 | 68.74 | **+3.42** | +5.99 | **5 / 6** |
| Llama-3.2-3B | 80.00 | 79.61 | 80.00 | 78.53 | **−0.39** | +1.08 | **1 / 6** |
| Qwen3-8B | 83.20 | 81.75 | 83.20 | 70.15 | **−1.45** | +11.60 | **1 / 6** |

**AHR's benefit over not training at all is a decreasing function of how good the base model is at ICL, and it is already negative for both of the paper's strongest models.** The ordering is perfectly monotone in base-model ICL competence (Kendall τ = **−1.00**, all 6 model pairs discordant). The reported average is carried by GPT2-XL, a 2019 model whose ICL is so poor (54.52) that nearly anything helps.

**This is a cliff, not a slope — and that is the finding.** The win-rate column does not decay gently; it collapses. On the two older/weaker models AHR beats ICL in **10 of 12** cells; on the two stronger models, in **2 of 12**. A Fisher exact test on that 2×2 gives **p = 0.0033**. Whatever AHR does, it stops doing it somewhere between Llama-3.2-1B and Llama-3.2-3B — a boundary the paper crosses with two of its four models and never remarks on. Per-cell across all 24 cells at |D|=10, **AHR beats ICL in exactly 12 — a coin flip**; at |D|=15, 9 of 23. The largest single gap in the entire table runs the *wrong* way: Jailbreak/Llama-3.2-1B at |D|=50, ICL 95.69 vs AHR 88.97 — **ICL wins by 6.72 points on a headline security task**, and by 6.38 at |D|=100.

**Significance, done properly.** The modern-3 AHR−ICL delta is **+0.529**. Testing it correctly matters, because two defensible tests disagree and the difference is instructive:

- **Seed noise only** (pooling the paper's reported per-cell stds, n=10 seeds): SE of the 18-cell mean = 0.175 → **+3.0 SE, "significant."**
- **Paired across the 18 model×dataset cells** (the test that matches the claim "AHR helps on average across tasks and models"): the per-cell deltas range from **+6.86 to −4.41** with std **2.87**, giving SE 0.676 → **t = 0.78, df = 17, p ≈ 0.45. Indistinguishable from zero.**

The second is the right test, and the gap between them is the point: AHR's effect is *precisely estimated within each cell and wildly heterogeneous across cells*, so the seed-level error bars the paper reports are not the uncertainty that matters for a claim about "average" performance. **Only 7 of 18 modern-model cells are positive at all.** Restricting to the two strongest models (3B + 8B, 12 cells), the mean is **−0.91**, positive in **2 of 12**, t = −1.86 — AHR is, if anything, *worse than doing nothing*, marginally so.

The deeper problem for the headline: **AHR's entire positive average on modern models rests on a single model.** Strip out Llama-3.2-1B and the modern-model advantage over ICL is negative. A method whose benefit is carried by one of four tested models, and which reverses sign on the two strongest, is not described by "AHR outperforms baselines when learning from limited samples."

---

## 5. Threats to validity

**T1 — The omitted baseline is the whole argument.** ICL appears in all 24 Appendix B tables and in **none** of the main paper's result tables (Table 2 lists AHR/LoRA/AdaLoRA/IA3 only). The abstract's "outperform standard baselines like LoRA" is true; "baselines" silently excludes the one baseline that wins. This is the paper's central framing choice and it is not defensible: the paper *ran* the comparison and reported it where it does the least damage. Fig. 2b's admission that PEFT "harms performance relative to the ICL baseline" is presented as motivation for AHR rather than as the finding it is.

The omission is not a rounding-error's worth of spin. At |D|=10, ICL is the *only* baseline with all 24 cells populated (context limits truncate it at larger |D|, which is the honest reason to drop it later — but not at 10, where the paper's own headline lives). Putting it back: ICL averages **72.26**, beating LoRA (68.27), AdaLoRA (71.22) and IA3 (67.88) — **the frozen model outranks every trained baseline the paper reports**, and Table 2's ranking is an artifact of which rows were printed. AHR's remaining margin over it is +2.70 overall, +0.53 on modern models (t = 0.78, p ≈ 0.45), and −0.91 on the two strongest. A reader of Table 2 alone cannot recover any of this.

**T2 — Model selection has more capacity than the data can support.** Per seed, CV searches **3 × 7 × 8 = 168 configurations** using 5 folds of **8 train / 2 val** examples at |D_train| = 10 — ten validation points total, selecting among 168 options, by **validation *loss*, not accuracy**. This is selection overfitting on a scale that dwarfs the effect being measured. And it is *not* method-neutral, which is the subtle part: **the grid contains `steps = 0`**, and since every method is zero-init (LoRA's B=0, γ=0), `steps = 0` means *fall back to the frozen model* — i.e. to ICL. So every method has "do nothing" available, and AHR ≈ max(ICL, trained) almost by construction, which is exactly the +0.53-and-never-much-worse pattern we observe. LoRA, meanwhile, lands 5.7 points *below* an option that was in its own search space, meaning CV systematically failed to select `steps = 0` for LoRA — plausibly because 2 held-out points cannot detect LoRA's overfitting, or because low val loss and good accuracy diverge under overconfidence. **The honest reading: part of AHR's win is that it is robust to a broken model-selection procedure, not that it is a better adaptation method.** A method whose hyperparameters barely matter cannot be hurt much by choosing them badly. This confound is invisible without the selected-hyperparameter table the paper does not print.

**T3 — The generalization result contradicts the framing.** §5.2/Fig. 4: finetuning on Web with AHR *improves* accuracy on SST2/AGNews/Emotion. The paper sells this as "avoids destructive changes" and "generalizable edits." But a genuine *task-specific* adapter trained on phishing URLs has no business improving emotion classification. That AHR does is evidence it is not learning phishing — it is turning up general ICL machinery that the frozen model was under-using. Combine with §5.3's finding that L15H14/L15H3 get large positive γ **on every dataset**, and the task-adaptation story is in serious trouble.

**T4 — The paper's own ablation prices AHR's benefit at zero.** Fig. A.3: manually setting γ=+1 on **just L15H14 and L15H3**, with **no training data and no task knowledge**, moves Llama-3.2-1B from 70.68 → **71.67 (+0.99)** averaged over 4 datasets. Compare to AHR's **+0.53** average margin over ICL on the modern-3 models. **A free, task-agnostic, two-head, zero-data edit buys more than the full method's average advantage over doing nothing.** On Llama-3.2-1B itself (where AHR gets +3.53 over ICL on those 4 datasets), the free edit captures **28%** of the entire gain while touching 2 of 512 heads and reading zero labels. The paper presents this as *interpretability evidence*. It is better read as an ablation that undercuts the method — and the authors appear not to have noticed.

**T5 — Weak baseline, uncited stronger one.** LoRA at r=1, α=1 is chosen for "training stability," i.e. it is already the crippled-for-safety configuration, and it *still* loses to the frozen model in 20/24 cells at |D|=10. No baseline is tuned toward the obvious fix (early stopping on accuracy, higher regularization, or simply `steps=0`). Per `lineage.md`, **LoFiT is uncited** and is the exact experiment that adjudicates AHR's thesis: scalar-per-head vs vector-per-head at |D_train|=10. Without it, "coarser is better in low data" is asserted at one point on a two-point curve.

**T6 — Contamination and dataset atypicality.** SST2 (2013), AG News (2015), Emotion (2018) and ToxiGen are near-certainly in the pretraining corpora of Llama-3.2 and Qwen3 — consistent with ICL already scoring 92.71/93.29 on SST2 at |D|=10, i.e. near ceiling with no training. The paper's own explanation for AHR's Web success ("less typical of the models' training distribution, so more to gain from finetuning") is a *contamination argument in disguise*, and it predicts exactly the gradient in §4: the more the base model already knows, the less AHR has to offer. Ceiling effects also compress every SST2 comparison into a 2-point band where nothing can be distinguished.

**T7 — Compute and cost are never accounted.** AHR trains 512 parameters but requires **full backpropagation through the entire network** — the backward pass cost is essentially LoRA's. Parameter count is an optimizer-state and storage metric, not a training-cost metric, yet "200–1000× fewer" invites the reading that AHR is 200–1000× cheaper. Meanwhile, the per-seed CV over 168 configs means AHR's *reported* numbers cost 168 training runs each. The genuinely cheap baseline — ICL, zero training runs — is the one omitted.

**Silent assumptions collected while reading.**
1. **That the natural comparison for a few-shot adapter is another adapter, not the frozen model.** The paper's entire evaluation architecture encodes this, and it is the assumption that makes the results look strong.
2. **That head function is stable across input distributions** — asserted from Web-only attention maps, and hedged only in Limitations.
3. **That validation cross-entropy on 2 held-out points ranks 168 configurations.** Never examined.
4. **That k=10 is the right IC-FT budget.** Never varied, despite being worth ~10 points.
5. **That accuracy on a balanced test set is the right metric for security tasks.** For phishing/jailbreak detection, operating points are set by false-positive budgets; nobody deploys at the accuracy-optimal threshold. No ROC, no precision/recall, no FPR at fixed TPR — and the balanced Jailbreak variant is explicitly chosen, which is the least security-realistic prior available.
6. **That "trainable parameters" is a meaningful axis of merit per se**, rather than a proxy for overfitting resistance — which is what actually matters and is measured only indirectly.

---

## 6. Limitations and future work

**What the authors admit.** (i) Classification only; fine-grained-output tasks may lack the requisite heads. (ii) Head analysis is context-limited. (iii) **"If you have sufficient training data (e.g. ≥300 examples), other finetuning methods outperform AHR"** — an honest, well-supported concession, visible in their own Table 2 where LoRA retakes the average at |D|=100 (83.05 vs 82.68).

**What they don't admit.**
1. **ICL is excluded from every main-paper table while being present in all 24 appendix tables, and it beats AHR in half the cells at |D|=10.** Not mentioned in Limitations.
2. **The method's value is a decreasing function of base-model quality and is already negative at 3B and 8B.** The trend is fully determined by their own data and is never plotted — plotting it would require putting ICL in a figure.
3. **The 6–7pp security headline is ~91% LoRA-is-broken** on modern models.
4. **Fig. A.3 is an ablation against the method**, presented as interpretability.
5. **LoFiT, Voita et al. (2019), Michel et al. (2019) uncited** (`lineage.md`) — the first is the natural baseline, the latter two established the mechanism *and* the functional-specialization premise the abstract invokes.
6. **Nothing about deployment economics**, though "no additional inference cost" is claimed while the 10-shot prompt stays in every query.

**Concrete follow-ups.**

- **F1 — Plot the gradient (the paper's missing figure).** x-axis: base-model ICL accuracy. y-axis: AHR − ICL. Four model-level points already exist (+9.20, +3.42, −0.39, −1.45), perfectly monotone (τ = −1.00), and the underlying 24 cells show the win rate collapsing from 10/12 to 2/12 across the same axis (p = 0.0033). Extend across a controlled model-quality sweep (Qwen3-0.6B/1.7B/4B/8B — one family, one axis, holding tokenizer and data constant). The confound to kill: per-*cell* ICL accuracy mixes model competence with task difficulty, which is why the pooled cell-level slope (−0.158) is far shallower than the within-dataset slope (−0.682 on SST-2). A single-family sweep separates the two. If the trend holds, AHR's addressable market is models too weak to do ICL, which is a shrinking market. **This is the paper the authors should have written.**
- **F2 — Run LoFiT.** Scalar-per-head vs vector-per-head at |D_train| = 10, same protocol. The paper's entire "coarser is better" thesis is untested against the one method that isolates granularity.
- **F3 — Fix model selection, re-run everything.** Replace val-loss CV with (a) accuracy-based selection, (b) `steps=0` as a hard-coded fallback whenever val accuracy doesn't improve. Prediction: **LoRA's catastrophic |D|=10 numbers largely disappear and AHR's margin collapses toward zero.** If so, the paper measures a model-selection artifact.
- **F4 — Security-realistic metrics.** Re-evaluate Web/Jailbreak/ToxiGen at fixed FPR (e.g. TPR @ 1% FPR) on the *natural* class prior instead of accuracy on the balanced variant.
- **F5 — Our minimal-change extension: "the universal ICL vector."** §5.3 says a few heads get positive γ on *every* dataset; §5.2 says training on Web *improves* other tasks; Fig. A.3 says γ=+1 on two heads helps everywhere for free. Take these seriously: fit **one γ vector on one dataset (or average γ across datasets), then apply it, frozen, to entirely unseen tasks with zero task data.** If task-agnostic γ recovers most of AHR's average gain over ICL — and the paper's own three results predict it will — then **AHR is not a data-efficient adaptation method at all; it is an ICL amplifier that a single fixed 512-number vector, shipped once, could replace.** That reframes the contribution from "few-shot PEFT" to "pretrained models systematically under-weight their own ICL heads, and one universal constant fixes it." It is cheap, it follows directly from three of the paper's own findings, and either outcome is publishable: confirmation kills the method's premise, refutation rescues it and is the ablation the paper needed.

---

## 7. Reproduction plan

**Hardware reality (verified this session, and it is binding).** No CUDA GPU: Intel Iris Xe integrated graphics, `torch 2.9.1+cpu`, `torch.cuda.is_available() == False`. CPU is a 12th-gen i5-1235U (low-power laptop part), 15.7 GB RAM. `transformers`, `peft`, `datasets` are **not installed**. The paper's protocol — 4 models × 6 datasets × 6 sizes × 10 seeds × 168-config CV on full test sets — is ~10⁵ training runs; we have 90 minutes on a laptop CPU. Anything resembling replication is off the table, so the plan targets the *cheapest decisive question* instead.

**What to test, and why not the headline.** Re-running "AHR beats LoRA" would burn the box confirming something the paper's own appendix already establishes and that isn't in dispute. The disputed, high-value, and much cheaper claim is **the gradient (§4): AHR's advantage over ICL shrinks as base-model ICL competence rises.** ICL costs *zero training* to evaluate, which is what makes this fit in the box at all. Critically, the gradient means **model choice determines the answer**: GPT-2-small would reproduce the *GPT2-XL regime* (+9.20) and tell us nothing about the interesting one. So we must span the axis, not sit at one end.

**Setup.** `pip install transformers peft datasets accelerate` (~5 min). Reimplement AHR from scratch (~30 lines, per `repo-notes.md` — the repo is an empty placeholder, no code exists): forward hook scaling the per-head slice of each attention layer's `o_proj` input by `(1 + γ_h)`; γ zero-init; everything else frozen; verify `γ = 0` reproduces base-model logits **bit-exactly** (a mandatory correctness gate — it is the one place a silent bug would invalidate everything).

**Configuration.** Dataset **SST-2** (`stanfordnlp/sst2`, ungated, short inputs — fits a 10-shot prompt in GPT-2's 1024-token window). |D_train| = **10** only. **3 seeds** (not 10 — the paper stresses seed variance, so a single seed is uninterpretable; 3 is the honest minimum and we report spread, not just means). Fixed hyperparameters (lr 3e-3, 50 steps, small L2) — **no CV**, so numbers will not match the paper's exactly and we will say so. **IC-FT with k=10 for both AHR and LoRA** (mandatory — it's worth ~10 points; omitting it for the baseline would manufacture AHR's win). Score **first answer token only**: one forward pass per test example, comparing " Positive"/" Negative" logits — no generation. Test on a **fixed 200-example subsample** of SST-2 validation (not the full set).

**Models — the axis, low end and high end of what CPU allows:**
- **GPT-2 small** (124M, 12×12 = 144 heads) — weak-ICL end.
- **Qwen2.5-0.5B** (ungated; Llama-3.2 requires HF license acceptance, and Qwen3-0.6B is an equally good same-family substitute for the paper's Qwen3-8B) — modern-ICL end.

**Three arms per model per seed:** ICL (no training), AHR (trained), LoRA r=1 α=1 (trained, via `peft`). Nine runs per model. **The comparative arm is mandatory and included** — LoRA is run at exactly the same scale, same prompt, same IC-FT, same seeds.

**Calibration — predictions must be keyed to SST-2, not to the pooled averages.** This is a trap worth flagging loudly for the reproduce stage. The pooled modern-3 numbers (AHR−LoRA = +6.23) do **not** describe SST-2, which is the dataset the box can afford. The paper's own SST-2 column at |D|=10:

| Model | ICL | AHR | LoRA | AHR − ICL | AHR − LoRA |
|---|---|---|---|---|---|
| GPT2-XL | 72.00 | 86.71 | 86.86 | **+14.71** | **−0.15** |
| Llama-3.2-1B | 86.87 | 90.53 | 84.93 | +3.66 | +5.60 |
| Llama-3.2-3B | 92.71 | 92.60 | 92.18 | −0.11 | +0.42 |
| Qwen3-8B | 93.29 | 94.20 | 89.29 | +0.91 | +4.91 |

Two consequences. **(a)** On SST-2 with a weak model, the paper predicts **AHR ≈ LoRA (−0.15)** — so a naive "AHR beats LoRA by ≥3" prediction would be *falsified on GPT-2-small even by a perfect reproduction*. **(b)** SST-2 is nonetheless the **best** dataset for our actual question: regressing AHR−ICL on base ICL across these 4 models gives slope **−0.682 per point of ICL** (vs −0.158 pooled across all cells) — the steepest gradient in the paper, crossing zero at **ICL ≈ 93%**. The dataset that is worst for the paper's headline is best for ours.

**Pre-registered falsifiable predictions.**

- **P0 (measured first, costs nothing, keys everything else).** Measure each model's 10-shot ICL accuracy on SST-2 before any training. Expect GPT-2-small ≈ 60–80, Qwen2.5-0.5B ≈ 85–92. All predictions below are conditional on these.
- **P1 (replication, SST-2-calibrated).** On **Qwen2.5-0.5B: AHR > LoRA by ≥ 3.0 points** (paper's SST-2 basis: +5.60 at Llama-1B, +4.91 at Qwen3-8B). On **GPT-2-small we predict no margin: |AHR − LoRA| < 3.0** (paper's basis: −0.15). Falsified by AHR ≤ LoRA on Qwen2.5-0.5B, or by a >3-point gap in either direction on GPT-2-small.
- **P2 (paper's own admission, Fig. 2b).** **LoRA < ICL on Qwen2.5-0.5B.** Paper's SST-2 basis: LoRA loses to ICL for Llama-1B/3B/Qwen3-8B but *wins* by +14.86 on GPT2-XL — so we predict **LoRA > ICL on GPT-2-small**. Falsified if either ordering reverses.
- **P3 (our core claim — the gradient; this is the experiment).** **(AHR − ICL) on GPT-2-small exceeds (AHR − ICL) on Qwen2.5-0.5B by ≥ 3.0 points.** Paper's SST-2 basis: a 13.8-point spread (+14.71 → +0.91), slope −0.682/point of ICL. **Falsified if the two deltas fall within 3.0 points of each other, or if the ordering reverses** — either would mean AHR's value is not a function of base-model ICL competence and our reframing is wrong.
- **P4 (the sharp end).** On **Qwen2.5-0.5B: AHR − ICL ≤ +3.0 points**, and quantitatively within ±5 of the SST-2 fit `63.60 − 0.682 × ICL_observed`. Falsified by a delta > 3.0 at ICL ≥ 88.
- **P5 (correctness gate, must pass before any of the above counts).** γ = 0 reproduces base-model logits to < 1e-4 max abs deviation.

Note P3/P4 are the ones that can embarrass us: P1 and P2 restate the paper, so passing them only validates the harness. **P3 is the experiment.** Report all deltas with per-seed spread, never means alone — §4 showed seed-level error bars badly understate cross-cell heterogeneity, and a 3-seed run on one dataset is exactly where that bites.

**Fallback tiers (degrade, don't stop).**
1. **Full plan** — 2 models × 3 arms × 3 seeds, 200 eval examples.
2. **Qwen too slow on CPU** (>25 min for its 9 runs) → cut to 2 seeds and 120 eval examples; if still slow, swap Qwen2.5-0.5B → **SmolLM2-360M** (modern, ungated, smaller). Keep both ends of the axis at all costs — a single-model result cannot test P3, which is the whole point.
3. **Training unstable / peft breaks** → drop the LoRA arm (P1, P2 lost) and report **ICL vs AHR only** on both models. P3 and P4 survive, and they are the valuable ones.
4. **AHR implementation fails the P5 gate or the box expires** → fall back to the **pure re-analysis of Appendix B already completed in §4 of this document**: the ICL-vs-AHR decomposition, the 12-of-24 coin flip, the 91%/9% split, and the monotone gradient. This requires no compute, is fully reproducible from `_appendixB.json`, is verified against the paper's own Fig. A.3 (the exact 70.68 match), and is — candidly — a stronger result than anything a laptop CPU can produce. **This tier is already banked.**

**Honest statement of what this can and cannot show.** It cannot settle whether AHR ≈ ICL at 3B–8B; that regime needs a GPU. It can show whether the *gradient exists* at the small end, which is the mechanism behind the claim. A confirmed P3 plus the banked Appendix B re-analysis is a complete argument: the paper's own data shows the trend across 4 models, and we independently reproduce its sign on 2 more.

---

*Word count ≈ 3,400. All figures recomputed from `ingest/_appendixB.json`; the Fig. A.3 / ICL-row identity (70.68) and the Table 2 Average row were both reproduced exactly, validating the transcription this analysis rests on.*
