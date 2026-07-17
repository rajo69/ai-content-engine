# Reddit post — r/MachineLearning

**Title:** [R] Reproducing "Attention Head Reweighting" (COLM 2026): the 6-point win over LoRA is mostly LoRA breaking, and the paper omits ICL from every main table

---

AHR (arXiv:2607.13425) adapts a frozen LLM to a text-classification task by learning a single scalar per attention head — 512 params for Llama-3.2-1B — applied as `(1 + γ_h)·h(z)`, everything else frozen, γ folded into W_O at test time. Headline claim: beats LoRA by ~6 points at |D_train|=10 with 200–1000× fewer trainable params. The parameter count checks out. I reimplemented the method (the official repo is a placeholder README) and re-aggregated the paper's own Appendix B. Two issues, then a reproduction with a prediction that failed.

**1. The main tables omit the ICL baseline.** The frozen model with the same 10 examples in its prompt (ICL) is reported in all 24 appendix cells and in none of the main-paper result tables (Table 2 = AHR/LoRA/AdaLoRA/IA3 only). Recomputed at |D|=10: ICL = 72.26 vs LoRA 68.27, AdaLoRA 71.22, IA3 67.88. ICL outranks every trained baseline the paper reports. I verified my appendix transcription by reconstructing the paper's stated Fig. A.3 "unmodified" value (70.68) exactly from the ICL row — which also confirms the "unmodified model" already carries the 10-shot prompt.

**2. AHR − ICL is monotone decreasing in base-model competence.** Per model: GPT2-XL +9.20, Llama-3.2-1B +3.42, Llama-3.2-3B −0.39, Qwen3-8B −1.45 (Kendall τ = −1.00). Win rate over ICL collapses from 10/12 cells on the two weaker models to 2/12 on the two stronger (Fisher exact p = 0.0033). On the 3 modern models, AHR beats LoRA by 6.23; decomposed, 5.70 is LoRA failing to beat ICL and 0.53 is AHR beating ICL (paired over 18 cells, t=0.78, p≈0.45). The headline is largely a fact about LoRA at 10 examples.

**Reproduction (laptop CPU, toy scale — stated up front).** 2 models (GPT-2-small, Qwen2.5-0.5B), 3 arms (ICL/AHR/LoRA r=1), 3 seeds, SST-2, 120 eval, fixed HPs, no CV. Correctness gate: γ=0 reproduces the frozen model bitwise (max dev 0.000e+00). Results (paired within seed):
- AHR − ICL: **−2.22 ± 3.14** (GPT-2-small), **−3.33 ± 1.36** (Qwen2.5-0.5B). The frozen model was the best arm on both.
- LoRA − ICL on Qwen2.5-0.5B: **−25.83**; one seed drove train loss to 1e-4 and became a constant classifier. AHR − LoRA = +22.50, i.e. entirely LoRA collapsing. AHR's γ stayed |γ|≈0.07–0.11 and never collapsed — the "too inexpressive to overfit" mechanism is visibly real.

**My pre-registered prediction failed.** I predicted ≥3 points of separation between the two models' AHR−ICL deltas (the "gradient"). Observed: 1.11, below both the threshold and the seed spread. Not supported. Reason: GPT-2-small is degenerate on SST-2 ICL (50.83, below the 54% majority-class baseline; one-class-dominant every seed via majority-label bias). It's broken, not weak, so there's no ICL machinery to amplify. I spanned broken→good, not weak→good. The gradient stands on the paper's 4 models, not mine.

**Other notes.** (a) Eq. 4's regularizer is the Lp *norm*, non-differentiable at the origin, and γ is zero-init — so the literal objective NaNs on the first backward pass at p=2. Paper almost certainly means squared L2 but doesn't say. (b) The closest prior work, LoFiT (NeurIPS 2024), learns a per-head *vector* `(1+A)⊙z` on small data with an L1 penalty, then discards it as a head-selection heuristic — same parameterization, uncited. Voita 2019 / Michel 2019 (per-head gates for pruning) also uncited. AHR's real delta is scalar-instead-of-vector + keeping it, which is legitimate but narrower than the framing, and makes LoFiT the natural baseline that was never run.

Method core is sound: inexpressiveness resisting overfitting is real and I reproduced it. What I can't reproduce is any benefit over doing nothing on a competent model. Critique welcome, especially: is there a defensible reason to exclude ICL from the main tables at |D|=10, and can anyone get AHR to beat ICL on a *non-degenerate* weak model (GPT-2-XL ~72 on SST-2, or a harder dataset)? That's the experiment I couldn't fit on a CPU.
