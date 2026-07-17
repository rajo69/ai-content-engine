# X / Twitter thread — AHR (arXiv:2607.13425)

*9 tweets. Tweet 1 is the hook: claim + one number, no hashtags. Viz attachment noted under each tweet in italics — those notes are production instructions, not posted text.*

---

**1/**
A COLM 2026 paper adapts an LLM by training 512 numbers — one volume knob per attention head — and beats LoRA by ~6 points at ten labeled examples. The parameter count is real. The 6 points are mostly LoRA breaking, not the method winning. Here's what the appendix hides.

*Attach: headline.png*

---

**2/**
The method (AHR): each attention head adds a contribution to the residual stream. Give each head one scalar, start at zero, scale the contribution by (1 + scalar), freeze everything else. It literally cannot memorize your data — it can only turn existing features up or down.

*Attach: method.png*

---

**3/**
The trick: the headline compares AHR to LoRA, AdaLoRA, IA3 — and quietly omits ICL, the frozen model with the same 10 examples in its prompt. ICL is in all 24 appendix cells and zero main tables. Put it back and it outranks *every* trained baseline the paper reports.

---

**4/**
Break AHR's edge over ICL out by model, sorted by base-model competence:
GPT2-XL: +9.2
Llama-3.2-1B: +3.4
Llama-3.2-3B: −0.4
Qwen3-8B: −1.5
Perfectly monotone (τ = −1.00). AHR's benefit vanishes as models improve, and is already negative for the two strongest.

---

**5/**
It's not a slope, it's a cliff. AHR beats the frozen model in 10 of 12 cells on the two weaker models — and 2 of 12 on the two stronger ones. Fisher exact p = 0.0033. The advertised "6 points over LoRA" on modern models is 91.5% just LoRA failing to beat doing nothing.

*Attach: decomposition.png*

---

**6/**
We reimplemented AHR (official repo is an empty README) and ran it on a laptop CPU. Correctness gate passed bitwise: scalars=0 reproduces the frozen model to 0.000e+00. Bonus bug: the paper's regularizer NaNs on the first backward step from its own zero-init.

---

**7/**
At our toy scale AHR never beat doing nothing: −3.3 pts on Qwen2.5-0.5B (outside seed noise) and −2.2 on GPT-2-small (within noise, so a tie there, not a loss). What reproduced hard: LoRA at 10 examples is a disaster (−25.8 vs frozen) — one seed collapsed to a constant classifier at loss 1e-4. AHR's scalars stayed tiny.

---

**8/**
Honest verdict: our pre-registered test of the vanishing-benefit trend FAILED. The weak model we picked couldn't do the task, so we spanned broken→good, not weak→good. The trend stands on the paper's 4 models, not ours. Also: closest prior work (LoFiT, 2024) is uncited.

---

**9/**
Paper: arxiv.org/abs/2607.13425. Verdict: the idea is sound — inexpressiveness really does resist overfitting — but "beats LoRA" is mostly a fact about LoRA, and we couldn't get AHR to beat leaving the model alone.
