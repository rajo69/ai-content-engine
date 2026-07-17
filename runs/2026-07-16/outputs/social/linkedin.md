# LinkedIn post — AHR (arXiv:2607.13425)

*Single post, ~210 words. Lead image: explainer.png. Professional framing for teams building with AI.*

---

A new COLM 2026 paper claims you can adapt an LLM to a new task by training just 512 numbers — one "volume knob" per attention head — and beat LoRA by ~6 points when you only have ten labeled examples.

We reproduced it and re-read its own appendix. Two lessons worth passing to any team doing few-shot adaptation:

**1. At tiny label budgets, capacity is a liability.** LoRA at ten examples was worse than the untouched model in 20 of the paper's 24 test cells — in our own run, one configuration memorized ten examples perfectly and turned into a constant classifier. The method that wins is the one too simple to overfit. If you're reaching for LoRA on a handful of labels, measure it against doing nothing first.

**2. "Beats the baselines" depends on which baselines are in the table.** The paper omits the cheapest one — the frozen model with the same ten examples in its prompt. Put it back and it outranks *every* trained method reported, and the new method's edge disappears as base models get stronger.

The mechanism is real and elegant, but its advantage is mostly a story about a broken baseline, and it fades exactly as models improve.

When you evaluate an adapter, is "do nothing with the same prompt" in your comparison table — or only the trainable methods?
