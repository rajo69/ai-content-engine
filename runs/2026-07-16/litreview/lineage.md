# Research Lineage: From Few-Shot Learners to a Single Scalar per Head

**Target:** Oikarinen, Chen, Siska, Weng, Singh & Gao, *Data-Efficient Adaptation of LLMs via Attention Head Reweighting* (COLM 2026, arXiv:2607.13425).

Companion to `bibliography.md`. Claims are grounded in sources read during this stage; inferences are marked.

---

## The narrative

### The problem the field was chasing

The question underneath this paper is old and simple: **how do you specialize a general model to a new task when you have almost no labels for it?** The paper's framing is security — phishing URLs, jailbreak prompts, toxic text — where labeled examples are scarce, adversaries move faster than annotation budgets, and the interesting cases are by construction rare. But the question generalizes, and the last six years have produced two rival answers to it that only recently began to converge.

### Answer one: don't learn at all

The first answer arrived with **GPT-3** (Brown et al., 2020): make the model big enough and it learns from demonstrations *in the forward pass*, with no gradient updates at all. In-context learning looked like the end of the data-efficiency problem — the adaptation budget dropped to zero parameters.

It did not hold up. **Min et al. (2022)** delivered the puncture: replacing demonstration labels with *random* ones barely hurts ICL performance. What matters is the label space, the input distribution, and the format — ICL largely *locates* a task the model already knows rather than *learning* one from the examples. **Webson & Pavlick (2022)** and **Wei et al. (2023)** filled in the picture: models are wildly sensitive to prompt phrasing, and irrelevant or even misleading prompts can work as well as good ones. So ICL's apparent data efficiency was partly an illusion. It cannot absorb much genuine task information from ten examples, it is brittle to formatting, and — mundane but decisive in practice — it is capped by the context window and pays the demonstrations' inference cost on every single query.

A theory branch grew alongside it, arguing ICL is *implicit optimization*: **Von Oswald et al. (2023)** and **Akyürek et al. (2022)** showed trained transformers implement gradient-descent-like algorithms over in-context examples. This matters for the story's ending. If ICL is an algorithm implemented by machinery, the machinery has a location, and things with locations can be turned up or down.

### Answer two: learn, but barely

The second answer kept gradients and shrank the parameter count instead. **Prefix-Tuning** (Li & Liang, 2021) froze the model and optimized a continuous prefix; **P-Tuning v2** (Liu et al., 2021) showed deep prompt tuning could match full finetuning across scales. Then **LoRA** (Hu et al., 2022) won the field outright with a better bargain: a zero-initialized low-rank update, a frozen backbone, and — the killer feature — a merge back into the weights at test time, so adaptation costs nothing at inference.

LoRA became the substrate rather than the destination, and the refinements that followed all left its parameterization intact while tuning *around* it. **AdaLoRA** (Zhang et al., 2023a) asked *where the rank budget should go* and reallocated it by importance. **LoRA+** (Hayou et al., 2024) asked *how to set learning rates*. **PLoP** (Hayou et al., 2025) asked *which modules to attach to*. A different branch tried to amortize the cost away entirely: **Library of LoRAs** (Ostapenko et al., 2024) built reusable adapters with zero-shot routing, and **Text-to-LoRA** (Charakorn et al., 2025) hypernetwork-generated an adapter from a task description alone. Both are flagged influential by the target paper.

The most important move in this branch, though, was quieter. **(IA)³** (Liu et al., 2022) argued few-shot PEFT beats ICL outright, and it got there by **rescaling frozen activations rather than editing weights** — learning element-wise multipliers on keys, values, and FFN activations. That is the pivot the target paper inherits: adaptation as *modulating computation the model already does*, not as *adding new computation*. What (IA)³ lacked was any principled answer to *which* activations deserve their own knob. It scaled per hidden dimension, a unit with no semantic meaning.

### The parallel track: heads have jobs

Meanwhile a separate literature was answering exactly that question, for entirely different reasons. **Elhage et al. (2021)** recast the transformer so attention heads are *independent, additive units writing into a shared residual stream*, with `W_O` decomposed into per-head blocks — making "this head's contribution" a well-defined, isolable object. **Olsson et al. (2022)** found **induction heads** and argued they are the primary mechanism of in-context learning; **Akyürek et al. (2024)** extended this to n-gram heads; **Todd et al. (2025)** showed **function vectors** — compact, causally manipulable task representations carried by a small set of heads.

The synthesis is: *heads are specialized, a few of them carry the task, and they can be manipulated causally.* Which invites the obvious method — and several people took it. **Voita et al. (2019)** and **Michel et al. (2019)** learned per-head gates, but pointed them at *compression*: their gates could only remove heads from a model already trained on the task. **PASTA** (Zhang et al., 2024a) and **TOAST** (Shi et al., 2023) steered attention at inference, but *within* heads over tokens, with heads chosen by profiling rather than learned. **LoFiT** (Yin et al., 2024) came closest of all — and then walked away, for reasons the novelty check below examines.

### The gap AHR steps into

By 2026 every ingredient was on the table: the residual-stream decomposition that makes a head's contribution isolable (Elhage); the evidence that few heads carry ICL (Olsson, Todd); the principle that rescaling frozen activations is enough for few-shot learning ((IA)³); and LoRA's merge-at-test-time ergonomics. Nobody had assembled them in the obvious way, because each community wanted something else — the pruning people wanted a smaller model, the interpretability people wanted an explanation, and the PEFT people were optimizing within LoRA's parameterization rather than questioning its granularity.

**AHR's move is to take the head-gate variable seriously as the adaptation itself.** One scalar `γ_h` per head, zero-initialized, multiplying that head's contribution as `(1 + γ_h)·h(z)`, trained by SFT under an `Lp` penalty, frozen everything else, merged into `W_O` at test time. 512–1,200 trainable parameters — ~0.0001% of the model. The bet is that this extreme constraint is not a limitation but the *mechanism*: with `|D_train| ≈ 10`, LoRA's thousands of parameters have enough capacity to overfit, and AHR's few hundred simply do not. The parameterization can only amplify or suppress computation the pretrained model already performs, which is exactly why it cannot memorize ten examples. **Inference:** this predicts AHR's advantage should erode as `|D_train|` grows and the capacity constraint turns from asset to ceiling — worth checking against the paper's own scaling results in the analyze stage.

The second claim rides free on the first: because each parameter has a mechanistic referent, the learned `γ` vector is an *interpretability readout*, indicating which heads a task recruits — connecting back to the induction-head literature that motivated the design.

### Position in the lineage

AHR is best read not as a new mechanism but as **a reclassification of an existing one**. The per-head scalar gate has been in the literature since 2019, always as a *means*: a tool for pruning (Voita, Michel), a scoring device for selection (LoFiT), a knob for training-free steering (PASTA). AHR's contribution is to argue that this instrument — pushed to the coarsest granularity anyone has tried, one number per head — is *sufficient by itself* to adapt an LLM, and that its coarseness is precisely what makes it win when data is scarce. It sits at the intersection of the (IA)³ rescaling principle and the interpretability literature's head-level semantics, taking the parameter budget one branch had spent per-dimension and re-spending it per-head. That is a real and useful finding. It is also a narrower delta than the paper's framing implies, and the reference list's omissions — detailed next — are what make it look wider.

*(~1,150 words)*

---

## Novelty check

**Verdict: NOT a rediscovery, but the delta is substantially narrower than the paper claims — and the closest prior work is uncited.**

The core idea is not novel as a *mechanism*. It is novel as a *method*, by a margin of roughly "one scalar instead of one vector, and we kept it instead of throwing it away."

### ⚠️ Primary finding: LoFiT (NeurIPS 2024) is uncited and does nearly this

**Yin, Ye & Durrett, *LoFiT: Localized Fine-Tuning on LLM Representations*, NeurIPS 2024 (arXiv:2406.01563)** — **not present in `ingest/references.json` and not cited anywhere in the paper.**

LoFiT's Step 1, quoted from its Section 3:

> "We incorporate a learnable scaling factor `A_i^l ∈ R^{d_head}` for any head at layer `l` and index `i`… the activation `z_t^(l,i)` is rescaled by `z_t^(l,i) ← (1 + A_i^l) ⊙ z_t^(l,i)`. We freeze all pre-trained weights and learn `A_i^l` end-to-end with the cross-entropy loss **on a small amount of labeled data** from the task of interest… We regularize the optimization with an **L1 normalization with a scaling hyperparameter λ** to encourage sparsity."

Set against AHR's Equations 3–4, the overlap is close to total:

| | LoFiT Step 1 (Jun 2024) | AHR (Jul 2026) |
|---|---|---|
| Intervention | `(1 + A_i^l) ⊙ z^(l,i)` on head activations | `(1 + γ_h) · h(z)` on head contribution |
| Granularity | **vector**, `d_head` per head | **scalar**, 1 per head |
| Init | `N(0, σ_A)` | zero |
| Objective | CE on small labeled data, backbone frozen | CE on few-shot data, backbone frozen |
| Regularizer | **L1, hyperparameter λ** | **`λ(Σ|γ_h|^p)^(1/p)`, p ∈ {1,2}** |
| Role of the scalars | **discarded** — used only to rank heads by `‖A_i^l‖` and select top-K | **kept — this is the method** |
| Then what | Step 2 learns additive bias vectors on selected heads | nothing; `γ` merges into `W_O` |
| Headline claim | "comparable to LoRA, 20–200× fewer parameters" | "outperforms LoRA, 200–1000× fewer parameters" |

Both papers freeze the backbone, both multiply head activations by `(1 + ·)`, both train on small labeled data with an L1/Lp sparsity penalty, and both sell the result as beating LoRA on a dramatic parameter-count ratio. LoFiT's Figure 1 states plainly: *"Only the tuned biases are used in the final model."* LoFiT **had AHR's method in hand and used it as a subroutine** — its authors never tested whether the scaling factors alone could serve as the adaptation.

**That untested possibility is exactly AHR's contribution, and it is a legitimate one.** "The thing you threw away works on its own, and works better if you make it 128× smaller" is a real result — LoFiT's Step 1 on Llama-2-7B is ~131k parameters (32×32×128), against AHR's ~1k. AHR's data-efficiency thesis lives entirely in that gap, and LoFiT's own footnote 6 shows its authors saw Step 1's parameters purely as overhead to apologize for, not as a method.

But three consequences follow, and downstream stages should carry all three:
1. **AHR's novelty is granularity + retention, not the parameterization.** Any framing of "we propose learning a scalar per head" as new is overclaiming.
2. **LoFiT is the natural baseline AHR never ran.** The obvious reviewer question — does scalar-per-head actually beat vector-per-head at `|D_train| = 10`, which is the entire data-efficiency thesis? — is unanswered by the paper. This is the sharpest hole in the evaluation.
3. **The omission is hard to attribute to obscurity.** LoFiT is a NeurIPS 2024 paper on head-level rescaling for low-data LLM adaptation with a fewer-parameters-than-LoRA headline. It is the single closest work in the literature.

### Secondary finding: the 2019 head-gating literature is also uncited

**Voita et al. (2019)** (arXiv:1905.09418, 1,554 citations) prune heads with *learnable stochastic gates under a differentiable L0 penalty*, and find surviving heads are specialized and interpretable — establishing *both* AHR's mechanism and the functional-specialization premise its abstract invokes. **Michel et al. (2019)** (arXiv:1905.10650, 1,409 citations) attach per-head mask variables `ξ_h` to head outputs — literally AHR's `γ_h` under `ξ_h = 1 + γ_h` — but only differentiate at `ξ = 1` for importance scoring.

Neither is cited. The honest distinction is real and should be stated rather than hidden: **pruning gates live in `[0,1]` and can only remove; AHR's `(1 + γ_h)` is unbounded and can amplify a head above its pretrained contribution.** Turning heads *up* is not something the pruning literature could do or wanted. But a paper whose stated premise is "functional specialization of individual attention heads" omitting the two papers that established it with ~3,000 combined citations is a gap.

### What was searched (and found clean)

- **S2 recommendations API** (`forpaper/arXiv:2607.13425`, 20 results, raw at `ingest/_s2_recs_raw.json`): generic PEFT/LoRA-variant noise (OLoRA+, MixedPEFT, SimLoRA, LoRA-S) and unrelated tabular/pretraining work. **No recommendation proposes per-head scalar reweighting.** Nearest was *Self-Improving In-Context Learning* (2605.23180), which optimizes prompt embeddings — a different intervention site.
- **S2 graph search:** "learnable scalar per attention head fine-tuning", "attention head scaling factor parameter efficient fine-tuning", "head importance gating few-shot adaptation LLM" — nothing equivalent.
- **WebSearch:** "learn one scalar per attention head parameter-efficient fine-tuning head gating LLM adaptation"; "attention head reweighting few-shot adaptation LLM scalar per head PEFT 2025". These surfaced LoFiT (above) plus near-misses worth logging, none of which is prior art for AHR's claim: **Soft Head Selection for Injecting ICL-Derived Task Embeddings** (2507.20906 — learns soft head-selection parameters, but *injects* ICL-derived task embeddings rather than scaling existing head output); **HyRA** (2510.04295 — hypernetwork generates low-rank matrices across heads, still low-rank); **head-specific gating** for perplexity/attention-sink suppression (architectural pretraining change, not task adaptation).
- **Forward citations** (`arXiv:2607.13425/citations`): **empty** — the paper was published 2026-07-15, one day before this run. No responses, extensions, or contradictions exist yet. Re-check is worthwhile in ~2 weeks but is out of scope for this run.

### Reference-list integrity

Three defects, all inherited by any stage trusting `references.json`:

1. **LoRA (arXiv:2106.09685) is missing** — the paper's primary baseline failed to resolve via Semantic Scholar. Recovered manually.
2. **A mis-resolved entry is flagged influential.** `"The Context"` (2019, 495 citations, `isInfluential: true`) has an abstract about change management and transformational leadership *in the Ukrainian economy*. It is a Semantic Scholar mis-link, not a reference of this paper. **Effective influential count is 6, not 7.**
3. **LoFiT, Voita et al., and Michel et al. are absent** — but this reflects the paper's actual bibliography, not an extraction failure. That is a finding about the paper, not about the pipeline.

Defects 1 and 2 are ingest/S2 artifacts. Defect 3 is the story.
