# Annotated Bibliography

**Target paper:** Oikarinen, Chen, Siska, Weng, Singh & Gao. *Data-Efficient Adaptation of LLMs via Attention Head Reweighting (AHR).* COLM 2026 / arXiv:2607.13425.

Selected from the 73 entries in `ingest/references.json`, prioritizing `isInfluential`, citation count, and — most importantly — whether the paper's own Section 2 leans on the work. Two entries (§B.1, §B.2) are **not cited by the target paper** and were added by this stage; see `lineage.md` → Novelty check for why they matter.

Citation keys follow the target paper's own Section 2 (e.g. "Zhang et al., 2023a"), so they can be matched back to its bibliography.

---

## A. Parameter-efficient finetuning — the baselines being beaten

### A.1 Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang & Chen (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. arXiv:2106.09685. (~20,900 citations)

LoRA freezes the pretrained weights and injects trainable rank-decomposition matrices (`B·A`, rank `r`) into transformer layers, cutting trainable parameters by orders of magnitude versus full finetuning while matching its quality on many tasks. Critically, the update merges into the base weights at inference, so it adds no serving latency. It became the default adaptation method for LLMs and the substrate for an entire sub-field.

**How it feeds AHR:** This is **the baseline being beaten** — the headline claim is that AHR outperforms LoRA in the few-shot regime with 200–1000× fewer trainable parameters. It is also the **template AHR imitates structurally**: zero-initialized update, frozen backbone, and merge-at-test-time (AHR folds `γ` into `W_O`, exactly LoRA's no-inference-cost bargain). AHR's argument is not that LoRA is wrong but that its parameter count — thousands to millions — is *overkill and actively harmful* when `|D_train|` is ~10, because there is enough capacity to overfit. AHR keeps LoRA's ergonomics and discards its capacity.

> ⚠️ **Data-quality note:** LoRA is **absent from `ingest/references.json`** — Semantic Scholar failed to resolve it from the PDF bibliography, even though it is the paper's single most load-bearing citation. This entry was recovered by a direct S2 lookup on `arXiv:2106.09685`. Any downstream stage treating `references.json` as complete will silently miss the primary baseline.

### A.2 Zhang, Chen, Bukharin, He, Cheng, Chen & Zhao (2023a). *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.* ICLR 2023. arXiv:2303.10512. (403 citations, **isInfluential**)

AdaLoRA observes that LoRA spreads its rank budget uniformly across modules, which is wasteful because weight matrices differ in importance. It parameterizes updates in SVD form and prunes singular values by an importance score, adaptively reallocating rank to the matrices that need it.

**How it feeds AHR:** A **baseline in the experiments**, and more interestingly the **closest philosophical predecessor**. AdaLoRA already accepts AHR's core premise — that adaptation budget should follow the model's own internal structure rather than be spread uniformly. The difference is the unit of allocation: AdaLoRA allocates *rank across weight matrices* and still learns the update contents; AHR allocates *one scalar per attention head* and learns nothing inside the head. AHR is AdaLoRA's importance-weighting logic pushed to its degenerate limit, where importance weighting is the *entire* method.

### A.3 Liu, Tam, Muqeeth, Mohta, Huang, Bansal & Raffel (2022). *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning.* NeurIPS 2022. arXiv:2205.05638. (1,372 citations)

Introduces **(IA)³**, which learns element-wise **rescaling vectors** that multiply the keys, values, and FFN inner activations of a frozen model, plus the T-Few recipe. Its central claim — that few-shot PEFT beats ICL at a fraction of the inference cost — is the thesis AHR inherits wholesale.

**How it feeds AHR:** **The method being modified, and the most important ancestor in the bibliography.** (IA)³ established the exact move AHR makes — *adapt by rescaling frozen activations rather than editing weights* — and established that this is enough for few-shot learning. AHR's genuine delta is one of **granularity and unit**: (IA)³ learns a scalar *per hidden dimension* of the key/value/FFN activations (thousands of parameters, no semantic grouping); AHR learns a scalar *per attention head* (hundreds of parameters, where each parameter has a mechanistic referent). This is the single most important comparison for judging AHR's novelty, and the honest framing is *"(IA)³ with the interpretability-motivated grouping and a 10× smaller budget"* rather than *"a new kind of method."*

### A.4 Hayou, Ghosh & Yu (2024). *LoRA+: Efficient Low Rank Adaptation of Large Models.* ICML 2024. arXiv:2402.12354. (439 citations)

Shows LoRA is suboptimal at large width because `A` and `B` share a learning rate, which blocks efficient feature learning; setting the `B` learning rate higher fixes it, improving accuracy and speed at no cost.

**How it feeds AHR:** A **baseline**, and evidence for AHR's rhetorical framing: it shows the LoRA line's ongoing progress comes from *optimization tuning within a fixed parameterization*. AHR argues the parameterization itself is the problem in the low-data regime. Cited alongside Hayou et al. (2025) (PLoP, arXiv:2506.20629), which asks *where* to place LoRA modules — again a question AHR dissolves by covering every head at negligible cost.

### A.5 Li & Liang (2021). *Prefix-Tuning: Optimizing Continuous Prompts for Generation.* ACL 2021. arXiv:2101.00190. (6,054 citations) — and Liu, Ji, Fu, Tam, Du, Yang & Tang (2021). *P-Tuning v2.* arXiv:2110.07602. (1,122 citations)

Prefix-Tuning freezes the LM and optimizes a small continuous task-specific prefix prepended to every layer's activations, matching finetuning with ~0.1% of the parameters. P-Tuning v2 shows properly-tuned deep prompt tuning matches full finetuning across scales and hard sequence-labeling tasks.

**How it feeds AHR:** **Baselines**, and the other branch of the "constrained adaptation" tree — they constrain the model's *input*, AHR constrains its *internal routing*. AHR's Section 2 groups these under prompting approaches that are "often very sensitive to minor variations," which is the vulnerability AHR positions against. Their existence establishes AHR's implicit claim: sub-0.1% adaptation had been reached before, but always by adding new activations, never by reweighting existing computation.

### A.6 Ostapenko, Su, Ponti, Charlin, Roux, Caccia & Sordoni (2024). *Towards Modular LLMs by Building and Reusing a Library of LoRAs.* arXiv:2405.11157. (83 citations, **isInfluential**) — and Charakorn, Cetin, Tang & Lange (2025). *Text-to-LoRA: Instant Transformer Adaption.* ICML 2025. arXiv:2506.06105. (38 citations, **isInfluential**)

Two steps in the same direction: adaptation should get *cheaper and more reusable*. Ostapenko et al. build a library of task LoRAs with zero-shot routing over them. Text-to-LoRA goes further, hypernetwork-generating a task-specific LoRA from a natural-language task description in a single forward pass, with no task data at all.

**How it feeds AHR:** Both flagged **isInfluential**, and they set the **frontier AHR is racing**. They matter because they represent the *rival answer* to the same problem: if adaptation must be cheap, one route is to keep LoRA and amortize it (route to a library, or generate it from text); AHR's route is to shrink the adapter until it is nearly free. *(Inference: the paper does not stage this as a rivalry, but the low-data motivation is shared, and a per-head scalar vector — 512–1,200 numbers — is an unusually attractive hypernetwork target. This is a natural "what's next" for downstream stages.)*

---

## B. Attention-head gating — the uncited prior art

> These three are **not in the target paper's reference list**. They are included because each learns *a multiplicative rescaling of an attention head's contribution* — AHR's exact parameterization. §B.0 is the closest prior work to AHR in the entire literature. See `lineage.md` → **Novelty check**.

### B.0 Yin, Ye & Durrett (2024). *LoFiT: Localized Fine-Tuning on LLM Representations.* NeurIPS 2024. arXiv:2406.01563. ⚠️ **UNCITED — closest prior work**

LoFiT freezes the backbone and adapts an LLM in two steps: **Step 1** learns a scaling factor vector `A_i^l ∈ R^{d_head}` per attention head, applied as `z^(l,i) ← (1 + A_i^l) ⊙ z^(l,i)`, trained with cross-entropy on *a small amount of labeled data* under an **L1 penalty with hyperparameter λ**; heads are then ranked by `‖A_i^l‖` and the top-K (3–10%) selected. **Step 2** learns additive offset vectors on those selected heads. It reports parity with LoRA on truthfulness and reasoning tasks while modifying **20–200× fewer parameters**.

**How it feeds AHR:** **It doesn't — and that is the finding.** LoFiT is uncited, yet its Step 1 is AHR's method with a vector where AHR has a scalar. Same `(1 + ·)` multiplicative reparameterization on head activations, same frozen backbone, same cross-entropy on small labeled data, same L1/λ sparsity penalty, same head-level granularity, same beat-LoRA-on-parameter-count framing. The decisive difference is what happens next: LoFiT's Figure 1 states *"Only the tuned biases are used in the final model"* — **the scaling factors are discarded**, having served only as a head-selection heuristic. LoFiT had AHR's method in hand and never tested it standalone.

AHR's genuine delta is therefore **granularity + retention**: scalar instead of vector (~1k vs ~131k parameters on Llama-2-7B — 128×), zero-init instead of `N(0, σ_A)`, and kept as the adaptation instead of thrown away. That delta is a real contribution — "the thing you discarded works alone, and works *better* the smaller you make it" is exactly AHR's data-efficiency thesis. But it is much narrower than "we propose a new method," and it means **LoFiT is the natural baseline AHR never ran**: the paper never shows scalar-per-head beats vector-per-head at `|D_train| = 10`, which is the entire claim. This is the sharpest hole in the evaluation and the highest-value input to the delta-to-claim scoring.

### B.1 Voita, Talbot, Moiseev, Sennrich & Titov (2019). *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.* ACL 2019. arXiv:1905.09418. (1,554 citations)

Evaluates the contribution of individual attention heads in a Transformer NMT encoder, finds that the important heads have consistent, linguistically interpretable roles (positional, syntactic, rare-word), and prunes the rest using **learnable stochastic gates with a differentiable L0 relaxation**. Specialized heads survive pruning longest; most heads can be removed with little loss.

**How it feeds AHR:** **The uncited mechanical ancestor and the source of AHR's core assumption.** Voita et al. established *both* halves of AHR's foundation seven years earlier: (i) empirically, that attention heads are functionally specialized and unequally important — the premise AHR's abstract states as its justification; and (ii) mechanically, that you can *learn one scalar gate per head, multiplied into that head's output, under a sparsity penalty* — which, modulo the stochastic relaxation, is Equations 3–4 of AHR. The differences are real and worth stating precisely: Voita's gates live in `[0,1]` and can only **remove** heads from a model *already trained on the task*, whereas AHR's `(1 + γ_h)` is unbounded, zero-initialized, and can **amplify** a head above its pretrained contribution — and AHR's purpose is few-shot *adaptation to a new task*, not compression. But "learn a scalar per head under an Lp penalty" is not new; its use as a data-efficient adaptation method is.

### B.2 Michel, Levy & Neubig (2019). *Are Sixteen Heads Really Better than One?* NeurIPS 2019. arXiv:1905.10650. (1,409 citations)

Shows that at test time a large fraction of attention heads can be removed with no significant performance loss — in some layers a single head suffices — and introduces per-head **mask variables `ξ_h`** multiplying each head's output, using the gradient w.r.t. `ξ_h` as a head-importance score for iterative pruning.

**How it feeds AHR:** **Uncited prior art on the parameterization.** Michel et al.'s `ξ_h` is *literally* AHR's `γ_h` up to reparameterization (`ξ_h = 1 + γ_h`): a single scalar per head, multiplying that head's output into the residual stream. Michel et al. never *train* `ξ` as an adaptation parameter — they only differentiate the loss with respect to it at `ξ = 1` to score importance, then threshold. AHR's move is to keep that variable and **optimize it on task data**. That is a genuine and useful step, and it is also a smaller step than the paper's framing implies. This reference is the strongest single piece of evidence for the delta-to-claim analysis in the next stage.

---

## C. In-context learning — the paradigm AHR competes with and explains

### C.1 Brown, Mann, Ryder, Subbiah, Kaplan, Dhariwal, et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS 2020. arXiv:2005.14165. (60,662 citations, **isInfluential**)

The GPT-3 paper. Established that scale alone yields in-context few-shot learning: a frozen model conditioned on a handful of demonstrations performs new tasks with no gradient updates, sometimes competitively with finetuned models.

**How it feeds AHR:** **The problem statement and the strongest-cited baseline.** ICL is AHR's real competition and the thing AHR's analysis claims to *explain*: the paper's second contribution is that learned `γ` weights identify heads responsible for ICL. AHR sits in the gap GPT-3 opened but did not close — the abstract's "continue to struggle when faced with few samples for difficult tasks." AHR is best read as *"ICL is real but brittle and capped by context length; let's get the same benefit by tuning the heads that implement it."*

### C.2 Min, Lyu, Holtzman, Artetxe, Lewis, Hajishirzi & Zettlemoyer (2022). *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?* EMNLP 2022. arXiv:2202.12837. (2,081 citations)

Shows ground-truth demonstration labels are largely *unnecessary* — randomly replacing labels barely hurts. What matters is the label space, the input distribution, and the format. ICL locates a task more than it learns one.

**How it feeds AHR:** **The assumption being challenged, and AHR's central justification.** This is why AHR exists rather than "just do ICL": if demonstrations only *point at* a task rather than teach it, ICL cannot absorb genuine task information from few samples, and its sensitivity to formatting is a liability. AHR's pitch is that gradient signal into head weights extracts what ICL leaves on the table. Cited alongside Wei et al. (2023, arXiv:2303.03846) and Webson & Pavlick (2022, arXiv:2109.01247) — the "prompts don't mean what you think" cluster. **This trio is the most tweetable part of the lineage.**

### C.3 Von Oswald, Niklasson, Randazzo, Sacramento, Mordvintsev, Zhmoginov & Vladymyrov (2023). *Transformers learn in-context by gradient descent.* ICML 2023. arXiv:2212.07677. (817 citations) — with Akyürek, Schuurmans, Andreas, Ma & Zhou (2022). *What learning algorithm is in-context learning?* arXiv:2211.15661. (746 citations)

The "ICL is implicit optimization" line: trained transformers implement gradient-descent-like (or higher-order) learning algorithms in their forward pass over in-context examples, demonstrated concretely on linear models.

**How it feeds AHR:** **Theoretical license for AHR's central conceit.** If ICL is an optimization algorithm implemented by specific circuits, then those circuits are *localized objects you can turn up or down* — which is exactly what a per-head scalar does. This line is what makes AHR's IC-FT variant (Section 3.1) coherent rather than a hack: fine-tuning `γ` on an in-context-formatted objective is *tuning the model's implicit learner*. **Inference:** the paper cites these as background and does not claim `γ` tunes an implicit optimizer; that connection is mine and downstream stages should mark it as interpretation, not a paper claim.

---

## D. Mechanistic interpretability — the tools and the assumption

### D.1 Elhage, Nanda, Olsson, Henighan, Joseph, Mann, et al. (2021). *A Mathematical Framework for Transformer Circuits.* Anthropic / Transformer Circuits Thread.

Recasts the transformer so that attention heads are **independent, additive units writing into a shared residual stream**, decomposing `W_O` into per-head blocks so each head's contribution `h(z)` can be read off and manipulated in isolation.

**How it feeds AHR:** **The tool being reused — and the enabling condition for the whole method.** AHR's Section 3 explicitly adopts this notation ("following Elhage et al., 2021, dividing `W_O` into per-head components"). This is more than bookkeeping: AHR's Equation 3 (`z + Σ (1+γ_h)·h(z)`) is only *well-defined and mergeable* because Elhage et al. showed the MHA update is an independent sum over heads with a per-head slice of `W_O`. Without this decomposition there is no "head's contribution" to scale and no way to fold `γ` into `W_O` for free inference. It is the least-cited and most load-bearing reference in the paper.

### D.2 Olsson, Elhage, Nanda, Joseph, DasSarma, Henighan, et al. (2022). *In-context Learning and Induction Heads.* Transformer Circuits Thread. — with Akyürek, Wang, Kim & Andreas (2024). *In-Context Language Learning: Architectures and Algorithms* (n-gram heads).

Identifies **induction heads** — heads implementing prefix-matching copy (`[A][B] … [A] → [B]`) — and argues they are the primary mechanism of in-context learning, with their formation coinciding with an ICL phase change during training. Akyürek et al. extend the story to n-gram heads.

**How it feeds AHR:** **The assumption the method rests on, and the target of its interpretability claim.** AHR's premise — "making use of the functional specialization of individual attention heads" — is only credible because this line established that heads implement *identifiable, nameable* functions, and specifically that ICL is carried by a small number of them. This grounds both halves of the paper: the method (few heads matter ⇒ few scalars suffice) and the analysis (learned `γ` should light up known ICL heads). **This is the paper's biggest scientific bet**: if head specialization is weaker or more distributed than the induction-head story suggests, AHR's parameterization is arbitrary and its interpretability claims are pareidolia. Downstream stages should check whether the paper's `γ` analysis actually recovers known induction heads or merely asserts the connection.

### D.3 Todd, Li, Sharma, Mueller, Wallace & Bau (2025). *Function Vectors in Large Language Models.* ICLR 2024. — with Oikarinen & Weng (2024) and Singh et al. (2023a).

Function vectors show a compact causal representation of a demonstrated task is carried by a small set of attention heads and can be extracted and transplanted to trigger the task elsewhere. Oikarinen & Weng and Singh et al. are the target authors' own prior work on automatically explaining neurons/heads in natural language.

**How it feeds AHR:** **The strongest evidence for AHR's premise, and self-citation establishing the authors' lineage.** Function vectors matter enormously here: they demonstrate that task information is *concentrated in a few heads and is causally manipulable* — which is the precondition for a per-head scalar being sufficient to specify a task. **Inference:** function vectors *add* a vector to the residual stream (additive intervention); AHR *scales* an existing head (multiplicative). AHR can only amplify what a head already computes, never make it compute something new — a real expressivity ceiling that the paper's data-efficiency argument treats as a feature (it is exactly why AHR resists overfitting) but which likely explains why AHR's advantage should vanish as `|D_train|` grows. Worth checking against the paper's own large-`|D_train|` results.

### D.4 Zhang, Singh, Liu, Liu, Yu, Gao & Zhao (2023/2024a). *Tell Your Model Where to Attend: Post-hoc Attention Steering for LLMs (PASTA).* ICLR 2024. arXiv:2311.02262. (83 citations) — with Shi et al. (2023), *TOAST: Transfer Learning via Attention Steering*, arXiv:2305.15542. (25 citations)

PASTA emphasizes user-specified spans by **downweighting attention to other tokens in a selected subset of heads**, chosen by a profiling stage — a training-free way to steer a frozen LLM. TOAST keeps a pretrained backbone frozen and steers attention top-down toward task-relevant features for transfer learning.

**How it feeds AHR:** **The nearest neighbours in method space, and the sharpest novelty test.** PASTA already: (i) treats a *subset of heads* as the intervention site, (ii) *reweights* their attention, and (iii) selects which heads via a profiling step. The delta is that PASTA reweights *within* a head's attention **over tokens** and picks heads by *search*, while AHR reweights *between* heads by **learned gradient descent** — differentiable, task-general, and needing no profiling. TOAST establishes attention steering specifically for *transfer learning*, which is AHR's problem. Shared authorship (Zhang, Singh, Gao) makes AHR plausibly a direct descendant of PASTA. *(Inference: the paper cites PASTA as related work, not as a baseline. A reader may reasonably ask why it isn't compared against; note this for the analyze stage.)*

---

## Coverage note

16 works annotated across 4 groups (A: 8 papers in 6 entries; B: 3; C: 5 papers in 3 entries; D: 8 papers in 4 entries) — 13 selected from `references.json`, plus 3 uncited works added by this stage (§B.0–B.2). Abstracts came from `ingest/references.json`; full detail was fetched from the Semantic Scholar Graph API for LoRA, (IA)³, PASTA, Voita et al., and Michel et al. LoFiT's method section was extracted directly from its PDF (`pdftotext`) because its abstract does not describe the per-head scaling factors — the mechanism only appears in its Section 3 and Figure 1. Excluded as boilerplate: dataset papers (SST-2, AG-News, Emotion, ToxiGen, Jailbreak — covered in `ingest/metadata.json`), model cards (Llama 3, Qwen3, GPT-2), the optimizer (AdamW), and the PEFT survey.

**Reference-list integrity (two defects found, both affecting downstream stages):**
1. **LoRA is missing from `references.json`** — the paper's primary baseline failed to resolve via Semantic Scholar. Recovered manually (§A.1).
2. **`references.json` contains a mis-resolved entry**: `"The Context"` (2019, 495 citations, **flagged `isInfluential`**) has an abstract about change management and transformational leadership *in the Ukrainian economy*. It is a Semantic Scholar mis-link, not a real reference of this paper — and it is one of only 7 entries flagged influential. It has been discarded here; downstream stages should not treat it as real. **Effective influential count is 6, not 7.**
