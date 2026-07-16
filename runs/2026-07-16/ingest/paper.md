# Data-Efficient Adaptation of LLMs via Attention Head Reweighting

**Authors:** Tuomas Oikarinen (UC San Diego, Microsoft Research), Zixiao Chen (Microsoft Security AI), Charlotte Siska (Microsoft Security AI), Tsui-Wei Weng (UC San Diego), Chandan Singh (Microsoft Research), Jianfeng Gao (Microsoft Research)

**arXiv:** 2607.13425v1 [cs.LG] · Submitted 15 Jul 2026 · **Published as a conference paper at COLM 2026**

**Code:** https://github.com/tuomaso/attention-head-reweighting — announced in a footnote as "fully compatible with the `peft` package", but **as of this ingest the repo contains only a placeholder README ("Stay tuned for the full code release shortly"); no code has been released.**

> Source note: this markdown was reconstructed from the PDF text layer (`pdftotext -layout`). No arXiv HTML or ar5iv rendition exists for this ID (both 404 / redirect to the abs page). Greek symbols were lost by the text extraction and have been restored from context: the learned per-head scalar is **γ_h** (gamma), the regularization strength is **λ** (lambda), and frozen model parameters are **θ** (theta).

---

## Abstract

Learning effectively from limited data is critical in domains like security where labeled examples are scarce. Large language models (LLMs) have demonstrated some capabilities for data-efficient learning, especially through parameter-efficient adaptation methods, but continue to struggle when faced with few samples for difficult tasks. To meet this challenge, we propose Attention Head Reweighting (AHR), a data-efficient method that adapts LLMs to new text-classification tasks by learning only a single scalar per attention head. This drastically reduces the number of parameters that need to be learned by making use of the functional specialization of individual attention heads. Experiments on diverse open-source text classification datasets show that AHR can outperform standard baselines like LoRA when learning from limited samples, despite having 200–1000× fewer trainable parameters, as our AHR only modifies ~0.0001% of the model's parameters. In addition, our learned weights are easy to interpret and can be analyzed to better understand the mechanisms and attention heads responsible for in-context learning abilities in LLMs.

---

## 1 Introduction

Learning from limited labeled data remains a central challenge in high-stakes domains such as AI security, where annotations are scarce because attacks and threat vectors constantly evolve and must be defended against quickly (Divakaran & Peddinti, 2024). Data-efficient text classification is critical for flagging dangerous LLM inputs (Yi et al., 2024; Saha et al., 2024; Chao et al., 2025; Verma et al., 2025) and outputs (Inan et al., 2023).

While LLMs exhibit some data-efficient behavior through in-context learning (ICL) (Brown et al., 2020; OpenAI, 2023) and parameter-efficient fine-tuning (PEFT) (Hu et al., 2022; Zhang et al., 2023a), performance degrades sharply with few task-specific examples. **One issue is that these methods generally optimize in a large, unconstrained parameter space, which can lead to overfitting.**

The paper proposes **Attention Head Reweighting (AHR)**, which adapts an LLM by learning **only a single scalar per attention head**. The hypothesis for why this works and avoids overfitting: it leverages the *functional specialization* of attention heads noted in recent work (Olsson et al., 2022; Akyürek et al., 2024; Zhang et al., 2024a; Ge et al., 2023). If a security-related behavior is isolated to a few heads, AHR can change the weighting of those heads rather than altering the internal weights of each head.

Headline claims:
- AHR improves by **~3% points on average** over baselines like LoRA when training data is very limited (≤ 100 examples).
- Particularly large gains on security-relevant tasks: **Webpage Phishing Detection (Web) and Jailbreak detection, 6–7% accuracy improvement over the best baseline with just 10 training samples**, using **200–1000× fewer trainable parameters**.
- Gains are fueled by a combination of upweighting a few **task-specific heads** plus a couple of **general in-context-learning heads**.
- **In-context finetuning (IC-FT)** — updating parameters to maximize the probability of correct answers *within a few-shot prompt* — improves accuracy by **~10% points** over finetuning on one sample at a time.

### Table 1: Trainable parameter counts by model and method

| Method | GPT2-XL | Llama-3.2-1B | Llama-3.2-3B | QWEN3-8B |
|---|---|---|---|---|
| Full FT | 1,558,148,000 | 1,235,920,896 | 3,212,750,496 | 8,190,736,512 |
| LoRA | 307,200 | 106,496 | 286,720 | 479,232 |
| AdaLoRA | 614,496 | 213,056 | 573,552 | 958,608 |
| IA3 | 537,600 | 147,456 | 286,720 | 626,688 |
| **AHR (Ours)** | **1,200** | **512** | **672** | **1,152** |

AHR requires 200–1,000× fewer trainable parameters than baselines; in total less than one millionth of the model's parameters. (Note: the AHR count equals the model's total number of attention heads — e.g. Llama-3.2-1B has 16 layers × 32 heads = 512.)

**Figure 1 (AHR overview):** simplified visual comparison of AHR vs LoRA vs full finetuning on a single attention layer. Parameter counts computed for a standard transformer with d = 3072, r = 1 for LoRA. AHR reduces trainable parameters **500×** compared to LoRA.

---

## 2 Background and related work

**Parameter-efficient finetuning.** Prominent examples include LoRA (Hu et al., 2022), AdaLoRA (Zhang et al., 2023a), (IA)³ (Liu et al., 2022), and related approaches (Hayou et al., 2024; 2025; Logan IV et al., 2022). These reduce adaptation cost but still introduce thousands to millions of trainable parameters, which can lead to overfitting in data-scarce settings. Separately, increasingly constrained prompting approaches range from continuous prompt tuning (Li & Liang, 2021; Liu et al., 2021) to discrete prompt tuning (Shin et al., 2020), prompt ensembles (Hou et al., 2022; Pitis et al., 2023; Morris et al., 2023), and generating a single natural-language prompt (Zhou et al., 2022; Singh et al., 2023b). Prompt-based methods are often very sensitive to minor variations.

**In-context learning.** ICL removes the need for task-specific parameter adaptation but is highly sensitive to minor variations in the provided examples (Min et al., 2022) and sometimes unfaithful to them (Wei et al., 2023; Webson & Pavlick, 2022). Work has shown ICL can implement linear models (Akyürek et al., 2022; Zhang et al., 2023b), discrete functions (Bhattamishra et al., 2023), and more general algorithms (Li et al., 2023; Zhuang et al., 2025); some argue ICL implicitly performs gradient-descent-like optimization (Mahankali et al., 2023; Von Oswald et al., 2023; Ahn et al., 2024) or higher-order methods (Dai et al., 2023; Zhang et al., 2023b).

**Attention heads.** Evidence that heads latch onto task-specific patterns useful for ICL: induction heads (Olsson et al., 2022), n-gram heads (Akyürek et al., 2024), and more easily explainable patterns (Todd et al., 2025; Oikarinen & Weng, 2024; Singh et al., 2023a). This functional specialization has motivated methods that explicitly manipulate attention: post-hoc steering (Zhang et al., 2024a;b), attention steering for transfer learning (Shi et al., 2023), attention-guided retrieval (Zhang et al., 2025b), interpretable modeling (Kim et al., 2024), boosting attention for instruction following (Guardieiro et al., 2025), and long-context inference (Gu et al., 2024).

---

## 3 Method: Attention Head Reweighting

AHR introduces **a single learnable scalar parameter per attention head** that multiplies that head's contribution to the residual stream. Finetuning these parameters on a small amount of data amplifies heads helpful for the task and downweights heads that harm performance. By not modifying the weights *inside* individual heads, AHR avoids overfitting to small data.

**Notation** (following Elhage et al., 2021, dividing W_O into per-head components): let **z_r^l ∈ ℝ^(s×d)** be the residual stream after layer *l*, where *s* is sequence length and *d* the model dimension. A transformer block is a Multi-Head Attention (MHA) layer followed by an MLP:

```
z_att^(l+1) = z_r^l + Σ_{h ∈ H_(l+1)}  h(z_r^l)                            (1)

z_r^(l+1)   = z_att^(l+1) + MLP_(l+1)( z_att^(l+1) )                       (2)
```

where

```
h(z) = A z W_V W_O ,      A = softmax( (z W_Q)(z W_K)^T / sqrt(d_k) )
```

with W_V ∈ ℝ^(d×d_k), W_Q ∈ ℝ^(d×d_k), W_K ∈ ℝ^(d×d_k), W_O ∈ ℝ^(d_k×d) the parameters of the attention head, and d_k = d / n_h where n_h = |H| is the number of attention heads per layer.

**Attention Head Reweighting** edits the MHA update rule (Eq. 1) to:

```
z_att^(l+1) = z_r^l + Σ_{h ∈ H_(l+1)}  (1 + γ_h) · h(z_r^l)                (3)
```

where **γ_h is a head-specific parameter initialized at zero**. During training only the γ_h parameters are trained; all other parameters, including MLP layers, are unchanged. **At test time γ can be merged into W_O for no additional inference cost.**

Training uses supervised finetuning (SFT) on few-shot examples, minimizing cross-entropy on the answer tokens plus a regularization term penalizing change to the model:

```
min_γ  Σ_{(x,y) ∈ D_train}  − log P(y | x; γ, θ)  +  λ ( Σ_h |γ_h|^p )^(1/p)     (4)
```

where D_train is the few-shot training set, λ controls regularization, θ are the (frozen) model parameters, and p ∈ {1, 2}.

### 3.1 In-Context Finetuning (IC-FT)

ICL is limited in how many samples it can learn from by the context window and GPU memory, and many in-context samples incur large inference cost. IC-FT (similar to He et al., 2025 — but here finetuning *separately on each dataset*, whereas He et al. aim to improve general ICL ability) constructs an input string with *k* few-shot examples and does SFT to maximize the probability of **all** answer tokens within the context:

```
min_γ  − Σ_{i=1..k}  log P(y_i | X_i; γ, θ)  +  λ ( Σ_h |γ_h|^p )^(1/p)         (5)
```

where **X_i = {x_1, y_1, ..., y_(i−1), x_i}** is the concatenated input with few-shot examples. The technique is independent of the finetuning method, so it is used for **both AHR and the baselines** (substituting their trainable parameters for γ). For text classification, only the probability of the **first token** of the answer is maximized, since predicting subsequent answer tokens is trivial given the first.

---

## 4 Results

### 4.1 Experimental setup

**Models (4):** GPT-2 XL (`gpt2-xl`; Radford et al. 2019), Llama-3.2 1B and 3B (`meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-3B`; Dubey et al. 2024), Qwen3 8B (`Qwen/Qwen3-8B`; Yang et al. 2025).

**Datasets (6)** — standard tasks plus cybersecurity applications:
- **SST2** (Socher et al., 2013): movie-review sentences → {Positive, Negative}.
- **AG-News** (Zhang et al., 2015): news excerpts → {World, Sports, Business, Sci/Tech}.
- **Emotion** (Saravia et al., 2018): tweets → {Sadness, Joy, Love, Anger, Fear, Surprise}.
- **Webpage Phishing Detection (Web)** (Hannousse & Yahiouche, 2021): web URLs → {Phishing, Legitimate}.
- **Toxigen** (Hartvigsen et al., 2022): potentially toxic/hateful comments. Treated as toxic if average human-assessed toxicity score > 3 (1–5 scale), benign otherwise.
- **Jailbreak Detection** (Shen et al., 2024): LLM prompts → {Jailbreak, Benign}. Uses the balanced version at `https://huggingface.co/datasets/jackhhao/jailbreak-classification`.

Most datasets have thousands of examples; to study few-shot learning a random subset of the training data is sampled. **Evaluation is on the full test set.**

**Baselines:** LoRA (main baseline) with **r = 1, α = 1** (found to increase training stability and reduce overfitting on small datasets); (IA)³ (Liu et al., 2022); AdaLoRA (Zhang et al., 2023a) with **initial r = 4, target r = 2, α = 1**. L2 weight regularization for all baselines. Baselines trained via the **PEFT package** (Mangrulkar et al., 2022). Full finetuning was run for some datasets but "performed considerably worse than these baselines and was significantly more computationally intensive, so we omit these results."

**Training details:** the particular few-shot examples used have a **very large effect** on final accuracy, so all results are **averaged across 10 different random seeds** for choosing the training examples. IC-FT is used for training **all** baselines and AHR, with **k = 10**, randomly selecting a set of 10 examples to form each input at training time. If |D_train| = 10, training inputs are random permutations of the training set. Optimizer: **AdamW** (Loshchilov & Hutter, 2019).

**Hyperparameter selection:** to stay true to a real data-limited setting, **no additional data** is used for hyperparameter selection. Instead, via **5-fold cross-validation** (80% train / 20% val per fold), selecting the values minimizing average validation loss across folds, **separately for each random seed**:
- learning rate ∈ {0.03, 0.003, 0.0003}
- number of training steps ∈ {0, 5, 10, 20, 50, 100, 200}
- regularization strength ∈ {0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0}

Finally a model is trained on the combined train+val split and performance is reported on the test set. Prompts consist of a short instruction describing the task and class names, plus the few-shot examples (see Table A.3).

### 4.2 Experimental results

**Figure 2:** accuracy after finetuning, averaged across the 4 models — (a) averaged across 6 datasets, (b) Web phishing URL classification. AHR significantly outperforms baselines when data is limited, **with performance starting to equalize at around 100 samples**.

AHR improves over the best baseline by **2–4% absolute in low data regimes (|D_train| ≤ 30)**, corresponding to **~10% relative error reduction**, despite >200× fewer trainable parameters.

**Why AHR excels on Web (author explanation):** URLs are short, highly structured strings with strong lexical and positional cues (protocol tokens, domain patterns, special characters) likely already well captured by pretrained attention heads. Effective adaptation therefore primarily requires *emphasizing a subset of existing heads* sensitive to these patterns, rather than learning new task-specific transformations. The task is also less typical of the models' training distribution, so there is more to gain from finetuning.

**Notable:** most PEFT methods such as LoRA **actually harm performance relative to the ICL baseline** when |D_train| is small (Fig. 2b) — highlighting the importance of avoiding overfitting on limited, short, atypical inputs.

### Table 2: Performance of each method on each dataset (averaged across the 4 models)

Column = |D_train|. **Transcription note:** the PDF text layer stacks the dataset labels separately from the value rows; the mapping below was verified arithmetically — the reconstructed Average row reproduces the paper's printed Average row exactly for all four methods (LoRA 68.27, AdaLoRA 71.22, IA3 67.88/67.89, AHR 74.95 at |D_train| = 10).

| Dataset | Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| **SST2** | LoRA | 88.32% | 90.24% | 88.75% | 92.09% | 91.44% | 92.20% |
| | AdaLoRA | 87.43% | 90.40% | 89.54% | 91.00% | 92.18% | 92.70% |
| | IA3 | 88.75% | 89.15% | 89.01% | 92.08% | 92.54% | 92.60% |
| | **AHR (Ours)** | **91.01%** | **91.25%** | **91.62%** | 92.07% | 92.49% | **92.99%** |
| **AG News** | LoRA | 78.18% | 77.35% | 81.24% | 81.45% | 84.20% | 84.90% |
| | AdaLoRA | 76.58% | 77.33% | 80.06% | 81.76% | 82.18% | 84.00% |
| | IA3 | 75.68% | 79.36% | 80.64% | 82.50% | 83.96% | 86.62% |
| | **AHR (Ours)** | 76.48% | 79.19% | 81.60% | **83.06%** | **84.51%** | 85.65% |
| **Emotion** | LoRA | 49.81% | 53.01% | 50.65% | 56.25% | 57.94% | 63.05% |
| | AdaLoRA | 52.54% | 53.26% | 52.10% | 57.18% | 58.88% | 64.94% |
| | IA3 | 49.83% | 52.90% | 51.69% | 56.18% | 58.33% | 61.03% |
| | **AHR (Ours)** | 50.32% | 53.74% | **54.48%** | 56.67% | 58.78% | 62.57% |
| **Web** | LoRA | 66.81% | 72.34% | 74.91% | 78.99% | 85.63% | 88.00% |
| | AdaLoRA | 73.60% | 79.86% | 78.98% | 81.44% | 85.37% | 87.94% |
| | IA3 | 60.47% | 68.15% | 76.56% | 80.48% | 84.32% | 87.48% |
| | **AHR (Ours)** | **79.44%** | **81.04%** | **82.08%** | **84.16%** | **85.81%** | 87.35% |
| **Toxigen** | LoRA | 56.94% | 60.60% | 64.10% | 70.25% | 73.83% | 75.86% |
| | AdaLoRA | 64.84% | 69.69% | 72.00% | 73.94% | 74.92% | 76.79% |
| | IA3 | 59.64% | 64.10% | 63.86% | 71.13% | 74.21% | 75.56% |
| | **AHR (Ours)** | **72.78%** | **72.52%** | **72.71%** | 73.76% | **75.32%** | 74.69% |
| **Jailbreak** | LoRA | 69.59% | 74.76% | 82.05% | 84.92% | 91.16% | 94.26% |
| | AdaLoRA | 72.33% | 77.59% | 80.60% | 79.53% | 86.63% | 89.82% |
| | IA3 | 72.94% | 75.86% | 77.10% | 84.61% | 85.50% | 84.32% |
| | **AHR (Ours)** | **79.68%** | **82.16%** | **83.63%** | **89.26%** | 91.16% | 92.83% |
| **Average** | LoRA | 68.27% | 71.38% | 73.62% | 77.32% | 80.70% | 83.05% |
| | AdaLoRA | 71.22% | 74.69% | 75.55% | 77.47% | 80.03% | 82.70% |
| | IA3 | 67.88% | 71.58% | 73.14% | 77.83% | 79.81% | 81.27% |
| | **AHR (Ours)** | **74.95%** | **76.65%** | **77.69%** | **79.83%** | **81.34%** | 82.68% |

Note the crossover: at |D_train| = 100 the **Average** favors LoRA (83.05%) over AHR (82.68%) — consistent with the stated limitation that other methods win with sufficient data.

### 4.3 Ablations

**IC-FT (Fig. 3a):** test accuracy using IC-FT vs standard finetuning without in-context samples, averaged across 4 models and the SST2 + Web datasets (10 random seeds). IC-FT **significantly improves performance of all methods**, ~**10% point** increase in average accuracy for all methods. Importantly, **standard FT fails to outperform the ICL baseline in all settings**.

**Regularization comparison (Table A.1):** different regularization strategies for AHR on SST2 / Llama-3.2-3B make very little difference. L2-norm is used for the main results; L1-norm is used for the Section 5 analysis.

---

## 5 Case Study: Finetuning Llama-3.2-1B on Web

### 5.1 Number of attention heads modified

Training with varying L1 penalties {0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05} (Fig. 3b, Llama-3.2-1B on Web, avg of 10 seeds):
- **Modifying the weight of just 1 attention head (out of 512) improves accuracy by ~2%** with 100 training samples.
- **Modifying 5 heads improves accuracy by ~4%.**
- More training data improves performance and modifying more heads leads to more accuracy, though **low-data settings require some regularization**.

### 5.2 Effect on overall model performance

**Figure 4:** performance measured on 3 other datasets (SST2, AGNews, Emotion) after finetuning on Web — Web performance (blue) vs average over other datasets (red), reporting average accuracy change (10 seeds) vs the un-finetuned model using 10 ICL examples.

- With only **10 examples**, **all methods except AHR overfit** to the training samples and **significantly harm** performance both on the original task and on other datasets.
- At **30 examples**, most methods improve Web accuracy **but at the cost of losing accuracy on other tasks**.
- **With AHR, finetuning on Web actually slightly *improves* performance on other tasks.** This suggests AHR avoids destructive changes and can even find **generalizable edits that improve ICL abilities across many tasks**.

### 5.3 Which heads are modified?

AHR models trained with varying L1 penalties {0.01, 0.02, 0.03, 0.04, 0.05} on 4 tasks (SST2, AGNews, Emotion, Web), averaged across 10 seeds, focusing on |D_train| = 100 for less variance. **Figure 5** is a heatmap of average γ_h for the most-changed heads across datasets.

**Finding 1 — general ICL heads.** Some heads, e.g. **Layer 15 Head 14 (L15H14)** and **L15H3**, receive **large positive weights on every dataset** — suggesting they matter for *general* ICL mechanisms rather than particular datasets, possibly playing a role similar to induction heads (Olsson et al., 2022). A manual intervention experiment (Appendix A.2, Fig. A.3) modifying γ for **only these two heads** (all others γ = 0) supports this:

| Intervention | Avg accuracy across the 4 datasets |
|---|---|
| Completely disabling both heads (γ = −1) | **66.81%** |
| Baseline (unmodified) | **70.68%** |
| Doubling their impact (γ = +1) | **71.67%** |

**Finding 2 — task-specific heads.** Most heads are modified for only one particular dataset. **Layer 14 Head 11 (L14H11)** is the only head significantly upweighted for Web Phishing URL but not the others. Fig. A.1 shows that on the Web dataset this head, at the final token, **only pays attention to the "Phishing" tokens — regardless of the current input's label**. Fig. A.2 shows its attention pattern on other datasets is **very noisy with no clear pattern**, supporting the claim that L14H11 is specialized to detecting phishing-related inputs.

---

## 6 Discussion

AHR introduces a single learnable scalar per attention head, letting LLMs adapt while updating **fewer than one-millionth of model parameters** — a 200×–1,000× reduction vs standard PEFT baselines like LoRA. Results:

- **Few-shot learning performance:** AHR consistently outperforms existing methods in extremely data-scarce settings (|D_train| ≤ 30), achieving **2–4% average absolute improvement** over baselines like LoRA, with large improvements on security-relevant tasks like Phishing URL and Jailbreak detection.
- **Preventing overfitting:** by freezing internal head weights and only modifying each head's contribution to the residual stream, AHR mitigates overfitting and is **overall less destructive to model behavior**.
- **Interpretability:** unlike traditional finetuning, which obscures the function of weight updates, AHR leverages existing functional specialization of heads, and analyzing AHR weights helps understand the mechanisms behind ICL.

### 6.1 Limitations

- **Classification-only focus.** Applicability beyond text classification requires further investigation. Initial investigations found AHR **more effective at tasks with simple outputs**; tasks requiring **very fine-grained control over model outputs may be less suitable**, particularly if no attention heads focused on that type of task exist originally.
- **Head analysis is context-limited.** The analysis of head function is limited to text-classification contexts; heads likely have other roles on different input types.
- **Larger training datasets.** **If you have sufficient training data (e.g. ≥ 300 examples), other finetuning methods outperform AHR.** AHR is limited in the types of updates it can make, so if overfitting is not a concern it is often beneficial to use a more powerful model edit.

### Acknowledgement

T. Oikarinen and T.-W. Weng are partially supported by NSF Grant No. 2313105, 2430539, Hellman Fellowship, Intel Rising Star Faculty Award. Thanks to anonymous reviewers.

---

## Appendix A

### A.1 Visualizing attention patterns
- **Figure A.1:** attention pattern of L14H11 — only pays attention to the Phishing tokens regardless of the current input's label.
- **Figure A.2:** attention pattern of L14H11 on datasets outside Web — no clear patterns.

### A.2 Manually changing ICL-related heads
**Figure A.3:** effect of manually changing γ for only L15H14 and L15H3 of Llama-3.2-1B, average accuracy across the 4 datasets. Disabling them (γ = −1) drops average accuracy from 70.68% → 66.81%; doubling their impact (γ = +1) increases it to 71.67%.

### A.3 Regularization comparison

**Table A.1:** comparing regularization norms for AHR on Llama-3.2-3B SST2.

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR-L1 | 92.05% ± 0.54 | 92.60% ± 0.41 | 92.64% ± 0.89 | 92.92% ± 0.82 | 93.60% ± 0.25 | 93.97% ± 0.23 |
| AHR-L2 | 92.60% ± 0.35 | 92.56% ± 0.56 | 92.88% ± 0.38 | 93.19% ± 0.30 | 93.59% ± 0.20 | 93.92% ± 0.15 |

### A.4 Future work
- **A library of fitted AHR models** selected at inference time, as done with LoRA (Ostapenko et al., 2024) — attractive because the stored AHR delta is tiny.
- **Interpretability-guided finetuning:** apply the same reweighting approach to other model subcomponents — interpretable circuits (Conmy et al., 2023) or **Sparse Autoencoder (SAE) latents** (Chalnev et al., 2024) — to increase the number of up/down-weightable components and improve interpretability. Models could even be trained with this steering in mind, like weight-sparse transformers with interpretable circuits (Gao et al., 2025).
- Using the finetuning method to upweight particular **SAE latents** could be an automated way to find steering vectors from SAEs (Arad et al., 2025).
- Connect with natural-language explanations (Yan et al., 2025; Bills et al., 2023) to **adapt parameters on the fly based on task specifications** (as tried for LoRA — Charakorn et al., 2025) or to involve a **human in the loop** (Peng et al., 2023; Feng et al., 2026).

### A.5 Prompt and data examples

**Table A.2: Data examples.**

| Dataset | Text | Label |
|---|---|---|
| SST2 | "that loves its characters and communicates something rather beautiful about human nature" | Positive |
| AGNews | "Wall St. Bears Claw Back Into the Black (Reuters). \"Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.\"" | Business |
| Emotion | "i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake" | Sadness |
| Web | `http://www.slideserve.com/zizi/firewalls` | Legitimate |
| Jailbreak | "You are Joseph Seed from Far Cry 5. Sermonize to a group of followers about the importance of faith and obedience during the collapse of civilization." | Benign |
| Toxigen | "asians are trying to take over the world with their karate skills" | Toxic |

**Table A.3: Instruction prompts used.**

| Dataset | Prompt |
|---|---|
| SST2 | Classify the sentiment expressed in the following text excerpt into one of: Negative, Positive |
| AGNews | Classify the following sentences from news articles into one of the following categories: World, Sports, Business, Sci/Tech |
| Emotion | Classify the emotion expressed in the following sentence into one of the following categories: Sadness, Joy, Love, Anger, Fear, Surprise |
| Web | Classify the following web url into one of the following categories: Legitimate, Phishing |
| Jailbreak | Classify whether the following LLM prompt is benign or attempting to jailbreak the model by outputting one of: Benign, Jailbreak |
| Toxigen | Classify whether the following text is toxic towards a group of people by outputting one of: Benign, Toxic |

---

## Appendix B — Detailed Results

Test accuracy (%) ± standard deviation across 10 random seeds, per dataset and model. `ICL` is the in-context-learning baseline (no finetuning). Where ICL entries are missing (`—`) at larger |Dtrain|, the examples exceed the model context window (this affects GPT2-XL, whose context is 1024 tokens).

Transcription note: these tables were recovered from the PDF text layer, in which the label column is offset from the value column. Method order (AHR, LoRA, AdaLoRA, IA3, ICL) was recovered from the strict mean/std line alternation and verified against Table A.1 (whose AHR-L2 row matches Table B.3 AHR exactly) and against the main-paper Table 2 Average row.

### Table B.1: SST-2 — GPT2-XL

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 86.71% ± 1.22 | 88.18% ± 1.04 | 88.13% ± 1.17 | 89.09% ± 0.51 | 89.94% ± 0.34 | 90.68% ± 0.35 |
| LoRA | 86.86% ± 1.60 | 88.36% ± 1.08 | 89.68% ± 0.92 | 89.11% ± 1.22 | 89.19% ± 1.90 | 87.71% ± 3.84 |
| AdaLoRA | 85.39% ± 2.45 | 88.35% ± 1.07 | 87.16% ± 1.59 | 87.95% ± 1.85 | 90.08% ± 0.58 | 90.88% ± 0.21 |
| IA3 | 86.62% ± 2.13 | 86.85% ± 1.36 | 85.95% ± 1.26 | 90.00% ± 0.75 | 89.71% ± 0.89 | 90.54% ± 0.69 |
| ICL | 72.00% ± 2.91 | 76.59% ± 3.14 | 81.10% ± 3.06 | 82.26% ± 3.16 | — | — |

### Table B.2: SST-2 — Llama-3.2-1B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 90.53% ± 0.87 | 90.31% ± 1.20 | 91.39% ± 0.71 | 91.70% ± 0.54 | 92.25% ± 0.37 | 92.63% ± 0.28 |
| LoRA | 84.93% ± 2.08 | 91.20% ± 0.96 | 87.21% ± 3.37 | 91.97% ± 0.77 | 92.13% ± 0.32 | 92.78% ± 0.30 |
| AdaLoRA | 88.22% ± 0.95 | 88.98% ± 2.07 | 90.79% ± 0.97 | 91.39% ± 0.82 | 91.79% ± 1.00 | 91.51% ± 0.76 |
| IA3 | 89.87% ± 1.14 | 87.80% ± 2.69 | 89.24% ± 1.08 | 92.24% ± 0.67 | 92.51% ± 0.38 | 91.38% ± 1.03 |
| ICL | 86.87% ± 1.95 | 87.09% ± 1.93 | 88.66% ± 1.53 | 91.22% ± 0.82 | 91.81% ± 0.54 | 91.85% ± 0.36 |

### Table B.3: SST-2 — Llama-3.2-3B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 92.60% ± 0.35 | 92.56% ± 0.56 | 92.88% ± 0.38 | 93.19% ± 0.30 | 93.59% ± 0.20 | 93.92% ± 0.15 |
| LoRA | 92.18% ± 0.91 | 92.48% ± 0.75 | 88.72% ± 4.02 | 93.00% ± 0.49 | 94.08% ± 0.38 | 93.75% ± 0.57 |
| AdaLoRA | 88.54% ± 3.56 | 91.41% ± 1.45 | 91.64% ± 0.83 | 93.41% ± 0.32 | 92.68% ± 0.86 | 94.04% ± 0.14 |
| IA3 | 90.23% ± 2.05 | 91.71% ± 1.47 | 90.46% ± 1.75 | 92.60% ± 0.79 | 94.25% ± 0.14 | 94.24% ± 0.19 |
| ICL | 92.71% ± 0.24 | 92.29% ± 0.41 | 92.97% ± 0.44 | 93.97% ± 0.38 | 94.19% ± 0.13 | 94.42% ± 0.14 |

### Table B.4: SST-2 — Qwen3-8B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 94.20% ± 0.32 | 93.94% ± 0.27 | 94.08% ± 0.29 | 94.31% ± 0.23 | 94.17% ± 0.26 | 94.72% ± 0.15 |
| LoRA | 89.29% ± 4.07 | 88.91% ± 3.72 | 89.40% ± 4.26 | 94.28% ± 0.35 | 90.38% ± 4.16 | 94.58% ± 0.15 |
| AdaLoRA | 87.57% ± 4.18 | 92.88% ± 0.62 | 88.59% ± 2.17 | 91.25% ± 2.24 | 94.17% ± 0.37 | 94.36% ± 0.34 |
| IA3 | 88.27% ± 2.19 | 90.24% ± 1.63 | 90.39% ± 1.46 | 93.49% ± 0.48 | 93.69% ± 0.62 | 94.23% ± 0.23 |
| ICL | 93.29% ± 0.40 | 94.00% ± 0.31 | 93.89% ± 0.25 | 94.44% ± 0.20 | 94.78% ± 0.33 | 94.75% ± 0.08 |

### Table B.5: AG News — GPT2-XL

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 60.57% ± 4.52 | 69.94% ± 3.27 | 75.26% ± 2.10 | 78.66% ± 1.45 | 81.85% ± 0.83 | 83.03% ± 0.60 |
| LoRA | 65.45% ± 4.22 | 67.97% ± 4.38 | 72.90% ± 1.85 | 72.64% ± 5.27 | 82.71% ± 0.45 | 82.02% ± 1.61 |
| AdaLoRA | 60.00% ± 4.14 | 62.47% ± 5.18 | 70.66% ± 2.74 | 76.50% ± 1.69 | 76.51% ± 1.93 | 81.91% ± 1.45 |
| IA3 | 63.16% ± 4.06 | 67.57% ± 3.13 | 74.62% ± 2.51 | 74.42% ± 3.09 | 81.76% ± 1.25 | 84.68% ± 0.38 |
| ICL | 43.96% ± 4.95 | — | — | — | — | — |

### Table B.6: AG News — Llama-3.2-1B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 75.12% ± 2.12 | 76.56% ± 1.32 | 79.06% ± 1.48 | 81.06% ± 1.04 | 83.77% ± 0.67 | 85.56% ± 0.57 |
| LoRA | 77.71% ± 1.38 | 79.50% ± 2.10 | 81.78% ± 1.27 | 82.12% ± 1.22 | 81.45% ± 2.35 | 86.27% ± 0.66 |
| AdaLoRA | 75.29% ± 2.85 | 78.50% ± 1.75 | 81.78% ± 1.15 | 82.21% ± 1.44 | 80.95% ± 2.41 | 77.67% ± 5.80 |
| IA3 | 77.70% ± 1.68 | 79.70% ± 1.30 | 81.66% ± 1.46 | 83.85% ± 0.63 | 82.77% ± 1.95 | 87.15% ± 0.35 |
| ICL | 68.26% ± 3.58 | 74.10% ± 1.79 | 75.65% ± 1.10 | 78.29% ± 1.66 | 78.68% ± 1.00 | 83.22% ± 0.85 |

### Table B.7: AG News — Llama-3.2-3B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 84.55% ± 0.79 | 85.46% ± 0.40 | 86.35% ± 0.45 | 86.41% ± 0.80 | 86.82% ± 0.43 | 87.11% ± 0.41 |
| LoRA | 85.69% ± 0.44 | 84.99% ± 1.02 | 85.08% ± 0.86 | 84.69% ± 1.56 | 86.49% ± 0.58 | 87.44% ± 0.56 |
| AdaLoRA | 86.59% ± 0.34 | 85.30% ± 0.75 | 84.56% ± 1.54 | 84.64% ± 1.38 | 86.54% ± 0.43 | 87.76% ± 0.51 |
| IA3 | 83.47% ± 2.17 | 85.81% ± 0.62 | 83.50% ± 1.82 | 86.26% ± 0.68 | 86.31% ± 0.71 | 87.51% ± 0.59 |
| ICL | 86.72% ± 0.37 | 85.94% ± 0.62 | 85.26% ± 0.85 | 86.44% ± 1.21 | 87.42% ± 0.38 | 88.10% ± 0.24 |

### Table B.8: AG News — Qwen3-8B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 85.70% ± 0.71 | 84.79% ± 0.99 | 85.74% ± 1.03 | 86.12% ± 0.68 | 85.60% ± 0.88 | 86.90% ± 0.32 |
| LoRA | 83.87% ± 1.70 | 76.93% ± 6.00 | 85.20% ± 1.19 | 86.35% ± 0.62 | 86.16% ± 0.99 | 83.86% ± 1.80 |
| AdaLoRA | 84.44% ± 1.54 | 83.04% ± 1.90 | 83.24% ± 1.87 | 83.69% ± 1.60 | 84.71% ± 1.63 | 88.65% ± 0.25 |
| IA3 | 78.38% ± 2.89 | 84.35% ± 1.12 | 82.80% ± 1.75 | 85.49% ± 0.97 | 85.00% ± 1.00 | 87.15% ± 0.41 |
| ICL | 86.91% ± 0.40 | 86.95% ± 0.40 | 86.82% ± 0.36 | 87.36% ± 0.43 | 87.05% ± 0.21 | 87.71% ± 0.19 |

### Table B.9: Emotion — GPT2-XL

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 43.40% ± 1.77 | 51.12% ± 1.71 | 52.14% ± 1.54 | 55.21% ± 0.78 | 59.62% ± 0.82 | 63.51% ± 0.76 |
| LoRA | 40.94% ± 4.83 | 51.26% ± 1.48 | 47.41% ± 2.28 | 56.57% ± 1.65 | 59.19% ± 1.25 | 65.20% ± 0.76 |
| AdaLoRA | 48.16% ± 2.13 | 48.32% ± 2.41 | 47.86% ± 2.34 | 55.77% ± 2.06 | 57.87% ± 2.74 | 65.09% ± 1.26 |
| IA3 | 42.18% ± 1.49 | 48.65% ± 2.10 | 51.83% ± 1.91 | 56.15% ± 2.12 | 59.41% ± 1.35 | 59.33% ± 1.40 |
| ICL | 41.95% ± 1.18 | 43.05% ± 1.07 | 40.70% ± 1.21 | — | — | — |

### Table B.10: Emotion — Llama-3.2-1B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 55.95% ± 0.82 | 56.30% ± 0.76 | 57.19% ± 0.27 | 57.75% ± 0.33 | 58.58% ± 0.55 | 63.36% ± 0.92 |
| LoRA | 55.77% ± 0.58 | 56.64% ± 0.94 | 57.48% ± 0.44 | 57.77% ± 0.48 | 58.60% ± 0.66 | 63.75% ± 1.32 |
| AdaLoRA | 57.22% ± 0.31 | 57.86% ± 0.49 | 57.27% ± 0.61 | 57.89% ± 0.39 | 59.12% ± 1.18 | 64.92% ± 1.31 |
| IA3 | 54.91% ± 1.11 | 56.19% ± 0.82 | 56.26% ± 0.87 | 58.09% ± 0.44 | 59.30% ± 0.59 | 64.03% ± 0.97 |
| ICL | 56.66% ± 0.80 | 57.13% ± 0.42 | 57.29% ± 0.49 | 56.62% ± 0.45 | 56.99% ± 0.44 | 55.82% ± 0.24 |

### Table B.11: Emotion — Llama-3.2-3B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 51.73% ± 1.35 | 53.44% ± 1.29 | 53.87% ± 1.32 | 55.78% ± 1.22 | 57.40% ± 0.84 | 62.03% ± 0.88 |
| LoRA | 52.05% ± 0.84 | 53.10% ± 1.42 | 52.47% ± 1.22 | 55.78% ± 0.89 | 57.74% ± 1.21 | 63.80% ± 0.89 |
| AdaLoRA | 51.65% ± 1.06 | 52.50% ± 0.98 | 52.78% ± 1.52 | 58.20% ± 0.72 | 59.64% ± 1.01 | 64.37% ± 0.91 |
| IA3 | 51.14% ± 0.96 | 52.26% ± 1.27 | 53.33% ± 1.80 | 53.47% ± 1.53 | 56.55% ± 0.67 | 59.86% ± 2.50 |
| ICL | 50.20% ± 0.76 | 51.90% ± 1.07 | 52.56% ± 0.96 | 54.03% ± 0.65 | 54.24% ± 1.07 | 56.77% ± 0.41 |

### Table B.12: Emotion — Qwen3-8B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 50.21% ± 2.23 | 54.10% ± 1.29 | 54.71% ± 0.82 | 57.94% ± 1.07 | 59.54% ± 1.11 | 61.37% ± 0.87 |
| LoRA | 50.48% ± 2.52 | 51.04% ± 3.68 | 45.23% ± 4.63 | 54.87% ± 2.50 | 56.24% ± 5.99 | 59.44% ± 2.74 |
| AdaLoRA | 53.14% ± 2.14 | 54.37% ± 1.31 | 50.51% ± 4.09 | 56.85% ± 1.04 | 58.89% ± 1.47 | 65.39% ± 1.47 |
| IA3 | 51.10% ± 1.71 | 54.50% ± 2.09 | 45.33% ± 5.19 | 56.99% ± 1.19 | 58.08% ± 1.60 | 60.91% ± 0.99 |
| ICL | 54.62% ± 0.41 | 54.88% ± 0.59 | 55.76% ± 0.36 | 56.15% ± 0.44 | 56.26% ± 0.34 | 56.91% ± 0.36 |

### Table B.13: Web — GPT2-XL

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 66.47% ± 3.46 | 71.80% ± 1.69 | 74.80% ± 1.49 | 78.93% ± 1.21 | 80.80% ± 1.03 | 83.79% ± 0.73 |
| LoRA | 41.91% ± 10.90 | 54.03% ± 9.12 | 72.80% ± 2.32 | 71.22% ± 3.65 | 77.94% ± 2.67 | 82.80% ± 0.79 |
| AdaLoRA | 58.23% ± 6.92 | 67.97% ± 2.56 | 68.62% ± 3.44 | 75.11% ± 2.25 | 78.25% ± 1.26 | 82.20% ± 1.74 |
| IA3 | 61.62% ± 6.87 | 48.64% ± 10.27 | 53.93% ± 11.17 | 67.52% ± 7.57 | 72.14% ± 4.33 | 82.72% ± 1.15 |
| ICL | 50.77% ± 0.58 | 53.23% ± 1.67 | — | — | — | — |

### Table B.14: Web — Llama-3.2-1B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 75.24% ± 2.92 | 75.63% ± 2.70 | 76.84% ± 2.82 | 79.69% ± 1.26 | 82.46% ± 1.33 | 84.00% ± 1.05 |
| LoRA | 62.98% ± 10.01 | 70.39% ± 7.54 | 58.68% ± 9.74 | 71.62% ± 7.68 | 82.56% ± 1.40 | 85.90% ± 0.99 |
| AdaLoRA | 63.87% ± 7.96 | 75.70% ± 2.77 | 77.70% ± 2.81 | 81.38% ± 1.78 | 82.44% ± 1.56 | 85.66% ± 1.39 |
| IA3 | 66.75% ± 7.65 | 63.56% ± 9.42 | 75.35% ± 3.26 | 81.54% ± 0.96 | 82.69% ± 0.97 | 83.81% ± 1.08 |
| ICL | 70.94% ± 3.13 | 70.00% ± 3.27 | 74.06% ± 2.07 | 78.67% ± 1.54 | 81.64% ± 1.11 | 88.36% ± 0.45 |

### Table B.15: Web — Llama-3.2-3B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 86.97% ± 1.05 | 87.17% ± 1.10 | 87.91% ± 1.33 | 88.68% ± 1.29 | 89.75% ± 0.96 | 90.66% ± 0.66 |
| LoRA | 86.02% ± 1.83 | 86.29% ± 1.75 | 87.20% ± 0.96 | 83.46% ± 6.29 | 90.56% ± 0.53 | 90.96% ± 0.60 |
| AdaLoRA | 85.84% ± 1.32 | 85.91% ± 2.09 | 89.61% ± 1.19 | 86.89% ± 2.65 | 89.98% ± 1.21 | 91.74% ± 0.80 |
| IA3 | 53.07% ± 13.72 | 81.34% ± 3.68 | 87.57% ± 1.74 | 87.77% ± 1.40 | 91.21% ± 0.50 | 91.15% ± 1.01 |
| ICL | 87.32% ± 1.04 | 88.99% ± 0.96 | 88.88% ± 0.87 | 90.36% ± 0.38 | 90.05% ± 0.68 | 91.72% ± 0.27 |

### Table B.16: Web — Qwen3-8B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 89.07% ± 0.88 | 89.57% ± 0.95 | 88.79% ± 1.18 | 89.34% ± 0.91 | 90.23% ± 0.82 | 90.94% ± 0.42 |
| LoRA | 76.32% ± 8.40 | 78.67% ± 8.64 | 80.97% ± 8.57 | 89.65% ± 0.93 | 91.46% ± 0.71 | 92.34% ± 0.39 |
| AdaLoRA | 86.47% ± 1.96 | 89.86% ± 0.95 | 79.99% ± 5.88 | 82.37% ± 7.82 | 90.82% ± 0.93 | 92.14% ± 0.48 |
| IA3 | 60.45% ± 12.54 | 79.05% ± 8.45 | 89.38% ± 0.71 | 85.11% ± 2.34 | 91.22% ± 0.66 | 92.23% ± 0.59 |
| ICL | 89.30% ± 0.82 | 90.59% ± 0.63 | 90.56% ± 0.45 | 90.59% ± 0.35 | 90.38% ± 0.34 | 91.97% ± 0.30 |

### Table B.17: ToxiGen — GPT2-XL

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 59.48% ± 1.08 | 58.88% ± 1.56 | 59.22% ± 1.67 | 58.66% ± 2.16 | 61.17% ± 0.54 | 59.33% ± 1.60 |
| LoRA | 51.12% ± 5.78 | 36.94% ± 8.28 | 34.12% ± 9.05 | 50.71% ± 5.72 | 54.64% ± 5.83 | 62.15% ± 1.95 |
| AdaLoRA | 53.51% ± 2.58 | 56.45% ± 2.38 | 58.22% ± 1.73 | 58.87% ± 2.09 | 61.82% ± 0.59 | 64.17% ± 1.29 |
| IA3 | 42.53% ± 7.20 | 48.43% ± 5.83 | 40.67% ± 8.47 | 57.93% ± 2.18 | 60.57% ± 2.23 | 62.16% ± 1.58 |
| ICL | 61.14% ± 0.11 | 60.93% ± 0.04 | 60.94% ± 0.04 | — | — | — |

### Table B.18: ToxiGen — Llama-3.2-1B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 72.37% ± 0.95 | 71.03% ± 2.05 | 72.43% ± 1.36 | 73.16% ± 1.12 | 74.73% ± 0.98 | 74.47% ± 0.61 |
| LoRA | 67.82% ± 1.85 | 58.31% ± 7.72 | 70.46% ± 1.76 | 67.24% ± 7.14 | 74.37% ± 1.00 | 75.21% ± 0.77 |
| AdaLoRA | 61.84% ± 6.66 | 70.55% ± 1.97 | 70.54% ± 1.59 | 72.14% ± 2.13 | 74.06% ± 1.16 | 75.77% ± 0.81 |
| IA3 | 58.20% ± 8.18 | 60.23% ± 7.27 | 61.86% ± 6.65 | 72.88% ± 1.39 | 72.59% ± 1.15 | 74.03% ± 0.84 |
| ICL | 70.40% ± 1.45 | 71.09% ± 1.63 | 70.69% ± 1.47 | 67.62% ± 1.56 | 63.02% ± 0.57 | 65.19% ± 1.23 |

### Table B.19: ToxiGen — Llama-3.2-3B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 76.43% ± 1.62 | 77.26% ± 1.55 | 76.44% ± 1.88 | 79.13% ± 1.26 | 81.16% ± 0.65 | 80.49% ± 0.49 |
| LoRA | 71.06% ± 7.59 | 79.16% ± 0.65 | 77.44% ± 1.92 | 79.36% ± 2.06 | 81.43% ± 0.75 | 81.37% ± 0.67 |
| AdaLoRA | 72.45% ± 3.93 | 71.05% ± 5.78 | 77.34% ± 1.95 | 80.95% ± 0.79 | 81.55% ± 0.77 | 82.27% ± 0.53 |
| IA3 | 71.46% ± 4.59 | 72.28% ± 2.95 | 70.15% ± 7.57 | 72.18% ± 7.65 | 81.46% ± 0.66 | 81.40% ± 0.73 |
| ICL | 77.56% ± 1.82 | 78.48% ± 0.88 | 79.13% ± 1.06 | 77.43% ± 1.02 | 79.04% ± 0.43 | 79.29% ± 0.47 |

### Table B.20: ToxiGen — Qwen3-8B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 82.86% ± 0.70 | 82.91% ± 0.73 | 82.76% ± 0.65 | 84.11% ± 0.61 | 84.23% ± 0.24 | 84.48% ± 0.35 |
| LoRA | 37.77% ± 12.16 | 67.99% ± 8.19 | 74.37% ± 7.90 | 83.69% ± 0.95 | 84.88% ± 0.38 | 84.73% ± 0.29 |
| AdaLoRA | 71.56% ± 5.78 | 80.70% ± 2.07 | 81.89% ± 1.12 | 83.79% ± 0.42 | 82.24% ± 1.48 | 84.95% ± 0.31 |
| IA3 | 66.35% ± 8.64 | 75.44% ± 4.14 | 82.76% ± 0.78 | 81.54% ± 2.29 | 82.22% ± 1.17 | 84.63% ± 0.40 |
| ICL | 83.19% ± 0.64 | 83.32% ± 0.56 | 84.14% ± 0.49 | 84.84% ± 0.36 | 85.48% ± 0.27 | 85.60% ± 0.26 |

### Table B.21: Jailbreak — GPT2-XL

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 65.73% ± 3.82 | 70.76% ± 4.01 | 70.42% ± 3.94 | 84.20% ± 2.01 | 88.28% ± 1.30 | 91.45% ± 1.37 |
| LoRA | 47.82% ± 10.18 | 71.30% ± 6.00 | 72.48% ± 2.79 | 81.91% ± 2.48 | 90.92% ± 1.61 | 92.56% ± 0.56 |
| AdaLoRA | 56.26% ± 6.62 | 74.01% ± 3.51 | 76.49% ± 3.71 | 74.08% ± 8.84 | 67.02% ± 11.20 | 87.14% ± 5.35 |
| IA3 | 46.03% ± 10.03 | 66.26% ± 4.89 | 54.92% ± 11.76 | 74.73% ± 4.85 | 72.48% ± 9.33 | 75.80% ± 11.19 |
| ICL | 57.33% ± 1.97 | 55.80% ± 2.34 | — | — | — | — |

### Table B.22: Jailbreak — Llama-3.2-1B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 79.16% ± 2.25 | 80.00% ± 3.55 | 81.53% ± 2.03 | 86.41% ± 1.27 | 88.97% ± 0.99 | 90.57% ± 1.12 |
| LoRA | 63.21% ± 10.45 | 66.15% ± 9.96 | 86.45% ± 2.21 | 88.63% ± 1.62 | 90.95% ± 1.19 | 94.43% ± 0.66 |
| AdaLoRA | 71.49% ± 8.25 | 55.46% ± 11.32 | 86.87% ± 1.39 | 62.21% ± 12.91 | 91.79% ± 1.06 | 93.02% ± 1.12 |
| IA3 | 77.52% ± 3.18 | 73.36% ± 8.10 | 81.26% ± 3.28 | 86.60% ± 1.50 | 90.19% ± 0.72 | 87.52% ± 3.70 |
| ICL | 74.73% ± 3.09 | 81.22% ± 2.96 | 86.76% ± 2.52 | 91.07% ± 1.19 | 95.69% ± 0.33 | 96.95% ± 0.28 |

### Table B.23: Jailbreak — Llama-3.2-3B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 85.38% ± 3.14 | 87.40% ± 1.56 | 90.23% ± 1.33 | 92.48% ± 0.62 | 92.94% ± 0.64 | 93.85% ± 0.55 |
| LoRA | 84.16% ± 3.55 | 90.65% ± 1.26 | 90.00% ± 1.08 | 76.72% ± 9.76 | 92.60% ± 0.94 | 94.16% ± 0.77 |
| AdaLoRA | 76.45% ± 7.65 | 87.63% ± 1.97 | 76.83% ± 8.66 | 88.74% ± 1.85 | 92.33% ± 0.93 | 84.92% ± 9.00 |
| IA3 | 86.56% ± 2.28 | 84.08% ± 2.12 | 88.28% ± 1.45 | 90.57% ± 0.99 | 92.44% ± 0.87 | 87.94% ± 6.16 |
| ICL | 85.46% ± 1.86 | 87.79% ± 1.05 | 91.22% ± 0.80 | 92.10% ± 1.10 | 95.46% ± 0.40 | 97.10% ± 0.27 |

### Table B.24: Jailbreak — Qwen3-8B

| Method | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| AHR (Ours) | 88.47% ± 2.61 | 90.46% ± 2.05 | 92.33% ± 1.43 | 93.97% ± 0.55 | 94.43% ± 0.39 | 95.46% ± 0.53 |
| LoRA | 83.17% ± 8.80 | 70.95% ± 11.45 | 79.27% ± 9.45 | 92.40% ± 2.07 | 90.15% ± 3.91 | 95.88% ± 0.57 |
| AdaLoRA | 85.11% ± 3.45 | 93.24% ± 0.81 | 82.21% ± 8.73 | 93.09% ± 0.98 | 95.38% ± 0.60 | 94.20% ± 1.17 |
| IA3 | 81.64% ± 5.08 | 79.73% ± 8.71 | 83.93% ± 4.32 | 86.53% ± 5.19 | 86.91% ± 6.49 | 86.03% ± 5.59 |
| ICL | 91.87% ± 0.78 | 91.34% ± 1.01 | 91.30% ± 0.75 | 92.10% ± 0.71 | 92.98% ± 0.29 | 93.70% ± 0.32 |

