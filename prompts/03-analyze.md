# Stage 3 — Deep analysis: dissect the core problem and solution

Inputs: `ingest/paper.md` (read it fully, twice if needed), `litreview/lineage.md`, `ingest/repo-notes.md` if present.

This is the intellectual heart of the pipeline. Think as hard as the paper deserves.

Read like a researcher (Keshav's method): first a structural pass — abstract, section headings, figures and their captions, conclusions — to build the map; then the deep pass — full text with the figures studied closely, noting every point where you'd challenge the authors. As you read, keep a running list of **assumptions the authors make silently** (about data distribution, scale, tuning budgets, metric choice) — these feed section 5.

Write `analysis/core.md` with these sections:

## 0. The research question
One crisp question this investigation answers — not the paper's question, ours (e.g., "Does reweighting existing attention heads really match adding new LoRA parameters, and does that hold at small scale?"). Every later section should serve this question.

## 1. The problem
State the core problem precisely — not the paper's framing, your own. What breaks, for whom, and why prior approaches (name them, cite from the lit review) fail at it.

## 2. The key insight
The one idea that makes this paper this paper, in one paragraph a smart undergraduate could follow. Then the same insight stated technically. If the insight is "minimal change, big effect", pin down exactly what the minimal change is.

## 3. Method mechanics
Walk the method end to end: inputs, transformations, outputs. Transcribe and explain the central equations/algorithms. Note every hyperparameter and design choice that carries load, and what the paper says (or fails to say) about sensitivity to them.

## 4. Evidence
What experiments support the claims? For each headline result: dataset, baseline, metric, margin, and whether the comparison is fair. Identify the single most impressive number and the single most fragile one.

## 5. Threats to validity
Play the skeptic. Confounds, weak baselines, cherry-picked benchmarks, missing ablations, compute mismatches, contamination risks — plus the silent assumptions you collected while reading. Be specific — vague skepticism is worthless.

## 6. Limitations and future work
What the authors admit, what they don't, and 3–5 concrete follow-up experiments or extensions — including at least one "minimal change" extension of our own that could itself be a viral finding.

## 7. Reproduction plan (feeds the next stage)
Given the code/dataset situation from ingest, specify the most convincing verification achievable on a consumer Windows machine within the time box: what to run, at what scale, what number to compare against which claim, and the fallback tiers if it doesn't work. State each check as a **falsifiable prediction** ("at toy scale N, if the claim holds we expect X to exceed Y by roughly Z") — the reproduce stage will pre-register these before running. If the headline claim is comparative ("beats baseline B"), the plan must include running B at the same scale.

Aim for 2000–4000 words of substance. No filler, no hedging boilerplate — every sentence should carry information.
