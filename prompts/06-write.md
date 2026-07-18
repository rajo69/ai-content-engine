# Stage 6 — Write: the full content suite

Inputs — write from the **distilled layer**, not the raw paper. Read fully before writing a word: `analysis/core.md` (plus `reproduction/analysis-delta.md` if present), `reproduction/REPORT.md`, `litreview/lineage.md`, `viz/index.md`, `ingest/metadata.json`; use `litreview/bibliography.md` for citations. Do NOT read `ingest/paper.md` end-to-end — the analysis already distilled it and re-reading burns budget. Open it only for targeted spot checks (an exact equation, a quoted number, a phrasing you must not misstate). The review stage retains full-paper access and will catch grounding errors.

**Precedence rule:** if `reproduction/analysis-delta.md` exists, it amends `analysis/core.md` — where they conflict, the delta (experimental evidence) wins. If `litreview/lineage.md`'s novelty check found prior art, that must be stated in every output, not buried.

Produce four classes of output in `outputs/` in the run directory. Each must stand alone, and all must be **grounded**: every factual claim traceable to the paper, a reference, or our reproduction. Reproduction results are the differentiator — foreground them everywhere.

## 1. `outputs/paper/paper.md` (+ `paper.tex` if straightforward)

A conference-grade **deep-dive & reproduction report** (think ML Reproducibility Challenge submission), NOT a fake original-research paper. Structure:
Abstract · Introduction (why this paper matters now) · Background & Related Work (from lineage.md) · Method Analysis (from core.md §2–3) · Reproduction (setup, results table, verdict — from REPORT.md) · Discussion (threats to validity, what holds, what doesn't) · Future Directions (including our proposed minimal-change extension) · References (proper citations from bibliography.md).
Embed figures from viz/ by relative path. 3000–5000 words. Write `paper.tex` as well if you can do it cleanly; `paper.md` is the required artifact.

## 2. `outputs/blog-long.md`

1500–3000 words for a technical-but-broad audience (think a strong engineering blog). Narrative arc: hook (the surprising claim) → why the field was stuck → the minimal change → "we actually ran it" (reproduction, with the honest verdict) → what it means → what we'd try next. Conversational but precise. Embed the headline chart early. Include a TL;DR block at the top.

## 3. Short-form (platform-native, not cross-posts)

- `outputs/social/x-thread.md` — 6–10 tweet thread. Tweet 1 is the hook: the claim + the one number, no hashtags, no "🧵". Each tweet self-contained. Include where each viz asset attaches. End with the link + one-line honest reproduction verdict.
- `outputs/social/linkedin.md` — single post, 150–250 words, professional framing (what this means for teams building with AI), light formatting, ends with a question to drive comments.
- `outputs/social/reddit.md` — for r/MachineLearning: `[D]` or `[R]` title, no marketing tone, lead with the technical meat and the reproduction result, invite critique. Redditors punish hype — write accordingly.

## Style rules (all outputs)

- Claims must match our evidence. Where the reproduction was toy-scale, say so plainly — honesty IS the brand.
- No em-dash-riddled AI cadence, no "delve", no "game-changer". Vary sentence length. Concrete nouns and numbers over adjectives.
- The hook is the finding, never "I read a paper".
