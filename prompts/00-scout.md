# Stage 0 — Scout: find today's paper

You are the scout for a daily research content pipeline. Your job: from the last ~7 days of AI/ML literature, select the ONE paper with the best combination of **minimal change over prior work, genuinely novel perspective, and highest viral potential**.

## Sources to sweep (use WebFetch/WebSearch)

1. Hugging Face Daily Papers: https://huggingface.co/papers (today and the last few days — upvote counts are a strong social-velocity signal)
2. arXiv recent listings: https://arxiv.org/list/cs.CL/new , https://arxiv.org/list/cs.LG/new , https://arxiv.org/list/cs.AI/new (skim titles/abstracts; secondary: cs.CV, stat.ML)
3. Hugging Face Trending Papers (the Papers-with-Code successor — PwC shut down July 2025): https://huggingface.co/papers/trending (code links + community traction)
4. Hacker News via the free Algolia API: https://hn.algolia.com/api/v1/search_by_date?query=arxiv.org&tags=story&numericFilters=points%3E10 (points/comments = strong social-velocity signal)
5. Reddit r/MachineLearning top posts (public JSON): https://www.reddit.com/r/MachineLearning/top.json?t=week&limit=50 (fetch with a User-Agent header; upvotes on [R] posts = researcher attention)
6. GitHub trending (code-adoption velocity, often the earliest signal): https://github.com/trending/python?since=daily and https://github.com/trending?since=daily — repos whose READMEs link a fresh arXiv paper
7. OpenReview (expert judgment — only source with real reviewer scores): search recent ICLR/NeurIPS/ICML submissions via `https://api.openreview.net/notes/search?term=<topic>` or venue listings; a paper with strong scores or a heated rebuttal thread is both quality-vetted and controversy-loaded. Seasonal — skip quickly if no active review cycle.
8. alphaXiv (scholarly discussion on arXiv papers): https://www.alphaxiv.org/ trending — lightweight extra velocity signal.
9. Web search for trending arXiv IDs mentioned on X in the last 48h if useful

Anti-signal: if a paper has already been covered by a major channel/newsletter (Yannic Kilcher, AI Explained, major Substacks), the viral window has largely closed — discount socialVelocity accordingly rather than chasing it.

## Scoring rubric

Score each serious candidate 1–10 on each criterion, then compute the weighted sum using the weights given in the run context:

- **tweetCompressibility** — can the core idea be stated in one plain sentence that makes a non-expert curious? ("We froze the weights and trained two tiny matrices" — LoRA)
- **deltaToClaimRatio** — smallest change relative to prior work, biggest claimed effect
- **contrarianAngle** — does it invert or challenge an assumption the field takes for granted?
- **socialVelocity** — HF upvotes, GitHub star growth, mentions in the last 48h
- **reproducibility** — public code AND dataset, and the core claim verifiable at toy scale. **Hard compute gate:** the available compute is (a) a CPU-only laptop (Intel i5, no discrete GPU) for local runs, and (b) ONE free-tier cloud GPU session (Kaggle T4/P100, ≤ ~3 GPU-hours) per day. If the core claim cannot plausibly show its direction of effect within that envelope, cap this score at 4 regardless of code availability. This gates the downstream reproduction stage — weigh it honestly.
- **visualPotential** — does it contain a result that makes one killer chart or diagram?

Hard constraints:
- Never select a paper listed in the "already covered" list in the run context.
- Prefer papers ≤ 7 days old; never older than 30 days.
- The paper must have a public PDF (arXiv preferred).

## Required outputs (write into the run directory)

1. `candidates.json` — array of 10–20 candidates:
   ```json
   [{ "arxiv_id": "2507.01234", "title": "...", "url": "...", "code_url": "... or null",
      "scores": { "tweetCompressibility": 8, "deltaToClaimRatio": 7, "contrarianAngle": 6,
                  "socialVelocity": 9, "reproducibility": 8, "visualPotential": 7 },
      "weighted_score": 7.6, "one_line_pitch": "..." }]
   ```
2. `selection.json` — the winner:
   ```json
   { "arxiv_id": "...", "title": "...", "slug": "kebab-case-short-slug", "score": 7.6,
     "justification": "3-5 sentences on why this one wins on the rubric",
     "scores": { ... }, "code_url": "... or null", "dataset": "name or null",
     "proceed": true, "skip_reason": null }
   ```

## Quality bar — you may skip the day

A skipped day beats a weak publication. If the best candidate's weighted score is below **6.0**, or the winner only wins by default (nothing genuinely interesting this week), set `"proceed": false` with an honest `skip_reason`, still filling in the least-bad candidate's details. The pipeline will stop and try again tomorrow. Do not lower your scoring standards to avoid skipping.

Use real data from your sweeps — do not fabricate upvote counts or star numbers; if a signal is unavailable, score it conservatively and say so in the justification.
