# Stage 5 — Visualize: make the insight visible

Inputs: `analysis/core.md`, `reproduction/results.json` and `reproduction/REPORT.md`, `ingest/paper.md`.

Produce the visual assets for the paper, blog, and social posts. Write everything into `viz/` in the run directory.

## Assets to produce

1. **The killer chart** (`viz/headline.png`) — the single chart that carries the paper's main claim, using OUR reproduction numbers where available (fall back to the paper's reported numbers, clearly labeled as such). Generate with matplotlib via a Python script saved as `viz/make_charts.py` so it's regenerable.
2. **Method diagram** (`viz/method.svg` or a mermaid block in `viz/method.md`) — the architecture/mechanism, drawn to make the "minimal change" visually obvious (e.g., the delta highlighted against the prior-work baseline).
3. **Concept explainer** (`viz/explainer.png` or `.svg`) — the one figure you'd show a smart non-expert to convey the key insight.
4. Optional: any supporting charts (ablations, comparisons) that earn their place.

## Design rules

- Every chart must make exactly one point, stated in its caption.
- Label axes with units; no chart junk; readable at social-media sizes (large fonts).
- Honest scales — no truncated axes that exaggerate effects; mark reproduction-vs-paper data provenance on the figure itself.

## Required output: `viz/index.md`

A catalog: for each asset, the filename, a one-sentence caption, the insight it demonstrates, and which downstream outputs should embed it (paper / blog / X thread / LinkedIn).
