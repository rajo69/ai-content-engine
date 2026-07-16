# Stage 2 — Literature review: walk the citation graph

Inputs (in the run directory): `ingest/paper.md`, `ingest/references.json`, `ingest/metadata.json`.

You are writing the literature review an actual researcher would write before publishing about this paper. Work like one: read the paper's related-work section, then go to the sources.

## Tasks

1. **Select the load-bearing references.** From `ingest/references.json`, pick the 10–15 references that actually matter — prioritize `isInfluential: true`, high citation counts, and anything the paper's method directly builds on or argues against. Ignore boilerplate citations.
2. **Read them.** For each selected reference, get its abstract (already in references.json) and, for the 4–6 most central ones, fetch more detail (arXiv abs page or Semantic Scholar `https://api.semanticscholar.org/graph/v1/paper/arXiv:<id>?fields=title,abstract,tldr,year,citationCount` — send `S2_API_KEY` env var as `x-api-key` header if set).
3. **Write `litreview/bibliography.md`** — annotated bibliography. For each reference:
   - Full citation (authors, title, venue, year)
   - What it contributed to the field (2-3 sentences)
   - **How it feeds the target paper specifically** — is it the baseline being beaten, the method being modified, the assumption being challenged, or the tool being reused?
4. **Novelty & concurrent-work check.** References only look backward — also check sideways and forward. Search (WebSearch + Semantic Scholar) for prior or concurrent papers proposing substantially the same core idea, and for any early responses/extensions/contradictions of the target paper. Add a `## Novelty check` section to `litreview/lineage.md`: either "no substantially-equivalent prior work found (searched: <queries>)" or an honest account of what came first and how the target paper differs. If the idea is a rediscovery, that is itself a key finding — flag it prominently for downstream stages, never bury it.
5. **Write `litreview/lineage.md`** — a narrative (600–1200 words) tracing the research lineage: what problem the field was chasing, the succession of key ideas that led here, where each fell short, and precisely what gap the target paper steps into. End with a one-paragraph statement of the paper's position in the lineage. This narrative becomes the Related Work section of our paper and the historical arc of the blog — write it well.

Ground every claim in the sources you actually read. Where you infer, mark it as inference.
