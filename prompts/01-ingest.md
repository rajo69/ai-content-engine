# Stage 1 — Ingest: gather all raw material

The selected paper is in `selection.json` in the run directory. Gather everything downstream stages need, into `ingest/` inside the run directory.

## Tasks

1. **PDF** — download the paper PDF to `ingest/paper.pdf` (arXiv: `https://arxiv.org/pdf/<id>`). Use `curl -L` via Bash.
2. **Full text** — extract the paper's text so later stages can read it without a PDF parser. Read the arXiv abs page and the HTML version (`https://arxiv.org/abs/<id>`, try `https://arxiv.org/html/<id>` or ar5iv `https://ar5iv.labs.arxiv.org/html/<id>`), and save a clean markdown rendition of the full paper (all sections, key equations transcribed) to `ingest/paper.md`.
3. **References** — fetch the citation graph from Semantic Scholar:
   `https://api.semanticscholar.org/graph/v1/paper/arXiv:<id>/references?fields=title,abstract,year,citationCount,isInfluential,externalIds&limit=100`
   If the `S2_API_KEY` environment variable is set, send it as the `x-api-key` header (much higher rate limits).
   Save the raw response, reshaped, to `ingest/references.json` (array of `{title, year, citationCount, isInfluential, arxivId, abstract}`). If the API rate-limits, retry with backoff; if it stays unavailable, extract references from the paper text itself and note the degradation.
4. **Code** — if `selection.json` has a `code_url`, shallow-clone it: `git clone --depth 1 <url> ingest/repo`. Then write `ingest/repo-notes.md`: repo structure, entry points, dependencies, and an assessment of what it would take to run the core experiment.
5. **Dataset** — identify the dataset(s) used; record name, size, license, and download URL in the metadata. Do NOT download large datasets now.
6. **Metadata** — write `ingest/metadata.json`:
   ```json
   { "arxiv_id": "...", "title": "...", "authors": [...], "published": "...", "venue": "...",
     "abstract": "...", "code_url": "...", "code_cloned": true, "datasets": [{ "name": "...", "size": "...", "url": "..." }],
     "pdf_saved": true, "fulltext_saved": true, "reference_count": 42 }
   ```

Required outputs: `ingest/metadata.json`, `ingest/references.json`. Everything else is strongly desired; record any failures honestly in metadata.
