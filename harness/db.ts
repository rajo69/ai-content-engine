/**
 * SQLite index over the run artifacts (files stay the source of truth).
 * The DB is disposable: `npm run sync` rebuilds it entirely from runs/ + state/.
 * Uses node:sqlite (built-in, Node >= 22.5) — no native dependencies.
 */
import { DatabaseSync } from "node:sqlite";
import * as fs from "node:fs";
import * as path from "node:path";

export const DB_PATH = (repoRoot: string) => path.join(repoRoot, "state", "engine.db");

const SCHEMA = `
CREATE TABLE IF NOT EXISTS runs (
  date TEXT PRIMARY KEY, paper_slug TEXT, paper_title TEXT, arxiv_id TEXT,
  proceed INTEGER, skip_reason TEXT, verdict TEXT, total_cost_usd REAL,
  created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS stages (
  run_date TEXT, name TEXT, ord INTEGER, status TEXT, cost_usd REAL,
  started_at TEXT, finished_at TEXT, error TEXT, note TEXT,
  PRIMARY KEY (run_date, name)
);
CREATE TABLE IF NOT EXISTS candidates (
  run_date TEXT, arxiv_id TEXT, rank INTEGER, title TEXT, url TEXT, code_url TEXT,
  one_line_pitch TEXT, simple_summary TEXT, weighted_score REAL, scores_json TEXT,
  selected INTEGER DEFAULT 0,
  PRIMARY KEY (run_date, arxiv_id)
);
CREATE TABLE IF NOT EXISTS papers (
  arxiv_id TEXT PRIMARY KEY, title TEXT, slug TEXT, date_covered TEXT, score REAL
);
CREATE TABLE IF NOT EXISTS outputs (
  run_date TEXT, rel_path TEXT, kind TEXT, bytes INTEGER, modified_at TEXT,
  PRIMARY KEY (run_date, rel_path)
);
CREATE TABLE IF NOT EXISTS review_findings (
  run_date TEXT, pass TEXT, idx INTEGER, file TEXT, issue TEXT, severity TEXT, resolution TEXT,
  PRIMARY KEY (run_date, pass, idx)
);
CREATE TABLE IF NOT EXISTS reproductions (
  run_date TEXT PRIMARY KEY, predictions_json TEXT, results_json TEXT, env_json TEXT
);
CREATE TABLE IF NOT EXISTS lessons (
  idx INTEGER, text TEXT, PRIMARY KEY (idx)
);
`;

export function openDb(repoRoot: string): DatabaseSync {
  const db = new DatabaseSync(DB_PATH(repoRoot));
  db.exec(SCHEMA);
  return db;
}

function readJson(p: string): any | null {
  try {
    return JSON.parse(fs.readFileSync(p, "utf-8"));
  } catch {
    return null;
  }
}

const OUTPUT_KINDS: Array<[RegExp, string]> = [
  [/^outputs[\\/]paper[\\/]/, "paper"],
  [/^outputs[\\/]blog/, "blog"],
  [/^outputs[\\/]social[\\/]x-/, "x"],
  [/^outputs[\\/]social[\\/]linkedin/, "linkedin"],
  [/^outputs[\\/]social[\\/]reddit/, "reddit"],
  [/^viz[\\/]/, "viz"],
];

function syncOneRun(db: DatabaseSync, runsRoot: string, date: string): void {
  const runDir = path.join(runsRoot, date);
  const manifest = readJson(path.join(runDir, "manifest.json"));
  if (!manifest) return;
  const selection = readJson(path.join(runDir, "selection.json"));
  const verdicts = readJson(path.join(runDir, "review", "verdicts.json"));

  db.prepare(
    `INSERT INTO runs (date, paper_slug, paper_title, arxiv_id, proceed, skip_reason, verdict, total_cost_usd, created_at, updated_at)
     VALUES (?,?,?,?,?,?,?,?,?,?)
     ON CONFLICT(date) DO UPDATE SET paper_slug=excluded.paper_slug, paper_title=excluded.paper_title,
       arxiv_id=excluded.arxiv_id, proceed=excluded.proceed, skip_reason=excluded.skip_reason,
       verdict=excluded.verdict, total_cost_usd=excluded.total_cost_usd, updated_at=excluded.updated_at`,
  ).run(
    date,
    manifest.paperSlug ?? null,
    manifest.paperTitle ?? null,
    manifest.arxivId ?? null,
    selection?.proceed === false ? 0 : 1,
    selection?.skip_reason ?? null,
    verdicts?.overall ?? null,
    manifest.totalCostUsd ?? 0,
    manifest.createdAt ?? null,
    manifest.updatedAt ?? null,
  );

  const stageNames = Object.keys(manifest.stages ?? {});
  stageNames.forEach((name, i) => {
    const s = manifest.stages[name] ?? {};
    db.prepare(
      `INSERT INTO stages (run_date, name, ord, status, cost_usd, started_at, finished_at, error, note)
       VALUES (?,?,?,?,?,?,?,?,?)
       ON CONFLICT(run_date, name) DO UPDATE SET status=excluded.status, cost_usd=excluded.cost_usd,
         started_at=excluded.started_at, finished_at=excluded.finished_at, error=excluded.error, note=excluded.note`,
    ).run(date, name, i, s.status ?? "pending", s.costUsd ?? null, s.startedAt ?? null, s.finishedAt ?? null, s.error ?? null, s.note ?? null);
  });

  const candidates = readJson(path.join(runDir, "candidates.json"));
  if (Array.isArray(candidates)) {
    const ranked = [...candidates].sort((a, b) => (b.weighted_score ?? 0) - (a.weighted_score ?? 0));
    ranked.forEach((c, i) => {
      if (!c?.arxiv_id) return;
      db.prepare(
        `INSERT INTO candidates (run_date, arxiv_id, rank, title, url, code_url, one_line_pitch, simple_summary, weighted_score, scores_json, selected)
         VALUES (?,?,?,?,?,?,?,?,?,?,?)
         ON CONFLICT(run_date, arxiv_id) DO UPDATE SET rank=excluded.rank, title=excluded.title,
           weighted_score=excluded.weighted_score, scores_json=excluded.scores_json, selected=excluded.selected,
           one_line_pitch=excluded.one_line_pitch, simple_summary=excluded.simple_summary`,
      ).run(
        date,
        c.arxiv_id,
        i + 1,
        c.title ?? null,
        c.url ?? null,
        c.code_url ?? null,
        c.one_line_pitch ?? null,
        c.simple_summary ?? null,
        c.weighted_score ?? null,
        JSON.stringify(c.scores ?? {}),
        selection?.arxiv_id === c.arxiv_id ? 1 : 0,
      );
    });
  }

  if (verdicts?.passes) {
    for (const [pass, body] of Object.entries<any>(verdicts.passes)) {
      (body?.findings ?? []).forEach((f: any, i: number) => {
        db.prepare(
          `INSERT INTO review_findings (run_date, pass, idx, file, issue, severity, resolution) VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(run_date, pass, idx) DO UPDATE SET file=excluded.file, issue=excluded.issue,
             severity=excluded.severity, resolution=excluded.resolution`,
        ).run(date, pass, i, f.file ?? null, f.issue ?? null, f.severity ?? null, f.resolution ?? null);
      });
    }
  }

  const predictions = readJson(path.join(runDir, "reproduction", "predictions.json"));
  const results = readJson(path.join(runDir, "reproduction", "results.json"));
  const env = readJson(path.join(runDir, "reproduction", "env-info.json"));
  if (predictions || results || env) {
    db.prepare(
      `INSERT INTO reproductions (run_date, predictions_json, results_json, env_json) VALUES (?,?,?,?)
       ON CONFLICT(run_date) DO UPDATE SET predictions_json=excluded.predictions_json,
         results_json=excluded.results_json, env_json=excluded.env_json`,
    ).run(date, JSON.stringify(predictions), JSON.stringify(results), JSON.stringify(env));
  }

  // Outputs + viz: walk shallow, classify by path.
  for (const base of ["outputs", "viz"]) {
    const baseDir = path.join(runDir, base);
    if (!fs.existsSync(baseDir)) continue;
    const walk = (dir: string): string[] =>
      fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
        const full = path.join(dir, e.name);
        return e.isDirectory() ? walk(full) : [full];
      });
    for (const file of walk(baseDir)) {
      const rel = path.relative(runDir, file);
      const kind = OUTPUT_KINDS.find(([re]) => re.test(rel))?.[1] ?? "other";
      const st = fs.statSync(file);
      db.prepare(
        `INSERT INTO outputs (run_date, rel_path, kind, bytes, modified_at) VALUES (?,?,?,?,?)
         ON CONFLICT(run_date, rel_path) DO UPDATE SET bytes=excluded.bytes, modified_at=excluded.modified_at, kind=excluded.kind`,
      ).run(date, rel, kind, st.size, st.mtime.toISOString());
    }
  }
}

/** Rebuild/refresh the whole index from disk. Cheap at daily-run scale. */
export function syncAll(repoRoot: string, db: DatabaseSync): void {
  const runsRoot = path.join(repoRoot, "runs");
  if (fs.existsSync(runsRoot)) {
    for (const d of fs.readdirSync(runsRoot).filter((d) => /^\d{4}-\d{2}-\d{2}$/.test(d)).sort()) {
      try {
        syncOneRun(db, runsRoot, d);
      } catch (err) {
        console.error(`sync: skipping ${d}: ${err}`);
      }
    }
  }
  const history = readJson(path.join(repoRoot, "state", "history.json"));
  if (Array.isArray(history)) {
    for (const h of history) {
      if (!h?.arxivId) continue;
      db.prepare(
        `INSERT INTO papers (arxiv_id, title, slug, date_covered, score) VALUES (?,?,?,?,?)
         ON CONFLICT(arxiv_id) DO UPDATE SET title=excluded.title, slug=excluded.slug,
           date_covered=excluded.date_covered, score=excluded.score`,
      ).run(h.arxivId, h.title ?? null, h.slug ?? null, h.date ?? null, h.score ?? null);
    }
  }
  const lessonsPath = path.join(repoRoot, "state", "lessons.md");
  if (fs.existsSync(lessonsPath)) {
    const bullets = fs
      .readFileSync(lessonsPath, "utf-8")
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.startsWith("-") || l.startsWith("*"))
      .map((l) => l.replace(/^[-*]\s*/, ""));
    db.exec("DELETE FROM lessons");
    bullets.forEach((text, i) => db.prepare("INSERT INTO lessons (idx, text) VALUES (?,?)").run(i, text));
  }
}
