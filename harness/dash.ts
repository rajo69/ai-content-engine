/**
 * Local ops dashboard: `npm run dash` then open http://localhost:4242
 * Sync-on-read: every API request re-indexes the run files into SQLite,
 * so the UI is always live without touching the pipeline.
 * Also usable as a one-shot backfill: `npm run sync`.
 */
import * as http from "node:http";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { openDb, syncAll } from "./db.js";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PORT = 4242;

const db = openDb(REPO_ROOT);

if (process.argv.includes("--sync-only")) {
  syncAll(REPO_ROOT, db);
  console.log("Index rebuilt from run artifacts -> state/engine.db");
  process.exit(0);
}

function all(sql: string, ...params: any[]): any[] {
  return db.prepare(sql).all(...params) as any[];
}

function overview() {
  syncAll(REPO_ROOT, db);
  const runs = all(`SELECT * FROM runs ORDER BY date DESC`);
  const stages = all(`SELECT * FROM stages ORDER BY run_date DESC, ord ASC`);
  const papers = all(`SELECT * FROM papers ORDER BY date_covered DESC`);
  const lessons = all(`SELECT text FROM lessons ORDER BY idx`);
  return { runs, stages, papers, lessons, now: new Date().toISOString() };
}

function runDetail(date: string) {
  syncAll(REPO_ROOT, db);
  return {
    run: all(`SELECT * FROM runs WHERE date = ?`, date)[0] ?? null,
    stages: all(`SELECT * FROM stages WHERE run_date = ? ORDER BY ord`, date),
    candidates: all(`SELECT * FROM candidates WHERE run_date = ? ORDER BY rank LIMIT 10`, date),
    findings: all(`SELECT * FROM review_findings WHERE run_date = ?`, date),
    outputs: all(`SELECT * FROM outputs WHERE run_date = ? ORDER BY kind, rel_path`, date),
    repro: all(`SELECT * FROM reproductions WHERE run_date = ?`, date)[0] ?? null,
  };
}

const INDEX_HTML = path.join(REPO_ROOT, "dashboard", "index.html");

const FILE_TYPES: Record<string, string> = {
  ".md": "text/markdown; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".log": "text/plain; charset=utf-8",
  ".py": "text/plain; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".pdf": "application/pdf",
};

/** Serve a run artifact for the in-dashboard viewer. Confined to runs/<date>/. */
function serveRunFile(res: http.ServerResponse, date: string, rel: string): void {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    res.writeHead(400);
    return void res.end("bad date");
  }
  const runDir = path.resolve(REPO_ROOT, "runs", date);
  const full = path.resolve(runDir, rel);
  if (full !== runDir && !full.startsWith(runDir + path.sep)) {
    res.writeHead(403);
    return void res.end("path escapes run directory");
  }
  if (!fs.existsSync(full) || !fs.statSync(full).isFile()) {
    res.writeHead(404);
    return void res.end("not found");
  }
  const type = FILE_TYPES[path.extname(full).toLowerCase()] ?? "application/octet-stream";
  res.writeHead(200, { "content-type": type });
  res.end(fs.readFileSync(full));
}

const server = http.createServer((req, res) => {
  try {
    const url = new URL(req.url ?? "/", `http://localhost:${PORT}`);
    if (url.pathname === "/file") {
      serveRunFile(res, url.searchParams.get("date") ?? "", url.searchParams.get("path") ?? "");
    } else if (url.pathname === "/api/overview") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(overview()));
    } else if (url.pathname === "/api/run") {
      const date = url.searchParams.get("date") ?? "";
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(runDetail(date)));
    } else if (url.pathname === "/" || url.pathname === "/index.html") {
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      res.end(fs.readFileSync(INDEX_HTML, "utf-8"));
    } else {
      res.writeHead(404);
      res.end("not found");
    }
  } catch (err) {
    res.writeHead(500, { "content-type": "text/plain" });
    res.end(String(err));
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Dashboard: http://localhost:${PORT}`);
});
