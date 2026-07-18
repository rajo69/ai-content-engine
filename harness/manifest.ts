import * as fs from "node:fs";
import * as path from "node:path";
import { CONFIG, type StageName } from "./config.js";
import type { HistoryEntry, RunManifest, StageRecord } from "./types.js";

const STAGE_NAMES = Object.keys(CONFIG.stages) as StageName[];

export function manifestPath(runDir: string): string {
  return path.join(runDir, "manifest.json");
}

export function loadOrCreateManifest(runDir: string, date: string): RunManifest {
  const p = manifestPath(runDir);
  if (fs.existsSync(p)) {
    const manifest = JSON.parse(fs.readFileSync(p, "utf-8")) as RunManifest;
    // runDir is machine-specific; a manifest created elsewhere (e.g. Windows)
    // must not steer writes to a path that doesn't exist here.
    if (manifest.runDir !== runDir) {
      manifest.runDir = runDir;
      saveManifest(manifest);
    }
    return manifest;
  }
  const stages = {} as Record<StageName, StageRecord>;
  for (const name of STAGE_NAMES) stages[name] = { status: "pending" };
  const manifest: RunManifest = {
    date,
    runDir,
    paperSlug: null,
    paperTitle: null,
    arxivId: null,
    stages,
    totalCostUsd: 0,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  saveManifest(manifest);
  return manifest;
}

export function saveManifest(manifest: RunManifest): void {
  manifest.updatedAt = new Date().toISOString();
  fs.writeFileSync(manifestPath(manifest.runDir), JSON.stringify(manifest, null, 2));
}

export function historyFile(repoRoot: string): string {
  return path.join(repoRoot, "state", "history.json");
}

export function loadHistory(repoRoot: string): HistoryEntry[] {
  const p = historyFile(repoRoot);
  if (!fs.existsSync(p)) return [];
  return JSON.parse(fs.readFileSync(p, "utf-8")) as HistoryEntry[];
}

export function appendHistory(repoRoot: string, entry: HistoryEntry): void {
  const entries = loadHistory(repoRoot);
  if (!entries.some((e) => e.arxivId === entry.arxivId)) {
    entries.push(entry);
    fs.writeFileSync(historyFile(repoRoot), JSON.stringify(entries, null, 2));
  }
}
