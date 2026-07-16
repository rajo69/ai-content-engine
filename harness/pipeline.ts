/**
 * Daily research pipeline orchestrator.
 *
 * Usage:
 *   npm run daily                       # run/resume today's pipeline
 *   npm run daily -- --date 2026-07-16  # run/resume a specific date
 *   npm run daily -- --from analyze     # force re-run from a stage onward
 *   npm run daily -- --only reproduce   # run a single stage
 *
 * Each stage is an autonomous agent session (Claude Agent SDK). Progress is
 * checkpointed in runs/<date>/manifest.json; re-running resumes at the first
 * incomplete stage. A stage counts as completed only when its required output
 * files exist on disk.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { CONFIG, type StageName } from "./config.js";
import { STAGES, type StageDef } from "./stages.js";
import { runStageAgent } from "./agent.js";
import { appendHistory, loadHistory, loadOrCreateManifest, saveManifest } from "./manifest.js";
import type { RunManifest, Selection } from "./types.js";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv: string[]) {
  const args: { date?: string; from?: StageName; only?: StageName } = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--date") args.date = argv[++i];
    if (argv[i] === "--from") args.from = argv[++i] as StageName;
    if (argv[i] === "--only") args.only = argv[++i] as StageName;
  }
  return args;
}

function today(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function missingOutputs(runDir: string, stage: StageDef): string[] {
  return stage.requiredOutputs.filter((rel) => !fs.existsSync(path.join(runDir, rel)));
}

function buildStagePrompt(stage: StageDef, manifest: RunManifest): string {
  const promptBody = fs.readFileSync(path.join(REPO_ROOT, stage.promptFile), "utf-8");
  const history = loadHistory(REPO_ROOT);
  const lessonsPath = path.join(REPO_ROOT, "state", "lessons.md");
  const lessons = fs.existsSync(lessonsPath) ? fs.readFileSync(lessonsPath, "utf-8").trim() : "";
  const context = [
    `# Run context`,
    `- Date: ${manifest.date}`,
    `- Repository root: ${REPO_ROOT}`,
    `- Run directory (write ALL outputs here): ${manifest.runDir}`,
    `- Paper: ${manifest.paperTitle ?? "not selected yet"} (${manifest.arxivId ?? "n/a"})`,
    `- Required output files for this stage (paths relative to the run directory): ${stage.requiredOutputs.join(", ")}`,
    `- Papers already covered on previous days (never select these again): ${
      history.length ? history.map((h) => h.arxivId).join(", ") : "none yet"
    }`,
    `- Scoring rubric weights: ${JSON.stringify(CONFIG.scoutWeights)}`,
    `- Reproduction time box: ${CONFIG.reproductionTimeBoxMinutes} minutes`,
    ...(lessons
      ? [``, `# Lessons from previous runs (apply them — they were earned the hard way)`, lessons]
      : []),
    ``,
  ].join("\n");
  return context + "\n" + promptBody;
}

/** After the scout completes, lift its selection into the manifest. */
function absorbSelection(manifest: RunManifest): void {
  const selPath = path.join(manifest.runDir, "selection.json");
  if (!fs.existsSync(selPath)) return;
  const sel = JSON.parse(fs.readFileSync(selPath, "utf-8")) as Selection;
  manifest.paperSlug = sel.slug ?? null;
  manifest.paperTitle = sel.title ?? null;
  manifest.arxivId = sel.arxiv_id ?? null;
}

async function runStage(stage: StageDef, manifest: RunManifest): Promise<boolean> {
  const cfg = CONFIG.stages[stage.name];
  const record = manifest.stages[stage.name];
  record.status = "running";
  record.startedAt = new Date().toISOString();
  saveManifest(manifest);
  console.log(`\n=== Stage: ${stage.name} (model=${cfg.model}, effort=${cfg.effort}, cap=$${cfg.maxBudgetUsd}) ===`);

  try {
    const result = await runStageAgent({
      prompt: buildStagePrompt(stage, manifest),
      cwd: REPO_ROOT,
      model: cfg.model,
      effort: cfg.effort,
      maxTurns: cfg.maxTurns,
      maxBudgetUsd: cfg.maxBudgetUsd,
      allowedTools: [...stage.allowedTools],
      label: stage.name,
    });
    record.costUsd = result.costUsd;
    manifest.totalCostUsd += result.costUsd;

    const missing = missingOutputs(manifest.runDir, stage);
    if (missing.length > 0) {
      record.status = "failed";
      record.missingOutputs = missing;
      record.error = `Stage finished but required outputs are missing: ${missing.join(", ")}`;
      saveManifest(manifest);
      console.error(`  [${stage.name}] FAILED — missing outputs: ${missing.join(", ")}`);
      return false;
    }

    record.status = "completed";
    record.finishedAt = new Date().toISOString();
    delete record.missingOutputs;
    delete record.error;
    if (stage.name === "scout") absorbSelection(manifest);
    saveManifest(manifest);
    console.log(`  [${stage.name}] completed. Run total so far: $${manifest.totalCostUsd.toFixed(2)}`);
    return true;
  } catch (err) {
    record.status = "failed";
    record.error = err instanceof Error ? err.message : String(err);
    saveManifest(manifest);
    console.error(`  [${stage.name}] FAILED — ${record.error}`);
    return false;
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const date = args.date ?? today();
  const runDir = path.join(REPO_ROOT, "runs", date);
  fs.mkdirSync(runDir, { recursive: true });
  const manifest = loadOrCreateManifest(runDir, date);

  const stageIndex = (name: StageName) => STAGES.findIndex((s) => s.name === name);

  let toRun: StageDef[];
  if (args.only) {
    const i = stageIndex(args.only);
    if (i < 0) throw new Error(`Unknown stage: ${args.only}`);
    toRun = [STAGES[i]];
  } else {
    const fromIdx = args.from ? stageIndex(args.from) : -1;
    if (args.from && fromIdx < 0) throw new Error(`Unknown stage: ${args.from}`);
    toRun = STAGES.filter(
      (s, i) => (fromIdx >= 0 && i >= fromIdx) || (fromIdx < 0 && manifest.stages[s.name].status !== "completed"),
    );
  }

  if (toRun.length === 0) {
    console.log(`Run ${date} is already complete. Nothing to do.`);
    return;
  }

  // Quality-bar gate on resume: if a completed scout declared proceed=false,
  // a later "npm run daily" must not sneak past it into the full pipeline.
  // Explicit --from/--only is treated as a deliberate human override.
  if (!args.from && !args.only) {
    const selPath = path.join(runDir, "selection.json");
    if (manifest.stages.scout.status === "completed" && fs.existsSync(selPath)) {
      const sel = JSON.parse(fs.readFileSync(selPath, "utf-8")) as Selection;
      if (sel.proceed === false) {
        console.log(`Scout previously skipped ${date} (reason: ${sel.skip_reason ?? "n/a"}). Not resuming.`);
        console.log(`Override with --from ingest if you disagree with the scout.`);
        return;
      }
    }
  }

  console.log(`Pipeline run ${date} — stages to execute: ${toRun.map((s) => s.name).join(" → ")}`);

  for (const stage of toRun) {
    if (manifest.totalCostUsd >= CONFIG.maxRunBudgetUsd) {
      console.error(`Run budget cap ($${CONFIG.maxRunBudgetUsd}) reached — aborting before ${stage.name}.`);
      process.exitCode = 1;
      return;
    }
    const ok = await runStage(stage, manifest);
    if (!ok) {
      console.error(`Pipeline halted at stage "${stage.name}". Re-run "npm run daily" to resume after fixing.`);
      process.exitCode = 1;
      return;
    }
    // Quality bar: the scout may decide no paper is worth a full run today.
    if (stage.name === "scout") {
      const sel = JSON.parse(
        fs.readFileSync(path.join(manifest.runDir, "selection.json"), "utf-8"),
      ) as Selection;
      if (sel.proceed === false) {
        console.log(`Scout found no paper meeting the quality bar today. Reason: ${sel.skip_reason ?? "n/a"}`);
        console.log(`Skipping the rest of the pipeline — a skipped day beats a weak publication.`);
        return;
      }
    }
    // Record the paper in history once the content suite exists, so a failed
    // early run doesn't burn the paper.
    if (stage.name === "write" && manifest.arxivId) {
      const sel = JSON.parse(
        fs.readFileSync(path.join(manifest.runDir, "selection.json"), "utf-8"),
      ) as Selection;
      appendHistory(REPO_ROOT, {
        date: manifest.date,
        arxivId: sel.arxiv_id,
        title: sel.title,
        slug: sel.slug,
        score: sel.score,
      });
    }
  }

  console.log(`\nRun ${date} complete. Total cost: $${manifest.totalCostUsd.toFixed(2)}`);
  console.log(`Outputs: ${manifest.runDir}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
