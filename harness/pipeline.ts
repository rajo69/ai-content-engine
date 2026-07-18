/**
 * Daily research pipeline orchestrator.
 *
 * Usage:
 *   npm run daily                       # run/resume today's pipeline
 *   npm run daily -- --date 2026-07-16  # run/resume a specific date
 *   npm run daily -- --from analyze     # force re-run from a stage onward
 *   npm run daily -- --only reproduce   # run a single stage
 *   npm run daily -- --dry-run          # show what would run, touch no agent
 *
 * Exit codes: 0 done/skipped, 1 stage failure or budget abort, 3 usage-limit
 * hit (see runs/<date>/_usage-limit.json for the parsed reset time).
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
import { runStageAgent, StageAgentError } from "./agent.js";
import { appendHistory, loadHistory, loadOrCreateManifest, saveManifest } from "./manifest.js";
import type { RunManifest, Selection, StageRecord, WaitingSpec } from "./types.js";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv: string[]) {
  const args: { date?: string; from?: StageName; only?: StageName; dryRun?: boolean } = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--date") args.date = argv[++i];
    if (argv[i] === "--from") args.from = argv[++i] as StageName;
    if (argv[i] === "--only") args.only = argv[++i] as StageName;
    if (argv[i] === "--dry-run") args.dryRun = true;
  }
  return args;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

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

/**
 * Async experiment handoff: consume runs/<date>/_waiting.json if present.
 * The file is renamed (audit trail) so it can never be double-processed.
 */
function consumeWaitingSpec(runDir: string, label: string): WaitingSpec | null {
  const p = path.join(runDir, "_waiting.json");
  if (!fs.existsSync(p)) return null;
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const consumed = path.join(runDir, `_waiting.consumed-${stamp}.json`);
  try {
    const spec = JSON.parse(fs.readFileSync(p, "utf-8")) as WaitingSpec;
    fs.renameSync(p, consumed);
    if (typeof spec.file !== "string" || !spec.file) throw new Error(`no "file" field`);
    return spec;
  } catch (err) {
    console.error(`  [${label}] ignoring malformed _waiting.json: ${err}`);
    if (fs.existsSync(p)) fs.renameSync(p, consumed);
    return null;
  }
}

/** fs-poll (zero tokens) until the awaited file exists or the timeout expires. */
async function pollForFile(target: string, timeoutMinutes: number, label: string): Promise<boolean> {
  const deadline = Date.now() + timeoutMinutes * 60_000;
  console.log(`  [${label}] waiting for ${target} (up to ${timeoutMinutes}min, polling every ${CONFIG.waitingPollSeconds}s)...`);
  let lastLog = Date.now();
  while (Date.now() < deadline) {
    if (fs.existsSync(target)) {
      console.log(`  [${label}] awaited file arrived.`);
      return true;
    }
    if (Date.now() - lastLog > 5 * 60_000) {
      console.log(`  [${label}] still waiting (${Math.round((deadline - Date.now()) / 60_000)}min left)...`);
      lastLog = Date.now();
    }
    await sleep(CONFIG.waitingPollSeconds * 1000);
  }
  console.warn(`  [${label}] wait timed out — resuming the agent anyway so it can degrade gracefully.`);
  return false;
}

function resumeContext(spec: WaitingSpec, arrived: boolean): string {
  return [
    ``,
    `# Resume context (async experiment handoff)`,
    `You previously ended your session after writing _waiting.json, handing the wait for a`,
    `long-running experiment to the harness. This is the SAME stage, resumed:`,
    `- Awaited file: ${spec.file} — ${arrived ? "it now EXISTS; interpret its contents and finish the stage" : `it did NOT appear within the timeout; diagnose (check logs/process state), degrade gracefully, and still produce every required output`}`,
    `- Your resume hint from last time: ${spec.resume_hint ?? "(none provided)"}`,
    `Build on the files already on disk — do not redo completed work. Do not write another`,
    `_waiting.json unless a genuinely new long-running experiment requires it.`,
    ``,
  ].join("\n");
}

function recordFailure(record: StageRecord, error: string): void {
  record.status = "failed";
  record.error = error;
  record.finishedAt = new Date().toISOString();
  (record.failures ??= []).push({ at: record.finishedAt, error });
}

async function runStage(stage: StageDef, manifest: RunManifest): Promise<boolean> {
  const cfg = CONFIG.stages[stage.name];
  const record = manifest.stages[stage.name];
  record.status = "running";
  record.startedAt = new Date().toISOString();
  saveManifest(manifest);
  console.log(
    `\n=== Stage: ${stage.name} (model=${cfg.model}, effort=${cfg.effort}, cap=$${cfg.maxBudgetUsd}, wall=${cfg.maxWallMinutes}min) ===`,
  );

  // Invocation loop: a stage may span several agent sessions when it hands a
  // long-running experiment to the harness via _waiting.json.
  let extraContext = "";
  while (true) {
    // A spec present here is either freshly written by the previous invocation
    // or left over from a killed launch — both mean "wait, then resume".
    const spec = consumeWaitingSpec(manifest.runDir, stage.name);
    if (spec) {
      if ((record.resumes ?? 0) >= CONFIG.maxWaitingResumes) {
        recordFailure(record, `exceeded ${CONFIG.maxWaitingResumes} async-handoff resume cycles`);
        saveManifest(manifest);
        console.error(`  [${stage.name}] FAILED — ${record.error}`);
        return false;
      }
      record.resumes = (record.resumes ?? 0) + 1;
      saveManifest(manifest);
      const timeoutMinutes = spec.timeout_minutes ?? CONFIG.reproductionTimeBoxMinutes + 15;
      const arrived = await pollForFile(path.join(manifest.runDir, spec.file), timeoutMinutes, stage.name);
      extraContext = resumeContext(spec, arrived);
    }

    try {
      const result = await runStageAgent({
        prompt: buildStagePrompt(stage, manifest) + extraContext,
        cwd: REPO_ROOT,
        model: cfg.model,
        effort: cfg.effort,
        maxTurns: cfg.maxTurns,
        maxBudgetUsd: cfg.maxBudgetUsd,
        maxWallMinutes: cfg.maxWallMinutes,
        allowedTools: [...stage.allowedTools],
        label: stage.name,
      });
      record.costUsd = (record.costUsd ?? 0) + result.costUsd;
      manifest.totalCostUsd += result.costUsd;
      saveManifest(manifest);
    } catch (err) {
      const cost = err instanceof StageAgentError ? err.costUsd : 0;
      record.costUsd = (record.costUsd ?? 0) + cost;
      manifest.totalCostUsd += cost;
      recordFailure(record, err instanceof Error ? err.message : String(err));
      saveManifest(manifest);
      console.error(`  [${stage.name}] FAILED — ${record.error}`);
      return false;
    }

    // The agent handed off another wait — loop to process it.
    if (fs.existsSync(path.join(manifest.runDir, "_waiting.json"))) continue;

    const missing = missingOutputs(manifest.runDir, stage);
    if (missing.length > 0) {
      recordFailure(record, `Stage finished but required outputs are missing: ${missing.join(", ")}`);
      record.missingOutputs = missing;
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
  }
}

/**
 * Usage/session-limit awareness: recognize the error and best-effort parse the
 * reset time ("resets 4pm", "resets at 16:30", or an ISO timestamp).
 */
function detectUsageLimit(message: string): { hit: boolean; resetAt: Date | null } {
  if (!/usage limit|session limit|rate.?limit|limit (?:will )?reset|resets\b/i.test(message)) {
    return { hit: false, resetAt: null };
  }
  const iso = message.match(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?/);
  if (iso) {
    const d = new Date(iso[0]);
    if (!Number.isNaN(d.getTime())) return { hit: true, resetAt: d };
  }
  const clock = message.match(/resets?\s*(?:at\s*)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?/i);
  if (clock && (clock[3] || clock[2])) {
    let h = parseInt(clock[1], 10);
    if (clock[3]) h = (h % 12) + (clock[3].toLowerCase() === "pm" ? 12 : 0);
    const d = new Date();
    d.setHours(h, clock[2] ? parseInt(clock[2], 10) : 0, 0, 0);
    if (d.getTime() <= Date.now()) d.setDate(d.getDate() + 1);
    return { hit: true, resetAt: d };
  }
  return { hit: true, resetAt: null };
}

function writeUsageLimitMarker(runDir: string, stage: string, message: string, resetAt: Date | null): void {
  fs.writeFileSync(
    path.join(runDir, "_usage-limit.json"),
    JSON.stringify(
      { stage, message, parsedResetAt: resetAt?.toISOString() ?? null, detectedAt: new Date().toISOString() },
      null,
      2,
    ),
  );
}

const RUN_STATS_HEADER = "## Run stats (auto-generated)";

/**
 * Lessons with numbers: keep a compact, auto-updated stats bullet per run at
 * the bottom of state/lessons.md so cross-run learning carries data, not just
 * impressions. Most recent 8 runs kept; the distilled bullets above the marker
 * stay agent-managed.
 */
function upsertRunStats(manifest: RunManifest): void {
  const lessonsPath = path.join(REPO_ROOT, "state", "lessons.md");
  let body = fs.existsSync(lessonsPath)
    ? fs.readFileSync(lessonsPath, "utf-8")
    : "# Cross-run lessons (injected into every future run)\n";
  const idx = body.indexOf(RUN_STATS_HEADER);
  const existing = new Map<string, string>();
  if (idx >= 0) {
    for (const l of body.slice(idx).split("\n")) {
      const m = l.trim().match(/^- (\d{4}-\d{2}-\d{2})/);
      if (m) existing.set(m[1], l.trim());
    }
    body = body.slice(0, idx);
  }

  const costliest = (Object.entries(manifest.stages) as [StageName, StageRecord][])
    .filter(([, r]) => (r.costUsd ?? 0) > 0)
    .sort((a, b) => (b[1].costUsd ?? 0) - (a[1].costUsd ?? 0))
    .slice(0, 2)
    .map(([n, r]) => `${n} $${(r.costUsd ?? 0).toFixed(2)}/$${CONFIG.stages[n].maxBudgetUsd}`);
  const resumes = Object.values(manifest.stages).reduce((a, r) => a + (r.resumes ?? 0), 0);
  const failed = (Object.entries(manifest.stages) as [StageName, StageRecord][])
    .filter(([, r]) => (r.failures?.length ?? 0) > 0)
    .map(([n, r]) => `${n}×${r.failures!.length}`);
  existing.set(
    manifest.date,
    `- ${manifest.date} — total $${manifest.totalCostUsd.toFixed(2)}/$${CONFIG.maxRunBudgetUsd}` +
      (costliest.length ? ` · costliest: ${costliest.join(", ")}` : "") +
      ` · resumes: ${resumes}` +
      ` · failed legs: ${failed.length ? failed.join(", ") : "none"}`,
  );

  const bullets = [...existing.keys()].sort().slice(-8).map((d) => existing.get(d)!);
  fs.writeFileSync(
    lessonsPath,
    body.trimEnd() +
      "\n\n" +
      RUN_STATS_HEADER +
      "\n\nMachine-written by the pipeline after each run — do not edit by hand; most recent 8 runs kept.\n\n" +
      bullets.join("\n") +
      "\n",
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const date = args.date ?? today();
  const runDir = path.join(REPO_ROOT, "runs", date);
  fs.mkdirSync(runDir, { recursive: true });
  const manifest = loadOrCreateManifest(runDir, date);

  // Stale-run hygiene: the pipeline is single-process, so any stage still
  // marked "running" at startup was orphaned by a killed launch.
  for (const [name, rec] of Object.entries(manifest.stages)) {
    if (rec.status === "running") {
      console.warn(`Stage "${name}" was left "running" by a killed launch — marking it failed (stale).`);
      recordFailure(rec, "stale: previous launch was killed while this stage was running");
      saveManifest(manifest);
    }
  }

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

  // Midnight footgun guard: a bare "npm run daily" just after midnight keys to
  // the NEW date and silently abandons yesterday's half-finished run. Warn.
  if (!args.date && !args.only && !args.from) {
    const fresh = Object.values(manifest.stages).every((s) => s.status === "pending");
    const runsRoot = path.join(REPO_ROOT, "runs");
    if (fresh && fs.existsSync(runsRoot)) {
      const prior = fs
        .readdirSync(runsRoot)
        .filter((d) => /^\d{4}-\d{2}-\d{2}$/.test(d) && d < date)
        .sort()
        .at(-1);
      if (prior) {
        const pmPath = path.join(runsRoot, prior, "manifest.json");
        const selPath = path.join(runsRoot, prior, "selection.json");
        if (fs.existsSync(pmPath)) {
          const prev = JSON.parse(fs.readFileSync(pmPath, "utf-8")) as RunManifest;
          const incomplete = Object.values(prev.stages).some((s) => s.status !== "completed");
          const skippedDay =
            fs.existsSync(selPath) &&
            (JSON.parse(fs.readFileSync(selPath, "utf-8")) as Selection).proceed === false;
          if (incomplete && !skippedDay) {
            console.warn(`\n!! Previous run ${prior} is INCOMPLETE. This command starts a NEW run for ${date}.`);
            console.warn(`!! To finish the previous run instead: npm run daily -- --date ${prior}\n`);
          }
        }
      }
    }
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

  if (args.dryRun) {
    console.log(`--dry-run: stopping before any agent session. Manifest hygiene (stale-marking, runDir fix) was applied.`);
    return;
  }

  for (const stage of toRun) {
    if (manifest.totalCostUsd >= CONFIG.maxRunBudgetUsd) {
      console.error(`Run budget cap ($${CONFIG.maxRunBudgetUsd}) reached — aborting before ${stage.name}.`);
      upsertRunStats(manifest);
      process.exitCode = 1;
      return;
    }
    let ok = await runStage(stage, manifest);
    if (!ok) {
      const rec = manifest.stages[stage.name];
      const limit = detectUsageLimit(rec.error ?? "");
      if (limit.hit) {
        writeUsageLimitMarker(manifest.runDir, stage.name, rec.error ?? "", limit.resetAt);
        const waitMs = limit.resetAt ? limit.resetAt.getTime() - Date.now() : Number.POSITIVE_INFINITY;
        if (waitMs > 0 && waitMs <= CONFIG.maxUsageLimitSleepHours * 3_600_000) {
          console.warn(
            `Usage limit hit at "${stage.name}" — sleeping until the reset (${limit.resetAt!.toISOString()}) + 2min, then retrying once.`,
          );
          await sleep(waitMs + 2 * 60_000);
          ok = await runStage(stage, manifest);
        }
        if (!ok) {
          console.error(`Usage limit halted the pipeline at "${stage.name}" — marker written to _usage-limit.json.`);
          console.error(`Relaunch after the reset with: npm run daily -- --date ${date}`);
          upsertRunStats(manifest);
          process.exitCode = 3;
          return;
        }
      } else {
        console.error(`Pipeline halted at stage "${stage.name}". Re-run "npm run daily" to resume after fixing.`);
        upsertRunStats(manifest);
        process.exitCode = 1;
        return;
      }
    }
    // Quality bar: the scout may decide no paper is worth a full run today.
    if (stage.name === "scout") {
      const sel = JSON.parse(
        fs.readFileSync(path.join(manifest.runDir, "selection.json"), "utf-8"),
      ) as Selection;
      if (sel.proceed === false) {
        console.log(`Scout found no paper meeting the quality bar today. Reason: ${sel.skip_reason ?? "n/a"}`);
        console.log(`Skipping the rest of the pipeline — a skipped day beats a weak publication.`);
        upsertRunStats(manifest);
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

  upsertRunStats(manifest);
  console.log(`\nRun ${date} complete. Total cost: $${manifest.totalCostUsd.toFixed(2)}`);
  console.log(`Outputs: ${manifest.runDir}`);
}

export { consumeWaitingSpec, detectUsageLimit, pollForFile, upsertRunStats };

// Only run when invoked as the entry script, so helpers stay importable/testable.
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
