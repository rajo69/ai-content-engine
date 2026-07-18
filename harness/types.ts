import type { StageName } from "./config.js";

export type StageStatus = "pending" | "running" | "completed" | "failed";

export interface StageRecord {
  status: StageStatus;
  startedAt?: string;
  finishedAt?: string;
  costUsd?: number;
  error?: string;
  missingOutputs?: string[];
  /** Async-handoff cycles consumed (_waiting.json → poll → re-invoke). */
  resumes?: number;
  /** Every failure this stage has hit across attempts (kept for run stats). */
  failures?: { at: string; error: string }[];
}

export interface RunManifest {
  date: string; // YYYY-MM-DD
  runDir: string;
  paperSlug: string | null;
  paperTitle: string | null;
  arxivId: string | null;
  stages: Record<StageName, StageRecord>;
  totalCostUsd: number;
  createdAt: string;
  updatedAt: string;
}

export interface HistoryEntry {
  date: string;
  arxivId: string;
  title: string;
  slug: string;
  score: number;
  /** Engagement metrics, filled in later by the (future) feedback loop. */
  engagement?: Record<string, number>;
}

/**
 * Async experiment handoff: a stage may write `_waiting.json` (this shape) into
 * the run directory and end its session instead of polling a long-running
 * experiment. The pipeline fs-polls for `file` (zero tokens), then re-invokes
 * the stage agent with `resume_hint` appended to its prompt.
 */
export interface WaitingSpec {
  /** Path, relative to the run directory, whose appearance means "results are ready". */
  file: string;
  /** Context the resumed agent needs to interpret the results and finish the stage. */
  resume_hint: string;
  /** Max minutes to wait before resuming anyway (default: reproduction time box + 15). */
  timeout_minutes?: number;
}

export interface Selection {
  arxiv_id: string;
  title: string;
  slug: string;
  score: number;
  justification: string;
  scores: Record<string, number>;
  code_url?: string | null;
  dataset?: string | null;
  /** false = no candidate met the quality bar today; pipeline stops gracefully. */
  proceed?: boolean;
  skip_reason?: string | null;
}
