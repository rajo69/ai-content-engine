import type { StageName } from "./config.js";

export type StageStatus = "pending" | "running" | "completed" | "failed";

export interface StageRecord {
  status: StageStatus;
  startedAt?: string;
  finishedAt?: string;
  costUsd?: number;
  error?: string;
  missingOutputs?: string[];
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
