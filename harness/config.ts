/**
 * Harness configuration — the knobs you tune without touching pipeline code.
 */

export const CONFIG = {
  /** arXiv categories the scout watches. Weighted toward LLM/NLP per project goals. */
  domains: {
    primary: ["cs.CL", "cs.LG", "cs.AI"],
    secondary: ["cs.CV", "stat.ML"],
  },

  /**
   * Scoring rubric weights (must sum to 1.0). The scout scores each candidate
   * 1-10 per criterion; weighted sum ranks them.
   */
  scoutWeights: {
    tweetCompressibility: 0.2, // core idea stateable in one sentence
    deltaToClaimRatio: 0.2, //    smallest change, biggest claimed effect
    contrarianAngle: 0.15, //     inverts an assumption the field holds
    socialVelocity: 0.15, //      HF upvotes, GitHub stars velocity, X mentions
    reproducibility: 0.2, //      public code+data, feasible on modest compute
    visualPotential: 0.1, //      has a result that makes one killer chart
  },

  /**
   * Per-stage model, effort, spend, and wall-clock limits.
   * maxWallMinutes bounds one agent invocation (a hung network call can't stall
   * the pipeline); fs-polling between async-handoff invocations doesn't count.
   * Ingest runs on Sonnet: it's pure plumbing (downloads, pdftotext, JSON
   * reshaping). Judgment stages stay on Opus — do not demote them without A/B
   * evidence via lessons.
   */
  stages: {
    scout: { model: "claude-opus-4-8", effort: "high", maxTurns: 60, maxBudgetUsd: 3, maxWallMinutes: 30 },
    ingest: { model: "claude-sonnet-5", effort: "medium", maxTurns: 60, maxBudgetUsd: 4, maxWallMinutes: 30 },
    litreview: { model: "claude-opus-4-8", effort: "high", maxTurns: 80, maxBudgetUsd: 5, maxWallMinutes: 45 },
    analyze: { model: "claude-opus-4-8", effort: "xhigh", maxTurns: 60, maxBudgetUsd: 6, maxWallMinutes: 60 },
    reproduce: { model: "claude-opus-4-8", effort: "xhigh", maxTurns: 150, maxBudgetUsd: 10, maxWallMinutes: 150 },
    visualize: { model: "claude-opus-4-8", effort: "high", maxTurns: 80, maxBudgetUsd: 4, maxWallMinutes: 45 },
    write: { model: "claude-opus-4-8", effort: "xhigh", maxTurns: 80, maxBudgetUsd: 8, maxWallMinutes: 60 },
    review: { model: "claude-opus-4-8", effort: "xhigh", maxTurns: 100, maxBudgetUsd: 8, maxWallMinutes: 75 },
  },

  /**
   * Hard ceiling for a full daily run. Checked BEFORE each stage starts, so the
   * true worst case is this value plus one stage cap (~$48 with current caps).
   * Per-stage caps sum to $46 — a run that maxes every stage will be stopped
   * by this ceiling before the final stages; that is intentional back-pressure.
   */
  maxRunBudgetUsd: 40,

  /** Wall-clock cap per reproduction attempt, communicated to the reproduce stage. */
  reproductionTimeBoxMinutes: 90,

  /** Max async-handoff cycles (_waiting.json → poll → re-invoke) per stage. */
  maxWaitingResumes: 3,

  /** Seconds between existence checks while fs-polling for an awaited file. */
  waitingPollSeconds: 30,

  /**
   * On a usage/session-limit error whose reset time is parseable and at most
   * this far away, the pipeline sleeps until the reset and retries the stage
   * once. Further away (or unparseable): exit code 3 + _usage-limit.json marker
   * so a scheduler/operator relaunches at the right moment.
   */
  maxUsageLimitSleepHours: 3,
} as const;

export type StageName = keyof typeof CONFIG.stages;
