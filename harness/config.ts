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

  /** Per-stage model, effort, and spend limits. */
  stages: {
    scout: { model: "claude-opus-4-8", effort: "high", maxTurns: 60, maxBudgetUsd: 3 },
    ingest: { model: "claude-opus-4-8", effort: "medium", maxTurns: 60, maxBudgetUsd: 2 },
    litreview: { model: "claude-opus-4-8", effort: "high", maxTurns: 80, maxBudgetUsd: 5 },
    analyze: { model: "claude-opus-4-8", effort: "xhigh", maxTurns: 60, maxBudgetUsd: 6 },
    reproduce: { model: "claude-opus-4-8", effort: "xhigh", maxTurns: 150, maxBudgetUsd: 10 },
    visualize: { model: "claude-opus-4-8", effort: "high", maxTurns: 80, maxBudgetUsd: 4 },
    write: { model: "claude-opus-4-8", effort: "xhigh", maxTurns: 80, maxBudgetUsd: 8 },
    review: { model: "claude-opus-4-8", effort: "xhigh", maxTurns: 100, maxBudgetUsd: 8 },
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
} as const;

export type StageName = keyof typeof CONFIG.stages;
