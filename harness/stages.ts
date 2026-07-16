import type { StageName } from "./config.js";

export interface StageDef {
  name: StageName;
  promptFile: string;
  /** Files (relative to run dir) that must exist for the stage to count as completed. */
  requiredOutputs: string[];
  /** Tools the stage agent may use without prompting. */
  allowedTools: string[];
}

const RESEARCH_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch", "Bash"];

/** Ordered pipeline. Each stage reads its predecessors' artifacts from the run dir. */
export const STAGES: StageDef[] = [
  {
    name: "scout",
    promptFile: "prompts/00-scout.md",
    requiredOutputs: ["candidates.json", "selection.json"],
    allowedTools: RESEARCH_TOOLS,
  },
  {
    name: "ingest",
    promptFile: "prompts/01-ingest.md",
    requiredOutputs: ["ingest/metadata.json", "ingest/references.json"],
    allowedTools: RESEARCH_TOOLS,
  },
  {
    name: "litreview",
    promptFile: "prompts/02-litreview.md",
    requiredOutputs: ["litreview/bibliography.md", "litreview/lineage.md"],
    allowedTools: RESEARCH_TOOLS,
  },
  {
    name: "analyze",
    promptFile: "prompts/03-analyze.md",
    requiredOutputs: ["analysis/core.md"],
    allowedTools: RESEARCH_TOOLS,
  },
  {
    name: "reproduce",
    promptFile: "prompts/04-reproduce.md",
    requiredOutputs: ["reproduction/REPORT.md"],
    allowedTools: RESEARCH_TOOLS,
  },
  {
    name: "visualize",
    promptFile: "prompts/05-visualize.md",
    requiredOutputs: ["viz/index.md"],
    allowedTools: RESEARCH_TOOLS,
  },
  {
    name: "write",
    promptFile: "prompts/06-write.md",
    requiredOutputs: [
      "outputs/paper/paper.md",
      "outputs/blog-long.md",
      "outputs/social/x-thread.md",
      "outputs/social/linkedin.md",
      "outputs/social/reddit.md",
    ],
    allowedTools: RESEARCH_TOOLS,
  },
  {
    name: "review",
    promptFile: "prompts/07-review.md",
    requiredOutputs: ["review/verdicts.json", "review/objections.md"],
    allowedTools: RESEARCH_TOOLS,
  },
];
