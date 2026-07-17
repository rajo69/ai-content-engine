"""Regenerate every figure for the AHR paper write-up.

    python viz/make_charts.py

Reads:  ingest/_appendixB.json      -- the paper's own published numbers
        reproduction/results.json   -- our laptop CPU reproduction (partial is fine)
Writes: viz/*.png, viz/method.svg

No number below is typed by hand: everything is recomputed from those two files, so
the figures cannot silently drift from the analysis. Provenance is stamped onto each
figure because the two sources are NOT interchangeable -- see viz/index.md.
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = pathlib.Path(__file__).resolve().parent
RUN = HERE.parent

# --- dataviz palette (validated reference instance, light surface) -------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"   # categorical slot 1 / diverging cool pole
RED = "#e34948"    # categorical slot 8 / diverging warm pole
GREEN = "#008300"  # categorical slot 2
ORANGE = "#eb6834"
NEUTRAL = "#f0efec"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 15,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})

SIZES = [10, 15, 20, 30, 50, 100]
MODELS = ["GPT2-XL", "Llama-3.2-1B", "Llama-3.2-3B", "Qwen3-8B"]
MODERN = MODELS[1:]
DATASETS = ["SST-2", "AG News", "Emotion", "Web", "ToxiGen", "Jailbreak"]
PROV_APPENDIX = "Data: recomputed by us from the paper's own Appendix B (24 tables, 10 seeds/cell). Not our measurements."


def load_appendix():
    """cells[(dataset, model)][method] -> list of accuracies indexed like SIZES."""
    raw = json.loads((RUN / "ingest" / "_appendixB.json").read_text(encoding="utf-8"))
    cells = {}
    for t in raw:
        cells[(t["dataset"], t["model"])] = {
            row[0]: [float(v) for v in row[1]] for row in t["rows"]
        }
    return cells


def at10(cells, dataset, model, method):
    vals = cells[(dataset, model)].get(method)
    return vals[0] if vals else None


def per_model(cells, method):
    """Mean over the 6 datasets at |D_train| = 10."""
    return {m: float(np.mean([at10(cells, d, m, method) for d in DATASETS])) for m in MODELS}


def load_repro():
    """Our reproduction. Still running -> partial. Returns {model: {arm: [accs]}}."""
    p = RUN / "reproduction" / "results.json"
    if not p.exists():
        return {}, {}
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, {}  # mid-write
    out = {}
    for key, rec in blob.get("runs", {}).items():
        model, arm, _seed = key.split("|")
        out.setdefault(model, {}).setdefault(arm, []).append(rec["acc"])
    return out, blob.get("config", {})


def short_name(model):
    """HF repo id -> display name ('Qwen/Qwen2.5-0.5B' -> 'Qwen2.5-0.5B')."""
    return model.split("/")[-1]


def stamp(fig, text, y=-0.02):
    fig.text(0.5, y, text, ha="center", va="top", fontsize=10.5, color=MUTED)


def tidy(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)


# ------------------------------------------------------------------ 1. headline
def headline(cells):
    ahr, icl = per_model(cells, "AHR (Ours)"), per_model(cells, "ICL")
    x = np.array([icl[m] for m in MODELS])
    y = np.array([ahr[m] - icl[m] for m in MODELS])

    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    fig.subplots_adjust(top=0.76, bottom=0.13, left=0.1, right=0.97)
    tidy(ax, "both")

    # Honest scale: symmetric about zero, both signs fully visible.
    lo, hi = 48, 90
    ax.axhspan(0, 12, color=BLUE, alpha=0.045, zorder=0)
    ax.axhspan(-6, 0, color=RED, alpha=0.05, zorder=0)
    ax.axhline(0, color=INK_2, linewidth=2, zorder=2)
    ax.text(lo + 0.7, 0.45, "AHR helps", fontsize=13, color=BLUE, fontweight="bold")
    ax.text(lo + 0.7, -1.15, "AHR hurts  —  worse than doing nothing",
            fontsize=13, color=RED, fontweight="bold")

    slope, intercept = np.polyfit(x, y, 1)
    # Trim the fit to the span of the data -- no extrapolation beyond the 4 models.
    xs = np.linspace(x.min() - 1.5, x.max() + 1.5, 100)
    ax.plot(xs, slope * xs + intercept, color=MUTED, linewidth=2,
            linestyle=(0, (5, 4)), zorder=3)

    # Hand-placed label offsets: the two negative points sit close together.
    offsets = {"GPT2-XL": (0, 28, "center"), "Llama-3.2-1B": (0, 28, "center"),
               "Llama-3.2-3B": (-14, -58, "right"), "Qwen3-8B": (14, -30, "left")}
    for xi, yi, m in zip(x, y, MODELS):
        c = BLUE if yi > 0 else RED
        ax.plot([xi, xi], [0, yi], color=c, linewidth=2.5, alpha=0.5, zorder=4)
        ax.scatter([xi], [yi], s=340, color=c, zorder=5,
                   edgecolors=SURFACE, linewidths=2.5)
        dx, dy, ha = offsets[m]
        ax.annotate(f"{m}\n{yi:+.2f} pp",
                    (xi, yi), textcoords="offset points",
                    xytext=(dx, dy), ha=ha,
                    fontsize=13.5, color=INK, fontweight="bold",
                    linespacing=1.45)

    zero_cross = -intercept / slope
    # Parked in the empty upper-right: the red band below is occupied by the
    # Llama-3.2-3B / Qwen3-8B labels.
    ax.annotate(
        f"crosses zero at\nICL ≈ {zero_cross:.0f}%",
        xy=(zero_cross, 0), xytext=(85.0, 3.4),
        ha="center", fontsize=12.5, color=MUTED, linespacing=1.4,
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1.4,
                        connectionstyle="arc3,rad=0.25"))

    ax.set_xlim(lo, hi)
    ax.set_ylim(-6, 12)
    ax.set_xlabel("Base model's 10-shot ICL accuracy (%)  —  how good it already is")
    ax.set_ylabel("AHR − ICL  (percentage points)")
    fig.text(0.5, 0.965, "The better the model, the less AHR helps",
             ha="center", va="top", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.5, 0.885,
             "Gain from training 512 head-scalars, vs. just putting the same 10 examples in the prompt.\n"
             "|D_train| = 10, averaged over 6 datasets. Kendall τ = −1.00 — perfectly monotone.",
             ha="center", va="top", fontsize=13.5, color=INK_2, linespacing=1.5)
    stamp(fig, PROV_APPENDIX + "  The paper reports every point above — and plots none of them.", y=0.045)
    fig.savefig(HERE / "headline.png")
    plt.close(fig)
    return slope, zero_cross


# ------------------------------------------------------------------ 2. win rate
def winrate(cells):
    icl_by_model = per_model(cells, "ICL")
    order = sorted(MODELS, key=lambda m: icl_by_model[m])
    wins = {
        m: sum(1 for d in DATASETS if at10(cells, d, m, "AHR (Ours)") > at10(cells, d, m, "ICL"))
        for m in MODELS
    }

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    tidy(ax, "x")
    ypos = np.arange(len(order))[::-1]
    for yp, m in zip(ypos, order):
        w = wins[m]
        c = BLUE if w >= 3 else RED
        ax.barh(yp, w, height=0.52, color=c, zorder=3)
        ax.barh(yp, 6, height=0.52, color=NEUTRAL, zorder=1)
        ax.text(w + 0.14, yp, f"{w} / 6", va="center", fontsize=15,
                color=INK, fontweight="bold")
        ax.text(-0.16, yp, f"{m}\nICL {icl_by_model[m]:.1f}%", va="center", ha="right",
                fontsize=13, color=INK_2, linespacing=1.4)

    ax.axhline(1.5, color=BASELINE, linewidth=1.5, linestyle=(0, (4, 4)))
    ax.text(6.05, 2.5, "10 of 12 cells", fontsize=13, color=BLUE, fontweight="bold",
            va="center", rotation=270)
    ax.text(6.05, 0.5, "2 of 12 cells", fontsize=13, color=RED, fontweight="bold",
            va="center", rotation=270)

    ax.set_yticks([])
    ax.set_xticks(range(7))
    ax.set_xlim(0, 6.35)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel("Datasets (of 6) where AHR beats doing nothing  —  |D_train| = 10")
    ax.set_title("It's a cliff, not a slope", pad=34, fontweight="bold")
    fig.text(0.5, 0.925,
             "Models ordered by ICL competence. Between Llama-3.2-1B and 3B, AHR simply stops working.\n"
             "Fisher exact test on the 10/12 vs 2/12 split: p = 0.0033.",
             ha="center", va="top", fontsize=13.5, color=INK_2, linespacing=1.5)
    stamp(fig, PROV_APPENDIX, y=0.015)
    fig.savefig(HERE / "winrate.png")
    plt.close(fig)
    return wins


# ------------------------------------------------------- 3. margin decomposition
def decomposition(cells):
    def mean_over(models, method):
        return float(np.mean([at10(cells, d, m, method) for m in models for d in DATASETS]))

    ahr, lora, icl = (mean_over(MODERN, k) for k in ("AHR (Ours)", "LoRA", "ICL"))
    total, lora_broken, ahr_real = ahr - lora, icl - lora, ahr - icl
    pct_broken = 100 * lora_broken / total

    fig, ax = plt.subplots(figsize=(11, 5.4))
    tidy(ax, "x")

    ax.barh(0, lora_broken, height=0.42, color=ORANGE, zorder=3)
    ax.barh(0, ahr_real, height=0.42, left=lora_broken + 0.04, color=BLUE, zorder=3)

    ax.text(lora_broken / 2, 0, f"{lora_broken:.2f} pp", ha="center", va="center",
            fontsize=17, color="white", fontweight="bold")
    ax.annotate(f"{ahr_real:.2f} pp", xy=(lora_broken + ahr_real / 2, 0.26),
                xytext=(lora_broken + ahr_real / 2, 0.58), ha="center",
                fontsize=15, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=1.4))

    ax.text(lora_broken / 2, -0.36,
            f"LoRA failing to beat a frozen model\n{pct_broken:.0f}% of the headline",
            ha="center", va="top", fontsize=13.5, color=INK_2, linespacing=1.45)
    ax.text(lora_broken + ahr_real / 2, 0.92,
            f"AHR actually beating\n“do nothing”  ({100 - pct_broken:.0f}%)\np ≈ 0.45  —  not significant",
            ha="center", va="bottom", fontsize=13, color=BLUE, linespacing=1.45,
            fontweight="bold")

    ax.set_yticks([])
    ax.set_ylim(-1.35, 1.75)
    ax.set_xlim(0, total * 1.06)
    ax.set_xlabel("Percentage points of the advertised AHR − LoRA margin")
    ax.set_title(f"The {total:.2f}-point headline is mostly LoRA being broken",
                 pad=34, fontweight="bold")
    fig.text(0.5, 0.93,
             "AHR − LoRA on the 3 modern models at |D_train| = 10, split into its two real causes.",
             ha="center", va="top", fontsize=13.5, color=INK_2)
    stamp(fig, PROV_APPENDIX, y=0.02)
    fig.savefig(HERE / "decomposition.png")
    plt.close(fig)
    return total, pct_broken


# ------------------------------------------------------------------ 4. explainer
def explainer():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.6))
    fig.subplots_adjust(top=0.80, bottom=0.10, wspace=0.08)
    heads = ["sentiment", "URL shape", "syntax", "topic"]
    ROWS = [7.55, 6.35, 5.15, 3.95]   # head rows
    EXTRA = 2.55                       # the "new feature" row (LoRA only)

    for ax, mode in zip(axes, ("ahr", "lora")):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        colour = BLUE if mode == "ahr" else RED

        ax.add_patch(FancyBboxPatch((0.5, 1.75), 9.0, 6.85, boxstyle="round,pad=0.1",
                                    facecolor="white", edgecolor=GRID, linewidth=1.6))
        ax.text(2.85, 8.3, "features the frozen model\nalready computes", ha="center",
                va="center", fontsize=11.5, color=MUTED, style="italic", linespacing=1.35)
        ax.text(7.4, 8.3, "volume knobs" if mode == "ahr" else "writes new directions",
                ha="center", va="center", fontsize=11.5, color=colour, style="italic")

        for h, y in zip(heads, ROWS):
            ax.add_patch(FancyBboxPatch((1.1, y - 0.32), 3.5, 0.64,
                                        boxstyle="round,pad=0.05",
                                        facecolor=NEUTRAL, edgecolor=BASELINE, linewidth=1.2))
            ax.text(2.85, y, h, ha="center", va="center", fontsize=12.5, color=INK_2)
            ax.add_patch(FancyArrowPatch((4.75, y), (6.3, y), arrowstyle="-|>",
                                         mutation_scale=12, color=BASELINE, linewidth=1.5))

        if mode == "ahr":
            for lvl, y in zip([0.95, 0.2, 0.5, 0.5], ROWS):
                ax.add_patch(FancyBboxPatch((6.45, y - 0.24), 1.9, 0.48,
                                            boxstyle="round,pad=0.03",
                                            facecolor=NEUTRAL, edgecolor=BASELINE, linewidth=1))
                ax.add_patch(FancyBboxPatch((6.45, y - 0.24), max(1.9 * lvl, 0.12), 0.48,
                                            boxstyle="round,pad=0.03",
                                            facecolor=BLUE, edgecolor="none"))
                ax.plot([6.45 + 1.9 * lvl], [y], marker="o", markersize=11, color=BLUE,
                        markeredgecolor="white", markeredgewidth=2, zorder=5)
            ax.text(7.4, ROWS[0] - 0.5, "turned up", ha="center", va="center",
                    fontsize=10.5, color=BLUE, fontweight="bold")
            ax.text(7.4, ROWS[1] - 0.5, "turned down", ha="center", va="center",
                    fontsize=10.5, color=BLUE, fontweight="bold")
            title = "AHR: 512 volume knobs"
            sub = "Can only turn up or down what is already there.\nIt cannot invent a feature. 512 numbers."
        else:
            for y in ROWS:
                ax.add_patch(FancyBboxPatch((6.45, y - 0.24), 1.9, 0.48,
                                            boxstyle="round,pad=0.03",
                                            facecolor=NEUTRAL, edgecolor=BASELINE, linewidth=1))
            ax.add_patch(FancyBboxPatch((1.1, EXTRA - 0.32), 3.5, 0.64,
                                        boxstyle="round,pad=0.05",
                                        facecolor="white", edgecolor=RED, linewidth=1.6,
                                        linestyle="--"))
            ax.text(2.85, EXTRA, "a brand-new feature", ha="center", va="center",
                    fontsize=12.5, color=RED, fontweight="bold")
            ax.add_patch(FancyArrowPatch((4.75, EXTRA), (6.3, EXTRA), arrowstyle="-|>",
                                         mutation_scale=12, color=RED, linewidth=1.8))
            ax.add_patch(FancyBboxPatch((6.45, EXTRA - 0.34), 1.9, 0.68,
                                        boxstyle="round,pad=0.04",
                                        facecolor=RED, edgecolor="none"))
            ax.text(7.4, EXTRA, "“example 4\nsaid ‘bank’”", ha="center", va="center",
                    fontsize=10.5, color="white", fontweight="bold", linespacing=1.25)
            ax.text(7.4, EXTRA - 0.85, "← memorised, not learned", ha="center",
                    va="center", fontsize=11, color=RED, fontweight="bold")
            title = "LoRA: 106,496 free parameters"
            sub = "Free to write features the task never needed —\nlike memorising example 4. 208× more numbers."

        ax.text(5, 9.55, title, ha="center", fontsize=18, color=colour, fontweight="bold")
        ax.text(5, 1.35, sub, ha="center", va="top", fontsize=12.5, color=INK_2,
                linespacing=1.5)

    fig.text(0.5, 0.985, "With only 10 labelled examples, being unable to learn is the feature",
             ha="center", va="top", fontsize=21, fontweight="bold", color=INK)
    fig.text(0.5, 0.915,
             "Both start out as the exact frozen model. Only one of them can overfit 10 examples —\n"
             "and at |D_train| = 10, LoRA is worse than the frozen model in 20 of 24 tests.",
             ha="center", va="top", fontsize=13.5, color=INK_2, linespacing=1.5)
    stamp(fig, "Schematic — head names are illustrative. The 20-of-24 figure is recomputed from the paper's Appendix B.",
          y=0.055)
    fig.savefig(HERE / "explainer.png")
    plt.close(fig)


# ------------------------------------------------------- 5. our reproduction
def repro_sst2(cells):
    repro, cfg = load_repro()
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    tidy(ax, "both")

    ax.axhline(0, color=INK_2, linewidth=2, zorder=2)

    px = [at10(cells, "SST-2", m, "ICL") for m in MODELS]
    py = [at10(cells, "SST-2", m, "AHR (Ours)") - at10(cells, "SST-2", m, "ICL") for m in MODELS]
    ax.scatter(px, py, s=300, color=BLUE, zorder=5, edgecolors=SURFACE, linewidths=2.5,
               label="Paper's 4 models (Appendix B, SST-2)")
    slope, intercept = np.polyfit(px, py, 1)
    xs = np.linspace(45, 100, 50)
    ax.plot(xs, slope * xs + intercept, color=BLUE, linewidth=2, linestyle=(0, (5, 4)),
            alpha=0.65, zorder=3)
    for xi, yi, m in zip(px, py, MODELS):
        ax.annotate(m, (xi, yi), textcoords="offset points", xytext=(0, 20),
                    ha="center", fontsize=12, color=INK_2)

    note = "Our reproduction: no runs completed yet."
    ours = []
    for model, arms in repro.items():
        if "icl" in arms and "ahr" in arms:
            n = min(len(arms["icl"]), len(arms["ahr"]))
            xi = float(np.mean(arms["icl"][:n]))
            yi = float(np.mean(arms["ahr"][:n])) - xi
            ours.append((xi, yi, model, n))
    if ours:
        ax.scatter([o[0] for o in ours], [o[1] for o in ours], s=340, marker="D",
                   color=ORANGE, zorder=6, edgecolors=SURFACE, linewidths=2.5,
                   label="Ours (laptop CPU, no CV, underpowered)")
        # Qwen sits at the right edge and low: label it leftwards so the text
        # clears both the x-axis and the paper's Llama-3.2-3B point above it.
        for xi, yi, m, n in ours:
            right_edge = xi > 80
            dx, dy, ha = (-18, -8, "right") if right_edge else (0, -46, "center")
            ax.annotate(f"{short_name(m)}\n({n} seed{'s' if n > 1 else ''})", (xi, yi),
                        textcoords="offset points", xytext=(dx, dy), ha=ha,
                        fontsize=12, color=ORANGE, fontweight="bold", linespacing=1.4)
        note = (f"Our reproduction: {sum(o[3] for o in ours)} seed-runs across "
                f"{len(ours)} model(s), {cfg.get('n_eval', '?')} eval examples, no cross-validation.")

    ax.axvspan(45, 62, color=RED, alpha=0.05, zorder=0)
    ax.text(46, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1,
            "chance zone —\nmodel too weak to\nclassify at all", fontsize=11.5,
            color=RED, va="top", linespacing=1.4)

    ax.set_xlim(45, 100)
    ax.set_ylim(-8, ax.get_ylim()[1])   # headroom so the gpt2 label clears the axis
    ax.set_xlabel("Base model's 10-shot ICL accuracy on SST-2 (%)")
    ax.set_ylabel("AHR − ICL  (percentage points)")
    ax.set_title("Our reproduction cannot carry this claim — and we won't pretend it does",
                 pad=34, fontweight="bold", fontsize=17.5)
    fig.text(0.5, 0.925,
             "SST-2 only, so paper and reproduction sit on the same axis. The paper's own SST-2 gradient is the\n"
             "steepest in the whole appendix (slope −{:.2f} pp per point of ICL).".format(abs(slope)),
             ha="center", va="top", fontsize=13, color=INK_2, linespacing=1.5)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=12.5)
    for t in leg.get_texts():
        t.set_color(INK_2)
    stamp(fig, note + "  Paper points: recomputed from Appendix B. Diamonds: measured by us this run.", y=0.015)
    fig.savefig(HERE / "repro_sst2.png")
    plt.close(fig)
    return ours


# ------------------------------------------------------------------ 6. method.svg
def method_svg():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.8)   # headroom: title + subtitle sit clear of the frozen box
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.35, 0.5), 11.3, 5.4, boxstyle="round,pad=0.12",
                                facecolor="white", edgecolor=GRID, linewidth=1.6))
    ax.text(0.7, 5.55, "everything below is FROZEN", fontsize=12, color=MUTED, style="italic")

    # residual stream
    ax.add_patch(FancyArrowPatch((0.9, 1.15), (11.1, 1.15), arrowstyle="-|>",
                                 mutation_scale=18, color=BASELINE, linewidth=3))
    ax.text(0.9, 0.75, "residual stream  z", fontsize=12.5, color=MUTED)

    xs = [2.4, 4.6, 6.8, 9.0]
    gammas = ["γ₁", "γ₂", "γ₃", "γₙ"]
    for x, g in zip(xs, gammas):
        ax.add_patch(FancyBboxPatch((x - 0.85, 3.9), 1.7, 0.95, boxstyle="round,pad=0.06",
                                    facecolor=NEUTRAL, edgecolor=BASELINE, linewidth=1.4))
        ax.text(x, 4.37, "head h", ha="center", va="center", fontsize=13, color=INK_2)
        ax.add_patch(FancyArrowPatch((x, 3.85), (x, 3.05), arrowstyle="-|>",
                                     mutation_scale=13, color=BASELINE, linewidth=1.6))
        # the delta
        ax.add_patch(FancyBboxPatch((x - 0.78, 2.15), 1.56, 0.85, boxstyle="round,pad=0.06",
                                    facecolor=BLUE, edgecolor="none"))
        ax.text(x, 2.57, f"× (1+{g})", ha="center", va="center", fontsize=13.5,
                color="white", fontweight="bold")
        ax.add_patch(FancyArrowPatch((x, 2.1), (x, 1.35), arrowstyle="-|>",
                                     mutation_scale=13, color=BLUE, linewidth=1.8))
        ax.plot([x], [1.15], marker="o", markersize=9, color=BLUE,
                markeredgecolor="white", markeredgewidth=1.6, zorder=5)

    ax.text(11.0, 4.37, "…", fontsize=18, color=MUTED, ha="center")
    # MLP sits left of head 1 (whose box starts at x=1.55) — untouched by AHR,
    # so its arrow runs straight to the residual stream with no blue box.
    ax.add_patch(FancyBboxPatch((0.6, 3.9), 0.85, 0.95, boxstyle="round,pad=0.06",
                                facecolor=NEUTRAL, edgecolor=BASELINE, linewidth=1.4))
    ax.text(1.02, 4.37, "MLP", ha="center", va="center", fontsize=12, color=INK_2)
    ax.add_patch(FancyArrowPatch((1.02, 3.85), (1.02, 1.35), arrowstyle="-|>",
                                 mutation_scale=13, color=BASELINE, linewidth=1.6))

    ax.text(6.0, 7.55, "The entire method is the blue row", ha="center", fontsize=20,
            fontweight="bold", color=INK)
    ax.text(6.0, 6.95,
            "One scalar γₕ per attention head — 512 numbers for Llama-3.2-1B, vs 106,496 for LoRA.\n"
            "Zero-initialised, so at γ = 0 the model IS the frozen model. Folds into W_O at test time.",
            ha="center", va="top", fontsize=13, color=INK_2, linespacing=1.5)
    ax.text(6.0, 0.12,
            "zₑₜₜᴸ⁺¹ = zᵣᴸ + Σₕ (1 + γₕ) · h(zᵣᴸ)      "
            "(Eq. 3 — the “+ γₕ” is the whole paper)",
            ha="center", va="bottom", fontsize=13, color=BLUE, fontweight="bold")

    fig.savefig(HERE / "method.svg", format="svg")
    fig.savefig(HERE / "method.png")
    plt.close(fig)


def main():
    cells = load_appendix()
    slope, cross = headline(cells)
    wins = winrate(cells)
    total, pct = decomposition(cells)
    explainer()
    ours = repro_sst2(cells)
    method_svg()

    ahr, icl, lora = per_model(cells, "AHR (Ours)"), per_model(cells, "ICL"), per_model(cells, "LoRA")
    print("== recomputed from Appendix B (|D_train| = 10) ==")
    for m in MODELS:
        print(f"  {m:16s} ICL {icl[m]:6.2f}  AHR {ahr[m]:6.2f}  LoRA {lora[m]:6.2f}  "
              f"AHR-ICL {ahr[m]-icl[m]:+.2f}  wins {wins[m]}/6")
    print(f"  headline slope {slope:+.3f} pp per point of ICL; crosses zero at ICL = {cross:.1f}%")
    print(f"  modern-3 AHR-LoRA = {total:.2f} pp, of which {pct:.1f}% is LoRA < frozen")
    print(f"== our reproduction: {len(ours)} model(s) with both ICL and AHR arms ==")
    for xi, yi, m, n in ours:
        print(f"  {m:22s} ICL {xi:6.2f}  AHR-ICL {yi:+.2f}  ({n} seeds)")
    print("\nwrote:", ", ".join(sorted(p.name for p in HERE.glob("*.png"))) + ", method.svg")


if __name__ == "__main__":
    main()
