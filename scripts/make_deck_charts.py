"""Presentation-native charts for the conference decks (matplotlib -> PNG, OpenWorld palette,
CVD-validated). Clean magnitude bars: direct value labels, recessive axes, one hue + an ochre
highlight, warm-paper background so they sit inside the slides seamlessly.

    python scripts/make_deck_charts.py      # -> presentations/assets/pres_*.png
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "presentations" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

PAPER = "#FBFAF6"; INK = "#16242E"; DEEP = "#0B2E4F"; MUTED = "#5B6B78"
TEAL = "#0E9E9E"; OCHRE = "#D98A2B"; BLUE = "#1E6FB0"; RED = "#9E2B25"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "font.size": 15, "font.family": "sans-serif", "text.color": INK,
    "axes.edgecolor": "#C9C2B4", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": INK, "axes.linewidth": 1.0,
})


def _clean(ax, keep_x=False):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if not keep_x:
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
    ax.tick_params(length=0)


def hbars(path, labels, values, colors, title, fmt="{:g}", xmax=None, note=None, figsize=(9, 4.3)):
    """A clean horizontal magnitude chart, bars top->bottom, direct value labels."""
    fig, ax = plt.subplots(figsize=figsize)
    y = list(range(len(labels)))[::-1]
    bars = ax.barh(y, values, height=0.62, color=colors, zorder=3)
    for yi, v, b in zip(y, values, bars):
        ax.text(v + (xmax or max(values)) * 0.015, yi, fmt.format(v), va="center", ha="left",
                fontsize=15, fontweight="bold", color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=15, color=INK)
    ax.set_xlim(0, (xmax or max(values) * 1.18))
    _clean(ax)
    ax.set_title(title, fontsize=16, fontweight="bold", color=DEEP, loc="left", pad=14)
    if note:
        ax.text(0.995, -0.13, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=13, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {Path(path).relative_to(ROOT)}")


# 1) ARC-AGI-3 model scaling: games completed source-free (of 25), Fable highlighted, baseline1 grey
hbars(OUT / "pres_arc3_scaling.png",
      ["Claude Fable 5", "Claude Opus 4.8", "baseline1 (prior SOTA)", "GPT-5.5"],
      [25, 16, 15, 12],
      [OCHRE, TEAL, MUTED, TEAL],
      "ARC-AGI-3 games completed source-free  (of 25)",
      fmt="{:g}", xmax=27,
      note="A stronger model clears more procedural walls at equal budget — Fable is the first perfect sweep.")

# 2) Rollout speed, log scale: LLM proxy vs verified code vs hand-written oracle
def speed_chart():
    labels = ["function oracle", "verified code", "LLM per-step"]
    vals = [243728, 14997, 0.32]
    cols = [MUTED, TEAL, RED]
    fig, ax = plt.subplots(figsize=(9, 3.7))
    y = list(range(len(labels)))
    ax.barh(y, vals, height=0.6, color=cols, zorder=3, log=True)
    for yi, v in zip(y, vals):
        ax.text(v * 1.35, yi, f"{v:,.2f}/s" if v < 1 else f"{v:,.0f}/s", va="center",
                ha="left", fontsize=15, fontweight="bold", color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=15, color=INK)
    ax.set_xscale("log"); ax.set_xlim(0.1, 5e6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0); ax.set_xticks([1, 1e2, 1e4, 1e6])
    ax.set_xlabel("rollout steps / second (log scale)", fontsize=13, color=MUTED)
    ax.set_title("Verified code rolls out ≈47,000× faster than a per-step LLM",
                 fontsize=16, fontweight="bold", color=DEEP, loc="left", pad=14)
    fig.tight_layout(); fig.savefig(OUT / "pres_speed.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("  wrote presentations/assets/pres_speed.png")
speed_chart()

# 3) World-time-compute lift by model size (the small-model multiplier)
hbars(OUT / "pres_smallmodel.png",
      ["0.5B", "1.5B", "3B", "7B", "32B"],
      [29, 18, 12, 7, 3],
      [OCHRE, TEAL, TEAL, TEAL, TEAL],
      "World-time-compute lift on held-out worlds  (accuracy, pts)",
      fmt="+{:g}", xmax=34,
      note="The smaller the model, the bigger the gain from traversing verified worlds.")

# 4) Exact 20-step rollouts: verified code vs per-step LLM (the compounding-error contrast)
hbars(OUT / "pres_exact.png",
      ["verified code", "per-step LLM"],
      [100, 0],
      [TEAL, RED],
      "Exact 20-step rollouts  (% of held-out trajectories, bit-exact)",
      fmt="{:g}%", xmax=112,
      note="Right once = exact at every depth. A per-step proxy diverges by ≈step 2.3.",
      figsize=(9, 2.9))

print("done.")
