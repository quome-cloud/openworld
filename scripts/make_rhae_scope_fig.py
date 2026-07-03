"""Reproducible RHAE-scope explainer: what the official RHAE metric scores (only the banked replay
path) vs the discovery cost it omits (exploration + LLM synthesis + offline search over the code world
model). Writes experiments/results/rhae_scope_diagram.png.

  python scripts/make_rhae_scope_fig.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments/results/rhae_scope_diagram.png")
BG, INK, BLUE, OCHRE, TEAL, GREY, RED = "#fcfbf8", "#16202e", "#1d4ed8", "#b45309", "#0f766e", "#8a94a3", "#b91c1c"


def main():
    fig, ax = plt.subplots(figsize=(12.6, 5.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")
    fig.patch.set_facecolor(BG)

    def box(x, y, w, h, fc, ec, title, sub, tc="#ffffff"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.4", fc=fc, ec=ec, lw=1.8))
        ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="top", fontsize=11.5, fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h - 8.6, sub, ha="center", va="top", fontsize=8.6, color=tc)

    def arrow(x1, x2, y):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=16, lw=2, color=BLUE))

    ax.text(50, 55.5, "How an ARC-AGI-3 game gets solved  →  what RHAE actually scores",
            ha="center", fontsize=13.5, fontweight="bold", color=INK)
    Y, H = 30, 15
    box(1.5, Y, 21, H, "#eef2fb", BLUE, "1 · EXPLORE", "act in the REAL env\n≥10⁴–10⁵ g.step() calls\nto discover dynamics", tc=INK)
    arrow(22.8, 25.5, Y + H / 2)
    box(25.5, Y, 21, H, "#f6efe3", OCHRE, "2 · SYNTHESIZE", "LLM writes predict()\ncode world model\n~10² turns / $ tokens", tc=INK)
    arrow(46.8, 49.5, Y + H / 2)
    box(49.5, Y, 21, H, "#e6f1ee", TEAL, "3 · SEARCH", "BFS/beam OVER the code\nCodeTransition, offline\n10⁴–10⁶ sim rollouts", tc=INK)
    arrow(70.8, 73.5, Y + H / 2)
    box(73.5, Y, 25, H, "#16202e", "#16202e", "4 · BANKED SOLUTION", "the winning action path\nK ≈ 86–558 actions\n(replayed + verified)")
    ax.add_patch(FancyBboxPatch((72.5, Y - 9.5), 27, 7.6, boxstyle="round,pad=0.4,rounding_size=1.2", fc="none", ec=RED, lw=2.2, ls="--"))
    ax.text(86, Y - 5.7, "⬆  RHAE scores ONLY this", ha="center", va="center", fontsize=10.5, fontweight="bold", color=RED)
    ax.text(86, Y - 8.4, "min((human_baseline / K)²·100, 115) per level", ha="center", va="center", fontsize=8.2, color=RED)
    ax.add_patch(FancyBboxPatch((1.0, Y + H + 2.0), 69.5, 6.8, boxstyle="round,pad=0.4,rounding_size=1.2", fc="none", ec=GREY, lw=1.8, ls="--"))
    ax.text(35.7, Y + H + 5.4, "NOT in RHAE:  all discovery cost — exploration env-steps + LLM sampling/tokens + world-model build & offline search",
            ha="center", va="center", fontsize=9.6, color=GREY, fontstyle="italic")
    ax.text(50, 6.5, "RHAE measures the ECONOMY OF THE FINAL PATH (vs a human's path), the same official metric baseline1 reports — apples-to-apples.",
            ha="center", fontsize=9.2, color=INK)
    ax.text(50, 2.8, "It is NOT a 'total cost to solve' metric: the search that FOUND the path (esp. offline over the code world model) is not counted.",
            ha="center", fontsize=9.2, color=RED, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT, dpi=175, facecolor=BG)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
