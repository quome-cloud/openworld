"""Fig: the model-scaling win. Across the three source-free arms (GPT-5.5 -> Claude Opus 4.8 -> Claude
Fable) capability RISES (games & levels completed) while the discovery cost per solve FALLS -- a stronger
model wins more games AND finds each solve with fewer tokens. Reads the committed arm archives + the
token-based discovery cost. Writes papers/assets/figs/arc3_scaling.png.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIGS = ROOT / "papers" / "assets" / "figs"
GREY, BLUE, TEAL = "#94a3b8", "#1f4e79", "#2a8a7f"

ARMS = ["GPT-5.5", "Claude Opus 4.8", "Claude Fable"]
COLOR = [GREY, BLUE, TEAL]


def nfull_and_levels(fname):
    d = json.load(open(RES / fname)); pg = d.get("per_game", {})
    full = sum(1 for v in pg.values() if isinstance(v, dict) and v.get("win")
               and (v.get("levels", 0) or 0) >= v.get("win", 99))
    levels = sum((v.get("levels", 0) or 0) for v in pg.values() if isinstance(v, dict))
    return full, levels


def main():
    g_cx, l_cx = nfull_and_levels("arc3_fullgame_sourcefree_codex.json")
    g_op, l_op = nfull_and_levels("arc3_fullgame_sourcefree.json")
    g_fb, l_fb = nfull_and_levels("arc3_fullgame_sourcefree_fable.json")
    games = [g_cx, g_op, g_fb]; levels = [l_cx, l_op, l_fb]
    disc = json.load(open(RES / "e149_discovery_cost.json"))["per_arm"]
    tok = [None, disc["opus"]["total_tokens"] / 1e6 / 25, disc["fable"]["total_tokens"] / 1e6 / 25]  # M/game
    base1_games, base1_levels = 15, 145

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(3)

    # ---- A: capability rises ----
    axA.bar(x, games, color=COLOR, width=0.66, zorder=3)
    for i, (g, l) in enumerate(zip(games, levels)):
        axA.text(i, g + 0.4, f"{g}/25", ha="center", fontsize=11, fontweight="bold", color=COLOR[i])
        axA.text(i, g / 2, f"{l}\nlevels", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    axA.axhline(base1_games, ls="--", color="#b00", lw=1.2)
    axA.text(2.42, base1_games + 0.3, f"baseline1 = {base1_games}", color="#b00", ha="right", fontsize=8.5)
    axA.annotate("", xy=(2, 25.9), xytext=(0, 12.3),
                 arrowprops=dict(arrowstyle="->", color="#334155", lw=1.4, alpha=0.55,
                                 connectionstyle="arc3,rad=-0.18"))
    axA.set_xticks(x); axA.set_xticklabels(ARMS, fontsize=9.5)
    axA.set_ylabel("games completed source-free"); axA.set_ylim(0, 28)
    axA.set_title("Capability scales with the model", fontsize=11, fontweight="bold")
    axA.grid(axis="y", alpha=0.25, zorder=0)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    # ---- B: cost per solve falls (Opus, Fable; GPT-5.5 logged no token telemetry, so it is omitted here) ----
    xb = [0, 1]  # opus, fable
    axB.bar(xb, [tok[1], tok[2]], color=[BLUE, TEAL], width=0.55, zorder=3)
    for xi, tv, c in zip(xb, [tok[1], tok[2]], [BLUE, TEAL]):
        axB.text(xi, tv + 2, f"{tv:.0f}M", ha="center", fontsize=11, fontweight="bold", color=c)
    axB.annotate(f"{tok[1] / tok[2]:.1f}$\\times$ cheaper\nper solve", xy=(1, tok[2] + 6), xytext=(0.42, 62),
                 ha="center", fontsize=9.5, color=TEAL, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4, connectionstyle="arc3,rad=0.2"))
    axB.set_xticks([0, 1]); axB.set_xticklabels([ARMS[1], ARMS[2]], fontsize=9.5)
    axB.set_xlim(-0.65, 1.65)
    axB.set_ylabel("mean tokens / solved game (millions)"); axB.set_ylim(0, 95)
    axB.set_title("…and each solve costs less", fontsize=11, fontweight="bold")
    axB.grid(axis="y", alpha=0.25, zorder=0)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    fig.suptitle("A stronger model wins more games AND finds each solve with fewer tokens",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGS / "arc3_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote arc3_scaling.png  (games {games}, levels {levels}, tok/game {[round(t,0) if t else None for t in tok]})")


if __name__ == "__main__":
    main()
