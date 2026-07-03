"""E140 arm comparison — per-game head-to-head under source-free full-budget.

Loads:
  - experiments/results/arc3_fullgame_sourcefree.json         (Opus @ max)
  - experiments/results/arc3_fullgame_sourcefree_codex.json   (Codex @ xhigh)
  - experiments/results/e140_budget_asymmetry.json            (before/after budget lift)
  - experiments/results/arc3_fullgame_sourcefree_fable.json   (optional; skip if missing)

Produces:
  - head_to_head.png            per-game levels, opus vs codex (and fable if present)
  - walls_shared.png            games where multiple arms hit the same wall (target set)
  - budget_lift_before_after.png the 11→16 and 2→7 story from e140_budget_asymmetry.json
  - overlap_venn.txt            plain-text set-diff so authors can cite directly
  - arm_comparison_summary.json all numbers dumped
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(ROOT, "experiments/results")

INK, BLUE, OCHRE, TEAL, GREY = "#16202e", "#1d4ed8", "#b45309", "#0f766e", "#9aa4b2"
COLOR = {"opus": BLUE, "codex": OCHRE, "fable": TEAL}


def load(path, default=None):
    try:
        return json.load(open(path))
    except Exception:
        return default


def full_solved(pg):
    """A game is 'full' when levels >= win and win > 0."""
    return int(pg.get("levels", 0)) >= int(pg.get("win", 0)) and int(pg.get("win", 0)) > 0


def main():
    opus = load(os.path.join(RES, "arc3_fullgame_sourcefree.json"))
    codex = load(os.path.join(RES, "arc3_fullgame_sourcefree_codex.json"))
    fable = load(os.path.join(RES, "arc3_fullgame_sourcefree_fable.json"))
    budget = load(os.path.join(RES, "e140_budget_asymmetry.json"))

    arms = {"opus": opus, "codex": codex}
    if fable and isinstance(fable, dict) and "per_game" in fable:
        arms["fable"] = fable

    # Common per-game view
    all_games = sorted(set(g for a in arms.values() for g in a["per_game"].keys()))
    per_arm = {a: [] for a in arms}
    wins = {}
    for g in all_games:
        for a, obj in arms.items():
            pg = obj["per_game"].get(g, {})
            per_arm[a].append(int(pg.get("levels", 0)))
            if pg.get("win"):
                wins[g] = int(pg["win"])

    # Head-to-head bar chart (grouped)
    fig, ax = plt.subplots(figsize=(13, 5.2))
    x = np.arange(len(all_games))
    w = 0.28 if len(arms) == 3 else 0.4
    offsets = {list(arms.keys())[i]: (i - (len(arms) - 1) / 2) * w for i in range(len(arms))}
    for a in arms:
        ax.bar(x + offsets[a], per_arm[a], w, label=f"{a}", color=COLOR[a],
               edgecolor=INK, linewidth=0.5)
    ax.plot(x, [wins.get(g, 0) for g in all_games], "--", color=GREY, linewidth=1.2, label="win_levels")
    ax.set_xticks(x)
    ax.set_xticklabels(all_games, rotation=45, ha="right", fontsize=9, color=INK)
    ax.set_ylabel("levels reached", color=INK)
    ax.set_title(f"E140 head-to-head: per-game levels reached per arm  "
                 f"(source-free, full-budget)  "
                 f"— opus {opus['n_full_games']}/25, codex {codex['n_full_games']}/25"
                 + (f", fable {fable['n_full_games']}/25" if fable else ""),
                 color=INK, fontsize=11)
    ax.legend(loc="upper right", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color=GREY, alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "head_to_head.png"), dpi=140, bbox_inches="tight")
    plt.close()

    # Set analysis
    full_by = {a: {g for g in all_games if full_solved(arms[a]["per_game"].get(g, {}))} for a in arms}
    all_full = set.intersection(*full_by.values()) if len(full_by) > 1 else set()
    arm_unique = {a: full_by[a] - set.union(*(full_by[b] for b in full_by if b != a))
                  for a in full_by}
    any_full = set.union(*full_by.values())

    # Walls: games where NO arm goes full but ALL arms reach same level within ±1
    walls = []
    for g in all_games:
        if g in any_full:
            continue
        levs = [arms[a]["per_game"].get(g, {}).get("levels", 0) for a in arms]
        if levs and (max(levs) - min(levs)) <= 1 and max(levs) > 0:
            walls.append((g, levs, wins.get(g, 0)))

    # Walls chart
    if walls:
        fig, ax = plt.subplots(figsize=(11, 4.8))
        wg = [w[0] for w in walls]
        x = np.arange(len(wg))
        wbar = 0.28 if len(arms) == 3 else 0.4
        offsets2 = {list(arms.keys())[i]: (i - (len(arms) - 1) / 2) * wbar for i in range(len(arms))}
        for a in arms:
            vals = [arms[a]["per_game"].get(g, {}).get("levels", 0) for g in wg]
            ax.bar(x + offsets2[a], vals, wbar, label=a, color=COLOR[a],
                   edgecolor=INK, linewidth=0.5)
        ax.plot(x, [w[2] for w in walls], "--", color=GREY, label="win_levels")
        ax.set_xticks(x)
        ax.set_xticklabels(wg, fontsize=10, color=INK)
        ax.set_ylabel("levels reached", color=INK)
        ax.set_title(f"Shared walls: {len(walls)} games where every arm gets stuck within 1 level of each other\n"
                     f"(target set for goal-predicate synthesis experiments)",
                     color=INK, fontsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", color=GREY, alpha=0.35)
        ax.legend(loc="upper right", frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(HERE, "walls_shared.png"), dpi=140, bbox_inches="tight")
        plt.close()

    # Budget-asymmetry chart
    if budget:
        capped = budget.get("results_under_our_capped_budget", {})
        # E140 uncapped result taken from opus + codex current archives
        before = [
            capped.get("claude_source_free", {}).get("full_games", 0),
            capped.get("codex_source_free", {}).get("full_games", 0),
        ]
        after = [opus["n_full_games"], codex["n_full_games"]]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(2)
        ax.bar(x - 0.2, before, 0.4, label="Capped (45 min · default reasoning)",
               color=GREY, edgecolor=INK, linewidth=0.5)
        ax.bar(x + 0.2, after, 0.4, label="E140 uncapped (4 hr · max reasoning)",
               color=[BLUE, OCHRE], edgecolor=INK, linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Claude / Opus", "Codex / GPT-5.5"], color=INK)
        ax.set_ylabel("full games / 25", color=INK)
        ax.set_title("E140 budget-asymmetry: removing time-cap + reasoning-cap lifts BOTH models\n"
                     f"opus {before[0]}→{after[0]} (+{after[0]-before[0]}),  "
                     f"codex {before[1]}→{after[1]} (+{after[1]-before[1]})",
                     color=INK, fontsize=11)
        for i, (b, a) in enumerate(zip(before, after)):
            ax.text(i - 0.2, b + 0.3, str(b), ha="center", color=INK, fontsize=10)
            ax.text(i + 0.2, a + 0.3, str(a), ha="center", color=INK, fontsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", color=GREY, alpha=0.35)
        ax.legend(loc="lower right", frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(HERE, "budget_lift_before_after.png"), dpi=140, bbox_inches="tight")
        plt.close()

    # Text overlap file
    overlap_lines = []
    overlap_lines.append("=== E140 source-free full-games overlap ===")
    overlap_lines.append(f"Games fully solved by EVERY arm ({len(all_full)}):")
    overlap_lines.append("  " + ", ".join(sorted(all_full)))
    for a, gs in arm_unique.items():
        overlap_lines.append(f"{a}-only full ({len(gs)}):")
        overlap_lines.append("  " + (", ".join(sorted(gs)) or "(none)"))
    overlap_lines.append("")
    overlap_lines.append(f"Union (any arm full): {len(any_full)}")
    overlap_lines.append(f"  " + ", ".join(sorted(any_full)))
    overlap_lines.append("")
    overlap_lines.append(f"Shared walls (no arm full, within ±1 level, {len(walls)}):")
    for g, levs, w in walls:
        levs_str = "/".join(str(v) for v in levs)
        overlap_lines.append(f"  {g:6s} — {levs_str} of {w}")
    with open(os.path.join(HERE, "overlap_venn.txt"), "w") as f:
        f.write("\n".join(overlap_lines))
    print("\n".join(overlap_lines))

    # Full summary JSON
    out = {
        "arms": list(arms.keys()),
        "per_arm_totals": {a: {"n_full_games": arms[a]["n_full_games"],
                               "total_levels": arms[a]["total_levels"],
                               "possible": arms[a]["total_possible"]} for a in arms},
        "per_game": {
            g: {a: arms[a]["per_game"].get(g, {}).get("levels", 0) for a in arms}
            for g in all_games
        },
        "win_levels": wins,
        "full_by_arm": {a: sorted(list(full_by[a])) for a in arms},
        "arm_unique_full": {a: sorted(list(arm_unique[a])) for a in arms},
        "shared_walls": [{"game": g, "per_arm": {a: arms[a]["per_game"].get(g, {}).get("levels", 0) for a in arms},
                          "win": w} for g, _, w in walls],
        "e140_budget_asymmetry": budget,
    }
    with open(os.path.join(HERE, "arm_comparison_summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
