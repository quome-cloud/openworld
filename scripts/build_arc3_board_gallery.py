"""Maps gallery = one visually-distinct tile per ARC-AGI-3 game: a representative board frame from
Claude Fable's source-free solve, rendered in the ARC 16-colour palette. The per-game world-model CARD
(state graph + rollout + schema) is inherently chain-like for a linear solve and reads the same at
thumbnail scale, so the gallery shows the actual boards (each obviously unique) while the full cards
appear in the per-game replay figures. Writes papers/assets/figs/arc3_maps_gallery.png.

Run with the arcv interpreter (has arc_agi):
    /Users/jim/.arcv/bin/python scripts/build_arc3_board_gallery.py
"""
import os, sys, json, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path("/Users/jim/Desktop/openworld")
sys.path.insert(0, str(ROOT / "scratch_arc" / "full_lf52"))          # arc3_harness
os.chdir(ROOT)                                                        # environment_files/ resolves here
FIGS = ROOT / "papers" / "assets" / "figs"
FABLE = json.load(open(ROOT / "experiments/results/arc3_fullgame_sourcefree_fable.json"))
SOL = FABLE["solutions"]; PG = FABLE["per_game"]
# game -> short modality/category label (metadata only)
CAT = {g: v.get("category") or v.get("modality") or ""
       for g, v in json.load(open(ROOT / "experiments/results/arc3_fullgame.json"))["games"].items()}

PAL = ["#000000","#0074D9","#FF4136","#2ECC40","#FFDC00","#AAAAAA","#F012BE","#FF851B",
       "#7FDBFF","#870C25","#39CCCC","#B10DC9","#01FF70","#85144b","#FFFFFF","#3D9970"]
CMAP = ListedColormap(PAL)


def solved_frame(game):
    """Replay the Fable solution; return the board at the FINAL level-up (the solved world)."""
    from arc3_harness import Game
    g = Game(game); g.reset()
    frame = g.frame.copy(); lv = g.levels
    for a in SOL[game]:
        g.step(*a) if isinstance(a, (list, tuple)) else g.step(a)
        if g.levels > lv:
            lv = g.levels; frame = g.frame.copy()          # keep latest level-up frame
        if g.done:
            break
    return frame, lv


def main():
    games = sorted(SOL, key=lambda g: (-(PG.get(g, {}).get("win") or 0), g))
    cols = 5; rows = math.ceil(len(games) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8.5, 1.72 * rows + 0.5))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for i, g in enumerate(games):
        try:
            fr, lv = solved_frame(g)
        except Exception as e:
            print(f"  {g}: ERROR {e}"); continue
        ax = axes[i]
        ax.imshow(fr, cmap=CMAP, vmin=0, vmax=15, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_edgecolor("#cbd5e1"); s.set_linewidth(0.8)
        cat = f" · {CAT.get(g,'')}" if CAT.get(g) else ""
        ax.set_title(f"{g}{cat}  ·  {lv}/{PG.get(g,{}).get('win',lv)}",
                     fontsize=7.2, color="#0f172a", pad=2.5)
        print(f"  {g}: solved board {lv}/{PG.get(g,{}).get('win','?')}", flush=True)
    fig.suptitle("The 25 ARC-AGI-3 games, each solved source-free into a serveable OpenWorld world model "
                 "(Claude Fable)", fontsize=10, y=0.997)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.06, hspace=0.28)
    out = FIGS / "arc3_maps_gallery.png"
    fig.savefig(out, dpi=190, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
