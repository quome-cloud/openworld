"""Appendix replay figures: for each FULLY-SOLVED ARC-AGI-3 game, pair the discovered OpenWorld
world-model MAP (papers/arc-3/maps/<g>.svg atlas card) with a level-completion REPLAY FILMSTRIP
(the real board screenshot at the start and at each level-up), replayed from the banked verified
action trace in experiments/results/arc3_fullgame_sourcefree.json. Screenshots use the ARC 16-colour
palette. Writes papers/assets/figs/arc3_replay_<g>.png (one per game).

Needs the arc venv (arc_agi) + cairosvg (run with DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib so
cairocffi finds libcairo):
    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ~/.arcv/bin/python scripts/make_arc3_replay_figs.py
"""
import os, sys, io, json, math, signal
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import cairosvg
from PIL import Image

ROOT = "/Users/jim/Desktop/openworld"
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "scratch_arc", "agent"))
from arc3_harness import Game

ARCH = json.load(open(os.path.join(ROOT, "experiments/results/arc3_fullgame_sourcefree.json")))
MAPS = os.path.join(ROOT, "papers/arc-3/maps")
FIGS = os.path.join(ROOT, "papers/assets/figs"); os.makedirs(FIGS, exist_ok=True)

# ARC-AGI 16-colour palette (0..15)
PAL = ["#000000","#0074D9","#FF4136","#2ECC40","#FFDC00","#AAAAAA","#F012BE","#FF851B",
       "#7FDBFF","#870C25","#39CCCC","#B10DC9","#01FF70","#85144b","#FFFFFF","#3D9970"]
CMAP = ListedColormap(PAL)

FULL = ARCH.get("full_games", [])
SOL = ARCH.get("solutions", {})
PG = ARCH.get("per_game", {})


def milestones(gid, actions):
    """Replay the verified trace; capture (label, frame) at start and at every level-up."""
    g = Game(gid); g.reset()
    out = [("start  ·  L0", g.frame.copy(), 0)]
    lv = g.levels
    for i, a in enumerate(actions):
        g.step(*a)
        if g.levels > lv:
            lv = g.levels
            out.append((f"L{lv} complete", g.frame.copy(), i + 1))
        if g.done:
            break
    return out, lv


def map_raster(gid, width=520):
    p = os.path.join(MAPS, f"{gid}.svg")
    if not os.path.exists(p):
        return None
    png = cairosvg.svg2png(url=p, output_width=width)
    return Image.open(io.BytesIO(png))


def figure(gid):
    actions = SOL.get(gid)
    if not actions:
        print(f"  {gid}: no banked solution, skip"); return None
    frames, reached = milestones(gid, actions)
    win = PG.get(gid, {}).get("win", reached)
    fcols = min(len(frames), 5)
    frows = math.ceil(len(frames) / fcols)
    fig = plt.figure(figsize=(2.05 * fcols, 3.4 + 2.0 * frows))
    gs = GridSpec(1 + frows, fcols, height_ratios=[3.0] + [2.0] * frows, hspace=0.32, wspace=0.08)
    # ---- discovered map (atlas card) across the top ----
    axm = fig.add_subplot(gs[0, :]); axm.axis("off")
    img = map_raster(gid)
    if img is not None:
        axm.imshow(img); axm.set_title("discovered OpenWorld world-model map (atlas card)",
                                       fontsize=9, color="#334155", pad=4)
    # ---- replay filmstrip ----
    for k, (label, fr, step) in enumerate(frames):
        ax = fig.add_subplot(gs[1 + k // fcols, k % fcols])
        ax.imshow(fr, cmap=CMAP, vmin=0, vmax=15, interpolation="nearest")
        ax.set_title(label + (f"\n(step {step})" if step else ""), fontsize=7.6, color="#0f172a")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"ARC-AGI-3  ·  {gid}  —  solved {reached}/{win} levels in {len(actions)} verified actions",
                 fontsize=12.5, fontweight="bold", y=0.995)
    out = os.path.join(FIGS, f"arc3_replay_{gid}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote arc3_replay_{gid}.png  ({len(frames)} frames, reached L{reached})")
    return out


def main():
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError("replay timeout")))
    signal.alarm(900)
    order = [g for g in ["ar25", "re86", "lp85", "sb26", "cn04", "ft09", "cd82", "tr87"] if g in FULL]
    print(f"building replay figures for {len(order)} full games: {order}")
    for gid in order:
        try:
            figure(gid)
        except Exception as e:
            print(f"  {gid}: ERROR {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
