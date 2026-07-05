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
sys.path.insert(0, os.path.join(ROOT, "scratch_arc", "full_lf52"))
sys.path.insert(0, os.path.join(ROOT, "scratch_arc", "agent"))
from arc3_harness import Game

# ARC3_ARCHIVE selects which source-free run to build filmstrips from; default = the Claude Fable
# 25/25 solve (arc3_fullgame_sourcefree_fable.json) so every game gets a replay figure.
ARCH = json.load(open(os.environ.get(
    "ARC3_ARCHIVE", os.path.join(ROOT, "experiments/results/arc3_fullgame_sourcefree_fable.json"))))
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


import re
_SCI = re.compile(r"-?\d(?:\.\d+)?e[+-]?\d+")   # scientific-notation board signatures


def _humanize_svg(svg):
    """The atlas card labels state nodes with the raw masked-frame signature (e.g. 'sig -7.68629e+17'),
    which is meaningless to a reader. Remap each distinct signature to a short stable id q0,q1,... so the
    graph reads like a state machine. We use 'q' for states because the card already names ACTIONS s1,s2,...
    -- keeping the two namespaces distinct (q = discovered board state, s = action)."""
    ids, order = {}, []
    for m in _SCI.findall(svg):
        if m not in ids:
            ids[m] = f"q{len(ids)}"; order.append(m)
    # node labels 'sig <num>' -> 'q_k'; bare schema/rollout signature values '<num>' -> 'q_k'
    for num in sorted(order, key=len, reverse=True):
        svg = svg.replace(f"sig {num}", ids[num]).replace(num, ids[num])
    return svg


def map_raster(gid, width=1700):
    p = os.path.join(MAPS, f"{gid}.svg")
    if not os.path.exists(p):
        return None
    svg = _humanize_svg(open(p, encoding="utf-8").read())
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=width)   # hi-res, legible text
    return Image.open(io.BytesIO(png))


def figure(gid):
    actions = SOL.get(gid)
    if not actions:
        print(f"  {gid}: no banked solution, skip"); return None
    frames, reached = milestones(gid, actions)
    win = PG.get(gid, {}).get("win", reached)
    fcols = min(len(frames), 5)
    frows = math.ceil(len(frames) / fcols)
    fig_w = 2.35 * fcols                         # filmstrip width drives the page width
    img = map_raster(gid)
    # size the map band to the card's TRUE aspect so it fills the full width (no letterbox)
    card_aspect = (img.width / img.height) if img is not None else 1.28   # w/h, ~1.28 for these cards
    map_h = fig_w / card_aspect                  # full-width card height (inches)
    frame_h = (fig_w / fcols) * 1.20             # per filmstrip row (square cell + label)
    fig_h = map_h + frows * frame_h + 0.55
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(1 + frows, fcols, height_ratios=[map_h] + [frame_h] * frows,
                  hspace=0.30, wspace=0.06)
    # ---- discovered map (atlas card): full-width, high-res, dominant ----
    axm = fig.add_subplot(gs[0, :]); axm.axis("off")
    if img is not None:
        axm.imshow(img, aspect="auto")           # axes already sized to card aspect -> fills, no distortion
        axm.set_title("discovered OpenWorld world-model map (atlas card)",
                      fontsize=10.5, color="#334155", pad=5)
    # ---- replay filmstrip ----
    for k, (label, fr, step) in enumerate(frames):
        ax = fig.add_subplot(gs[1 + k // fcols, k % fcols])
        ax.imshow(fr, cmap=CMAP, vmin=0, vmax=15, interpolation="nearest")
        ax.set_title(label + (f"\n(step {step})" if step else ""), fontsize=8.2, color="#0f172a")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"ARC-AGI-3  ·  {gid}  —  solved {reached}/{win} levels in {len(actions)} verified actions",
                 fontsize=13.5, fontweight="bold", y=0.997)
    out = os.path.join(FIGS, f"arc3_replay_{gid}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote arc3_replay_{gid}.png  ({len(frames)} frames, reached L{reached}, card {img.width}x{img.height})")
    return out


def main():
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError("replay timeout")))
    signal.alarm(900)
    order = sorted(FULL)                          # every fully-solved game (all 25 for the Fable archive)
    print(f"building replay figures for {len(order)} full games: {order}")
    for gid in order:
        try:
            figure(gid)
        except Exception as e:
            print(f"  {gid}: ERROR {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
