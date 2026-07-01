"""Animated source-free solutions for the appendix: replay each banked source-free solution and render
a subsampled sequence of board frames as PNGs, so the paper can loop them with \\animategraphics.

Writes papers/assets/figs/anim/<game>/fr-000.png ... fr-NNN.png and prints the per-game last index for
the LaTeX. Frames are subsampled (target ~TARGET per game, every level-up frame forced in) to keep the
PDF size and frame count reasonable. Uses the fixed source-free SandboxGame (no game source).

  ~/.arcv/bin/python scripts/make_arc3_animations.py            # all source-free games
  ~/.arcv/bin/python scripts/make_arc3_animations.py sk48 tu93  # a subset
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = "/Users/jim/Desktop/openworld"
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from arc3_sandbox import SandboxGame

PAL = ["#000000","#0074D9","#FF4136","#2ECC40","#FFDC00","#AAAAAA","#F012BE","#FF851B",
       "#7FDBFF","#870C25","#39CCCC","#B10DC9","#01FF70","#85144b","#FFFFFF","#3D9970"]
CMAP = ListedColormap(PAL)
ANIM = os.path.join(ROOT, "papers/assets/figs/anim")
ARCH = json.load(open(os.path.join(ROOT, "experiments/results/arc3_fullgame_sourcefree.json")))
SOL, PG = ARCH["solutions"], ARCH["per_game"]
TARGET = 28          # ~frames per game (subsampled); level-up frames are always included

def board(g):
    a = np.asarray(g.frame)
    while a.ndim > 2: a = a[-1]
    return a.reshape(64, 64).astype(int) if a.size == 4096 else np.zeros((64, 64), int)

def capture(game, actions):
    """Replay; return a list of (frame, level) keeping every level-up frame + an even subsample."""
    g = SandboxGame(game); g.reset()
    seq = [(board(g), g.levels)]; lvl = g.levels
    for a in actions:
        try: g.step(*a)
        except Exception: break
        seq.append((board(g), g.levels))
        if g.levels > lvl: lvl = g.levels
        if getattr(g, "done", False): break
    try: g.close()
    except Exception: pass
    # subsample to ~TARGET, but force-keep first, last, and every level-up frame
    keep = {0, len(seq) - 1}
    prev = -1
    for i, (_, lv) in enumerate(seq):
        if lv != prev: keep.add(max(0, i - 1)); keep.add(i); prev = lv
    stride = max(1, len(seq) // TARGET)
    keep |= set(range(0, len(seq), stride))
    return [seq[i] for i in sorted(keep)]

def render(game, frames):
    d = os.path.join(ANIM, game); os.makedirs(d, exist_ok=True)
    for old in os.listdir(d):
        if old.endswith(".png"): os.remove(os.path.join(d, old))
    for i, (f, lv) in enumerate(frames):
        fig = plt.figure(figsize=(1.7, 1.7)); ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(f, cmap=CMAP, vmin=0, vmax=15, interpolation="nearest"); ax.axis("off")
        ax.text(0.02, 0.98, f"L{lv}", transform=ax.transAxes, va="top", ha="left",
                fontsize=6, color="white", bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.6))
        fig.savefig(os.path.join(d, f"fr-{i}.png"), dpi=90); plt.close(fig)  # unpadded for \animategraphics
    return len(frames)

def main():
    games = sys.argv[1:] or sorted(g for g in SOL if SOL[g])
    os.makedirs(ANIM, exist_ok=True)
    rows = []
    for g in games:
        acts = SOL.get(g) or []
        if not acts:
            print(f"  {g}: no solution -- skip"); continue
        try:
            frames = capture(g, acts); n = render(g, frames)
        except Exception as e:
            print(f"  {g}: error {str(e)[:50]}"); continue
        last_lv = frames[-1][1]; win = PG.get(g, {}).get("win", "?")
        rows.append((g, n, last_lv, win))
        print(f"  {g}: {n} frames, reaches L{last_lv}/{win}", flush=True)
    # emit a manifest the LaTeX generator can read (game -> last frame index)
    man = {g: {"frames": n, "last": n - 1, "level": lv, "win": w} for g, n, lv, w in rows}
    json.dump(man, open(os.path.join(ANIM, "manifest.json"), "w"), indent=1)
    emit_grid(man)
    print(f"\n  wrote {len(rows)} games to {ANIM} (+ anim_grid.tex for the appendix)")

def emit_grid(man, fps=8, per_row=5):
    """Emit the \\animategraphics grid the appendix \\inputs (paths are figs/anim/<game>/fr-)."""
    cells = []
    for g in sorted(man):
        m = man[g]
        cells.append(
            "\\begin{minipage}[t]{0.19\\textwidth}\\centering\n"
            f"\\animategraphics[autoplay,loop,width=\\linewidth]{{{fps}}}{{figs/anim/{g}/fr-}}{{0}}{{{m['last']}}}\\\\\n"
            f"{{\\scriptsize\\texttt{{{g}}} $\\cdot$ L{m['level']}/{m['win']}}}\n\\end{{minipage}}"
        )
    rows = ["\\hfill\n".join(cells[i:i + per_row]) for i in range(0, len(cells), per_row)]
    body = "\n\n\\vspace{4pt}\n\n".join(rows)
    open(os.path.join(ANIM, "anim_grid.tex"), "w").write(body + "\n")

if __name__ == "__main__":
    main()
