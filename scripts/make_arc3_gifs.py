"""Animated GIFs of the source-free Fable solutions for the README's "Watch it solve" grid.

Replays each banked Claude Fable source-free solution through the fixed SandboxGame (no game
source), renders the 64x64 board frames in the official ARC-AGI palette with a level label, and
saves an animated GIF per game to assets/arc3/<game>.gif. Frames are subsampled (~TARGET/game,
every level-up frame forced in) to keep each GIF small; the final winning frame is held longer.

  ~/.arcv/bin/python scripts/make_arc3_gifs.py            # all 25 Fable source-free games
  ~/.arcv/bin/python scripts/make_arc3_gifs.py dc22 su15  # a subset
"""
import os, sys, json
import numpy as np
from PIL import Image, ImageDraw

ROOT = "/Users/jim/Desktop/openworld"
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from arc3_sandbox import SandboxGame

PAL = ["#000000","#0074D9","#FF4136","#2ECC40","#FFDC00","#AAAAAA","#F012BE","#FF851B",
       "#7FDBFF","#870C25","#39CCCC","#B10DC9","#01FF70","#85144b","#FFFFFF","#3D9970"]
RGB = np.array([[int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)] for h in PAL], dtype=np.uint8)

OUT  = os.path.join(ROOT, "assets/arc3")
ARCH = json.load(open(os.path.join(ROOT, "experiments/results/arc3_fullgame_sourcefree_fable.json")))
SOL, PG = ARCH["solutions"], ARCH["per_game"]

TARGET = 30       # ~frames per game after subsampling
SCALE  = 3        # 64 -> 192 px (nearest)
MS     = 110      # ms per frame
HOLD   = 14       # extra copies of the final frame (hold the win)


def board(g):
    a = np.asarray(g.frame)
    while a.ndim > 2: a = a[-1]
    return a.reshape(64, 64).astype(int) if a.size == 4096 else np.zeros((64, 64), int)


def capture(game, actions):
    """Source-free replay; keep first, last, every level-up frame, and an even subsample."""
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
    keep = {0, len(seq) - 1}; prev = -1
    for i, (_, lv) in enumerate(seq):
        if lv != prev: keep.add(max(0, i - 1)); keep.add(i); prev = lv
    stride = max(1, len(seq) // TARGET)
    keep |= set(range(0, len(seq), stride))
    return [seq[i] for i in sorted(keep)]


def to_image(frame, lv, win):
    """64x64 int board -> upscaled RGB PIL image with an 'L{lv}/{win}' badge."""
    img = Image.fromarray(RGB[np.clip(frame, 0, 15)], "RGB").resize(
        (64 * SCALE, 64 * SCALE), Image.NEAREST)
    d = ImageDraw.Draw(img)
    txt = f"L{lv}/{win}"
    d.rectangle([3, 3, 7 + 6 * len(txt), 15], fill=(0, 0, 0))
    d.text((5, 4), txt, fill=(255, 255, 255))
    return img


def make_gif(game, actions):
    win = PG.get(game, {}).get("win", "?")
    frames = capture(game, actions)
    imgs = [to_image(f, lv, win) for f, lv in frames]
    imgs += [imgs[-1]] * HOLD                     # hold the winning frame
    os.makedirs(OUT, exist_ok=True)
    imgs[0].save(os.path.join(OUT, f"{game}.gif"), save_all=True, append_images=imgs[1:],
                 duration=MS, loop=0, optimize=True, disposal=2)
    return len(frames), frames[-1][1], win


def main():
    games = sys.argv[1:] or sorted(g for g in SOL if SOL[g])
    rows = []
    for g in games:
        acts = SOL.get(g) or []
        if not acts:
            print(f"  {g}: no solution -- skip"); continue
        try:
            n, lv, win = make_gif(g, acts)
            rows.append((g, win)); print(f"  {g}: {n} frames, reaches L{lv}/{win}", flush=True)
        except Exception as e:
            print(f"  {g}: error {str(e)[:60]}", flush=True)
    print(f"\n  wrote {len(rows)} GIFs to {OUT}")
    # per-game win-level manifest for the README grid
    json.dump({g: PG.get(g, {}).get("win", "?") for g, _ in rows},
              open(os.path.join(OUT, "wins.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
