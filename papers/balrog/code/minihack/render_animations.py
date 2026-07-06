"""Render per-episode GIF animations from saved trajectories.

Frames were captured as tty text screens each step. We draw them with the
bundled NetHack Hack-Regular.ttf, colorize key glyph classes, and annotate
step counter + last action + message line.

Usage:
    python3 render_animations.py            # representative set
    python3 render_animations.py --all      # every episode
"""

import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
TRAJ = os.path.join(HERE, "results", "trajectories")
OUT = os.path.join(HERE, "results", "animations")
os.makedirs(OUT, exist_ok=True)

FONT_PATH = os.path.join(HERE, "balrog", "environments", "nle", "Hack-Regular.ttf")
FONT = ImageFont.truetype(FONT_PATH, 14)
FONT_SMALL = ImageFont.truetype(FONT_PATH, 12)
CW, CH = 9, 16          # cell size
COLS, ROWS = 80, 24

PALETTE = {
    "@": (255, 221, 51),      # agent: yellow
    ">": (80, 250, 123),      # stairs down: green
    "<": (139, 233, 253),     # stairs up: cyan
    "{": (98, 114, 164),      # fountain: blue-gray
    "}": (255, 85, 85),       # lava/water: red
    "`": (241, 250, 140),     # boulder: pale yellow
    "#": (130, 130, 130),     # corridor/bars
    ".": (90, 90, 90),        # floor
    "+": (255, 184, 108),     # door
    "|": (170, 170, 170), "-": (170, 170, 170),
    "^": (255, 121, 198),
    "(": (189, 147, 249), ")": (189, 147, 249), "/": (189, 147, 249),
    "!": (189, 147, 249), "?": (189, 147, 249), "*": (189, 147, 249),
}
MONSTER_COLOR = (255, 85, 85)
DEFAULT = (220, 220, 220)


def frame_image(screen, header, footer):
    img = Image.new("RGB", (COLS * CW + 16, (ROWS + 3) * CH + 12), (12, 12, 20))
    d = ImageDraw.Draw(img)
    d.text((8, 4), header, font=FONT_SMALL, fill=(139, 233, 253))
    y0 = CH + 8
    for r, line in enumerate(screen[:ROWS]):
        for c, ch in enumerate(line[:COLS]):
            if ch == " ":
                continue
            col = PALETTE.get(ch) if 1 <= r <= 21 else None
            if col is None:
                col = (MONSTER_COLOR if 1 <= r <= 21 and ch.isalpha()
                       else DEFAULT)
            if ch == "@":
                d.rectangle([8 + c * CW - 1, y0 + r * CH - 1,
                             8 + (c + 1) * CW, y0 + (r + 1) * CH - 2],
                            fill=(60, 60, 0))
            d.text((8 + c * CW, y0 + r * CH), ch, font=FONT, fill=col)
    d.text((8, y0 + ROWS * CH + 2), footer[:110], font=FONT_SMALL,
           fill=(241, 250, 140))
    return img


def render(fn, out_fn, max_frames=140):
    with open(fn) as f:
        t = json.load(f)
    res = t["result"]
    frames = t["frames"]
    n = len(frames)
    stride = max(1, (n + max_frames - 1) // max_frames)
    imgs = []
    for i in range(0, n, stride):
        act = t["actions"][i - 1] if 0 < i <= len(t["actions"]) else "start"
        msg = t["messages"][i - 1] if 0 < i <= len(t["messages"]) else ""
        hp = t["hp"][i - 1] if 0 < i <= len(t["hp"]) else ["-", "-"]
        header = (f"{res['task']}  ep{res['episode']} seed {res['seed']}   "
                  f"step {i}/{res['steps']}   action: {act}   "
                  f"HP {hp[0]}/{hp[1]}")
        footer = msg or " "
        imgs.append(frame_image(frames[i], header, footer))
    # verdict frame
    verdict = "SOLVED" if res["progression"] >= 1.0 else "FAILED"
    header = (f"{res['task']}  ep{res['episode']}   {verdict}   "
              f"steps={res['steps']}  progression={res['progression']}")
    imgs.append(frame_image(frames[-1], header,
                            f"end: {res['end_reason']}"))
    durations = [140] * len(imgs)
    durations[0] = 800
    durations[-1] = 2500
    imgs[0].save(out_fn, save_all=True, append_images=imgs[1:],
                 duration=durations, loop=0, optimize=True)
    return len(imgs)


def main():
    render_all = "--all" in sys.argv
    files = sorted(os.listdir(TRAJ))
    by_task = {}
    for fn in files:
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(TRAJ, fn)) as f:
            t = json.load(f)
        by_task.setdefault(t["result"]["task"], []).append(
            (fn, t["result"]["progression"]))
    chosen = []
    for task, eps in sorted(by_task.items()):
        if render_all:
            chosen += [fn for fn, _ in eps]
        else:
            solved = [fn for fn, p in eps if p >= 1.0]
            failed = [fn for fn, p in eps if p < 1.0]
            if solved:
                chosen.append(solved[0])
            if failed:
                chosen.append(failed[0])
    for fn in chosen:
        out = os.path.join(OUT, fn.replace(".json", ".gif"))
        k = render(os.path.join(TRAJ, fn), out)
        print(f"{fn} -> {os.path.basename(out)} ({k} frames)")


if __name__ == "__main__":
    main()
