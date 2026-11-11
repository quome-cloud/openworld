"""Render per-episode GIF animations from saved trajectory tty frames.

Overlays: step counter, dungeon level, HP, last action, and a fired-memory
banner (condition B). Frames were subsampled at capture time (see
nh_runner.want_frame); GIFs further subsample to <= MAXF frames.

Usage:
    python3 render_animations.py <traj.json> <out.gif> [title]
    python3 render_animations.py --auto     # representative set
"""

import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
TRAJ = os.path.join(HERE, "results", "trajectories")
OUT = os.path.join(HERE, "results", "animations")
os.makedirs(OUT, exist_ok=True)

FONT_PATH = os.path.join(HERE, "balrog", "environments", "nle",
                         "Hack-Regular.ttf")
FONT = ImageFont.truetype(FONT_PATH, 14)
FONT_SMALL = ImageFont.truetype(FONT_PATH, 12)
CW, CH = 9, 16
COLS, ROWS = 80, 24
MAXF = 420

PALETTE = {
    "@": (255, 221, 51), ">": (80, 250, 123), "<": (139, 233, 253),
    "{": (98, 114, 164), "}": (255, 85, 85), "`": (241, 250, 140),
    "#": (130, 130, 130), ".": (90, 90, 90), "+": (255, 184, 108),
    "|": (170, 170, 170), "-": (170, 170, 170), "^": (255, 121, 198),
    "(": (189, 147, 249), ")": (189, 147, 249), "/": (189, 147, 249),
    "!": (189, 147, 249), "?": (189, 147, 249), "*": (189, 147, 249),
    "$": (255, 215, 0), "%": (255, 165, 90),
}
DEFAULT = (220, 220, 220)
MONSTER = (255, 85, 85)


def frame_image(screen, header, footer, banner=None):
    img = Image.new("RGB", (COLS * CW + 16, (ROWS + 3) * CH + 12),
                    (12, 12, 20))
    d = ImageDraw.Draw(img)
    d.text((8, 2), header, font=FONT, fill=(255, 221, 51))
    for r, line in enumerate(screen[:ROWS]):
        for c, ch in enumerate(line[:COLS]):
            if ch == " ":
                continue
            color = PALETTE.get(ch)
            if color is None:
                color = MONSTER if ch.isalpha() and 1 <= r <= 21 else DEFAULT
            d.text((8 + c * CW, (r + 1) * CH + 4), ch, font=FONT_SMALL,
                   fill=color)
    d.text((8, (ROWS + 1) * CH + 8), footer, font=FONT_SMALL,
           fill=(139, 233, 253))
    if banner:
        d.rectangle([(8, (ROWS + 2) * CH + 6),
                     (COLS * CW + 8, (ROWS + 3) * CH + 8)],
                    fill=(60, 20, 60))
        d.text((12, (ROWS + 2) * CH + 8), banner, font=FONT_SMALL,
               fill=(255, 121, 198))
    return img


def render(traj_file, out_gif, title=""):
    t = json.load(open(traj_file))
    frames = t["frames"]
    fsteps = t["frame_steps"]
    n = len(frames)
    stride = max(1, n // MAXF)
    idxs = list(range(0, n, stride))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    # memory banner lookup: step -> text
    mem = {}
    for line in t.get("mem_fired", []):
        try:
            s = int(line.split(":")[0].replace("step", "").strip())
            mem[s] = line.split(":", 1)[1].strip()
        except Exception:
            pass
    imgs = []
    for i in idxs:
        step = fsteps[i]
        # nearest trajectory index for hp/depth/action (arrays are per step)
        j = min(step, len(t["actions"])) - 1
        hp = t["hp"][j] if j >= 0 else [0, 0]
        depth = t["depth"][j] if j >= 0 else 1
        act = t["actions"][j] if j >= 0 else "-"
        header = (f"{title}  step {step}  Dlvl {depth}  "
                  f"HP {hp[0]}/{hp[1]}  last: {act}")
        banner = None
        for s, txt in mem.items():
            if 0 <= step - s < 200:
                banner = f"MEMORY: {txt}"
        imgs.append(frame_image(frames[i], header,
                                t["messages"][j][:100] if j >= 0 else "",
                                banner))
    durs = [90] * len(imgs)
    durs[-1] = 2500
    imgs[0].save(out_gif, save_all=True, append_images=imgs[1:],
                 duration=durs, loop=0, optimize=True)
    print(f"wrote {out_gif} ({len(imgs)} frames from {n} captured)")


def main():
    if sys.argv[1] == "--auto":
        picks = json.load(open(os.path.join(OUT, "picks.json")))
        for p in picks:
            render(os.path.join(TRAJ, p["traj"]),
                   os.path.join(OUT, p["gif"]), p.get("title", ""))
    else:
        render(sys.argv[1], sys.argv[2],
               sys.argv[3] if len(sys.argv) > 3 else "")


if __name__ == "__main__":
    main()
