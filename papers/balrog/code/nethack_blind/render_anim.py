"""Render logged tty frames to an animated GIF.
Usage: python3 render_anim.py <transitions.jsonl.gz> <out.gif> [stride] [max_frames]
Uses PIL from the pylib runtime + the Hack-Regular.ttf font asset (a font file,
not game knowledge)."""
import gzip
import json
import os
import sys

FABLE_NH = "/data/doh/teams/researchy/work/fable_nethack"
sys.path.insert(0, os.path.join(FABLE_NH, "pylib"))
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

FONT = os.path.join(FABLE_NH, "balrog", "environments", "nle", "Hack-Regular.ttf")


def render_frame(text, font, cw=8, chh=15):
    lines = text.split("\n")
    img = Image.new("RGB", (cw * 80 + 8, chh * 24 + 8), (12, 12, 16))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines[:24]):
        color = (200, 200, 200)
        if i == 0:
            color = (255, 230, 120)
        elif i >= 22:
            color = (120, 200, 255)
        d.text((4, 4 + i * chh), line, fill=color, font=font)
    return img


def main():
    path, out = sys.argv[1], sys.argv[2]
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    maxf = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    font = ImageFont.truetype(FONT, 13)
    frames = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "frame" in r:
                frames.append((r.get("t", 0), r["frame"]))
    sel = frames[::stride]
    if len(sel) > maxf:
        sel = sel[:: max(1, len(sel) // maxf)][:maxf]
    imgs = [render_frame(t, font) for _, t in sel]
    if not imgs:
        print("no frames"); return
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=120, loop=0,
                 optimize=True)
    print("wrote", out, len(imgs), "frames")


if __name__ == "__main__":
    main()
