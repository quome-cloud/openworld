"""Generate the OpenWorld repo QR for the decks: deep-blue modules on paper, nested-worlds
mark in the centre. -> presentations/assets/qr_openworld.png   (pip install segno pillow)"""
import io
from pathlib import Path
import segno
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "presentations" / "assets" / "qr_openworld.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
URL = "https://github.com/quome-cloud/openworld"
DEEP, PAPER, BLUE, TEAL = "#0B2E4F", "#FBFAF6", "#1E6FB0", "#0F8C8C"
qr = segno.make(URL, error="h")
try:
    from PIL import Image, ImageDraw
    buf = io.BytesIO()
    qr.save(buf, kind="png", scale=20, border=4, dark=DEEP, light="#FFFFFF")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)
    # clear a centre patch and draw the nested-worlds mark (error='h' tolerates ~30% occlusion)
    m = int(W * 0.16); cx, cy = W // 2, H // 2
    d.rectangle([cx - m, cy - m, cx + m, cy + m], fill="#FFFFFF")
    def sq(frac, color, fill=False, wpx=None):
        r = int(m * frac)
        box = [cx - r, cy - r, cx + r, cy + r]
        d.rounded_rectangle(box, radius=int(r * 0.18), outline=None if fill else color,
                            fill=color if fill else None, width=wpx or max(3, int(W * 0.010)))
    sq(0.92, DEEP); sq(0.60, BLUE); sq(0.30, TEAL, fill=True)
    img.save(OUT)
    print("wrote", OUT.relative_to(ROOT), "(with mark)")
except Exception as e:
    qr.save(str(OUT), scale=20, border=3, dark=DEEP, light=PAPER)
    print("wrote", OUT.relative_to(ROOT), "(plain, no PIL:", str(e)[:40], ")")
