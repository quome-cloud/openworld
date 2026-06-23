"""Generate the SVG title-page assets and rasterless PDF copies for the papers.

The papers embed vector PDFs (tectonic cannot include SVG directly), but the SVG
is the editable source -- same house style as openworld/card.py and assets/logo.svg
(atlas aesthetic: nested-worlds mark, blue/ochre/teal depth ramp, grid, corner
registration ticks, blue->cyan accent).

    python3 scripts/make_title_assets.py     # writes papers/assets/figs/title_band.{svg,pdf}

Requires cairosvg (pip install cairosvg) for the SVG->PDF step; the committed PDF
lets the papers build with tectonic alone, so CI does not need cairosvg.
"""
from pathlib import Path

FIGS = Path(__file__).parent.parent / "papers" / "assets" / "figs"

INK, SLATE = "#16202e", "#5b6675"
BLUE, CYAN, OCHRE, TEAL = "#1d4ed8", "#0891b2", "#b45309", "#0f766e"

SERIF = "Georgia,'Times New Roman','Iowan Old Style',Palatino,serif"
MONO = "'SFMono-Regular',Menlo,Consolas,'DejaVu Sans Mono',monospace"


def nested_mark(cx, cy, s, opacity=1.0, shadow=False):
    """Three concentric rounded squares: blue outer / ochre mid / teal core."""
    sh = ' filter="url(#sh)"' if shadow else ""
    o = f' opacity="{opacity}"' if opacity != 1.0 else ""
    return f"""  <g transform="translate({cx},{cy})"{o}>
    <rect x="{-1.0*s:.1f}" y="{-1.0*s:.1f}" width="{2.0*s:.1f}" height="{2.0*s:.1f}" rx="{0.33*s:.1f}" fill="#ffffff" stroke="{BLUE}" stroke-width="{0.09*s:.2f}"{sh}/>
    <rect x="{-0.62*s:.1f}" y="{-0.62*s:.1f}" width="{1.24*s:.1f}" height="{1.24*s:.1f}" rx="{0.22*s:.1f}" fill="none" stroke="{OCHRE}" stroke-width="{0.075*s:.2f}"/>
    <rect x="{-0.26*s:.1f}" y="{-0.26*s:.1f}" width="{0.52*s:.1f}" height="{0.52*s:.1f}" rx="{0.12*s:.1f}" fill="{TEAL}"/>
  </g>"""


def build_band(W=1680, H=336):
    ticks = []
    for (x, y, dx, dy) in [(30, 30, 1, 1), (W - 30, 30, -1, 1), (30, H - 30, 1, -1), (W - 30, H - 30, -1, -1)]:
        ticks.append(f'<path d="M{x} {y+16*dy} L{x} {y} L{x+16*dx} {y}"/>')
    ticks = "\n    ".join(ticks)

    # depth ramp: "many worlds" receding to the right, joined by a composition bridge
    ramp = "\n".join(nested_mark(cx, 168, s, op) for cx, s, op in
                     [(1276, 50, 0.92), (1396, 38, 0.62), (1500, 28, 0.4)])
    bridge = (f'<path d="M1334 168 H1366 M1442 168 H1472" stroke="{BLUE}" '
              f'stroke-width="2.2" stroke-dasharray="3 6" opacity="0.55" fill="none"/>')

    # a faint world-rollout sparkline behind the wordmark band
    pts = [(286, 286), (340, 272), (394, 280), (448, 256), (502, 266),
           (556, 240), (610, 250), (664, 228), (718, 236), (772, 216),
           (826, 224), (880, 204), (934, 212), (988, 196)]
    spark = ("M" + " L".join(f"{x} {y}" for x, y in pts))

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#fdfcf9"/><stop offset="1" stop-color="#eceef0"/>
    </linearGradient>
    <linearGradient id="acc" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{BLUE}"/><stop offset="1" stop-color="{CYAN}"/>
    </linearGradient>
    <pattern id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
      <path d="M30 0 H0 V30" fill="none" stroke="#8aa0c2" stroke-width="0.7" opacity="0.18"/>
    </pattern>
    <filter id="sh" x="-40%" y="-40%" width="180%" height="180%">
      <feDropShadow dx="0" dy="7" stdDeviation="11" flood-color="#16202e" flood-opacity="0.18"/>
    </filter>
  </defs>

  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <rect width="{W}" height="{H}" fill="url(#grid)"/>
  <rect x="0" y="0" width="{W}" height="13" fill="url(#acc)"/>
  <rect x="0" y="{H-7}" width="{W}" height="7" fill="url(#acc)" opacity="0.85"/>
  <g stroke="{BLUE}" stroke-width="1.8" opacity="0.5" fill="none">
    {ticks}
  </g>

  <path d="{spark}" fill="none" stroke="{CYAN}" stroke-width="2.0" opacity="0.28"/>

  {nested_mark(140, 168, 66, shadow=True)}

  <text x="280" y="160" font-family="{SERIF}" font-size="88" font-weight="700" fill="{INK}" letter-spacing="-1.3">OpenWorld</text>
  <rect x="285" y="186" width="330" height="5" rx="2.5" fill="url(#acc)"/>
  <text x="287" y="230" font-family="{MONO}" font-size="19" font-weight="700" fill="{SLATE}" letter-spacing="2.8">VERIFIED CODE WORLD MODELS</text>
  <text x="287" y="261" font-family="{MONO}" font-size="16" font-weight="600" fill="{BLUE}" letter-spacing="1.4">build &#183; optimize &#183; deploy</text>

  {bridge}
  {ramp}
</svg>
"""


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    svg = build_band()
    svg_path = FIGS / "title_band.svg"
    svg_path.write_text(svg)
    print("wrote", svg_path)
    try:
        import cairosvg
        cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(FIGS / "title_band.pdf"))
        print("wrote", FIGS / "title_band.pdf")
    except Exception as e:  # pragma: no cover
        print("WARNING: SVG->PDF skipped (need `pip install cairosvg`):", e)


if __name__ == "__main__":
    main()
