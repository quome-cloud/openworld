"""README 'Three ways to use a world model' diagram -> assets/three-routes.svg (+ .png).

One verified world is a reusable artifact; there are three routes to spend it, plus a hybrid.
Brand vocabulary matches assets/pipeline.svg (bg gradient, blue/ochre/teal ramp, grid, ticks,
nested-worlds mark, dark ribbon). Chip x-offsets are computed so pills never overlap.
"""
import os
try:
    import cairosvg
except Exception:
    cairosvg = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 1200, 636
BLUE, OCHRE, TEAL, INK, MUTE = "#1d4ed8", "#b45309", "#0f766e", "#16202e", "#5b6675"

ROUTES = [
    dict(n="1", grad="gBuild", color=BLUE, tint="#eaeef6", dark="#1e3a8a",
         title="USE IT AS A TOOL", tag="no training",
         desc="Serve the world and call it — plan, query exact next-states, or verify.",
         mono="openworld serve specs/*.json --allow-code",
         chips=[("exact", True), ("tool at inference", None), ("any world", None)]),
    dict(n="2", grad="gOpt", color=OCHRE, tint="#f6eee2", dark="#9a4408",
         title="DISTILL ONE WORLD", tag="test-time training",
         desc="Generate exact trajectories from one world; QLoRA-tune its skill into weights.",
         mono="experiments/e80_*_ttt.py",
         chips=[("approximate", False), ("no tool at inference", None), ("within the world", None)]),
    dict(n="3", grad="gDep", color=TEAL, tint="#e3f1ee", dark="#0b5c56",
         title="TRAIN ACROSS MANY", tag="world-time compute",
         desc="Traverse a family of worlds and generalize to held-out worlds you never trained on.",
         mono="experiments/e74_scaling.py · e76_world_count.py",
         chips=[("approximate", False), ("no tool at inference", None), ("across the family", True)]),
]

CX, CY, CW, CH, GAP = 452, 60, 688, 156, 24     # route card geometry
def chip_w(label, has_mark): return int(len(label) * 6.7) + (34 if has_mark is not None else 18)

def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def card(r, i):
    Y = CY + i * (CH + GAP); X = CX
    p = [f'<g filter="url(#sh)"><rect x="{X}" y="{Y}" width="{CW}" height="{CH}" rx="17" fill="#ffffff"/></g>',
         f'<g clip-path="url(#clip{i})"><rect x="{X}" y="{Y}" width="{CW}" height="7" fill="url(#{r["grad"]})"/></g>',
         f'<rect x="{X}" y="{Y}" width="{CW}" height="{CH}" rx="17" fill="none" stroke="#dde2ea" stroke-width="1.5"/>',
         f'<circle cx="{X+38}" cy="{Y+42}" r="15" fill="{r["tint"]}"/>',
         f'<text x="{X+38}" y="{Y+48}" text-anchor="middle" font-size="17" font-weight="800" fill="{r["color"]}" font-family="ui-monospace,Menlo,monospace">{r["n"]}</text>',
         f'<text x="{X+66}" y="{Y+49}" font-size="21" font-weight="800" letter-spacing="2" fill="{r["color"]}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">{r["title"]}</text>']
    # tag pill (top-right)
    tw = chip_w(r["tag"], None)
    p.append(f'<rect x="{X+CW-tw-20}" y="{Y+30}" width="{tw}" height="24" rx="12" fill="{r["tint"]}"/>')
    p.append(f'<text x="{X+CW-20-tw/2:.0f}" y="{Y+46}" text-anchor="middle" font-size="12.5" font-weight="700" letter-spacing="0.5" fill="{r["dark"]}" font-family="ui-monospace,Menlo,monospace">{r["tag"]}</text>')
    # description
    p.append(f'<text x="{X+30}" y="{Y+84}" font-size="15.5" fill="{INK}">{esc(r["desc"])}</text>')
    # attribute chips row
    x = X + 30; cy = Y + 104
    for label, mark in r["chips"]:
        cw = chip_w(label, mark)
        fill = "#f4f6fb"; stroke = "#dde2ea"
        p.append(f'<rect x="{x}" y="{cy}" width="{cw}" height="24" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
        tx = x + 14
        if mark is True:                                  # drawn check (font glyph is unreliable in cairosvg)
            p.append(f'<path d="M{x+12},{cy+13} l3.2,4 l7,-9" fill="none" stroke="{TEAL}" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/>'); tx = x + 28
        elif mark is False:
            p.append(f'<path d="M{x+11},{cy+11.5} q3,-3 6,0 q3,3 6,0 M{x+11},{cy+16} q3,-3 6,0 q3,3 6,0" fill="none" stroke="{MUTE}" stroke-width="1.6"/>'); tx = x + 28
        p.append(f'<text x="{tx}" y="{cy+16.5}" font-size="12.5" fill="{INK if mark is None else (TEAL if mark else MUTE)}" font-family="ui-monospace,Menlo,monospace">{esc(label)}</text>')
        x += cw + 10
    # mono path (bottom-right)
    p.append(f'<text x="{X+CW-20}" y="{Y+CH-16}" text-anchor="end" font-size="12.5" fill="{r["dark"]}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">{esc(r["mono"])}</text>')
    # connector arrow from the world artifact
    ay = Y + CH / 2
    p.append(f'<path d="M372 318 C 410 318, 414 {ay:.0f}, {CX-8} {ay:.0f}" fill="none" stroke="#1d4ed8" stroke-width="2.2" opacity="0.8" marker-end="url(#arr)"/>')
    return "\n  ".join(p)

def svg():
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="\'Iowan Old Style\',\'Palatino Linotype\',Palatino,Georgia,serif">']
    clips = "".join(f'<clipPath id="clip{i}"><rect x="{CX}" y="{CY+i*(CH+GAP)}" width="{CW}" height="{CH}" rx="17"/></clipPath>' for i in range(3))
    parts.append(f'''<defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fcfbf8"/><stop offset="1" stop-color="#eef0ec"/></linearGradient>
    <linearGradient id="acc" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#1d4ed8"/><stop offset="1" stop-color="#0891b2"/></linearGradient>
    <linearGradient id="gBuild" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#1d4ed8"/><stop offset="1" stop-color="#3b82f6"/></linearGradient>
    <linearGradient id="gOpt" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#b45309"/><stop offset="1" stop-color="#d97706"/></linearGradient>
    <linearGradient id="gDep" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#0f766e"/><stop offset="1" stop-color="#0891b2"/></linearGradient>
    <pattern id="grid" width="26" height="26" patternUnits="userSpaceOnUse"><path d="M26 0 H0 V26" fill="none" stroke="#8aa0c2" stroke-width="0.6" opacity="0.20"/></pattern>
    <filter id="sh" x="-40%" y="-40%" width="180%" height="180%"><feDropShadow dx="0" dy="6" stdDeviation="11" flood-color="#16202e" flood-opacity="0.15"/></filter>
    <marker id="arr" markerWidth="10" markerHeight="10" refX="6.5" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#1d4ed8"/></marker>
    {clips}
  </defs>''')
    parts.append(f'<rect width="{W}" height="{H}" fill="url(#bg)"/><rect width="{W}" height="{H}" fill="url(#grid)"/><rect x="0" y="0" width="{W}" height="6" fill="url(#acc)"/>')
    parts.append(f'<g stroke="#1d4ed8" stroke-width="1.5" opacity="0.45" fill="none"><path d="M28 44 L28 28 L44 28"/><path d="M{W-28} 44 L{W-28} 28 L{W-44} 28"/><path d="M28 {H-32} L28 {H-18} L44 {H-18}"/><path d="M{W-28} {H-32} L{W-28} {H-18} L{W-44} {H-18}"/></g>')

    # ---- world artifact (left) ----
    parts.append('<g filter="url(#sh)"><rect x="44" y="216" width="330" height="204" rx="18" fill="#f4f6fb"/></g>')
    parts.append('<rect x="44" y="216" width="330" height="204" rx="18" fill="none" stroke="#c7d2ea" stroke-width="1.6"/>')
    parts.append('''<g transform="translate(140,300)">
    <rect x="-40" y="-40" width="80" height="80" rx="14" fill="#ffffff" stroke="#1d4ed8" stroke-width="4.4"/>
    <rect x="-24" y="-24" width="48" height="48" rx="9" fill="none" stroke="#b45309" stroke-width="3.8"/>
    <rect x="-10" y="-10" width="20" height="20" rx="5" fill="#0f766e"/></g>''')
    parts.append(f'<text x="140" y="372" text-anchor="middle" font-size="19" font-weight="800" letter-spacing="1.5" fill="{INK}">ONE VERIFIED</text>')
    parts.append(f'<text x="140" y="396" text-anchor="middle" font-size="19" font-weight="800" letter-spacing="1.5" fill="{INK}">WORLD</text>')
    parts.append(f'<text x="209" y="248" font-size="13.5" fill="{MUTE}">a reusable artifact —</text>')
    parts.append(f'<text x="209" y="268" font-size="13.5" fill="{MUTE}">spend it any of 3 ways,</text>')
    parts.append(f'<text x="209" y="288" font-size="13.5" fill="{MUTE}">fine-tune or not.</text>')

    for i, r in enumerate(ROUTES):
        parts.append(card(r, i))

    # ---- hybrid ribbon ----
    ry = CY + 3 * (CH + GAP) - GAP + 10
    parts.append(f'<g filter="url(#sh)"><rect x="{CX}" y="{ry}" width="{CW}" height="40" rx="20" fill="#16202e"/></g>')
    parts.append(f'<text y="{ry+26}" font-size="14.5" text-anchor="middle">'
                 f'<tspan x="{CX+70}" font-weight="800" letter-spacing="1.5" fill="#f0d9b8">HYBRID</tspan>'
                 f'<tspan x="{CX+CW/2+40:.0f}" font-weight="600" fill="#eef1f6">tool makes exact data  ·  training amortizes it  ·  tool stays the exact oracle</tspan></text>')
    # little label on the left pointing to the ribbon
    parts.append(f'<text x="{CX-14}" y="{ry+26}" text-anchor="end" font-size="12.5" fill="{MUTE}">the powerful one:</text>')

    parts.append('</svg>')
    return "\n".join(parts)

def main():
    out_svg = os.path.join(ROOT, "assets/three-routes.svg")
    open(out_svg, "w").write(svg())
    print("wrote", out_svg)
    if cairosvg:
        cairosvg.svg2png(url=out_svg, write_to=os.path.join(ROOT, "assets/three-routes.png"),
                         output_width=W * 2, output_height=H * 2)
        print("wrote", os.path.join(ROOT, "assets/three-routes.png"))
    else:
        print("cairosvg unavailable — PNG not written")

if __name__ == "__main__":
    main()
