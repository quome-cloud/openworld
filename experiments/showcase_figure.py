"""Render the showcase world's spec as the framework's legend figure:
external inputs -> perceptors -> worlds (composite) -> emit -> act.
Generated from the real spec (showcase_world), so labels can't drift from the code.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

TEAL, BLUE, OCHRE, INK, GRAY = "#0d8a8a", "#1f5fbf", "#c8861a", "#1b1b1b", "#8a8a8a"
MOD_ICON = {"text": "{ }", "image": "img", "audio": "wav", "video_frame": "vid", "video_segment": "vid"}


def _box(ax, x, y, w, h, text, fc, ec=None, fs=8.5, tc="white", round=0.04):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.005,rounding_size={round}",
                                linewidth=1.1, edgecolor=ec or fc, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3)


def _arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11,
                                 lw=lw, color=color, zorder=1, shrinkA=2, shrinkB=2))


def render(spec, path):
    perc = spec.get("perception", [])
    comp = spec.get("composite", {})
    children = list(comp.get("children", {}).items())   # {name: childspec}
    emit = spec.get("emit", [])
    objs = spec.get("objectives", [])

    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.8); ax.axis("off")

    # column headers
    for x, t in [(0.55, "External inputs"), (2.5, "Perceptors"), (5.4, "Worlds (composite)"),
                 (7.85, "Objectives & dials"), (9.85, "Emit"), (11.7, "Act")]:
        ax.text(x, 4.55, t, ha="center", fontsize=9.5, color=INK, weight="bold")

    n = max(len(perc), 1)
    ys = [3.7 - i * (3.3 / max(n, 1)) for i in range(n)]
    # inputs + perceptors
    for p, y in zip(perc, ys):
        mod = p.get("modality", "text")
        ax.text(0.55, y + 0.18, MOD_ICON.get(mod, "?"), ha="center", va="center", fontsize=10,
                color=INK, family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", fc="#eef2f7", ec=GRAY, lw=0.8))
        _arrow(ax, 0.9, y + 0.18, 1.45, y + 0.18)  # centered in the input->perceptor gap
        prod = ",".join(p.get("produces", []) or [p.get("kind", "")])[:18]
        _box(ax, 1.55, y, 1.9, 0.36, f"{p.get('kind','perceptor')}\n[{mod}] →{prod}", TEAL, fs=7.0)
        _arrow(ax, 3.45, y + 0.18, 4.05, 2.4, color=TEAL)

    # composite container + child worlds + bridge + aggregator
    ax.add_patch(FancyBboxPatch((4.05, 0.7), 2.8, 3.2, boxstyle="round,pad=0.01,rounding_size=0.06",
                                linewidth=1.4, edgecolor=BLUE, facecolor="#eaf1fb", zorder=1))
    ax.text(5.45, 3.72, f"composite: {spec.get('name','')}", ha="center", fontsize=8, color=BLUE, style="italic")
    cys = [2.7, 1.4]
    for (nm, ch), cy in zip(children, cys):
        kind = (ch.get("transition", {}) or {}).get("kind", "code")
        _box(ax, 4.35, cy, 2.2, 0.7, f"{nm}\n(transition: {kind})", BLUE, fs=8)
    if comp.get("bridges"):
        _arrow(ax, 5.45, cys[0], 5.45, cys[1] + 0.7, color=OCHRE, lw=1.6)
        ax.text(5.38, 2.4, "bridge", ha="right", va="center", fontsize=7, color=OCHRE)
    if comp.get("aggregators"):
        ax.text(5.45, 0.95, "▲ aggregator: " + ",".join(a.get("name", "") for a in comp["aggregators"]),
                ha="center", fontsize=7, color=BLUE)

    # objectives + dials (the steerable scoring layer)
    VIOLET = "#6a4c93"
    names = ", ".join(o.get("name", "") for o in objs)
    dial = next((o["weight"].name for o in objs if hasattr(o.get("weight"), "name")), None)
    otext = "objectives:\n" + names + (f"\n◐ dial: {dial}" if dial else "")
    _arrow(ax, 6.85, 2.4, 7.15, 2.4, color=BLUE)
    _box(ax, 7.15, 1.9, 1.55, 1.0, otext, VIOLET, fs=7.5, round=0.06)

    # emit + act
    eys = [2.7, 1.4]
    for e, ey in zip(emit, eys):
        _arrow(ax, 8.7, 2.4, 8.95, ey + 0.18, color=VIOLET)
        _box(ax, 8.95, ey, 1.85, 0.36, f"{e.get('kind','emit')}\n→{','.join(e.get('fields',[]) or ['report'])[:16]}",
             OCHRE, fs=7.0)
        _arrow(ax, 10.8, ey + 0.18, 11.3, 2.4, color=OCHRE)
    _box(ax, 11.3, 2.05, 1.05, 0.7, "report\n+ tools", INK, fs=8)

    fig.suptitle("A code world model in OpenWorld: perceive → world → emit → act  (every box is a few lines you write)",
                 fontsize=10.5, x=0.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from showcase_world import build_showcase_world
    from openworld.spec import to_spec
    render(to_spec(build_showcase_world()), "/tmp/showcase.png")
    print("wrote /tmp/showcase.png")
