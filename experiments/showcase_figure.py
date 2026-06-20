"""Render the showcase world's spec as the framework's legend figure:
external inputs -> perceptors -> worlds (composite) -> emit/act, with objectives as a
parallel steering branch. Generated from the real spec (showcase_world), so labels can't
drift from the code.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

TEAL, BLUE, OCHRE, INK, GRAY = "#0d8a8a", "#1f5fbf", "#c8861a", "#1b1b1b", "#8a8a8a"
VIOLET, COUPLE = "#6a4c93", "#15407f"   # steer (objectives/dials); world-to-world coupling
# Input-chip icon: by perceptor KIND first (so the two text perceptors differ), else modality.
KIND_ICON = {"JSONPerceptor": "{ }", "CodePerceptor": "</>"}
MOD_ICON = {"text": "txt", "image": "img", "audio": "wav", "graph": "dag",
            "video_frame": "vid", "video_segment": "vid"}


def _join(tokens, maxlen=20):
    """Token-aware truncation: keep whole comma-separated tokens, never break mid-word."""
    out = ""
    for t in tokens:
        cand = t if not out else out + "," + t
        if len(cand) > maxlen:
            return (out + ",…") if out else t[:maxlen]
        out = cand
    return out


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

    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    # column headers (Objectives is no longer a column -- it is a parallel branch below)
    for x, t in [(0.55, "External inputs"), (2.5, "Perceptors"), (5.45, "Worlds (composite)"),
                 (9.9, "Emit"), (11.85, "Act")]:
        ax.text(x, 4.78, t, ha="center", fontsize=9.5, color=INK, weight="bold")
    ax.text(2.5, 4.55, "one real perceptor per modality -- an open set",
            ha="center", fontsize=7, color=GRAY, style="italic")

    # child-world layout, and a var -> child-row map so perceptor arrows reach their target
    cys = [2.7, 1.4]
    varmap = {}
    for (nm, ch), cy in zip(children, cys):
        for v in (ch.get("initial_state", {}) or {}):
            varmap.setdefault(v, cy)

    # external inputs + perceptors (one row each); arrows fan in to the consuming child
    n = max(len(perc), 1)
    ys = [3.85 - i * (3.4 / n) for i in range(n)]
    for p, y in zip(perc, ys):
        mod = p.get("modality", "text")
        kind = p.get("kind", "perceptor")
        icon = KIND_ICON.get(kind) or MOD_ICON.get(mod, "?")
        ax.text(0.55, y + 0.18, icon, ha="center", va="center", fontsize=9.5, color=INK,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", fc="#eef2f7", ec=GRAY, lw=0.8))
        _arrow(ax, 0.9, y + 0.18, 1.45, y + 0.18)
        prod = _join(p.get("produces", []) or [kind])
        _box(ax, 1.55, y, 1.95, 0.36, f"{kind}\n[{mod}] →{prod}", TEAL, fs=7.0)
        tgt = [varmap[v] for v in (p.get("produces") or []) if v in varmap]
        ty = sum(tgt) / len(tgt) if tgt else 3.55   # graph/schema declares -> container top
        _arrow(ax, 3.5, y + 0.18, 4.05, ty, color=TEAL)

    # composite container + child worlds + bridge (a coupling, not an output) + aggregator
    ax.add_patch(FancyBboxPatch((4.05, 0.7), 2.8, 3.2, boxstyle="round,pad=0.01,rounding_size=0.06",
                                linewidth=1.4, edgecolor=BLUE, facecolor="#eaf1fb", zorder=1))
    ax.text(5.45, 3.72, f"composite: {spec.get('name', '')}", ha="center", fontsize=8,
            color=BLUE, style="italic")
    for (nm, ch), cy in zip(children, cys):
        kind = (ch.get("transition", {}) or {}).get("kind", "code")
        _box(ax, 4.35, cy, 2.2, 0.6, f"{nm}\n(transition: {kind})", BLUE, fs=8)
    if comp.get("bridges"):
        _arrow(ax, 5.45, cys[0], 5.45, cys[1] + 0.6, color=COUPLE, lw=1.6, style="<|-|>")
        ax.text(5.38, 2.35, "bridge", ha="right", va="center", fontsize=7, color=COUPLE)
    if comp.get("aggregators"):
        ax.text(5.45, 0.95, "▲ aggregator: "
                + ",".join(a.get("name", "") for a in comp["aggregators"]),
                ha="center", fontsize=7, color=BLUE)

    # emit reads WORLD STATE (from the composite), not objectives -> report + tool-calls -> act
    eys = [2.75, 1.95]
    ax.text(7.9, 2.66, "state", fontsize=6.5, color=BLUE, style="italic")
    for e, ey in zip(emit, eys):
        _arrow(ax, 6.85, 2.4, 9.0, ey + 0.18, color=BLUE)
        fields = _join(e.get("fields", []) or ["report"])
        _box(ax, 9.0, ey, 1.85, 0.36, f"{e.get('kind', 'emit')}\n→{fields}", OCHRE, fs=7.0)
        _arrow(ax, 10.85, ey + 0.18, 11.3, 2.4, color=OCHRE)
    _box(ax, 11.3, 2.05, 1.1, 0.7, "report\n+ tools", INK, fs=8)

    # objectives + dial: a PARALLEL steering branch off world state (terminal, not upstream)
    names = ", ".join(o.get("name", "") for o in objs)
    dial = next((o["weight"].name for o in objs if hasattr(o.get("weight"), "name")), None)
    otext = "objectives: " + names + (f"\n◐ dial: {dial}  (retune at inference)" if dial else "")
    _arrow(ax, 6.4, 0.7, 7.0, 1.05, color=VIOLET)
    ax.text(6.62, 1.0, "steer", fontsize=6.6, color=VIOLET, style="italic", ha="left")
    _box(ax, 7.0, 0.42, 2.4, 0.78, otext, VIOLET, fs=7.0, round=0.06)

    # legend: color semantics (color-blind redundancy) + the ->x notation
    lx, ly = 0.45, 0.2
    for c, lab in [(TEAL, "perceive"), (BLUE, "world state"), (OCHRE, "emit"),
                   (VIOLET, "steer"), (COUPLE, "couple")]:
        ax.add_patch(plt.Rectangle((lx, ly), 0.16, 0.16, facecolor=c, edgecolor="none", zorder=3))
        ax.text(lx + 0.22, ly + 0.08, lab, fontsize=6.6, va="center", color=INK, zorder=3)
        lx += 0.62 + 0.072 * len(lab)
    ax.text(lx + 0.05, ly + 0.08, "·  →x : writes state var x",
            fontsize=6.6, va="center", color=GRAY)

    fig.suptitle("A code world model in OpenWorld: perceive → world → emit → act  "
                 "(every box is a few lines you write)", fontsize=10.5, x=0.5, y=0.99)
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
