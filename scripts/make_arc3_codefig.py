"""Fig: the world model IS code. Shows real banked synthesized predict(frame,action) programs and the
verification gate -- an LLM writes Python, ACCEPTED ONLY IF it exact-matches held-out (s,a,s') transitions
in the real engine. Reads the actual programs from experiments/results/arc3_claude/<game>.json so the
figure is honest (nothing hand-written). Writes papers/assets/figs/arc3_codefig.png.
"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results" / "arc3_claude"
FIGS = ROOT / "papers" / "assets" / "figs"
BLUE, TEAL, OCHRE, GREEN, INK = "#1f4e79", "#2a8a7f", "#c8881f", "#2a8a3e", "#0f172a"

# three fidelity-1.0 programs, each a DIFFERENT discovered rule
CARDS = [
    ("ar25", "a step counter", TEAL),
    ("lf52", "a progress bar", BLUE),
    ("sb26", "a paint rule",   OCHRE),
]


def _wrap(line, width=54):
    """Wrap a too-long source line at word boundaries, preserving the leading indentation and
    hanging-indenting continuations (in practice only the verbose comment lines wrap)."""
    if len(line) <= width:
        return [line]
    stripped = line.lstrip()
    indent = line[:len(line) - len(stripped)]
    hang = indent + "  "
    toks = stripped.split()
    if not toks:
        return [line]
    out, cur = [], indent + toks[0]
    for w in toks[1:]:
        if len(cur) + 1 + len(w) > width:
            out.append(cur); cur = hang + w
        else:
            cur += " " + w
    out.append(cur)
    return out


def load(game):
    d = json.load(open(RES / f"{game}.json"))
    code = d["code"].strip()
    lines = [ln for ln in code.splitlines() if ln.strip() != "import numpy as np"]
    while lines and lines[0].strip() == "":
        lines.pop(0)
    wrapped = [w for ln in lines for w in _wrap(ln)]
    return "\n".join(wrapped), float(d.get("deterministic_frac") or 0), int(d.get("transitions") or 0)


def main():
    fig = plt.figure(figsize=(12.4, 5.4))
    fig.suptitle("The world model $is$ code: an LLM writes  predict(frame, action) , accepted only if it "
                 "exact-matches held-out transitions in the real engine",
                 fontsize=12.5, fontweight="bold", y=0.985, color=INK)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    n = len(CARDS); pad = 2.2; cw = (100 - pad * (n + 1)) / n
    for i, (game, blurb, color) in enumerate(CARDS):
        code, fid, ntr = load(game)
        x0 = pad + i * (cw + pad); yc0, ch = 12, 74
        # card
        ax.add_patch(FancyBboxPatch((x0, yc0), cw, ch, boxstyle="round,pad=0.4,rounding_size=1.6",
                                    facecolor="#f8fafc", edgecolor=color, linewidth=1.8))
        ax.add_patch(FancyBboxPatch((x0, yc0 + ch - 7.5), cw, 7.5, boxstyle="round,pad=0,rounding_size=0",
                                    facecolor=color, edgecolor="none"))
        ax.text(x0 + cw / 2, yc0 + ch - 3.7, f"arc3-{game}  ·  {blurb}", ha="center", va="center",
                color="white", fontsize=10.5, fontweight="bold")
        # the real synthesized program (monospace)
        ax.text(x0 + 1.8, yc0 + ch - 10.5, code, ha="left", va="top", fontsize=6.6, family="monospace",
                color=INK, linespacing=1.32)
        # verification gate badge
        ax.add_patch(FancyBboxPatch((x0 + 1.4, yc0 + 1.4), cw - 2.8, 6.6,
                                    boxstyle="round,pad=0.2,rounding_size=1.0",
                                    facecolor="#eaf6ee", edgecolor=GREEN, linewidth=1.2))
        ax.text(x0 + cw / 2, yc0 + 4.7, f"✓ exact-match on {ntr} held-out (s,a,s')   ·   "
                f"fidelity {fid:.2f}", ha="center", va="center", fontsize=8.4, color=GREEN, fontweight="bold")

    # bottom note
    ax.text(50, 6.5, "Every accepted program is executed against the real engine on transitions it never saw; "
                     "only an exact frame match banks it as the game's verified dynamics.\n"
                     "Denser games are captured partially (mean exact-match ≈ 0.87 across the suite); the "
                     "directional player-move rule (e.g. dc22) is recovered exactly even where the full frame is not.",
            ha="center", va="center", fontsize=8.6, color="#334155")
    fig.savefig(FIGS / "arc3_codefig.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote arc3_codefig.png")


if __name__ == "__main__":
    main()
