"""Architecture figure: how OpenWorld worlds are used in the codex-steered autonomous deep search (E124).

A clean top-to-bottom control flow in the OpenWorld atlas aesthetic, colour-coded so a reader sees at a
glance which stages are OpenWorld primitives (teal/blue), which is the environment ground truth (ochre), and
which is the codex brain (indigo). The right rail spells out the four slots of an OpenWorld world model and
which stage fills each. Writes papers/assets/figs/e124_openworld_flow.png.

    ~/.arcv/bin/python scripts/make_e124_flow_fig.py
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = "/Users/jim/Desktop/openworld"
FIGS = os.path.join(ROOT, "papers/assets/figs"); os.makedirs(FIGS, exist_ok=True)

BG = "#FBFAF6"
# (fill, edge, text)
ENV = ("#FCEBD2", "#D97706", "#7C2D12")          # ochre  — ground truth
OW_T = ("#E4F1F4", "#0E7490", "#0E4A57")         # teal   — OpenWorld primitive
OW_B = ("#E7EDFB", "#1D4ED8", "#1E3A8A")         # blue   — OpenWorld primitive
CODEX = ("#ECEBFB", "#4F46E5", "#312E81")        # indigo — the codex brain (new)
ARROW = "#475569"
INK = "#1F2937"

fig, ax = plt.subplots(figsize=(13.6, 11.2))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 14); ax.set_ylim(-0.8, 12.2); ax.axis("off")


def box(x, y, w, h, title, sub, tag, colors, lw=2.0, fontsize=12.5):
    fill, edge, txt = colors
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.14", linewidth=lw,
                 facecolor=fill, edgecolor=edge, zorder=3))
    ax.text(x, y + h * 0.20, title, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=txt, zorder=4)
    if sub:
        ax.text(x, y - h * 0.12, sub, ha="center", va="center", fontsize=10.0, color=INK, zorder=4)
    if tag:
        ax.text(x, y - h * 0.36, tag, ha="center", va="center", fontsize=8.8, style="italic",
                color=edge, zorder=4)


def arrow(x1, y1, x2, y2, label=None, rad=0.0, color=ARROW, lw=2.2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=20,
                 linewidth=lw, color=color, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + (0.0 if rad == 0 else 0.5), my, label, ha="center", va="center",
                fontsize=8.6, color=color, style="italic",
                bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none"), zorder=5)


CX = 5.2            # main column centre
W, H = 6.4, 1.15

# ---- stages (top -> bottom) ----
box(CX, 11.0, W, H, "ARC-AGI-3 environment", "replay-only  ·  the GROUND TRUTH",
    "a level-up (levels_completed↑) = a verified solve", ENV)
box(CX, 9.35, W, H, "Perceptor", "mask the status bar  →  discrete state  state_key σ(s)",
    "OpenWorld: the perceive→world boundary  ·  e119/perceive", OW_T)
box(CX, 7.70, W, H, "Codex goal compiler   (source-free)",
    "frames + actions  →  ordered SUBGOALS  +  MACROS",
    "the brain — supplies the OBJECTIVE  ·  best-of-N, abstain  ·  E124 (design)", CODEX, lw=2.6,
    fontsize=13.5)
box(CX, 6.05, W, H, "Search   (subgoal hill-climb, macros = options)",
    "learns  (state, action) → next-state",
    "OpenWorld FunctionTransition  =  the world's DYNAMICS  ·  e119/planner + e124", OW_B)
box(CX, 4.40, W, H, "Surprise  →  replay-to-boundary",
    "rules changed → re-compile the goal for the new regime",
    "E122 (causal monitor)  ·  E123 (resynthesis)", OW_T)
box(CX, 2.75, W, H, "Compose the worlds",
    "PhasedTransition over regimes   ·   ConsensusTransition vote",
    "OpenWorld: self-rebuilding + multi-hypothesis  ·  E120 / E121 / E123", OW_B)
box(CX, 1.15, W, H, "Emit",
    "to_spec → preview.graph (MAP) → render_card (ATLAS)",
    "OpenWorld: inspectable in  openworld serve /view", OW_T)

# ---- straight-down arrows ----
for y1, y2 in [(10.42, 9.93), (8.77, 8.28), (7.12, 6.63), (5.47, 4.98), (3.82, 3.33), (2.17, 1.73)]:
    arrow(CX, y1, CX, y2)

# ---- verification loop: search <-> env (left side) ----
arrow(CX - W / 2 - 0.05, 6.05, CX - W / 2 - 1.15, 8.0, rad=-0.35, color="#0E7490", lw=2.0)
arrow(CX - W / 2 - 1.15, 9.9, CX - W / 2 - 0.05, 10.7, rad=-0.35, color="#0E7490", lw=2.0)
ax.text(1.25, 8.2, "act  ↑\nverify ↓\n(env decides)", ha="center", va="center", fontsize=9.0,
        color="#0E7490", style="italic", zorder=6)

# ---- right rail: the four slots of an OpenWorld world ----
rx, rw = 11.6, 4.0
ax.add_patch(FancyBboxPatch((rx - rw / 2, 1.6), rw, 8.6, boxstyle="round,pad=0.03,rounding_size=0.12",
             linewidth=1.6, facecolor="#FFFFFF", edgecolor="#94A3B8", zorder=1))
ax.text(rx, 9.75, "An OpenWorld world =", ha="center", fontsize=12.5, fontweight="bold", color=INK)
slots = [("Perceptor", "input → discrete state", OW_T, "σ(s)"),
         ("Dynamics", "state,action → next", OW_B, "FunctionTransition"),
         ("Objective", "what counts as winning", CODEX, "codex subgoals  ← NEW"),
         ("Compose", "regimes & hypotheses", OW_B, "Phased / Consensus"),
         ("Emit", "serialize → map / atlas", OW_T, "to_spec · render_card")]
yy = 8.7
for name, desc, (fill, edge, txt), prim in slots:
    ax.add_patch(FancyBboxPatch((rx - rw / 2 + 0.25, yy - 0.55, ), 0.5, 0.5,
                 boxstyle="round,pad=0.01,rounding_size=0.08", facecolor=fill, edgecolor=edge,
                 linewidth=1.8, zorder=2))
    ax.text(rx - rw / 2 + 0.95, yy - 0.18, name, ha="left", va="center", fontsize=11,
            fontweight="bold", color=txt)
    ax.text(rx - rw / 2 + 0.95, yy - 0.46, desc, ha="left", va="center", fontsize=8.6, color=INK)
    ax.text(rx - rw / 2 + 0.30, yy - 0.86, prim, ha="left", va="center", fontsize=8.4,
            style="italic", color=edge)
    yy -= 1.45

# ---- legend chips (bottom) ----
chips = [("environment (ground truth)", ENV), ("OpenWorld primitive", OW_T),
         ("codex brain (new)", CODEX)]
cx0 = 1.3
for label, (fill, edge, _) in chips:
    ax.add_patch(FancyBboxPatch((cx0, -0.45, ), 0.42, 0.32, boxstyle="round,pad=0.01,rounding_size=0.06",
                 facecolor=fill, edgecolor=edge, linewidth=1.8, zorder=4))
    ax.text(cx0 + 0.55, -0.29, label, ha="left", va="center", fontsize=9.6, color=INK, zorder=4)
    cx0 += 0.7 + len(label) * 0.125

ax.text(7.0, 11.78, "The ARC-AGI-3 solver IS an OpenWorld world model — codex supplies the objective that "
        "blind search lacked", ha="center", fontsize=13.5, fontweight="bold", color=INK)

out = os.path.join(FIGS, "e124_openworld_flow.png")
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("wrote", out)
