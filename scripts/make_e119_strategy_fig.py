"""E119 method figure: the MSA (arXiv 2507.12547) Bayesian ideas made concrete in an OpenWorld solver.

Flow: ARC frame -> OpenWorld PERCEPTORS -> BAYESIAN SUBGOAL (on-demand model synthesis + posterior over
predicate-programs + tau-abstain) -> ENV-GROUND-TRUTH SEARCH (ordered by the synthesized model) ->
replay-verify + EMIT as an openworld.World map. Highlights: OpenWorld primitives (teal) and the MSA
Bayesian core (indigo). Writes papers/arc-3/figs/e119_msa_strategy.{png,pdf}. Run with a python w/ matplotlib.
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

FIG = Path("papers/arc-3/figs"); FIG.mkdir(parents=True, exist_ok=True)

# palette (shared with arc3_architecture.png) + an indigo accent for the MSA Bayesian core
OW_FILL="#ccfbf1"; OW_EDGE="#0d9488"; OW_DARK="#0f766e"          # OpenWorld primitive (teal)
ENV_FILL="#e2e8f0"; ENV_EDGE="#64748b"                            # environment / data (slate)
RES_FILL="#dcfce7"; RES_EDGE="#16a34a"; RES_DARK="#166534"        # result (green)
OUT_FILL="#fef3c7"; OUT_EDGE="#d97706"; OUT_DARK="#92400e"        # map output (ochre)
MSA_FILL="#eef2ff"; MSA_EDGE="#4f46e5"; MSA_DARK="#3730a3"        # MSA Bayesian core (indigo)
INK="#0f172a"; SUB="#334155"

fig, ax = plt.subplots(figsize=(14.6, 7.7)); ax.set_xlim(0, 14.6); ax.set_ylim(0, 7.7); ax.axis("off")


def box(x, y, w, h, title, sub, fill, edge, tcol=INK, lw=1.8, fs=10, badge=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                 linewidth=lw, edgecolor=edge, facecolor=fill, mutation_scale=1))
    ax.text(x+w/2, y+h-0.26, title, ha="center", va="top", fontsize=fs, fontweight="bold", color=tcol)
    if sub:
        ax.text(x+w/2, y+h-0.26-0.40, sub, ha="center", va="top", fontsize=fs-2.6, color=SUB)
    if badge:
        ax.text(x+0.13, y+0.12, badge, ha="left", va="bottom", fontsize=6.6, style="italic", color=edge)


def arrow(x1, y1, x2, y2, col="#475569", lw=2.0, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=15, lw=lw,
                 color=col, shrinkA=3, shrinkB=3, connectionstyle=f"arc3,rad={rad}"))


# ── title ──────────────────────────────────────────────────────────────────────
ax.text(7.3, 7.46, "On-demand model synthesis for ARC-AGI-3: the MSA Bayesian loop inside an OpenWorld solver",
        ha="center", fontsize=14.5, fontweight="bold", color=INK)
ax.text(7.3, 7.12, "synthesize a goal-program on demand · keep a posterior over programs · abstain under "
        "uncertainty · let the replay-only env be ground truth",
        ha="center", fontsize=9.3, color=SUB, style="italic")

# ── Stage 1: ARC env frame ─────────────────────────────────────────────────────
box(0.15, 3.05, 1.75, 1.9, "ARC-AGI-3", "64×64×16 grid\nlevels_completed\n(replay-only env)",
    ENV_FILL, ENV_EDGE, fs=10)
rng = np.array([[0,1,0,2,0,0],[0,0,3,0,0,5],[4,0,0,0,0,0],[0,0,6,0,7,0],[0,8,0,0,0,0],[0,0,0,9,0,0]])
ax.imshow(rng, extent=(0.42, 1.63, 3.18, 3.92), cmap="tab10", vmin=0, vmax=9, aspect="auto", zorder=5)

# ── Stage 2: OpenWorld perceptors (perceive→world boundary) ─────────────────────
px, pw = 2.15, 3.05
box(px, 2.35, pw, 3.3, "Perceptors",
    "status_mask · zero the\nalways-changing status bar\n\nstate_key σ(s) · masked-frame\nhash = STATE IDENTITY\n\n"
    "object_json · relational scene\nclick_candidates · sprite targets\nprobe · 1-step transitions",
    OW_FILL, OW_EDGE, badge="OpenWorld", fs=10)
ax.text(px+pw/2, 2.18, "perceive → world boundary", ha="center", fontsize=7.6, color=OW_DARK, style="italic")
arrow(1.9, 4.0, px, 4.0)

# ── Stage 3: Bayesian subgoal — the MSA core ───────────────────────────────────
bx, bw = 5.5, 3.95
box(bx, 1.55, bw, 4.45, "Bayesian subgoal  ·  MSA core",
    "synthesize:  SLM samples N goal\nPROGRAMS  πᵢ ~ p(π | object_json)\nπ ∈ {reach · count · align} →\ncompile_predicate (executable)",
    MSA_FILL, MSA_EDGE, tcol=MSA_DARK, lw=2.6, badge="arXiv 2507.12547", fs=10.5)

# posterior-over-programs bar inset (cluster mass = p̂(π|obs); behavior = effect on frames)
ax.text(bx+0.28, 3.62, "posterior by BEHAVIOR  (cluster mass = p̂(π|obs))", ha="left", fontsize=7.7,
        color=MSA_DARK, fontweight="bold")
bars = [("reach(5)", 2.55, MSA_EDGE), ("count(2,≥,3)", 1.35, "#818cf8"), ("align(1,4)", 0.55, "#c7d2fe")]
by = 3.18
for lab, wbar, col in bars:
    ax.add_patch(Rectangle((bx+1.55, by-0.13), wbar, 0.24, facecolor=col, edgecolor=MSA_DARK, lw=0.8, zorder=4))
    ax.text(bx+1.5, by, lab, ha="right", va="center", fontsize=7.4, color=INK)
    by -= 0.42
# tau gate line + argmax
ax.plot([bx+1.55+1.95, bx+1.55+1.95], [3.30, 1.96], ls=(0,(3,2)), color="#ef4444", lw=1.4, zorder=6)
ax.text(bx+1.55+1.97, 1.86, "τ", ha="center", va="top", fontsize=8.5, color="#ef4444", fontweight="bold")
arrow(bx+1.55+2.55, 3.18, bx+3.7, 3.18, col=MSA_DARK, lw=1.6)
ax.text(bx+bw-0.12, 3.40, "argmax\n= π*", ha="right", va="bottom", fontsize=7.2, color=MSA_DARK, fontweight="bold")
ax.text(bx+bw/2, 1.78, "τ-GATE:  top mass ≥ τ → π*   else  ABSTAIN (search runs unguided)",
        ha="center", fontsize=7.6, color=MSA_DARK)
arrow(1.0, 3.05, bx+0.6, 1.55, col=OW_DARK, lw=1.4, rad=-0.28)   # object_json feeds the proposer
ax.text(3.1, 1.5, "object_json", ha="center", fontsize=7.0, color=OW_DARK, style="italic")

# ── Stage 4: env-ground-truth search (correctness lives here) ──────────────────
sx, sw = 9.85, 4.55
box(sx, 4.05, sw, 1.95, "Env-ground-truth search",
    "search_level: BFS  or  BEST-FIRST ordered by π*'s score_fn ·\nevery node = action prefix REPLAYED from reset() ·\n"
    "a child that raises levels_completed ⇒ level solved",
    RES_FILL, RES_EDGE, tcol=RES_DARK, fs=9.6)
arrow(bx+bw, 4.4, sx, 4.7, col=MSA_DARK)
ax.text((bx+bw+sx)/2, 4.95, "π* → score_fn\n(orders search)", ha="center", fontsize=7.2, color=MSA_DARK, style="italic")
arrow(1.0, 3.05, sx, 4.35, col=ENV_EDGE, lw=1.5, rad=-0.34)      # env replay = ground truth
ax.text(6.0, 6.18, "replay = ground truth", ha="center", fontsize=7.2, color=ENV_EDGE, style="italic")

# ── Stage 5: emit as an openworld.World ────────────────────────────────────────
box(sx, 1.05, sw, 2.35, "Replay-verify → bank → emit  openworld.World",
    "state = σ(s) ·  learned (σ(s),a)→(σ(s′),levels) table = FunctionTransition\n"
    "induced reward = levels_completed ↑  (CodeObjective)\n"
    "to_spec → preview.graph = the MAP · render_card = atlas · serve /view",
    OUT_FILL, OUT_EDGE, tcol=OUT_DARK, badge="OpenWorld", fs=9.6)
arrow(sx+sw/2, 4.05, sx+sw/2, 3.40, col=RES_DARK)
ax.text(sx+sw/2, 3.72, "verified action sequence", ha="center", fontsize=7.2, color=RES_DARK, style="italic")

# ── invariant banner ───────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.15, 0.12), 14.3, 0.66, boxstyle="round,pad=0.02,rounding_size=0.08",
             linewidth=1.6, edgecolor=MSA_EDGE, facecolor="#fafaff"))
ax.text(7.3, 0.45, "INVARIANT:  the synthesized model only ORDERS search — the replay-only env decides "
        "correctness.   A wrong or abstained subgoal costs SPEED, never a false solve   (so  slm-rung ⊒ search-rung control).",
        ha="center", va="center", fontsize=9.0, color=MSA_DARK, fontweight="bold")

# ── legend ─────────────────────────────────────────────────────────────────────
lx = 0.3
for lab, fc, ec in [("OpenWorld primitive", OW_FILL, OW_EDGE), ("MSA Bayesian core", MSA_FILL, MSA_EDGE),
                    ("environment", ENV_FILL, ENV_EDGE), ("search", RES_FILL, RES_EDGE), ("World map", OUT_FILL, OUT_EDGE)]:
    ax.add_patch(FancyBboxPatch((lx, 6.62), 0.30, 0.24, boxstyle="round,pad=0.01", facecolor=fc, edgecolor=ec, lw=1.4))
    ax.text(lx+0.38, 6.74, lab, ha="left", va="center", fontsize=7.8, color=SUB); lx += 2.18

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
for ext, dpi in [("png", 200), ("pdf", None)]:
    plt.savefig(FIG/f"e119_msa_strategy.{ext}", dpi=dpi, bbox_inches="tight")
plt.close()
print("wrote", FIG/"e119_msa_strategy.png", "and .pdf")
