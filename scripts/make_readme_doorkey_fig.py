"""README figure: steps-to-solve MiniGrid DoorKey — OpenWorld vs DreamerV3 vs V-JEPA-2.

Reads experiments/results/e65_minigrid_bench.json (the single source of truth for
E65) and writes assets/doorkey_bench.png, embedded in the README's "Empirical
baselines" section. Rerun after E65 changes:

    python scripts/make_readme_doorkey_fig.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "experiments/results/e65_minigrid_bench.json").read_text())

ow_steps = d["openworld"]["plan_length"]                     # 11
dr_first = d["dreamerv3"]["steps_to_first_solve"]            # 9640
dr_curve = d["dreamerv3"]["curve"]
dr_reliable = next(c["step"] for c in dr_curve if c["solve_rate"] >= 1.0)

BLUE, OCHRE, TEAL, GREY = "#1d4ed8", "#d97706", "#0d9488", "#6b7280"

fig, (ax, ax2) = plt.subplots(
    2, 1, figsize=(10, 10), dpi=160,
    gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.42})
fig.patch.set_facecolor("white")

# ---- Top: log-scale bar chart, steps to solve ----
labels = ["OpenWorld\nverified-code world model\n(zero training data)",
          "DreamerV3\n(pixels, A100)",
          "V-JEPA-2\n(frozen features)"]
vals = [ow_steps, dr_first, np.nan]
y = np.arange(3)[::-1]

ax.barh(y[0], vals[0], color=TEAL, height=0.55, zorder=3)
ax.barh(y[1], vals[1], color=OCHRE, height=0.55, zorder=3)
ax.set_xscale("log")
ax.set_xlim(1, 4e5)
ax.set_ylim(-0.9, 2.55)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=12.5)
ax.set_xlabel("environment interactions to solve DoorKey (log scale)", fontsize=12)
ax.tick_params(axis="x", labelsize=11)
ax.grid(axis="x", which="both", color="#e5e7eb", zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

ax.text(vals[0] * 1.4, y[0], f"{ow_steps} steps — one optimal plan, zero-shot",
        va="center", fontsize=13, fontweight="bold", color=TEAL)
ax.text(vals[1] * 1.15, y[1],
        f"{dr_first:,} steps to first solve\n(~{dr_reliable // 1000}k to 100% reliable)",
        va="center", fontsize=12.5, fontweight="bold", color=OCHRE)
ax.text(1.3, y[2] + 0.18,
        "no controller we tried could solve it\n(great features, no usable dynamics for control)",
        va="center", fontsize=12, color=GREY, style="italic")

ratio = dr_first // ow_steps
ax.set_title(
    f"MiniGrid DoorKey: {ratio:,}× fewer interactions\n"
    "a truck at 3 mpg vs. an EV at 200+ MPGe — and the analogy undersells it",
    fontsize=15.5, fontweight="bold", loc="left", pad=14)

# ---- Bottom: DreamerV3 learning curve vs the zero-shot line ----
steps = [c["step"] for c in dr_curve]
solve = [c["solve_rate"] for c in dr_curve]
ax2.plot([0] + steps, [0] + solve, color=OCHRE, lw=3, marker="o", ms=5,
         label="DreamerV3 solve rate", zorder=3)
ax2.axhline(1.0, color=TEAL, lw=3, zorder=2)
ax2.text(2500, 1.035,
         f"OpenWorld: 100% from step 0 (verified {ow_steps}-step plan, no training)",
         color=TEAL, fontsize=12, fontweight="bold")
ax2.axvline(dr_first, color=GREY, lw=1, ls="--")
ax2.text(dr_first * 1.15, 0.08, f"first Dreamer solve:\nstep {dr_first:,}",
         fontsize=10.5, color=GREY)
ax2.set_xlim(0, 180000)
ax2.set_ylim(0, 1.12)
ax2.set_xlabel("environment steps", fontsize=12)
ax2.set_ylabel("solve rate", fontsize=12)
ax2.tick_params(labelsize=11)
ax2.xaxis.set_major_formatter(lambda x, _: f"{int(x / 1000)}k" if x else "0")
ax2.grid(color="#e5e7eb", zorder=0)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.legend(loc="lower right", fontsize=11, frameon=False)

fig.text(0.01, 0.012,
         "E65 (experiments/e65_minigrid_bench.py) · OpenWorld transition validated bit-exact vs Farama "
         "MiniGrid (600/600) · DreamerV3 from 48×48 RGB on A100 · V-JEPA-2 vitl-fpc64-256 frozen",
         fontsize=8.5, color=GREY)

out = ROOT / "assets/doorkey_bench.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(out)
