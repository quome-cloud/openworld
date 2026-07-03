"""README 'Empirical baselines' figure — generated from the real E36 results.

Two honest panels straight out of experiments/results/e36_representations.json:
  A. Exact accuracy on UNSEEN part-combinations (K=5): verified symbolic composition = 1.00
     with 0 training samples, while all nine monolithic learned families (trained on thousands
     of samples) collapse to <=0.20.
  B. The same, swept over task size K=2..5: symbolic stays flat at 1.0; the strongest learner
     (hist-grad-boost) erodes toward chance as the joint combination space grows.

  ~/.arcv/bin/python scripts/make_readme_baselines_fig.py   # -> assets/baselines.png

Brand palette matches assets/pipeline.svg (blue #1d4ed8, ochre #b45309, teal #0f766e).
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E36 = json.load(open(os.path.join(ROOT, "experiments/results/e36_representations.json")))

BG = "#fcfbf8"; INK = "#16202e"; MUTE = "#5b6675"
BLUE = "#1d4ed8"; OCHRE = "#b45309"; TEAL = "#0f766e"; GREY = "#aab3c0"; GRID = "#d7dce4"

# the nine MONOLITHIC learned families in the generalization leg (composite_* excluded), + labels
FAMILIES = [
    ("hist_grad_boost", "Hist grad boost"), ("monolith", "Monolithic MLP"),
    ("knn5", "k-NN (5)"), ("knn1", "k-NN (1)"), ("random_forest", "Random forest"),
    ("ridge", "Ridge"), ("svr", "SVR"), ("koopman", "Koopman (EDMD)"), ("gp", "Gaussian proc."),
]

def _serif():
    for name in ("Iowan Old Style", "Palatino Linotype", "Palatino", "Georgia"):
        if any(name.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Serif"

def main():
    plt.rcParams.update({"font.family": _serif(), "font.size": 11, "axes.edgecolor": "#c7ccd4",
                         "axes.linewidth": 1.0, "figure.facecolor": BG, "savefig.facecolor": BG})
    ks = {r["k"]: r for r in E36["leg_generalization"]}
    k5 = ks[5]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 4.9), gridspec_kw={"width_ratios": [1.32, 1]})
    fig.subplots_adjust(left=0.175, right=0.985, top=0.85, bottom=0.205, wspace=0.34)

    # ---- Panel A: horizontal bars at K=5 ------------------------------------------------
    rows = [("composite_symbolic", "OpenWorld — verified code", k5["composite_symbolic"]["acc"], BLUE)]
    rows += [(key, lab, k5[key]["acc"], GREY) for key, lab in FAMILIES]
    rows.sort(key=lambda r: r[2])                       # low -> high (bottom -> top)
    labels = [r[1] for r in rows]; vals = [r[2] for r in rows]; cols = [r[3] for r in rows]
    y = range(len(rows))
    axA.barh(list(y), vals, color=cols, height=0.66, zorder=3,
             edgecolor=[INK if c == BLUE else "none" for c in cols], linewidth=1.1)
    for yi, v, c in zip(y, vals, cols):
        axA.text(v + 0.018, yi, f"{v:.2f}", va="center", ha="left", fontsize=10,
                 color=INK if c == BLUE else MUTE, fontweight="bold" if c == BLUE else "normal", zorder=4)
    axA.set_yticks(list(y)); axA.set_yticklabels(labels, fontsize=10.5)
    for tick, c in zip(axA.get_yticklabels(), cols):      # bold + ink the OpenWorld row
        if c == BLUE: tick.set_color(INK); tick.set_fontweight("bold")
    axA.set_xlim(0, 1.12); axA.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axA.set_xlabel("exact accuracy on unseen combinations", fontsize=10.5, color=MUTE)
    axA.xaxis.grid(True, color=GRID, linewidth=0.9, zorder=0); axA.set_axisbelow(True)
    for s in ("top", "right", "left"): axA.spines[s].set_visible(False)
    axA.tick_params(length=0)
    axA.set_title("A. Generalize to unseen part-combinations  (E36, K=5)", fontsize=11.5,
                  loc="left", color=INK, fontweight="bold", pad=10)
    # training-data annotation
    axA.text(1.0, len(rows) - 1, "0 samples", va="center", ha="right", fontsize=9,
             color=BLUE, fontstyle="italic", transform=axA.transData)
    axA.annotate("thousands of\ntraining samples", xy=(0.20, 0.4), xytext=(0.52, 1.7),
                 fontsize=9, color=MUTE, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=GREY, lw=0.9))

    # ---- Panel B: decay over K ----------------------------------------------------------
    K = sorted(ks)
    sym = [ks[k]["composite_symbolic"]["acc"] for k in K]
    hgb = [ks[k]["hist_grad_boost"]["acc"] for k in K]
    mlp = [ks[k]["monolith"]["acc"] for k in K]
    axB.plot(K, sym, "-o", color=BLUE, lw=2.6, ms=7, zorder=4, label="OpenWorld (verified code)")
    axB.plot(K, hgb, "--o", color=OCHRE, lw=2.2, ms=6, zorder=3, label="best learned (hist grad boost)")
    axB.plot(K, mlp, ":o", color=GREY, lw=2.0, ms=5, zorder=2, label="monolithic MLP")
    for k, v in zip(K, sym): axB.text(k, v + 0.035, "1.00" if v == 1 else f"{v:.2f}",
                                      ha="center", fontsize=8.5, color=BLUE, fontweight="bold")
    axB.text(K[-1], hgb[-1] - 0.075, f"{hgb[-1]:.2f}", ha="center", fontsize=8.5, color=OCHRE)
    axB.set_ylim(-0.06, 1.12); axB.set_xticks(K)
    axB.set_xlabel("task size  K  (parts composed)", fontsize=10.5, color=MUTE)
    axB.set_ylabel("exact accuracy", fontsize=10.5, color=MUTE)
    axB.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0); axB.set_axisbelow(True)
    for s in ("top", "right"): axB.spines[s].set_visible(False)
    axB.tick_params(length=0)
    axB.legend(loc="center right", fontsize=8.8, frameon=False, bbox_to_anchor=(1.0, 0.52))
    axB.set_title("B. Learners erode as K grows — symbolic holds", fontsize=11.5,
                  loc="left", color=INK, fontweight="bold", pad=10)

    fig.text(0.175, 0.035, "Structure, not scale: verified symbolic composition needs only per-part "
             "marginals; the monolithic learners need the full joint and never see it.  "
             "Source: experiments/results/e36_representations.json",
             fontsize=8.0, color=MUTE, ha="left")

    out = os.path.join(ROOT, "assets/baselines.png")
    fig.savefig(out, dpi=200); print("wrote", out)

if __name__ == "__main__":
    main()
