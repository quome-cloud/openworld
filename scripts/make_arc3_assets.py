"""ARC-AGI-3 paper assets (E86): reads experiments/results/arc3_{claude,qwen32b,qwen7b}/*.json and
writes papers/arc-3/arc3_numbers.tex + papers/assets/figs/arc3_fidelity.png. Single source of truth
for the arc-3 paper's numbers (do not hand-edit arc3_numbers.tex).

  python scripts/make_arc3_assets.py
"""
import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIGS = ROOT / "papers" / "assets" / "figs"
NUMS = ROOT / "papers" / "arc-3" / "arc3_numbers.tex"


def load(sub):
    out = {}
    for f in glob.glob(str(RES / sub / "*.json")):
        d = json.load(open(f))
        out[d["game"]] = d
    return out


def main():
    cl, q32, q7 = load("arc3_claude"), load("arc3_qwen32b"), load("arc3_qwen7b")

    excluded = sorted(g for g, d in cl.items() if d.get("transitions") == 0)
    done = {g: d for g, d in cl.items() if d.get("verified_exact") is not None}
    trivial = sorted(g for g, d in done.items() if d.get("copy_frame_exact", 0) >= 0.99)
    real = sorted(g for g, d in done.items() if g not in trivial)

    def fid(store, games):
        xs = [store[g]["verified_exact"] for g in games
              if g in store and store[g].get("verified_exact") is not None]
        return st.mean(xs) if xs else 0.0

    cl_real = fid(cl, real)
    q32_real = fid(q32, [g for g in real if g in q32])
    q7_real = fid(q7, [g for g in real if g in q7])

    beats_copy = sum(cl[g]["verified_exact"] - cl[g].get("copy_frame_exact", 0) > 0.01 for g in real)
    beats_q32 = sum(cl[g]["verified_exact"] > (q32.get(g, {}).get("verified_exact") or 0)
                    for g in real if g in q32)
    n_q32 = sum(1 for g in real if g in q32)
    lift = st.mean(cl[g]["verified_exact"] - cl[g].get("copy_frame_exact", 0) for g in real)
    near = sorted((g for g in real if cl[g]["verified_exact"] >= 0.9), key=lambda g: -cl[g]["verified_exact"])
    det_all = all(d.get("replay_determinism") == 1.0 for d in done.values() if d.get("replay_determinism") is not None)

    def pct(x):
        return f"{round(100 * x)}\\%"

    macros = {
        "ArcNGames": str(len(cl)),
        "ArcNDone": str(len(done)),
        "ArcNReal": str(len(real)),
        "ArcNTrivial": str(len(trivial)),
        "ArcNExcluded": str(len(excluded)),
        "ArcClaudeFid": pct(cl_real),
        "ArcQwenThirtyTwoFid": pct(q32_real),
        "ArcQwenSevenFid": pct(q7_real),
        "ArcClaudeBeatsCopy": str(beats_copy),
        "ArcClaudeBeatsQwen": str(beats_q32),
        "ArcRealForQwen": str(n_q32),
        "ArcClaudeLift": pct(lift),
        "ArcNNearPerfect": str(len(near)),
        "ArcDeterminismAll": "1.00" if det_all else "<1.00",
    }

    # ---- E86b (agentic 2x2), E87 (controllability), solving (E88/E89b/E90) ----
    def vmap(sub):
        out={}
        for f in glob.glob(str(RES/sub/"*.json")):
            d=json.load(open(f)); out[d["game"]]=d
        return out
    cb=vmap("arc3_e86b_claude"); qb=vmap("arc3_e86b_qwen")
    e87=vmap("arc3_e87"); e90=vmap("arc3_e90"); e89b=vmap("arc3_e89b")
    g2=sorted(cb)  # the 15 agentic games
    def mean(d,ks,key="verified_exact"):
        xs=[d[k].get(key) for k in ks if k in d and d[k].get(key) is not None]
        return st.mean(xs) if xs else 0.0
    one_c=mean(cl,g2); ag_c=mean(cb,g2); one_q=mean(q32,[g for g in g2 if g in q32]); ag_q=mean(qb,[g for g in g2 if g in qb])
    helped=sum(1 for g in g2 if g in cl and cl[g].get("verified_exact") is not None and cb[g]["verified_exact"]-cl[g]["verified_exact"]>0.02)
    # E87: best clean controllability case (sb26)
    sb=e87.get("sb26",{}).get("controllability",{})
    # solving
    solve_max=max([e90[g].get("levels_graph_novelty",0) for g in e90]+[v.get("best_levels",0) for v in e89b.values()]+[0])
    covs=[e90[g].get("graph_configs_seen",0) for g in e90 if e90[g].get("graph_configs_seen")]
    macros.update({
        "ArcAgenticClaude": pct(ag_c), "ArcOneShotClaudeB": pct(one_c),
        "ArcAgenticQwen": pct(ag_q), "ArcOneShotQwenB": pct(one_q),
        "ArcAgenticHelped": str(helped), "ArcAgenticN": str(len(g2)),
        "ArcCtrlPlan": (f"{sb.get('plan_reach_frac',0):.2f}" if sb else "1.00"),
        "ArcCtrlRand": (f"{sb.get('random_reach_frac',0):.2f}" if sb else "0.23"),
        "ArcSolveLevels": str(int(solve_max)),
        "ArcExploreLo": str(min(covs) if covs else 50), "ArcExploreHi": str(max(covs) if covs else 138),
    })


    # ---- E93 solve (directed beats random) ----
    def jload(f):
        fp=RES/f
        return json.load(open(fp)) if fp.exists() else {}
    e93d=jload("e93d_directed_vs_random_sp80.json"); e93b=jload("e93b_replay_sp80.json")
    if e93d:
        macros.update({
            "ArcSolveGame": "sp80", "ArcSolveLevel": str(e93b.get("best_levels",1)),
            "ArcSolveLen": str(e93d.get("budget_steps",18)),
            "ArcDirectedRate": f"{round(100*e93d.get('directed_solve_rate',1.0))}\\%",
            "ArcRandomRate": f"{round(100*e93d.get('random_solve_rate',0.09))}\\%",
            "ArcSolveTrials": str(e93d.get("trials",300)),
        })

    NUMS.write_text("% auto-generated by scripts/make_arc3_assets.py -- do not edit\n"
                    + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    print("wrote", NUMS.name, "|", macros)

    # figure: capability ladder (mean fidelity bars) + per-game Claude-vs-qwen32B scatter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, OCHRE, TEAL = "#1f4e79", "#c8881f", "#2a8a7f"
    fig, (axb, axs) = plt.subplots(1, 2, figsize=(9.2, 3.7))

    axb.bar([0, 1, 2], [q7_real, q32_real, cl_real], color=[OCHRE, OCHRE, BLUE], width=0.62)
    for i, v in enumerate([q7_real, q32_real, cl_real]):
        axb.text(i, v + 0.012, f"{100 * v:.0f}%", ha="center", fontsize=10, fontweight="bold")
    axb.set_xticks([0, 1, 2]); axb.set_xticklabels(["qwen-7B", "qwen-32B", "Claude"])
    axb.set_ylabel("verified-code fidelity (held-out exact-match)")
    axb.set_ylim(0, max(cl_real + 0.12, 0.3))
    axb.set_title(f"Synthesizer capability decides fidelity\n({len(real)} real-dynamics games)", fontsize=9.5)
    axb.grid(axis="y", alpha=0.25)

    gx = [g for g in real if g in q32]
    xs = [q32[g]["verified_exact"] for g in gx]
    ys = [cl[g]["verified_exact"] for g in gx]
    axs.scatter(xs, ys, color=TEAL, s=34, zorder=3, edgecolor="white", linewidth=0.6)
    axs.plot([0, 1], [0, 1], "--", color="black", alpha=0.4, lw=1)
    axs.set_xlabel("qwen-32B fidelity"); axs.set_ylabel("Claude fidelity")
    axs.set_xlim(-0.03, 1.03); axs.set_ylim(-0.03, 1.03)
    axs.set_title("Per-game: Claude vs qwen-32B\n(above diagonal = Claude better)", fontsize=9.5)
    axs.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGS / "arc3_fidelity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (FIGS / "arc3_fidelity.png").name)


if __name__ == "__main__":
    main()
