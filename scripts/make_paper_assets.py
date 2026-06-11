"""Generate every paper figure, table, and numbers.tex from experiments/results/.

Single command reproducibility:  python3 scripts/make_paper_assets.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "experiments" / "results"
FIGS = ROOT / "paper" / "figs"
TABLES = ROOT / "paper" / "tables"

BLUE, ORANGE, TEAL, SLATE, PURPLE = "#1D4ED8", "#D97706", "#0F766E", "#475569", "#6D28D9"

EXPERIMENTS = [
    "e01_fidelity", "e02_synthesis", "e03_verifier_ablation",
    "e04_rollout_speed", "e05_codefix_agent", "e06_judge_selection",
    "e07_judge_alignment", "e08_morality_pareto", "e09_tuning_efficiency",
    "e10_ood_generalization", "e11_multiworld_fidelity",
    "e12_learned_baseline", "e13_judge_controls", "e15_judge_robustness",
    "e16_cross_model", "e17_judge_power", "e18_repair_loop",
    "e19_scale_ladder",
]


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def pct(x):
    return f"{100 * x:.0f}\\%"


def ci_str(ci):
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


# ---------------------------------------------------------------------------
def fig_hero(e01, e10):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    ax = axes[0]
    engines = {e["engine"]: e for e in e01["engines"]}
    steps = range(1, e01["rollout_steps"] + 1)
    ax.plot(steps, engines["code_transition"]["per_step_match_rate"], "-o",
            color=BLUE, markersize=3.5, label="Symbolic (verified code, ours)")
    ax.plot(steps, engines["llm_transition"]["per_step_match_rate"], "-s",
            color=ORANGE, markersize=3.5, label="LLM next-state (proxy)")
    ax.set_xlabel("Rollout depth (steps)")
    ax.set_ylabel("Exact state-match rate")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("A. Compounding rollout error", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    rows = e10["rows"]

    def rate(engine, probes):
        return next(r["exact_match_rate"] for r in rows
                    if r["engine"] == engine and r["probes"] == probes)

    groups = ["in_distribution", "scaled_10x"]
    x = [0, 1]
    width = 0.36
    ax.bar([i - width / 2 for i in x], [rate("code_transition", g) for g in groups],
           width, color=BLUE, label="Symbolic (ours)")
    ax.bar([i + width / 2 for i in x], [rate("llm_transition", g) for g in groups],
           width, color=ORANGE, label="LLM next-state")
    ax.set_xticks(x)
    ax.set_xticklabels(["In-distribution", "10× scaled (OOD)"])
    ax.set_ylabel("Exact transition accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("B. Out-of-distribution scale transfer", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(FIGS / "hero.png", dpi=200)
    plt.close(fig)


def fig_learned(e12):
    """Sample-efficiency of trained baselines vs the zero-transition program."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    rows = e12["rows"]
    ks = e12["ks"]

    def series(model, field):
        return [next(r[field] for r in rows
                     if r["model"] == model and r["k_transitions"] == k) for k in ks]

    for ax, field, title in (
        (axes[0], "probe_in_dist", "A. Branch-covering probes (in-distribution)"),
        (axes[1], "probe_ood_10x", "B. The same probes at 10× scale (OOD)"),
    ):
        ax.plot(ks, series("mlp", field), "-o", color=PURPLE, label="MLP (trained)")
        ax.plot(ks, series("knn1", field), "-s", color=SLATE, label="1-NN (memorizer)")
        ax.axhline(1.0, color=BLUE, lw=2, ls="--",
                   label="Synthesized code (0 transitions)")
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("Training transitions (random policy)")
        ax.set_ylabel("Exact transition accuracy")
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "learned.png", dpi=200)
    plt.close(fig)


def fig_judge(e17_pooled):
    """Controlled selection comparison, pooled across all paired rounds."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    order = ["first", "random", "judge", "oracle"]
    labels = ["first\ncandidate", "random\nof 3", "judge\nof 3", "oracle\nof 3\n(ceiling)"]
    colors = [SLATE, SLATE, TEAL, "#9CA3AF"]
    strategies = e17_pooled["strategies"]
    x = list(range(len(order)))
    for i, name in enumerate(order):
        s = strategies[name]
        ax.bar(i, s["rate"], 0.55, color=colors[i])
        lo, hi = s["ci"]
        ax.errorbar(i, s["rate"], yerr=[[s["rate"] - lo], [hi - s["rate"]]],
                    fmt="none", ecolor="black", capsize=3, lw=1)
        ax.text(i, min(s["rate"] + 0.1, 1.02), f"{s['rate']:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Solve rate over {e17_pooled['n_rounds']} paired rounds")
    ax.set_ylim(0, 1.12)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "judge.png", dpi=200)
    plt.close(fig)


def fig_pareto(e08):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    points = e08["points"]
    xs = [p["welfare"] for p in points]
    ys = [p["fairness"] for p in points]
    lam = [p["lambda"] for p in points]
    sc = ax.scatter(xs, ys, c=lam, cmap="viridis", s=55, zorder=3)
    ax.plot(xs, ys, color="black", alpha=0.3, lw=1, zorder=2)
    nash = e08["nash_optimum_lambda"]
    for p in points:
        if p["lambda"] == nash:
            ax.annotate(f"Nash optimum ($\\lambda$={nash})",
                        (p["welfare"], p["fairness"]),
                        textcoords="offset points", xytext=(10, 10), fontsize=8,
                        arrowprops=dict(arrowstyle="->", lw=0.8))
    fig.colorbar(sc, label="moral weight $\\lambda$")
    ax.set_xlabel("Individual welfare (total harvest)")
    ax.set_ylabel("Collective fairness ($-$harvest gap)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "pareto.png", dpi=200)
    plt.close(fig)


def fig_verifier(e03):
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    order = ["none", "syntax", "full", "critic"]
    summary = {s["condition"]: s for s in e03["summary"]}
    accs = [summary[c]["mean_probe_accuracy"] for c in order]
    x = list(range(len(order)))
    ax.bar(x, accs, 0.55, color=[SLATE, SLATE, BLUE, TEAL])
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["accept\nblindly", "syntax\nonly", "+sandbox\n+invariants",
                        "+7B LLM\ncritic"])
    ax.set_ylabel("Mean ground-truth probe accuracy")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "verifier.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def table_main(e01, e04, e10):
    engines01 = {e["engine"]: e for e in e01["engines"]}
    speed = {e["engine"]: e for e in e04["engines"]}
    ood = {(r["engine"], r["probes"]): r for r in e10["rows"]}

    def row(label, key):
        e = engines01[key]
        ood_rate = ood[(key, "scaled_10x")]["exact_match_rate"]
        sps = speed[key]["steps_per_second"]
        return (f"{label} & {e['mean_first_divergence_step']:.1f} & "
                f"{e['mean_final_l1_error']:.1f} & {100 * ood_rate:.0f}\\% & "
                f"{sps:,.0f} \\\\")

    body = "\n".join([
        row("\\textbf{Symbolic (verified code, ours)}", "code_transition"),
        row("LLM next-state (proxy)", "llm_transition"),
    ])
    (TABLES / "main.tex").write_text(
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Engine & First divergence (step) & Final L1 error & OOD exact (10$\\times$) & Steps/s \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_fidelity(e11):
    rows = []
    worlds = sorted({r["world"] for r in e11["rows"]})
    for world in worlds:
        code = next(r for r in e11["rows"]
                    if r["world"] == world and r["engine"] == "code_transition")
        llm = next(r for r in e11["rows"]
                   if r["world"] == world and r["engine"] == "llm_transition")
        llm_div = (f"{llm['mean_first_divergence']:.1f}"
                   if llm["mean_first_divergence"] else "---")
        rows.append(
            f"{world} & {code['exact_rollouts']}/{code['n_rollouts']} & "
            f"{llm['exact_rollouts']}/{llm['n_rollouts']} & {llm_div} \\\\"
        )
    totals = e11["totals"]
    code_t, llm_t = totals["code_transition"], totals["llm_transition"]
    rows.append("\\midrule")
    rows.append(
        f"\\textbf{{Total}} & \\textbf{{{code_t['exact_rollouts']}/{code_t['n']}}} "
        f"{ci_str(code_t['ci'])} & {llm_t['exact_rollouts']}/{llm_t['n']} "
        f"{ci_str(llm_t['ci'])} & --- \\\\"
    )
    (TABLES / "fidelity.tex").write_text(
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "World & Code: exact rollouts & LLM: exact rollouts & LLM first divergence \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_judge_controls(e13):
    rows = []
    labels = {"first": "First candidate (no selection)",
              "random": "Random of 3 (diversity only)",
              "judge": "\\textbf{Judge of 3 (ours)}",
              "oracle": "Oracle of 3 (ceiling)"}
    for name in ("first", "random", "judge", "oracle"):
        s = e13["strategies"][name]
        rows.append(f"{labels[name]} & {s['solves']}/{s['n']} & "
                    f"{pct(s['rate'])} {ci_str(s['ci'])} \\\\")
    (TABLES / "judge_controls.tex").write_text(
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "Selection strategy & Solves & Rate (95\\% CI) \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_synthesis(e02, e16):
    """Synthesis reliability across model scale AND family (E2 + E16)."""
    families = {"qwen2.5:7b": "Qwen", "qwen2.5:3b": "Qwen",
                "llama3.1:8b": "Llama", "gemma2:9b": "Gemma"}
    rows = []
    for s in e02["summary"]:
        rows.append(
            f"{s['model']} ({families.get(s['model'], '?')}) & {s['n']} & "
            f"{pct(s['acceptance_rate'])} {ci_str(s['acceptance_ci'])} & "
            f"{s['mean_probe_accuracy_accepted']:.2f} & --- \\\\"
        )
    for s in e16["summary"]:
        rows.append(
            f"{s['model']} ({families.get(s['model'], '?')}) & {s['n']} & "
            f"{pct(s['acceptance_rate'])} {ci_str(s['acceptance_ci'])} & "
            f"{s['mean_probe_accuracy_accepted']:.2f} & {s['mean_wall_seconds']:.0f}s \\\\"
        )
    (TABLES / "synthesis.tex").write_text(
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Generator (family) & Runs & Verified acceptance (95\\% CI) & "
        "Probe acc.\\ of accepted & Mean synthesis time \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_ladder(e19):
    order = ["code_synthesized", "delta_mlp_strong", "mlp", "knn1", "llm_next_state"]
    labels = {
        "code_synthesized": "\\textbf{Synthesized code (0 transitions)}",
        "delta_mlp_strong": "Delta-MLP, 128h (10k transitions)",
        "mlp": "MLP, 64h (10k transitions)",
        "knn1": "1-NN memorizer (10k transitions)",
        "llm_next_state": "LLM next-state (7B, 0 transitions)",
    }
    scales = ["1x", "10x", "100x"]
    by = {(r["engine"], r["scale"]): r for r in e19["rows"]}
    rows = []
    for engine in order:
        cells = " & ".join(
            f"{by[(engine, s)]['exact']}/{by[(engine, s)]['n']}" for s in scales)
        rows.append(f"{labels[engine]} & {cells} \\\\")
    (TABLES / "ladder.tex").write_text(
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Engine & 1$\\times$ & 10$\\times$ & 100$\\times$ \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_repair(e18):
    rows = []
    for s in e18["summary"]:
        label = ("Filter only (max\\_iters=1)" if s["max_iters"] == 1
                 else "Filter + repair loop (max\\_iters=4)")
        rows.append(
            f"{label} & {pct(s['acceptance_rate'])} & "
            f"{s['mean_probe_accuracy_accepted']:.2f} & "
            f"{s['mean_probe_accuracy_all']:.2f} \\\\"
        )
    (TABLES / "repair.tex").write_text(
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Regime & Acceptance & Probe acc.\\ (accepted) & Probe acc.\\ (all runs) \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_tuning(e09):
    rows = []
    for s in e09["summary"]:
        ttfs = (f"{s['mean_trials_to_first_solve']:.0f}"
                if s["mean_trials_to_first_solve"] else "---")
        rows.append(
            f"{s['strategy']} & {s['seeds_solving']}/{s['seeds']} & {ttfs} & "
            f"{s['mean_best_score']:.2f} \\\\"
        )
    (TABLES / "tuning.tex").write_text(
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Strategy & Seeds solving & Mean trials to first solve & Mean best score \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def numbers_tex(d):
    e01, e02, e03 = d["e01_fidelity"], d["e02_synthesis"], d["e03_verifier_ablation"]
    e04, e05, e06 = d["e04_rollout_speed"], d["e05_codefix_agent"], d["e06_judge_selection"]
    e07, e08, e09 = d["e07_judge_alignment"], d["e08_morality_pareto"], d["e09_tuning_efficiency"]
    e10, e11, e12 = d["e10_ood_generalization"], d["e11_multiworld_fidelity"], d["e12_learned_baseline"]
    e13, e15 = d["e13_judge_controls"], d["e15_judge_robustness"]
    e16, e17, e18, e19 = (d["e16_cross_model"], d["e17_judge_power"],
                          d["e18_repair_loop"], d["e19_scale_ladder"])

    engines01 = {e["engine"]: e for e in e01["engines"]}
    speed = {e["engine"]: e for e in e04["engines"]}
    ood = {(r["engine"], r["probes"]): r for r in e10["rows"]}
    e03s = {s["condition"]: s for s in e03["summary"]}
    audit = e06["judge_audit"]
    code_total = e11["totals"]["code_transition"]
    llm_total = e11["totals"]["llm_transition"]
    e12_rows = {(r["model"], r["k_transitions"]): r for r in e12["rows"]}
    bias = e13["position_bias"]
    mcn = e13["mcnemar_judge_vs_random"]

    def macro(name, value):
        return f"\\newcommand{{\\{name}}}{{{value}}}"

    speed_ratio = (speed["code_transition"]["steps_per_second"]
                   / speed["llm_transition"]["steps_per_second"])
    lines = [
        "% Auto-generated by scripts/make_paper_assets.py - do not edit.",
        macro("CodeFirstDivergence", f"{engines01['code_transition']['mean_first_divergence_step']:.0f}"),
        macro("LLMFirstDivergence", f"{engines01['llm_transition']['mean_first_divergence_step']:.1f}"),
        macro("LLMFinalLOne", f"{engines01['llm_transition']['mean_final_l1_error']:.1f}"),
        macro("RolloutSteps", str(e01["rollout_steps"])),
        macro("SynthAcceptBig", pct(e02["summary"][0]["acceptance_rate"])),
        macro("SynthAcceptSmall", pct(e02["summary"][1]["acceptance_rate"])),
        macro("SynthProbeBig", f"{e02['summary'][0]['mean_probe_accuracy_accepted']:.2f}"),
        macro("VerifierNoneAcc", f"{e03s['none']['mean_probe_accuracy']:.2f}"),
        macro("VerifierFullAcc", f"{e03s['full']['mean_probe_accuracy']:.2f}"),
        macro("VerifierCriticAcc", f"{e03s['critic']['mean_probe_accuracy']:.2f}"),
        macro("CodeStepsPerSec", f"{speed['code_transition']['steps_per_second']:,.0f}"),
        macro("LLMStepsPerSec", f"{speed['llm_transition']['steps_per_second']:.2f}"),
        macro("SpeedRatio", f"{speed_ratio:,.0f}"),
        macro("CodeOODRate", pct(ood[("code_transition", "scaled_10x")]["exact_match_rate"])),
        macro("LLMOODRate", pct(ood[("llm_transition", "scaled_10x")]["exact_match_rate"])),
        macro("LLMInDistRate", pct(ood[("llm_transition", "in_distribution")]["exact_match_rate"])),
        macro("LLMProbeErrorRate", pct(1 - ood[("llm_transition", "in_distribution")]["exact_match_rate"])),
        macro("BaselinePassOne", pct(e05["summary"]["pass_at_1"])),
        macro("BaselinePassBudget", pct(e05["summary"]["pass_at_budget"])),
        macro("JudgePassOne", pct(e06["summary"]["pass_at_1"])),
        macro("JudgePassBudget", pct(e06["summary"]["pass_at_budget"])),
        macro("JudgeAccuracy",
              pct(audit["judge_accuracy_when_solvable"]) if audit["judge_accuracy_when_solvable"] is not None else "n/a"),
        macro("JudgeRounds", str(audit["rounds_with_passing_candidate"])),
        macro("JudgeSpearman", f"{e07['spearman_judge_vs_aggregate']:.2f}"),
        macro("ParetoFrontierSize", str(e08["pareto_frontier_size"])),
        macro("ParetoLambdas", str(len(e08["lambdas"]))),
        macro("NashLambda", str(e08["nash_optimum_lambda"])),
        macro("TuningBudget", str(e09["budget_trials"])),
        macro("NumTasks", str(e05["summary"]["n_tasks"])),
        macro("NumExperiments", "18"),
        # E11 multi-world fidelity
        macro("MultiCodeExact", f"{code_total['exact_rollouts']}/{code_total['n']}"),
        macro("MultiCodeCI", ci_str(code_total["ci"])),
        macro("MultiLLMExact", f"{llm_total['exact_rollouts']}/{llm_total['n']}"),
        macro("MultiLLMCI", ci_str(llm_total["ci"])),
        # E12 learned baselines (10k-transition column)
        macro("MLPInDistTenK", pct(e12_rows[("mlp", 10000)]["probe_in_dist"])),
        macro("MLPOODTenK", pct(e12_rows[("mlp", 10000)]["probe_ood_10x"])),
        macro("KNNInDistTenK", pct(e12_rows[("knn1", 10000)]["probe_in_dist"])),
        macro("KNNRolloutsTenK",
              f"{e12_rows[('knn1', 10000)]['exact_rollouts']}/{e12_rows[('knn1', 10000)]['n_rollouts']}"),
        # E13 judge controls
        macro("CtrlFirst", pct(e13["strategies"]["first"]["rate"])),
        macro("CtrlRandom", pct(e13["strategies"]["random"]["rate"])),
        macro("CtrlJudge", pct(e13["strategies"]["judge"]["rate"])),
        macro("CtrlOracle", pct(e13["strategies"]["oracle"]["rate"])),
        macro("CtrlRounds", str(e13["n_rounds"])),
        macro("McNemarP", f"{mcn['p']:.3f}"),
        macro("McNemarB", str(mcn["b"])),
        macro("McNemarC", str(mcn["c"])),
        macro("OrderConsistency", pct(bias["order_consistency"])),
        macro("OrderConsistencyDisc", pct(
            (lambda disc: sum(r["order_consistent"] for r in disc) / len(disc) if disc else 0)
            ([r for r in e13["rounds"] if any(r["passing"]) and not all(r["passing"])])
        )),
        macro("JudgeFwdAcc",
              pct(bias["judge_accuracy_forward"]) if bias["judge_accuracy_forward"] is not None else "n/a"),
        macro("JudgeRevAcc",
              pct(bias["judge_accuracy_reversed"]) if bias["judge_accuracy_reversed"] is not None else "n/a"),
        macro("DiscRounds", str(bias["n_discriminative_rounds"])),
        # E15 judge robustness
        macro("RubricOrigRho", f"{e15['results']['original']['spearman']:.2f}"),
        macro("RubricOrigP", f"{e15['results']['original']['permutation_p']:.4f}"),
        macro("RubricParaRho", f"{e15['results']['paraphrase']['spearman']:.2f}"),
        macro("RubricParaP", f"{e15['results']['paraphrase']['permutation_p']:.4f}"),
    ]
    # E16 cross-family synthesis
    e16s = {s["model"]: s for s in e16["summary"]}
    lines += [
        macro("LlamaAccept", pct(e16s["llama3.1:8b"]["acceptance_rate"])),
        macro("LlamaProbe", f"{e16s['llama3.1:8b']['mean_probe_accuracy_accepted']:.2f}"),
        macro("LlamaWall", f"{e16s['llama3.1:8b']['mean_wall_seconds']:.0f}"),
        macro("GemmaAccept", pct(e16s["gemma2:9b"]["acceptance_rate"])),
        macro("GemmaProbe", f"{e16s['gemma2:9b']['mean_probe_accuracy_accepted']:.2f}"),
        macro("GemmaWall", f"{e16s['gemma2:9b']['mean_wall_seconds']:.0f}"),
    ]
    # E17 pooled judge power
    pooled = e17["pooled"]
    pm = pooled["mcnemar_judge_vs_random"]
    pb = pooled["position_bias"]
    lines += [
        macro("PooledRounds", str(pooled["n_rounds"])),
        macro("PooledFirst", pct(pooled["strategies"]["first"]["rate"])),
        macro("PooledRandom", pct(pooled["strategies"]["random"]["rate"])),
        macro("PooledJudge", pct(pooled["strategies"]["judge"]["rate"])),
        macro("PooledOracle", pct(pooled["strategies"]["oracle"]["rate"])),
        macro("PooledMcNemarP", f"{pm['p']:.3f}"),
        macro("PooledMcNemarB", str(pm["b"])),
        macro("PooledMcNemarC", str(pm["c"])),
        macro("PooledFwdAcc", pct(pb["judge_accuracy_forward"])),
        macro("PooledRevAcc", pct(pb["judge_accuracy_reversed"])),
        macro("PooledConsistency", pct(pb["order_consistency"])),
    ]
    # E18 repair loop
    e18s = {s["max_iters"]: s for s in e18["summary"]}
    lines += [
        macro("FilterAccept", pct(e18s[1]["acceptance_rate"])),
        macro("FilterProbeAccepted", f"{e18s[1]['mean_probe_accuracy_accepted']:.2f}"),
        macro("LoopAccept", pct(e18s[4]["acceptance_rate"])),
        macro("LoopProbeAccepted", f"{e18s[4]['mean_probe_accuracy_accepted']:.2f}"),
    ]
    # E19 scale ladder
    ladder = {(r["engine"], r["scale"]): r for r in e19["rows"]}
    lines += [
        macro("CodeHundredX", pct(ladder[("code_synthesized", "100x")]["rate"])),
        macro("DeltaMLPInDist", pct(ladder[("delta_mlp_strong", "1x")]["rate"])),
        macro("DeltaMLPHundredX", pct(ladder[("delta_mlp_strong", "100x")]["rate"])),
        macro("LLMHundredX", pct(ladder[("llm_next_state", "100x")]["rate"])),
    ]
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n")


def main():
    FIGS.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    data = {name: load(name) for name in EXPERIMENTS}
    fig_hero(data["e01_fidelity"], data["e10_ood_generalization"])
    fig_learned(data["e12_learned_baseline"])
    fig_judge(data["e17_judge_power"]["pooled"])
    fig_pareto(data["e08_morality_pareto"])
    fig_verifier(data["e03_verifier_ablation"])
    table_main(data["e01_fidelity"], data["e04_rollout_speed"], data["e10_ood_generalization"])
    table_fidelity(data["e11_multiworld_fidelity"])
    table_judge_controls(data["e13_judge_controls"])
    table_synthesis(data["e02_synthesis"], data["e16_cross_model"])
    table_tuning(data["e09_tuning_efficiency"])
    table_ladder(data["e19_scale_ladder"])
    table_repair(data["e18_repair_loop"])
    numbers_tex(data)
    print("assets written to paper/figs, paper/tables, paper/numbers.tex")


if __name__ == "__main__":
    main()
