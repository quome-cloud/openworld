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

BLUE, ORANGE, TEAL, SLATE = "#1D4ED8", "#D97706", "#0F766E", "#475569"


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def pct(x):
    return f"{100 * x:.0f}\\%"


def ci_str(ci):
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


# ---------------------------------------------------------------------------
def fig_hero(e01, e10):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    # Panel A: compounding rollout error.
    ax = axes[0]
    engines = {e["engine"]: e for e in e01["engines"]}
    steps = range(1, e01["rollout_steps"] + 1)
    ax.plot(steps, engines["code_transition"]["per_step_match_rate"], "-o",
            color=BLUE, markersize=3.5, label="Symbolic (verified code, ours)")
    ax.plot(steps, engines["llm_transition"]["per_step_match_rate"], "-s",
            color=ORANGE, markersize=3.5, label="Learned-style (LLM next-state)")
    ax.set_xlabel("Rollout depth (steps)")
    ax.set_ylabel("Exact state-match rate")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("A. Compounding rollout error", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.25)

    # Panel B: OOD scale generalization.
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
           width, color=ORANGE, label="Learned-style")
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


def fig_judge(e05, e06):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    b, j = e05["summary"], e06["summary"]
    metrics = ["pass_at_1", "pass_at_budget"]
    labels = ["pass@1", f"pass@{e05['max_attempts']}"]
    x = [0, 1]
    width = 0.36
    ax.bar([i - width / 2 for i in x], [b[m] for m in metrics], width,
           color=SLATE, label="Single proposal (baseline)")
    ax.bar([i + width / 2 for i in x], [j[m] for m in metrics], width,
           color=TEAL, label=f"{e06['n_candidates']} candidates + judge")
    for i, m in enumerate(metrics):
        lo, hi = b[f"{m}_ci"]
        ax.errorbar(i - width / 2, b[m], yerr=[[b[m] - lo], [hi - b[m]]],
                    fmt="none", ecolor="black", capsize=3, lw=1)
        lo, hi = j[f"{m}_ci"]
        ax.errorbar(i + width / 2, j[m], yerr=[[j[m] - lo], [hi - j[m]]],
                    fmt="none", ecolor="black", capsize=3, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Solve rate (10 repair tasks)")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8)
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
    bars = ax.bar(x, accs, 0.55, color=[SLATE, SLATE, BLUE, TEAL])
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
        row("Learned-style (LLM next-state)", "llm_transition"),
    ])
    (TABLES / "main.tex").write_text(
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Engine & First divergence (step) & Final L1 error & OOD exact (10$\\times$) & Steps/s \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_synthesis(e02):
    rows = []
    for s in e02["summary"]:
        rows.append(
            f"{s['model']} & {s['n']} & {pct(s['acceptance_rate'])} "
            f"{ci_str(s['acceptance_ci'])} & {s['mean_probe_accuracy_accepted']:.2f} \\\\"
        )
    (TABLES / "synthesis.tex").write_text(
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Generator model & Runs & Verified acceptance (95\\% CI) & Probe accuracy of accepted \\\\\n"
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


def numbers_tex(e01, e02, e03, e04, e05, e06, e07, e08, e09, e10):
    engines01 = {e["engine"]: e for e in e01["engines"]}
    speed = {e["engine"]: e for e in e04["engines"]}
    ood = {(r["engine"], r["probes"]): r for r in e10["rows"]}
    e03s = {s["condition"]: s for s in e03["summary"]}
    audit = e06["judge_audit"]

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
        macro("VerifierNonePerfect", pct(e03s["none"]["perfect_rate"])),
        macro("VerifierFullPerfect", pct(e03s["full"]["perfect_rate"])),
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
        macro("NumExperiments", "10"),
    ]
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n")


def main():
    FIGS.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    data = {name: load(name) for name in (
        "e01_fidelity", "e02_synthesis", "e03_verifier_ablation",
        "e04_rollout_speed", "e05_codefix_agent", "e06_judge_selection",
        "e07_judge_alignment", "e08_morality_pareto", "e09_tuning_efficiency",
        "e10_ood_generalization",
    )}
    fig_hero(data["e01_fidelity"], data["e10_ood_generalization"])
    fig_judge(data["e05_codefix_agent"], data["e06_judge_selection"])
    fig_pareto(data["e08_morality_pareto"])
    fig_verifier(data["e03_verifier_ablation"])
    table_main(data["e01_fidelity"], data["e04_rollout_speed"], data["e10_ood_generalization"])
    table_synthesis(data["e02_synthesis"])
    table_tuning(data["e09_tuning_efficiency"])
    numbers_tex(*[data[k] for k in (
        "e01_fidelity", "e02_synthesis", "e03_verifier_ablation",
        "e04_rollout_speed", "e05_codefix_agent", "e06_judge_selection",
        "e07_judge_alignment", "e08_morality_pareto", "e09_tuning_efficiency",
        "e10_ood_generalization",
    )])
    print("assets written to paper/figs, paper/tables, paper/numbers.tex")


if __name__ == "__main__":
    main()
