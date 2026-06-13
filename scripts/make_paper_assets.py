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
    "e19_scale_ladder", "e20_complexity", "e21_stochastic",
    "e22_planning", "e23_self_check", "e24_aggregators",
    "e25_constraints", "e26_parliament", "e27_rubric_pluralism",
    "e28_swebench_ablation", "e29_swebench_staged",
    "e30_composition", "e31_nested_fidelity", "e32_regime_switch",
    "e33_dynamic_traversal", "e34_composite_swe", "e36_representations",
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


def table_swebench(e28, e29):
    def block(label, summary):
        rows = []
        for s in summary:
            rows.append(
                f"{label} & \\texttt{{{s['model']}}} & "
                f"{s['single_shot_pass1']:.2f} & {s['in_world_pass1']:.2f} & "
                f"{s['in_world_pass_budget']:.2f} & "
                f"${s['delta_budget_minus_ss']:+.2f}$ & {s['mean_attempts']:.1f} \\\\"
            )
            label = ""  # only print the suite name on its first row
        return rows

    atomic = block(f"Atomic ($n={e28['n_instances']}$)", e28["summary"])
    staged = block(f"Staged ($n={e29['n_instances']}$)", e29["summary"])
    (TABLES / "swebench.tex").write_text(
        "\\begin{tabular}{llccccc}\n\\toprule\n"
        "Suite & Model & SS pass@1 & IW pass@1 & IW pass@4 & "
        "$\\Delta$ & Mean att. \\\\\n\\midrule\n"
        + "\n".join(atomic) + "\n\\midrule\n" + "\n".join(staged)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_composition(e30, e32):
    """E30 (composition vs the cliff) + E32 (regime switch) in two blocks."""
    s30 = {s["condition"]: s for s in e30["summary"]}
    n_sectors = len(e30["sectors"])
    n_bridges = len(e30["bridges"])
    # 4 internal rules per sector: the parameter groups prod_*, rec_*,
    # decay_* in the results JSON, plus the unparameterized wait rule.
    child_rules = len({k.split("_")[0] for k in e30["sectors"][0]}) + 1
    labels30 = {
        "monolithic": f"Monolithic (one {n_sectors * child_rules + n_bridges}-rule prompt, flat state)",
        "compositional": (f"\\textbf{{Compositional ({n_sectors} sectors $\\times$ "
                          f"{child_rules} rules + {n_bridges} bridges)}}"),
    }
    rows30 = []
    for cond in ("monolithic", "compositional"):
        s = s30[cond]
        rows30.append(
            f"{labels30[cond]} & {pct(s['acceptance_rate'])} & "
            f"{s['mean_probe_accuracy']:.2f} {ci_str(s['pooled_ci'])} \\\\"
        )

    s32 = {s["condition"]: s for s in e32["summary"]}
    labels32 = {
        "phased": "\\textbf{PhasedTransition (two verified phases)}",
        "monolithic": "Monolithic (combined rules-with-change text)",
        "llm_proxy": "LLM next-state (proxy)",
    }
    rows32 = []
    for cond in ("phased", "monolithic", "llm_proxy"):
        s = s32[cond]
        rows32.append(
            f"{labels32[cond]} & {s['mean_pre_boundary_accuracy']:.2f} & "
            f"{s['boundary_step_match_rate']:.2f} & "
            f"{s['mean_post_boundary_accuracy']:.2f} & "
            f"{s['exact_full_rollouts']}/{s['replicates']} \\\\"
        )

    (TABLES / "composition.tex").write_text(
        "\\begin{tabular}{lcc}\n\\toprule\n"
        f"E30 condition (same {n_sectors * child_rules}-rule system) & "
        "Accepted & Probe acc.\\ (95\\% CI) \\\\\n"
        "\\midrule\n" + "\n".join(rows30) + "\n\\bottomrule\n\\end{tabular}\n"
        "\n\\vspace{0.7em}\n\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        f"E32 engine (switch at step {e32['switch_step']}) & Pre & Boundary & "
        "Post & Exact rollouts \\\\\n"
        "\\midrule\n" + "\n".join(rows32) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def fig_representations(e36):
    """E36: composition vs monolithic learners on three representation tests."""
    gen = e36["leg_generalization"]
    intf = e36["leg_interference"]
    se = e36["leg_sample_efficiency"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3.3))

    ks = [r["k"] for r in gen]
    ax1.plot(ks, [r["composite_symbolic"]["acc"] for r in gen], "-D",
             color=TEAL, lw=2, markersize=4, label="composite-symbolic (ours)")
    ax1.plot(ks, [r["composite_learned"]["acc"] for r in gen], "-o",
             color=BLUE, lw=2, markersize=4, label="composite-learned (ours)")
    ax1.plot(ks, [r["monolith"]["acc"] for r in gen], "-s",
             color=RED, lw=2, markersize=4, label="monolithic MLP")
    ax1.plot(ks, [r["knn1"]["acc"] for r in gen], "--^",
             color=SLATE, lw=1.6, markersize=4, label="1-NN memorizer")
    ax1.set_xlabel("Number of parts (K)")
    ax1.set_ylabel("Exact accuracy on novel combinations")
    ax1.set_xticks(ks)
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_title("A. Compositional generalization", fontsize=9.5, loc="left")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=6.5, loc="center right")

    bars = [("monolith\n(sequential)", intf["monolith_sequential_retained"], RED),
            ("monolith\n(joint)", intf["monolith_joint_retained"], "#9CA3AF"),
            ("composite\n(learned)", intf["composite_learned_retained"], BLUE),
            ("composite\n(symbolic)", intf["composite_symbolic_retained"], TEAL)]
    ax2.bar(range(len(bars)), [b[1] for b in bars],
            color=[b[2] for b in bars], width=0.62)
    for i, b in enumerate(bars):
        ax2.text(i, b[1] + 0.03, f"{b[1]:.2f}", ha="center", fontsize=8)
    ax2.set_xticks(range(len(bars)))
    ax2.set_xticklabels([b[0] for b in bars], fontsize=7)
    ax2.set_ylabel("Retained accuracy on part 0")
    ax2.set_ylim(0, 1.12)
    ax2.set_title("B. Interference (K=4)", fontsize=9.5, loc="left")
    ax2.grid(alpha=0.25, axis="y")

    ns = [r["n_train_actual"] for r in se["rows"]]
    ax3.plot(ns, [r["composite_learned_acc"] for r in se["rows"]], "-o",
             color=BLUE, lw=2, markersize=4, label="composite-learned")
    ax3.plot(ns, [r["monolith_acc"] for r in se["rows"]], "-s",
             color=RED, lw=2, markersize=4, label="monolithic MLP")
    ax3.axhline(1.0, color=TEAL, lw=1.6, ls=(0, (4, 3)),
                label="composite-symbolic (0 data)")
    ax3.set_xscale("log")
    ax3.set_xlabel("Training transitions")
    ax3.set_ylabel("Exact accuracy (joint test)")
    ax3.set_ylim(-0.03, 1.05)
    ax3.set_title("C. Sample efficiency (K=3)", fontsize=9.5, loc="left")
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=6.5, loc="center right")

    fig.tight_layout()
    fig.savefig(FIGS / "representations.png", dpi=200)
    plt.close(fig)


def fig_complexity(e20):
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    summary = e20["summary"]
    xs = [s["n_rules"] for s in summary]
    ys = [s["mean_probe_accuracy"] for s in summary]
    los = [s["pooled_ci"][0] for s in summary]
    his = [s["pooled_ci"][1] for s in summary]
    ax.plot(xs, ys, "-o", color=BLUE, lw=2)
    ax.fill_between(xs, los, his, color=BLUE, alpha=0.15,
                    label="95% CI (pooled probes)")
    ax.set_xlabel("Declared interacting rules (R)")
    ax.set_ylabel("Mean probe accuracy of synthesized dynamics")
    ax.set_xticks(xs)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "complexity.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Composition figures (E30/E31): drawn with patches in the house palette.
# ---------------------------------------------------------------------------
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RED = "#B91C1C"


def _card(ax, x, y, w, h, title, lines, face="white", edge="#CBD5E1",
          alpha=1.0, title_color="#0F172A", bold_edge=None, shadow=True):
    """A rounded state card with a soft shadow."""
    if shadow:
        ax.add_patch(FancyBboxPatch((x + 0.045, y - 0.045), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="black", ec="none", alpha=0.10 * alpha, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=face, ec=bold_edge or edge,
                                lw=2.0 if bold_edge else 0.9, alpha=alpha, zorder=3))
    ax.text(x + w / 2, y + h - 0.21, title, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=title_color, alpha=alpha, zorder=4)
    for i, line in enumerate(lines):
        ax.text(x + 0.12, y + h - 0.50 - 0.26 * i, line, ha="left", va="center",
                fontsize=7, family="monospace", color="#334155",
                alpha=alpha, zorder=4)


def _chip(ax, cx, cy, text, color, alpha=1.0, fontsize=7):
    ax.add_patch(FancyBboxPatch((cx - 0.78, cy - 0.14), 1.56, 0.30,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec="none", alpha=0.14 * alpha, zorder=4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            family="monospace", color=color, alpha=alpha, zorder=5)


def _panel(ax, x, y, w, h, label, color, alpha_fill=0.05):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.14",
                                fc=color, ec=color, lw=1.1, alpha=alpha_fill, zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.14",
                                fc="none", ec=color, lw=1.1, alpha=0.55, zorder=1))
    ax.text(x + 0.14, y + h - 0.18, label, ha="left", va="center",
            fontsize=9, fontweight="bold", color=color, zorder=4)


def _city_lines(leaves, country, city):
    return [f"treasury {leaves[f'{country}_{city}_treasury']:>3}",
            f"goods    {leaves[f'{country}_{city}_goods']:>3}",
            f"gdp      {leaves[f'{country}_{city}_gdp']:>3}"]


_GEOM = {  # shared layout for fig_composition / fig_traversal
    "region": (0.30, 1.00, 9.40, 4.10),
    "c0": (0.62, 1.28, 4.20, 3.06),
    "c1": (5.18, 1.28, 4.20, 3.06),
    "cards": {("c0", "a"): (0.92, 1.55), ("c0", "b"): (2.92, 1.55),
              ("c1", "a"): (5.48, 1.55), ("c1", "b"): (7.48, 1.55)},
    "cw": 1.62, "ch": 1.38,
}


def fig_composition(e31):
    leaves = e31["per_step"][0]["leaves"]
    fig, ax = plt.subplots(figsize=(10, 6.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.05)
    ax.axis("off")

    rx, ry, rw, rh = _GEOM["region"]
    _panel(ax, rx, ry, rw, rh, "region", SLATE, alpha_fill=0.03)
    _chip(ax, rx + rw - 1.15, ry + rh - 0.18, "_agg gdp=Σ", SLATE)
    for name, color in (("c0", BLUE), ("c1", BLUE)):
        px, py, pw, ph = _GEOM[name]
        _panel(ax, px, py, pw, ph, f"country {name}", color, alpha_fill=0.05)
        _chip(ax, px + pw / 2, py + ph - 0.18, "_agg gdp=Σ cities", BLUE)
    cw, ch = _GEOM["cw"], _GEOM["ch"]
    for (country, city), (x, y) in _GEOM["cards"].items():
        _card(ax, x, y, cw, ch, f"city {country}:{city}",
              _city_lines(leaves, country, city))

    # aggregator arrows: city tops -> country chip (derived, dashed, one-way up)
    for (country, city), (x, y) in _GEOM["cards"].items():
        px, py, pw, ph = _GEOM[country]
        ax.add_patch(FancyArrowPatch((x + cw / 2, y + ch + 0.06),
                                     (px + pw / 2, py + ph - 0.34),
                                     arrowstyle="-|>", mutation_scale=7,
                                     ls=(0, (2, 2)), color=BLUE, lw=0.9,
                                     alpha=0.55, zorder=2))
    for name in ("c0", "c1"):
        px, py, pw, ph = _GEOM[name]
        ax.add_patch(FancyArrowPatch((px + pw / 2, py + ph + 0.05),
                                     (rx + rw - 1.15, ry + rh - 0.34),
                                     arrowstyle="-|>", mutation_scale=7,
                                     ls=(0, (2, 2)), color=SLATE, lw=0.9,
                                     alpha=0.5, zorder=2))

    # bridge: c0:b <-> c1:a (trade, conserved); label in the clear top band
    bx0 = _GEOM["cards"][("c0", "b")][0] + cw
    bx1 = _GEOM["cards"][("c1", "a")][0]
    by = _GEOM["cards"][("c0", "b")][1] + ch * 0.55
    ax.add_patch(FancyArrowPatch((bx0 + 0.03, by), (bx1 - 0.03, by),
                                 connectionstyle="arc3,rad=-0.35",
                                 arrowstyle="<|-|>", mutation_scale=13,
                                 color=TEAL, lw=2.6, zorder=5))
    label_y = ry + rh - 0.32  # the clear band inside the region, between panels
    _chip(ax, (bx0 + bx1) / 2, label_y, "bridge: trade · Σ conserved", TEAL,
          fontsize=7.5)
    ax.add_patch(FancyArrowPatch(((bx0 + bx1) / 2, label_y - 0.16),
                                 ((bx0 + bx1) / 2, by + 0.30),
                                 arrowstyle="-", color=TEAL, lw=0.8,
                                 ls=(0, (1, 2)), alpha=0.7, zorder=4))

    # route: c0:b <-> c1:b along the bottom, with the agent mid-crossing
    px0 = _GEOM["cards"][("c0", "b")][0] + cw * 0.55
    px1 = _GEOM["cards"][("c1", "b")][0] + cw * 0.45
    py0 = _GEOM["cards"][("c0", "b")][1] - 0.06
    ax.add_patch(FancyArrowPatch((px0, py0), (px1, py0),
                                 connectionstyle="arc3,rad=0.18",
                                 arrowstyle="<|-|>", mutation_scale=11,
                                 color=ORANGE, lw=2.0, ls=(0, (5, 2)), zorder=5))
    mid_x, mid_y = (px0 + px1) / 2, py0 - 0.34
    ax.plot([mid_x], [mid_y], "o", color=PURPLE, markersize=9, zorder=6)
    ax.text(mid_x, 0.78, "agent · toll −2 → treasury", ha="center",
            fontsize=7.5, color=ORANGE, zorder=6)

    # binding: region -> c0 (downward parameter)
    ax.add_patch(FancyArrowPatch((rx + 0.55, ry + rh - 0.30),
                                 (rx + 0.55, _GEOM["c0"][1] + _GEOM["c0"][3] - 0.05),
                                 arrowstyle="-|>", mutation_scale=9,
                                 color=SLATE, lw=1.4, zorder=4))
    ax.text(rx + 0.66, ry + rh - 0.62, "binding: policy ↓", fontsize=7.5,
            color=SLATE, ha="left", zorder=4)

    # legend strip: four fixed columns
    legend = [(TEAL, "bridge — verified coupling"),
              (ORANGE, "route — agents may cross"),
              (BLUE, "aggregator — derived, never simulated"),
              (SLATE, "binding — downward parameter")]
    for lx, (color, label) in zip((0.55, 2.62, 4.72, 7.62), legend):
        ax.plot([lx, lx + 0.26], [0.42, 0.42], color=color, lw=2.4,
                ls="--" if color == ORANGE else "-",
                solid_capstyle="round")
        ax.text(lx + 0.36, 0.42, label, fontsize=6.8, color="#334155",
                ha="left", va="center")
    ax.text(0.32, 5.78, "A composite world is a world",
            fontsize=11.5, fontweight="bold", color="#0F172A")
    ax.text(0.32, 5.50,
            "children run unmodified; every coupling channel is an explicit, verifiable object",
            fontsize=8, color="#475569")
    fig.savefig(FIGS / "composition.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_composition_cliff(e20, e30):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    summary = e20["summary"]
    xs = [s["n_rules"] for s in summary]
    ys = [s["mean_probe_accuracy"] for s in summary]
    los = [s["pooled_ci"][0] for s in summary]
    his = [s["pooled_ci"][1] for s in summary]
    ax.plot(xs, ys, "-o", color=SLATE, lw=2, markersize=4.5,
            label="monolithic synthesis (E20)")
    ax.fill_between(xs, los, his, color=SLATE, alpha=0.13)

    by_cond = {s["condition"]: s for s in e30["summary"]}
    mono, comp = by_cond["monolithic"], by_cond["compositional"]
    ax.errorbar([16], [mono["mean_probe_accuracy"]],
                yerr=[[mono["mean_probe_accuracy"] - mono["pooled_ci"][0]],
                      [mono["pooled_ci"][1] - mono["mean_probe_accuracy"]]],
                fmt="o", color=RED, markersize=7, capsize=4, lw=1.4,
                label="monolithic, 16 rules (E30)", zorder=5)
    ax.errorbar([16], [comp["mean_probe_accuracy"]],
                yerr=[[comp["mean_probe_accuracy"] - comp["pooled_ci"][0]],
                      [comp["pooled_ci"][1] - comp["mean_probe_accuracy"]]],
                fmt="*", color=TEAL, markersize=17, capsize=4, lw=1.4,
                markeredgecolor="white", markeredgewidth=0.6,
                label="compositional, 4×4 + bridges (E30)", zorder=6)
    ax.axhline(comp["mean_probe_accuracy"], color=TEAL, lw=0.8,
               ls=(0, (1, 3)), alpha=0.6, zorder=1)
    ax.annotate("same 16 rules:\n4×4 children + verified bridges",
                xy=(16, comp["mean_probe_accuracy"]),
                xytext=(9.2, 0.62), fontsize=8, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.1,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_xlabel("Declared interacting rules (R)")
    ax.set_ylabel("Probe accuracy")
    ax.set_xticks(xs)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "composition_cliff.png", dpi=200)
    plt.close(fig)


def fig_traversal(e31):
    step = e31["per_step"][0]
    leaves, agent = step["leaves"], step["agent"]
    here_c, here_city = agent["at"].split(":")
    fig, ax = plt.subplots(figsize=(10, 6.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.05)
    ax.axis("off")

    rx, ry, rw, rh = _GEOM["region"]
    _panel(ax, rx, ry, rw, rh, "region", SLATE, alpha_fill=0.02)
    _chip(ax, rx + rw - 1.15, ry + rh - 0.18, "_agg: visible", SLATE)
    for name in ("c0", "c1"):
        px, py, pw, ph = _GEOM[name]
        mine = name == here_c
        _panel(ax, px, py, pw, ph, f"country {name}", BLUE,
               alpha_fill=0.05 if mine else 0.02)
        _chip(ax, px + pw / 2, py + ph - 0.18,
              "_agg: visible" if mine else "_agg only", BLUE,
              alpha=1.0 if mine else 0.75)
    cw, ch = _GEOM["cw"], _GEOM["ch"]
    neighbor = ("c1", "b")  # route-adjacent to the agent's city
    for (country, city), (x, y) in _GEOM["cards"].items():
        if (country, city) == (here_c, here_city):
            _card(ax, x, y, cw, ch, f"YOU: {country}:{city}",
                  _city_lines(leaves, country, city), bold_edge=PURPLE,
                  title_color=PURPLE)
            ax.plot([x + cw / 2], [y - 0.16], "o", color=PURPLE, markersize=8)
            ax.text(x + cw / 2, y - 0.40, f"coins {agent['coins']}",
                    ha="center", fontsize=7.5, color=PURPLE)
        elif (country, city) == neighbor:
            _card(ax, x, y, cw, ch, f"{country}:{city} · neighbor",
                  ["summary only", "via route"], face="#FFF7ED",
                  edge=ORANGE, alpha=0.95)
        else:
            _card(ax, x, y, cw, ch, f"{country}:{city}",
                  ["not observable"], face="#F1F5F9", alpha=0.38,
                  shadow=False)
    hx, hy = _GEOM["cards"][(here_c, here_city)]
    nx, ny = _GEOM["cards"][neighbor]
    ax.add_patch(FancyArrowPatch((hx + cw * 0.55, hy - 0.05),
                                 (nx + cw * 0.45, ny - 0.05),
                                 connectionstyle="arc3,rad=0.18",
                                 arrowstyle="<|-|>", mutation_scale=11,
                                 color=ORANGE, lw=2.0, ls=(0, (5, 2)), zorder=5))
    ax.add_patch(FancyBboxPatch((0.55, 0.18), 8.9, 0.52,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#F8FAFC", ec="#CBD5E1", lw=0.8))
    ax.text(0.75, 0.44, "legal_actions:", fontsize=8, color="#475569",
            fontweight="bold", va="center")
    ax.text(2.05, 0.44,
            f"{agent['at']}:work   ·   {agent['at']}:trade   ·   travel:c1:b",
            fontsize=8.5, family="monospace", color=PURPLE, va="center")
    ax.text(0.32, 5.78, "What an agent sees",
            fontsize=11.5, fontweight="bold", color="#0F172A")
    ax.text(0.32, 5.50,
            "full detail at its own node · ancestor aggregates · route-adjacent summaries · nothing else",
            fontsize=8, color="#475569")
    fig.savefig(FIGS / "traversal.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_dynamic_traversal(e33):
    s = e33["summary"]
    wr, st = e33["with_route"], e33["stranded"]
    steps = [r["step"] for r in wr]
    switch, travel = s["c0_phase1_first_seen_at_step"], s["travel_step"]

    fig, (ax, lane) = plt.subplots(
        2, 1, figsize=(8.6, 4.7), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 0.8], "hspace": 0.08})

    ax.axvspan(switch - 0.5, steps[-1] + 0.3, color=RED, alpha=0.05, zorder=0)
    ax.axvline(switch - 0.5, color=RED, lw=1.0, ls=(0, (4, 3)), alpha=0.7)
    ax.text(switch - 0.1, max(r["world_gdp"] for r in wr) * 0.98,
            "c0 enters austerity\n(work yields 0)", fontsize=7.5, color=RED,
            ha="left", va="top")
    ax.plot(steps, [r["world_gdp"] for r in wr], "-o", color=TEAL, lw=2.2,
            markersize=3.5, label="with route (agent re-locates)", zorder=5)
    ax.plot(steps, [r["world_gdp"] for r in st], "--s", color=SLATE, lw=1.8,
            markersize=3, label="no route (agent stranded)", zorder=4)
    travel_gdp = next(r["world_gdp"] for r in wr if r["step"] == travel)
    ax.annotate(f"agent crosses · toll −{s['toll_paid']}",
                xy=(travel, travel_gdp), xytext=(travel + 1.6, travel_gdp - 9),
                fontsize=7.5, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1,
                                connectionstyle="arc3,rad=0.25"))
    final_w, final_s = wr[-1]["world_gdp"], st[-1]["world_gdp"]
    ax.annotate("", xy=(steps[-1] + 0.25, final_w),
                xytext=(steps[-1] + 0.25, final_s),
                arrowprops=dict(arrowstyle="<->", color=TEAL, lw=1.2))
    ax.text(steps[-1] - 0.15, (final_w + final_s) / 2,
            f"+{s['mobility_gain']}\nmobility\ngain", fontsize=7.5, color=TEAL,
            ha="right", va="center")
    ax.set_ylabel("World GDP (derived aggregate)")
    ax.set_ylim(0, final_w * 1.12)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7.5, loc="upper left")

    # location lane (with-route scenario)
    lane.axvspan(switch - 0.5, steps[-1] + 0.3, color=RED, alpha=0.05, zorder=0)
    lane.broken_barh([(0.5, travel - 0.5 - 0.5)], (0.2, 0.6),
                     facecolors=PURPLE, alpha=0.75)
    lane.broken_barh([(travel + 0.5, steps[-1] - travel - 0.5 + 0.3)],
                     (0.2, 0.6), facecolors=ORANGE, alpha=0.75)
    lane.plot([travel], [0.5], marker="D", color="white", markersize=7,
              markeredgecolor=ORANGE, markeredgewidth=1.6, zorder=5)
    lane.text((0.5 + travel - 0.5) / 2, 0.5, "working in c0", fontsize=7.5,
              color="white", ha="center", va="center", fontweight="bold")
    lane.text((travel + 0.5 + steps[-1] + 0.3) / 2, 0.5, "working in c1",
              fontsize=7.5, color="white", ha="center", va="center",
              fontweight="bold")
    lane.set_ylim(0, 1)
    lane.set_yticks([0.5])
    lane.set_yticklabels(["agent\nlocation"], fontsize=7)
    lane.set_xlabel("Composite step")
    lane.set_xlim(0.3, steps[-1] + 0.7)
    lane.set_xticks(range(2, steps[-1] + 1, 2))
    for spine in ("top", "right", "left"):
        lane.spines[spine].set_visible(False)
    lane.tick_params(left=False)
    fig.subplots_adjust(left=0.09, right=0.975, top=0.97, bottom=0.13)
    fig.savefig(FIGS / "dynamic_traversal.png", dpi=200)
    plt.close(fig)


def fig_sprint(e34):
    """E34: solved-vs-budget curves per allocation condition on owsb-atomic."""
    styles = {
        "fixed": (SLATE, "-", "o", "fixed 4/task (standard protocol)"),
        "round_robin": (BLUE, "--", "s", "round-robin (pinned seed)"),
        "greedy": (RED, ":", "v", "greedy min-failing (pinned seed)"),
        "round_robin_jitter": (TEAL, "-", "D", "round-robin + per-attempt seeds"),
        "greedy_jitter": (ORANGE, "-.", "^", "greedy + per-attempt seeds"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    # jitter variants first so the coincident pinned-seed dashes stay visible
    order = ["round_robin_jitter", "greedy_jitter",
             "fixed", "round_robin", "greedy"]
    by_name = {c["condition"]: c for c in e34["conditions"]}
    for z, name in enumerate(n for n in order if n in by_name):
        cond = by_name[name]
        color, ls, marker, label = styles[name]
        xs = [0] + [e["attempt_index"] for e in cond["events"]]
        ys = [0] + [e["cumulative_solved"] for e in cond["events"]]
        ax.plot(xs, ys, ls, color=color, lw=1.9, marker=marker, markersize=3,
                markevery=max(1, len(xs) // 16), label=label, zorder=3 + z)
    ax.axhline(e34["conditions"][0]["n_tasks"], color="#9CA3AF", lw=0.8,
               ls=(0, (1, 3)))
    ax.text(1, e34["conditions"][0]["n_tasks"] + 0.25, "all tasks",
            fontsize=7, color="#9CA3AF")
    ax.set_xlabel(f"Repair attempts consumed (budget {e34['total_budget']})")
    ax.set_ylabel("Tasks solved")
    ax.set_xlim(0, e34["total_budget"] + 1)
    ax.set_ylim(0, e34["conditions"][0]["n_tasks"] + 2)
    ax.set_yticks(range(0, e34["conditions"][0]["n_tasks"] + 1, 5))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(FIGS / "sprint.png", dpi=200)
    plt.close(fig)


def table_planning(e22):
    labels = {
        "code_d3": "\\textbf{Lookahead d=3 via synthesized code}",
        "oracle_d3": "Lookahead d=3 via ground truth (bound)",
        "llm_d2": "Lookahead d=2 via LLM next-state",
        "reactive_heuristic": "Reactive heuristic (no model)",
        "random": "Random policy (5 seeds, mean)",
    }
    rows = []
    for r in e22["rows"]:
        secs = ("---" if r["episode_seconds"] is None
                else f"{r['episode_seconds']:.2f}")
        rows.append(f"{labels[r['planner']]} & {r['score']:.1f} & {secs} \\\\")
    (TABLES / "planning.tex").write_text(
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "Policy & Episode score & Planning time (s) \\\\\n"
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
        macro("NumExperiments", "34"),
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
    # E20-E23 (round 3)
    e20, e21 = d["e20_complexity"], d["e21_stochastic"]
    e22, e23 = d["e22_planning"], d["e23_self_check"]
    c20 = {s["n_rules"]: s for s in e20["summary"]}
    plan = {r["planner"]: r for r in e22["rows"]}
    lines += [
        macro("CplxFour", f"{c20[4]['mean_probe_accuracy']:.2f}"),
        macro("CplxEight", f"{c20[8]['mean_probe_accuracy']:.2f}"),
        macro("CplxTwelve", f"{c20[12]['mean_probe_accuracy']:.2f}"),
        macro("CplxSixteen", f"{c20[16]['mean_probe_accuracy']:.2f}"),
        macro("StochAccept", pct(e21["summary"]["acceptance_rate"])),
        macro("StochArrivalErr", f"{100 * e21['summary']['mean_arrival_abs_error']:.1f}"),
        macro("StochDetAcc", pct(e21["summary"]["mean_deterministic_accuracy"])),
        macro("StochOracleExact", pct(e21["summary"]["mean_oracle_bit_exact"])),
        macro("PlanCode", f"{plan['code_d3']['score']:.1f}"),
        macro("PlanOracle", f"{plan['oracle_d3']['score']:.1f}"),
        macro("PlanLLM", f"{plan['llm_d2']['score']:.1f}"),
        macro("PlanReactive", f"{plan['reactive_heuristic']['score']:.1f}"),
        macro("PlanRandom", f"{plan['random']['score']:.1f}"),
        macro("PlanLLMSeconds", f"{plan['llm_d2']['episode_seconds']:.0f}"),
        macro("PlanCodeSeconds", f"{plan['code_d3']['episode_seconds']:.2f}"),
        macro("SelfCheckPrograms", str(e23["n_programs"])),
        macro("SelfCheckPairs", str(e23["n_program_probe_pairs"])),
        macro("SelfCheckPrecision", pct(e23["flag_precision"])),
        macro("SelfCheckRecall", pct(e23["flag_recall"])),
        macro("SelfCheckSpearman", f"{e23['spearman_agreement_vs_accuracy']:.2f}"),
    ]
    # E24-E27 (round 4: moral structure)
    e24, e25 = d["e24_aggregators"], d["e25_constraints"]
    e26, e27 = d["e26_parliament"], d["e27_rubric_pluralism"]
    rhos = sorted(v["spearman"] for v in e27["pairwise"].values())
    lines += [
        macro("AggUtilWorst", str(e24["summary"]["utilitarian_sum"]["worst_off"])),
        macro("AggMaximinWorst", str(e24["summary"]["maximin"]["worst_off"])),
        macro("AggMaximinR", str(e24["summary"]["maximin"]["r"])),
        macro("ViolationsDialOnly", str(e25["summary"]["dial_only_total_violations"])),
        macro("ConstraintOutcomeCost", str(e25["summary"]["max_outcome_cost_of_constraint"])),
        macro("ParliamentHedges",
              "never" if e26["parliament_never_strictly_worst"] else "sometimes"),
        macro("RubricPluralismRange", f"{rhos[0]:.2f}--{rhos[-1]:.2f}"),
        macro("RubricDeontVsUtil", f"{e27['pairwise']['utilitarian_vs_deontological']['spearman']:.2f}"),
        macro("RubricCareVsUtil", f"{e27['pairwise']['utilitarian_vs_care_ethics']['spearman']:.2f}"),
    ]
    # E28-E29 (benchmark-scale repair ablation: single-shot vs in-world)
    e28, e29 = d["e28_swebench_ablation"], d["e29_swebench_staged"]
    a = {s["model"]: s for s in e28["summary"]}
    g = {s["model"]: s for s in e29["summary"]}
    lines += [
        macro("SweAtomicN", str(e28["n_instances"])),
        macro("SweStagedN", str(e29["n_instances"])),
        macro("SweBudget", str(e28["budget"])),
        macro("SweAtomicSSSmall", f"{a['qwen2.5:1.5b']['single_shot_pass1']:.2f}"),
        macro("SweAtomicSSMid", f"{a['qwen2.5:3b']['single_shot_pass1']:.2f}"),
        macro("SweAtomicSSBig", f"{a['qwen2.5:7b']['single_shot_pass1']:.2f}"),
        macro("SweAtomicIWBudgetBig", f"{a['qwen2.5:7b']['in_world_pass_budget']:.2f}"),
        macro("SweStagedSSMid", f"{g['qwen2.5:3b']['single_shot_pass1']:.2f}"),
        macro("SweStagedSSBig", f"{g['qwen2.5:7b']['single_shot_pass1']:.2f}"),
        macro("SweStagedIWBudgetMid", f"{g['qwen2.5:3b']['in_world_pass_budget']:.2f}"),
        macro("SweStagedIWBudgetBig", f"{g['qwen2.5:7b']['in_world_pass_budget']:.2f}"),
        macro("SweStagedDeltaSmall", f"{g['qwen2.5:1.5b']['delta_budget_minus_ss']:+.2f}"),
        macro("SweStagedDeltaMid", f"{g['qwen2.5:3b']['delta_budget_minus_ss']:+.2f}"),
        macro("SweStagedDeltaBig", f"{g['qwen2.5:7b']['delta_budget_minus_ss']:+.2f}"),
    ]
    # E30-E32 (composition, nesting, and changing rules)
    e30, e31, e32 = (d["e30_composition"], d["e31_nested_fidelity"],
                     d["e32_regime_switch"])
    s30 = {s["condition"]: s for s in e30["summary"]}
    s32 = {s["condition"]: s for s in e32["summary"]}
    # 4 internal rules per sector: the parameter groups prod_*, rec_*,
    # decay_* in the results JSON, plus the unparameterized wait rule.
    child_rules = len({k.split("_")[0] for k in e30["sectors"][0]}) + 1
    n31 = e31["counts"]
    lines += [
        macro("CompMonoAcc", f"{s30['monolithic']['mean_probe_accuracy']:.2f}"),
        macro("CompMonoCI", ci_str(s30["monolithic"]["pooled_ci"])),
        macro("CompCompAcc", f"{s30['compositional']['mean_probe_accuracy']:.2f}"),
        macro("CompCompCI", ci_str(s30["compositional"]["pooled_ci"])),
        macro("CompRules", str(child_rules * len(e30["sectors"]))),
        macro("CompChildRules", str(child_rules)),
        macro("NestSteps", str(n31["steps"])),
        macro("NestExact", f"{n31['leaf_exact']}/{n31['steps']}"),
        macro("RegimePhasedExact",
              f"{s32['phased']['exact_full_rollouts']}/{s32['phased']['replicates']}"),
        macro("RegimeMonoExact",
              f"{s32['monolithic']['exact_full_rollouts']}/{s32['monolithic']['replicates']}"),
        macro("RegimeLLMPre", f"{s32['llm_proxy']['mean_pre_boundary_accuracy']:.2f}"),
        macro("RegimeLLMPost", f"{s32['llm_proxy']['mean_post_boundary_accuracy']:.2f}"),
    ]
    # E33 (dynamic rules x composition x traversal demonstration)
    s33 = d["e33_dynamic_traversal"]["summary"]
    lines += [
        macro("DynSwitchStep", str(s33["c0_phase1_first_seen_at_step"])),
        macro("DynTravelStep", str(s33["travel_step"])),
        macro("DynToll", str(s33["toll_paid"])),
        macro("DynGdpWith", str(s33["final_world_gdp_with_route"])),
        macro("DynGdpStranded", str(s33["final_world_gdp_stranded"])),
        macro("DynMobilityGain", str(s33["mobility_gain"])),
    ]
    # E34 (sprint composite: attempt allocation on owsb-atomic)
    s34 = {c["condition"]: c for c in d["e34_composite_swe"]["summary"]}

    def sprint_solved(cond):
        return f"{s34[cond]['solved']}/{s34[cond]['n_tasks']}"

    lines += [
        macro("SprintBudget", str(d["e34_composite_swe"]["total_budget"])),
        macro("SprintFixedSolved", sprint_solved("fixed")),
        macro("SprintFixedAttempts", str(s34["fixed"]["attempts_consumed"])),
        macro("SprintRRSolved", sprint_solved("round_robin")),
        macro("SprintGreedySolved", sprint_solved("greedy")),
        macro("SprintGreedyTarpit",
              str(max(s34["greedy"]["attempts_per_task"].values()))),
        macro("SprintRRJitterSolved", sprint_solved("round_robin_jitter")),
        macro("SprintGreedyJitterSolved", sprint_solved("greedy_jitter")),
    ]
    # E36 (representations: composition vs monolithic learners)
    e36 = d["e36_representations"]
    g36 = {r["k"]: r for r in e36["leg_generalization"]}
    i36 = e36["leg_interference"]
    se36 = {r["n_cap"]: r for r in e36["leg_sample_efficiency"]["rows"]}

    def acc(cell):
        return f"{cell:.2f}"

    lines += [
        macro("RepMonoGenLo", acc(g36[2]["monolith"]["acc"])),
        macro("RepMonoGenHi", acc(g36[5]["monolith"]["acc"])),
        macro("RepCompGenLo", acc(g36[2]["composite_learned"]["acc"])),
        macro("RepCompGenHi", acc(g36[5]["composite_learned"]["acc"])),
        macro("RepKnnGenHi", acc(g36[5]["knn1"]["acc"])),
        macro("RepIntfMono", acc(i36["monolith_sequential_retained"])),
        macro("RepIntfComp", acc(i36["composite_learned_retained"])),
        macro("RepSampMonoLo", acc(se36[100]["monolith_acc"])),
        macro("RepSampCompLo", acc(se36[100]["composite_learned_acc"])),
        macro("RepMonoParams", str(g36[5]["monolith"]["n_params"])),
        macro("RepCompParams", str(g36[5]["composite_learned"]["n_params"])),
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
    fig_complexity(data["e20_complexity"])
    fig_composition(data["e31_nested_fidelity"])
    fig_composition_cliff(data["e20_complexity"], data["e30_composition"])
    fig_traversal(data["e31_nested_fidelity"])
    fig_dynamic_traversal(data["e33_dynamic_traversal"])
    fig_sprint(data["e34_composite_swe"])
    fig_representations(data["e36_representations"])
    table_planning(data["e22_planning"])
    table_swebench(data["e28_swebench_ablation"], data["e29_swebench_staged"])
    table_composition(data["e30_composition"], data["e32_regime_switch"])
    numbers_tex(data)
    print("assets written to paper/figs, paper/tables, paper/numbers.tex")


if __name__ == "__main__":
    main()
