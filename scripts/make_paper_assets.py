"""Generate every paper figure, table, and numbers.tex from experiments/results/.

Single command reproducibility:  python3 scripts/make_paper_assets.py
"""

import json
import re
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
    "e28_repairbench_ablation", "e29_repairbench_staged",
    "e30_composition", "e31_nested_fidelity", "e32_regime_switch",
    "e33_dynamic_traversal", "e34_composite_swe", "e35_sprint_ladder", "e36_representations",
    "e37_induction", "e38_induction_scale", "e39_perception_fidelity", "e40_perceive_forecast",
    "e41_nonstationary", "e42_agent_traversal", "e43_active_induction",
    "e44_emergent_economy", "e46_many_worlds", "e45_next_token",
    "e47_relativity", "e49_path_integral", "e48_corporate_world", "e50_trading",
    "e51_startups", "e52_denoise", "e53_sheaf",
    "e54_bounds", "e55_infogeom", "e56_transport", "e57_world_specs", "e58_brain", "e59_brain_arch",
    "e60_io_boundary", "e61_trained_wm_control", "e62_branch_gate",
    "e63_world_model_bakeoff", "e65_minigrid_bench",
]


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def pct(x):
    return f"{100 * x:.0f}\\%"


def repo_metrics():
    """Count the live codebase so the Implementation paragraph can't drift.
    Core = openworld/*.py minus the serve/CLI layer (serve, cli, _tmux), which is
    the only place third-party deps (fastapi/uvicorn/click/rich) are allowed."""
    pkg = ROOT / "openworld"
    serve_layer = {"serve.py", "cli.py", "_tmux.py"}
    core = [p for p in sorted(pkg.glob("*.py")) if p.name not in serve_layer]
    core_loc = sum(len(p.read_text().splitlines()) for p in core)
    tests = sorted((ROOT / "tests").glob("test_*.py"))
    test_fns = sum(t.read_text().count("def test_") for t in tests)
    return {
        "core_modules": len(core),
        "core_loc": core_loc,
        "test_functions": test_fns,
        "models": scan_models(),
    }


_MODEL_RE = re.compile(r"^[a-z][a-z0-9._-]*:[0-9][0-9.]*b$")


def scan_models():
    """Every Ollama model id that appears anywhere in the cached result JSONs, so
    the Reproducibility section's model list is derived from the runs, not hand-kept."""
    found = set()

    def walk(o):
        if isinstance(o, str):
            if _MODEL_RE.match(o):
                found.add(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for f in sorted(RESULTS.glob("*.json")):
        try:
            walk(json.loads(f.read_text()))
        except Exception:
            pass
    # family-then-size ordering for a stable, readable list
    def key(m):
        name, size = m.split(":")
        return (name, float(size[:-1]))
    return sorted(found, key=key)


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


def fig_induction_scale(e38):
    """E38: induction-from-traces does not improve with generator capability."""
    ladder = e38["ladder"]
    short = {"qwen2.5:7b": "qwen2.5\n7b", "qwen3-coder:30b": "qwen3-coder\n30b",
             "gpt-oss:20b": "gpt-oss\n20b"}
    models = [m["model"] for m in ladder]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    x = range(len(models))
    ax.axhline(1.0, color=TEAL, lw=1.8, ls=(0, (4, 3)),
               label="rule-text synthesis (E37 anchor)")
    ax.plot(x, [m["mean_in_dist_bigK"] for m in ladder], "-o", color=BLUE,
            lw=2, markersize=6, label="induction from traces (in-dist)")
    ax.plot(x, [m["mean_ood_bigK"] for m in ladder], "--s", color=RED,
            lw=2, markersize=6, label="induction from traces (10× OOD)")
    for xi, m in zip(x, ladder):
        ax.text(xi, m["mean_in_dist_bigK"] + 0.04, f"{m['mean_in_dist_bigK']:.2f}",
                ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([short.get(m, m) for m in models], fontsize=8)
    ax.set_ylabel("Exact probe accuracy")
    ax.set_xlabel("Generator (increasing capability →)")
    ax.set_ylim(0, 1.08)
    # Add horizontal margin so the rightmost marker and its centered value
    # label sit fully inside the axes instead of running off the right edge.
    ax.set_xlim(-0.4, len(models) - 1 + 0.4)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("Induction from traces vs the rule-text anchor", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "induction_scale.png", dpi=200)
    plt.close(fig)


def fig_active_induction(e43):
    """E43: acting to disambiguate identifies the rule where passive observation
    plateaus. Left: candidate-elimination curves for a hidden rule the passive
    policy never resolves. Right: mean steps-to-identify across hidden rules."""
    rows = e43["rows"]
    s = e43["summary"]
    # representative rule: one the passive policy never resolves (high k)
    rep = next((r for r in rows if r["passive_steps"] is None), rows[0])
    fig, (axc, axb) = plt.subplots(1, 2, figsize=(7.4, 3.2),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    ac, pc = rep["active_curve"], rep["passive_curve"]
    axc.plot(range(1, len(ac) + 1), ac, "-o", color=BLUE, lw=2, markersize=3,
             label="active (acts to disambiguate)")
    axc.plot(range(1, len(pc) + 1), pc, "--s", color=RED, lw=2, markersize=3,
             label="passive (random policy)")
    axc.axhline(1, color=SLATE, lw=1, ls=":")
    if rep["active_steps"]:
        # Offset the "identified" label up and to the right into clear space so
        # it no longer sits on top of the descending curves / the arrow.
        axc.annotate("identified", xy=(rep["active_steps"], 1),
                     xytext=(rep["active_steps"] + 9, 13), fontsize=8, color=BLUE,
                     ha="left", va="center",
                     arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
    axc.text(len(pc), pc[-1] + 1.5, "passive plateaus", fontsize=8, color=RED,
             ha="right")
    axc.set_yscale("log")
    axc.set_xlabel("Transitions observed")
    axc.set_ylabel("Candidate rules remaining")
    axc.set_title(f"Eliminating {s['n_candidates']} candidates "
                  f"($k$={rep['true_rule']['k']})", fontsize=9, loc="left")
    axc.legend(fontsize=7.5, loc="upper right")

    labels = ["passive", "active", "clairvoyant\n(knows rule)"]
    vals = [s["passive_mean_steps"], s["active_mean_steps"],
            s["clairvoyant_mean_steps"]]
    colors = [RED, BLUE, TEAL]
    bars = axb.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        axb.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}",
                 ha="center", fontsize=8)
    axb.text(0, 1.2, f"{s['passive_unresolved']}/{s['n_rules']}\nnever", ha="center",
             fontsize=7.5, color="white", weight="bold")
    # Headroom so the tallest bar's value label is not clipped off the top.
    axb.set_ylim(0, max(vals) * 1.15)
    axb.set_ylabel("Mean steps to identify")
    axb.set_title("Lower is better", fontsize=9, loc="left")
    axb.tick_params(axis="x", labelsize=7.5)
    fig.suptitle("Active world-model induction: acting beats observing",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "active_induction.png", dpi=200)
    plt.close(fig)


def fig_emergent_economy(e44):
    """E44: four macro phenomena emerging from composed verified rules, each
    isolated by a causal toggle on one rule."""
    c1, c2 = e44["claim1_price_formation"], e44["claim2_inflation"]
    c3, c4 = e44["claim3_inequality"], e44["claim4_dial"]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0))
    (a1, a2), (a3, a4) = axes

    # A: price formation
    a1.plot(c1["scarce_traj"], color=BLUE, lw=2, label="scarce supply")
    a1.plot(c1["abundant_traj"], color=ORANGE, lw=2, label="abundant supply")
    a1.axhline(c1["scarce_supply_price"], color=BLUE, lw=0.8, ls=":")
    a1.axhline(c1["abundant_supply_price"], color=ORANGE, lw=0.8, ls=":")
    a1.set_title("A. Price formation: scarcity sets the price", fontsize=9.5, loc="left")
    a1.set_xlabel("Tick"); a1.set_ylabel("Market price")
    a1.legend(fontsize=8, loc="center right")

    # B: inflation (money supply, log scale), burn sink off vs on
    a2.plot(c2["off_money_traj"], color=RED, lw=2, label="burn sink OFF (faucet only)")
    a2.plot(c2["on_money_traj"], color=TEAL, lw=2, label="burn sink ON")
    a2.set_yscale("log")
    a2.set_title("B. Inflation: a sink curbs the money supply", fontsize=9.5, loc="left")
    a2.set_xlabel("Tick"); a2.set_ylabel("Money supply (log)")
    a2.legend(fontsize=8, loc="lower right")

    # C: inequality (Gini), redistribution off vs on
    a3.plot(c3["off_gini_traj"], color=RED, lw=2, label="redistribution OFF")
    a3.plot(c3["on_gini_traj"], color=TEAL, lw=2, label="redistribution ON")
    a3.set_title("C. Inequality emerges; redistribution flattens it",
                 fontsize=9.5, loc="left")
    a3.set_xlabel("Tick"); a3.set_ylabel("Wealth Gini")
    a3.set_ylim(bottom=0)
    a3.legend(fontsize=8, loc="center right")

    # D: selfish vs cooperative dial (welfare + top-agent gold)
    groups = ["total\nwelfare", "richest\nagent"]
    selfish = [c4["selfish_welfare"], c4["selfish_max_gold"]]
    coop = [c4["cooperative_welfare"], c4["cooperative_max_gold"]]
    x = range(len(groups))
    w = 0.38
    a4.bar([i - w / 2 for i in x], selfish, w, color=PURPLE, label="selfish")
    a4.bar([i + w / 2 for i in x], coop, w, color=TEAL, label="cooperative")
    for i, (sv, cv) in enumerate(zip(selfish, coop)):
        a4.text(i - w / 2, sv, f"{sv:.0f}", ha="center", va="bottom", fontsize=7.5)
        a4.text(i + w / 2, cv, f"{cv:.0f}", ha="center", va="bottom", fontsize=7.5)
    a4.set_xticks(list(x)); a4.set_xticklabels(groups, fontsize=8.5)
    a4.set_title("D. Dial: cooperation lifts the total, selfishness the top",
                 fontsize=9.5, loc="left")
    a4.set_ylabel("Gold")
    a4.legend(fontsize=8, loc="upper right")

    fig.suptitle("Emergent economy from composed verified rules (E44)",
                 fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGS / "emergent_economy.png", dpi=200)
    plt.close(fig)


def fig_corporate_world(e48):
    """E48: a composite company world. Top: the org as a nested CompositeWorld
    with revenue aggregators and agents at three levels. Then: individual-vs-
    collective Pareto, value-of-action by level, granularity-bound perception
    from real transcripts, and per-role policy."""
    divs = e48["divisions"]
    names = list(divs)
    fig = plt.figure(figsize=(8.4, 9.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.5, wspace=0.28)
    sch = fig.add_subplot(gs[0, :])
    a = fig.add_subplot(gs[1, 0]); b = fig.add_subplot(gs[1, 1])
    c = fig.add_subplot(gs[2, 0]); dax = fig.add_subplot(gs[2, 1])

    # --- top: org composite schematic ---
    sch.set_xlim(0, 10); sch.set_ylim(0, 2.9); sch.axis("off")
    _panel(sch, 0.2, 0.25, 9.6, 2.5, "company  ·  Nimbus Cloud (CompositeWorld)",
           SLATE, alpha_fill=0.03)
    _chip(sch, 8.3, 2.55, "_agg: revenue=Σ", SLATE, fontsize=6.8)
    xs = [0.55 + i * 1.85 for i in range(len(names))]
    for x, n in zip(xs, names):
        _card(sch, x, 0.95, 1.62, 1.15, n,
              [f"${divs[n]['R0']}M", f"a={divs[n]['a']}"], bold_edge=BLUE)
        for k in range(min(4, divs[n]["hc"])):              # ICs as dots
            sch.plot([x + 0.25 + k * 0.32], [0.7], "o", ms=4, color=TEAL, zorder=5)
        sch.add_patch(FancyArrowPatch((x + 0.81, 2.12), (8.3, 2.42),
                                      arrowstyle="-|>", mutation_scale=6,
                                      ls=(0, (2, 2)), color=BLUE, lw=0.7, alpha=0.5))
    sch.text(0.4, 2.42, "CEO → budget across divisions", fontsize=6.6, color=SLATE, zorder=6)
    sch.text(0.4, 2.2, "director → within division", fontsize=6.6, color=BLUE, zorder=6)
    sch.text(0.4, 0.5, "SWE → own project   (ICs ●)", fontsize=6.6, color=TEAL, zorder=6)
    sch.set_title("The composite world: company > divisions > individuals; revenue "
                  "aggregates upward; agents act at their level",
                  fontsize=8.6, loc="left", color="#334155")

    # A: individual vs collective Pareto
    par = e48["pareto"]
    rho = [p["rho"] for p in par]
    a.plot(rho, [p["company_growth"] for p in par], "-o", color=BLUE, lw=2,
           markersize=3, label="company growth")
    a.set_xlabel("selfishness dial $\\rho$"); a.set_ylabel("company growth", color=BLUE)
    a.tick_params(axis="y", labelcolor=BLUE)
    a2 = a.twinx()
    a2.plot(rho, [p["promo_gini"] for p in par], "--s", color=RED, lw=2,
            markersize=3, label="promotion Gini")
    a2.set_ylabel("promotion Gini", color=RED); a2.tick_params(axis="y", labelcolor=RED)
    a.set_title("A. Individual vs collective", fontsize=9.5, loc="left")

    # B: value-of-action by level
    voa = e48["value_of_action"]
    levels = ["ceo", "director", "ic"]
    bars = b.bar(["CEO", "director", "IC"], [voa[f"{l}_pct"] for l in levels],
                 color=[SLATE, BLUE, TEAL])
    for bar, l in zip(bars, levels):
        b.text(bar.get_x() + bar.get_width() / 2, voa[f"{l}_pct"] + 0.3,
               f"+{voa[f'{l}_pct']:.0f}%", ha="center", fontsize=8)
    b.set_ylabel("marginal company growth (%)")
    b.set_title("B. Value-of-action by level", fontsize=9.5, loc="left")

    # C: granularity-bound perception (from real transcripts)
    isig = e48["perception"]["individual_signal"]
    groups = ["division metric\n(growth)", "individual\n(promo signal)"]
    src_a = [1.0, isig["recover_from_all_hands"]]            # all-hands
    src_b = [1.0, isig["recover_from_one_on_one"]]           # review / 1:1
    xg = range(len(groups)); w = 0.36
    c.bar([i - w / 2 for i in xg], src_a, w, color=SLATE, label="all-hands (aggregate)")
    c.bar([i + w / 2 for i in xg], src_b, w, color=TEAL, label="review / 1:1 (local)")
    c.set_xticks(list(xg)); c.set_xticklabels(groups, fontsize=8)
    c.set_ylabel("signal recovered"); c.set_ylim(0, 1.15)
    c.set_title(f"C. Perception ({e48['perception']['corpus']['n_records']} real transcripts)",
                fontsize=9.5, loc="left")
    c.legend(fontsize=7, loc="upper right")

    # D: per-role policy, principled vs greedy
    pol = e48["policies"]
    roles = ["ceo", "director", "ic"]
    xr = range(len(roles))
    dax.bar([i - w / 2 for i in xr], [pol[r]["greedy"] for r in roles], w,
            color=ORANGE, label="greedy")
    dax.bar([i + w / 2 for i in xr], [pol[r]["principled"] for r in roles], w,
            color=BLUE, label="principled")
    dax.set_xticks(list(xr)); dax.set_xticklabels(["CEO", "director", "IC"], fontsize=8.5)
    dax.set_ylabel("company growth")
    dax.set_title("D. Navigation policy per role", fontsize=9.5, loc="left")
    dax.legend(fontsize=7.5)

    fig.suptitle("A composite corporate world: individual, division, and company goals (E48)",
                 fontsize=10.5, x=0.02, ha="left", y=0.995)
    fig.subplots_adjust(top=0.945, bottom=0.05, left=0.09, right=0.95)
    fig.savefig(FIGS / "corporate_world.png", dpi=200)
    plt.close(fig)


def table_corporate_world(e48):
    """E48: the four agent-level results."""
    par, voa = e48["pareto"], e48["value_of_action"]
    isig = e48["perception"]["individual_signal"]
    aligned, selfish = par[0], par[-1]
    lines = ["\\begin{tabular}{lll}", "\\toprule",
             "Question & Result & Takeaway \\\\", "\\midrule",
             f"Individual vs collective & growth {aligned['company_growth'] * 100:.0f}\\% "
             f"(aligned) vs {selfish['company_growth'] * 100:.0f}\\% (selfish), Gini "
             f"{aligned['promo_gini']:.2f}$\\to${selfish['promo_gini']:.2f} & "
             "selfish concentration hurts the aggregate \\\\",
             f"Value-of-action & CEO +{voa['ceo_pct']:.0f}\\%, director "
             f"+{voa['director_pct']:.0f}\\%, IC +{voa['ic_pct']:.0f}\\% & "
             "leverage lives at the top \\\\",
             f"Perception (real transcripts) & individual signal "
             f"{isig['recover_from_one_on_one'] * 100:.0f}\\% (1:1) vs "
             f"{isig['recover_from_all_hands'] * 100:.0f}\\% (all-hands) & "
             "perception is granularity-bound \\\\",
             "\\bottomrule", "\\end{tabular}"]
    (TABLES / "corporate_world.tex").write_text("\n".join(lines) + "\n")


def fig_trading(e50):
    """E50: same-day trading world model. Equity curves (honest OOS vs lookahead
    vs SPY vs random), cost fragility, synthetic-edge validation, and risk-
    adjusted honesty."""
    cur = e50["equity_curves"]
    real = e50["real"]
    fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(8.4, 6.4))

    # A: equity curves
    a.plot(cur["lookahead"], color=SLATE, lw=1.6, ls=":", label="lookahead (cheating)")
    a.plot(cur["honest"], color=BLUE, lw=2, label="honest OOS (after cost)")
    a.plot(cur["spy"], color=TEAL, lw=2, label="buy-and-hold SPY")
    a.plot(cur["random"], color=RED, lw=1.4, label="random")
    a.axhline(1.0, color="k", lw=0.6)
    a.set_ylabel("equity (×)"); a.set_xlabel("trading day")
    a.set_yscale("log")
    a.set_title("A. Equity curves (walk-forward)", fontsize=9.5, loc="left")
    a.legend(fontsize=7, loc="upper left")

    # B: cost fragility
    cs = e50["cost_sweep_annualized"]
    ks = list(cs)
    bars = b.bar(ks, [cs[k] * 100 for k in ks],
                 color=[BLUE if cs[k] > 0 else RED for k in ks])
    b.axhline(0, color="k", lw=0.7)
    b.set_ylabel("annualized return (%)")
    b.set_title("B. Cost fragility (edge vanishes)", fontsize=9.5, loc="left")
    b.tick_params(axis="x", labelsize=8)

    # C: synthetic validation (detector recovers a known edge)
    syn = e50["synthetic"]
    c.bar(["strategy", "random"], [syn["strategy"]["sharpe"], syn["random"]["sharpe"]],
          color=[BLUE, RED])
    c.set_ylabel("Sharpe")
    c.set_title("C. Synthetic validation (known +40bps edge)", fontsize=9.5, loc="left")

    # D: risk-adjusted honesty (Sharpe) vs SPY
    series = [("honest\n(cost)", real["honest_oos"]["sharpe"], BLUE),
              ("honest\n(0bps)", real["honest_no_cost"]["sharpe"], "#94A3B8"),
              ("lookahead", real["lookahead"]["sharpe"], SLATE),
              ("SPY", real["spy_buy_hold"]["sharpe"], TEAL)]
    d.bar([s[0] for s in series], [s[1] for s in series], color=[s[2] for s in series])
    for i, s in enumerate(series):
        d.text(i, s[1] + 0.02, f"{s[1]:.2f}", ha="center", fontsize=7.5)
    d.set_ylabel("Sharpe (out-of-sample)")
    d.set_title("D. Risk-adjusted: honest < SPY", fontsize=9.5, loc="left")
    d.tick_params(axis="x", labelsize=7.5)

    fig.suptitle("Same-day trading world model: honest out-of-sample on real data (E50)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGS / "trading.png", dpi=200)
    plt.close(fig)


def table_trading(e50):
    """E50: honest OOS vs baselines (after cost)."""
    r = e50["real"]
    rows = [("Honest OOS (after cost)", r["honest_oos"]),
            ("Honest (no cost)", r["honest_no_cost"]),
            ("Lookahead (cheating)", r["lookahead"]),
            ("Buy-and-hold SPY", r["spy_buy_hold"]),
            ("Random", r["random"])]
    lines = ["\\begin{tabular}{lrrr}", "\\toprule",
             "Strategy & Annualized & Sharpe & Hit rate \\\\", "\\midrule"]
    for name, m in rows:
        hit = f"{m.get('hit_rate', float('nan')):.2f}" if "hit_rate" in m else "--"
        lines.append(f"{name} & {m['annualized'] * 100:+.1f}\\% & {m['sharpe']:.2f} & {hit} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "trading.tex").write_text("\n".join(lines) + "\n")


def fig_startups(e51):
    """E51: startup growth world model. Causal value-of-factor, the power law of
    returns, the no-PMF counterfactual, and early-vs-final predictability."""
    fig, ((a, b), (c, dax)) = plt.subplots(2, 2, figsize=(8.4, 6.6))

    # A: value-of-factor (causal)
    voa = e51["value_of_factor"]
    facs = sorted(voa, key=lambda k: -voa[k]["delta_value_pct"])
    vals = [voa[f]["delta_value_pct"] for f in facs]
    cols = [BLUE if f == "pmf" else (RED if f == "capital" else TEAL) for f in facs]
    bars = a.bar(facs, vals, color=cols)
    for bar, v in zip(bars, vals):
        a.text(bar.get_x() + bar.get_width() / 2, v + 3, f"+{v:.0f}%", ha="center", fontsize=8)
    a.set_ylabel("Δ batch value when lifted (%)")
    a.set_title("A. What drives growth (causal): PMF ≫ capital", fontsize=9.3, loc="left")
    a.tick_params(axis="x", labelsize=8.5)

    # B: power law (Lorenz-style cumulative value)
    cum = e51["cum_value_share"]
    n = len(cum)
    x = [100 * (i + 1) / n for i in range(n)]
    b.plot(x, [100 * c for c in cum], color=BLUE, lw=2.2)
    b.plot([0, 100], [0, 100], color=SLATE, lw=1, ls=":")
    td = e51["power_law"]["top_decile_share"]
    b.axvline(10, color=RED, lw=1, ls="--")
    b.text(12, 40, f"top 10% =\n{td:.0%} of value", color=RED, fontsize=8)
    b.set_xlabel("startups (ranked by value, %)"); b.set_ylabel("cumulative value (%)")
    b.set_title("B. Power law of returns", fontsize=9.3, loc="left")

    # C: counterfactual attribution
    cf = e51["counterfactual"]
    labels = ["base", "no PMF", "2× capital\n(low PMF)"]
    vc = [100, cf["no_pmf_value_pct_of_base"], cf["double_capital_lowpmf_pct_of_base"]]
    bars = c.bar(labels, vc, color=[TEAL, RED, ORANGE])
    for bar, v in zip(bars, vc):
        c.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=8)
    c.set_ylabel("batch value (% of base)")
    c.set_title("C. Money can't buy growth without PMF", fontsize=9.3, loc="left")
    c.tick_params(axis="x", labelsize=8)

    # D: predictability (month-6 traction vs final value)
    sc = e51["scatter"]
    rev6 = [max(x, 0.1) for x in sc["rev6"]]; val = [max(x, 0.1) for x in sc["value"]]
    dax.scatter(rev6, val, s=10, alpha=0.5, color=PURPLE)
    dax.set_xscale("log"); dax.set_yscale("log")
    dax.set_xlabel("month-6 revenue ($k)"); dax.set_ylabel("final value ($k)")
    pr = e51["predictability"]
    dax.set_title(f"D. Early signal informative, not decisive (ρ={pr['spearman_m6_vs_final']})",
                  fontsize=9.0, loc="left")

    fig.suptitle("Startup growth world model: a YC-style batch and what drives it (E51)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGS / "startups.png", dpi=200)
    plt.close(fig)


def table_startups(e51):
    """E51: value-of-factor and the power law."""
    voa, pw = e51["value_of_factor"], e51["power_law"]
    facs = sorted(voa, key=lambda k: -voa[k]["delta_value_pct"])
    lines = ["\\begin{tabular}{lr}", "\\toprule",
             "Factor (lifted to 90th pct) & $\\Delta$ batch value \\\\", "\\midrule"]
    for f in facs:
        suffix = " (2$\\times$)" if f == "capital" else ""
        lines.append(f"{f}{suffix} & {voa[f]['delta_value_pct']:+.0f}\\% \\\\")
    lines += ["\\midrule",
              f"Survival rate & {pw['survival_rate'] * 100:.0f}\\% \\\\",
              f"Top-decile share of value & {pw['top_decile_share'] * 100:.0f}\\% \\\\",
              f"Value Gini & {pw['value_gini']:.2f} \\\\",
              "\\bottomrule", "\\end{tabular}"]
    (TABLES / "startups.tex").write_text("\n".join(lines) + "\n")


def fig_denoise(e52):
    """E52: wavelet denoising as perceive->world->emit. Per-signal SNR gain,
    edge preservation vs low-pass, the sparse symbolic state, and the
    edge-vs-smooth basis-match summary."""
    fig, ((a, b), (c, dax)) = plt.subplots(2, 2, figsize=(8.4, 6.6))
    ps = e52["per_signal"]
    sigs = list(ps)

    # A: per-signal SNR gain by method
    x = range(len(sigs)); w = 0.27
    for k, (m, col) in enumerate([("wavelet", BLUE), ("lowpass", ORANGE),
                                  ("tuned_lowpass", SLATE)]):
        a.bar([i + (k - 1) * w for i in x], [ps[s][m] for s in sigs], w,
              color=col, label=m.replace("_", " "))
    a.axhline(0, color="k", lw=0.6)
    a.set_xticks(list(x)); a.set_xticklabels(sigs, fontsize=7.5, rotation=20)
    a.set_ylabel("SNR gain (dB)")
    a.set_title("A. SNR gain by signal (basis match)", fontsize=9.3, loc="left")
    a.legend(fontsize=7, loc="upper right")

    # B: edge preservation (blocks segment)
    wf = e52["waveform"]
    b.plot(wf["noisy"], color="0.75", lw=0.6, label="noisy")
    b.plot(wf["clean"], color="k", lw=1.4, label="clean")
    b.plot(wf["wavelet"], color=BLUE, lw=1.4, label="wavelet")
    b.plot(wf["lowpass"], color=SLATE, lw=1.2, ls="--", label="tuned low-pass")
    b.set_title("B. Wavelet keeps edges; low-pass blurs/rings", fontsize=9.3, loc="left")
    b.set_xlabel("sample"); b.legend(fontsize=6.5, loc="upper right")

    # C: sparse symbolic state (sorted detail-coefficient magnitudes)
    cm = e52["coeff_mag"]
    c.semilogy(range(1, len(cm) + 1), [max(v, 1e-4) for v in cm], color=BLUE, lw=1.6)
    c.axhline(e52["threshold"], color=RED, lw=1.3, ls="--", label="universal threshold")
    c.set_xlabel("coefficient rank"); c.set_ylabel("|coefficient| (log)")
    c.set_title(f"C. Sparse state: {e52['noisy_sparsity']:.0%} of coeffs zeroed",
                fontsize=9.3, loc="left")
    c.legend(fontsize=7.5)

    # D: edge-vs-smooth, wavelet vs tuned low-pass
    ed, sm = e52["edges_delta"], e52["smooth_delta"]
    cats = ["edge\n(blocks,\nheavisine)", "smooth\n(bumps,\ndoppler)"]
    xc = range(len(cats))
    dax.bar([i - 0.2 for i in xc], [ed["wavelet"], sm["wavelet"]], 0.4, color=BLUE,
            label="wavelet (no tuning)")
    dax.bar([i + 0.2 for i in xc], [ed["tuned_lowpass"], sm["tuned_lowpass"]], 0.4,
            color=SLATE, label="tuned low-pass")
    dax.set_xticks(list(xc)); dax.set_xticklabels(cats, fontsize=7.5)
    dax.set_ylabel("SNR gain (dB)")
    dax.set_title("D. Edges: wavelet wins; smooth: Fourier wins", fontsize=9.0, loc="left")
    dax.legend(fontsize=7.5)

    fig.suptitle("Wavelet denoising as a perception->world->emit loop (E52)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGS / "denoise.png", dpi=200)
    plt.close(fig)


def table_denoise(e52):
    """E52: SNR gain (dB) by signal and method."""
    ps = e52["per_signal"]
    methods = ["wavelet", "lowpass", "tuned_lowpass"]
    lines = ["\\begin{tabular}{lrrr}", "\\toprule",
             "Signal & Wavelet & Low-pass & Tuned low-pass \\\\", "\\midrule"]
    for s in ps:
        lines.append(f"{s} & {ps[s]['wavelet']:+.1f} & {ps[s]['lowpass']:+.1f} & "
                     f"{ps[s]['tuned_lowpass']:+.1f} \\\\")
    lines += ["\\midrule",
              f"\\textbf{{Edge mean}} & \\textbf{{{e52['edges_delta']['wavelet']:+.1f}}} & "
              f"{e52['edges_delta']['lowpass']:+.1f} & {e52['edges_delta']['tuned_lowpass']:+.1f} \\\\",
              "\\bottomrule", "\\end{tabular}"]
    (TABLES / "denoise.tex").write_text("\n".join(lines) + "\n")


def fig_sheaf(e53):
    """E53: sheaf consistency. Field recovery, detection, correction vs averaging,
    and localization across fault counts."""
    fig, ((a, b), (c, dax)) = plt.subplots(2, 2, figsize=(8.4, 6.4))
    rows = e53["rows"]
    nf = [r["n_faults"] for r in rows]

    # A: the field at 3 faults - truth, corrupted average, majority recovery
    snap = e53["snapshots"]["3"]
    xs = range(len(snap["truth"]))
    a.plot(xs, snap["truth"], color="k", lw=1.6, label="true field")
    a.plot(xs, snap["average"], color=RED, lw=1.2, ls="--", label="naive average (corrupted)")
    a.plot(xs, snap["majority"], color=BLUE, lw=1.4, label="sheaf majority (recovered)")
    a.set_xlabel("location"); a.set_ylabel("value")
    a.set_title("A. Glue the global field (3 faults)", fontsize=9.3, loc="left")
    a.legend(fontsize=6.8, loc="upper right")

    # B: detection - obstruction norm vs faults
    b.plot(nf, [r["obstruction"] for r in rows], "-o", color=PURPLE, lw=2, markersize=4)
    b.set_xlabel("# faulty sensors"); b.set_ylabel("gluing obstruction")
    b.set_title("B. Detection: obstruction = 0 iff consistent", fontsize=9.3, loc="left")

    # C: correction error - majority vs average
    c.plot(nf, [r["majority_rmse"] for r in rows], "-o", color=BLUE, lw=2,
           markersize=4, label="sheaf majority")
    c.plot(nf, [r["average_rmse"] for r in rows], "--s", color=RED, lw=2,
           markersize=4, label="naive average")
    c.set_xlabel("# faulty sensors"); c.set_ylabel("field RMSE")
    c.set_title("C. Correction: consensus beats averaging", fontsize=9.3, loc="left")
    c.legend(fontsize=7.5)

    # D: localization accuracy
    dax.plot(nf, [r["localize_acc"] for r in rows], "-o", color=TEAL, lw=2, markersize=4)
    dax.set_ylim(0, 1.08); dax.set_xlabel("# faulty sensors")
    dax.set_ylabel("localization accuracy")
    dax.set_title("D. Localize the fault to its source", fontsize=9.3, loc="left")

    fig.suptitle("Sheaf consistency: gluing local views into a global world (E53)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGS / "sheaf.png", dpi=200)
    plt.close(fig)


def table_sheaf(e53):
    lines = ["\\begin{tabular}{rccrr}", "\\toprule",
             "Faults & Detected & Localize & Majority RMSE & Average RMSE \\\\",
             "\\midrule"]
    for r in e53["rows"]:
        lines.append(f"{r['n_faults']} & {'yes' if r['detected'] else 'no'} & "
                     f"{r['localize_acc']:.2f} & {r['majority_rmse']:.3f} & "
                     f"{r['average_rmse']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "sheaf.tex").write_text("\n".join(lines) + "\n")


def fig_bounds(e54):
    """E54: abstract interpretation. Affine bounds stay tight where intervals
    explode and Monte Carlo under-covers, across the rollout horizon."""
    rows = e54["rows"]
    H = [r["T"] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.5))

    # A: bound width vs horizon (lower is tighter; affine wins)
    a.plot(H, [r["interval_width"] for r in rows], "--s", color=RED, lw=2,
           markersize=4, label="interval (loses correlation)")
    a.plot(H, [r["affine_width"] for r in rows], "-o", color=BLUE, lw=2,
           markersize=4, label="affine (tracks correlation)")
    a.plot(H, [r["mc_width"] for r in rows], ":^", color=SLATE, lw=2,
           markersize=4, label="Monte Carlo (unsound)")
    a.plot(H, [r["true_width"] for r in rows], "-", color="k", lw=1.3,
           label="true reachable width")
    a.set_xlabel("rollout horizon (steps)"); a.set_ylabel("bound width")
    a.set_title("A. Affine stays tight; intervals explode", fontsize=9.3, loc="left")
    a.legend(fontsize=7)

    # B: soundness — sound methods enclose the truth; MC misses the high end
    a_sound = all(r["affine_sound"] for r in rows)
    i_sound = all(r["interval_sound"] for r in rows)
    b.plot(H, [r["mc_misses_hi"] for r in rows], "-^", color=SLATE, lw=2,
           markersize=4, label="Monte Carlo over-shoot missed")
    b.axhline(0, color=BLUE, lw=2, label="affine / interval (always enclose)")
    b.set_xlabel("rollout horizon (steps)")
    b.set_ylabel("truth outside the bound")
    b.set_title(f"B. Soundness: affine={'sound' if a_sound else 'UNSOUND'}, "
                f"MC under-covers", fontsize=9.3, loc="left")
    b.legend(fontsize=7.5)

    fig.suptitle("Abstract interpretation: sound, tight bounds on world rollouts (E54)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "bounds.png", dpi=200)
    plt.close(fig)


def table_bounds(e54):
    lines = ["\\begin{tabular}{rrrrr}", "\\toprule",
             "Horizon & True width & Affine & Interval & MC miss \\\\",
             "\\midrule"]
    for r in e54["rows"]:
        lines.append(f"{r['T']} & {r['true_width']:.1f} & {r['affine_width']:.1f} & "
                     f"{r['interval_width']:.1f} & {r['mc_misses_hi']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "bounds.tex").write_text("\n".join(lines) + "\n")


def fig_infogeom(e55):
    """E55: information geometry. Expected-info-gain probing collapses posterior
    entropy fastest, identifying the world in fewer probes than heuristic/random."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.5))
    curves = e55["entropy_curves"]
    styles = {"eig": (BLUE, "-o", "expected info gain"),
              "heuristic": (ORANGE, "--s", "version-space heuristic"),
              "random": (SLATE, ":^", "random probing")}

    # A: posterior entropy collapse
    for s, (col, ls, lab) in styles.items():
        y = curves[s]
        a.plot(range(len(y)), y, ls, color=col, lw=2, markersize=3.5, label=lab)
    a.set_xlabel("probes"); a.set_ylabel("posterior entropy (bits)")
    a.set_title("A. Information-guided probes identify faster", fontsize=9.3, loc="left")
    a.legend(fontsize=7.5)

    # B: mean probes to identify
    summ = e55["summary"]
    order = ["eig", "heuristic", "random"]
    vals = [summ[s]["mean_steps"] for s in order]
    cols = [styles[s][0] for s in order]
    bars = b.bar([styles[s][2].replace(" ", "\n") for s in order], vals, color=cols)
    for bar, v in zip(bars, vals):
        b.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.1f}",
               ha="center", fontsize=8.5)
    b.set_ylabel("mean probes to identify")
    b.set_title(f"B. {e55['n_worlds']} worlds, {e55['n_probes']} probes",
                fontsize=9.3, loc="left")

    fig.suptitle("Information geometry: identify the world by maximizing information (E55)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "infogeom.png", dpi=200)
    plt.close(fig)


def table_infogeom(e55):
    lines = ["\\begin{tabular}{lrr}", "\\toprule",
             "Strategy & Mean probes & Accuracy \\\\", "\\midrule"]
    names = {"eig": "Expected info gain", "heuristic": "Version-space heuristic",
             "random": "Random probing"}
    for s in ["eig", "heuristic", "random"]:
        r = e55["summary"][s]
        lines.append(f"{names[s]} & {r['mean_steps']:.2f} & {r['accuracy']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "infogeom.tex").write_text("\n".join(lines) + "\n")


def fig_transport(e56):
    """E56: optimal transport. Wasserstein localizes regime shifts and gives a
    usable calibration gradient from a cold start where the KL objective is flat."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.5))

    # A: drift detection — consecutive-window Wasserstein spikes at each shift
    centers, wc = e56["centers"], e56["wasserstein_curve"]
    a.plot(centers, wc, "-", color=BLUE, lw=1.6, label="consecutive-window $W_1$")
    for i, c in enumerate(e56["true_changes"]):
        a.axvline(c, color="k", ls=":", lw=1.2,
                  label="true shift" if i == 0 else None)
    for i, c in enumerate(e56["detected_changes"]):
        a.axvline(c, color=RED, ls="--", lw=1.4,
                  label="detected" if i == 0 else None)
    a.set_xlabel("time"); a.set_ylabel("Wasserstein distance")
    a.set_title("A. Drift localized by transport distance", fontsize=9.3, loc="left")
    a.legend(fontsize=7.5)

    # B: calibration objective — Wasserstein V vs flat/saturated KL
    cal = e56["calibration"]
    grid = cal["grid"]
    a2 = b
    a2.plot(grid, cal["w_objective"], "-", color=BLUE, lw=2, label="Wasserstein (gradient)")
    a2.set_xlabel(r"world mean $\mu$"); a2.set_ylabel("Wasserstein to target", color=BLUE)
    a2.tick_params(axis="y", labelcolor=BLUE)
    a2.axvline(cal["target_mean"], color="k", ls=":", lw=1.2, label="target $\\mu^*$")
    a2.axvline(cal["cold_start"], color=SLATE, ls="--", lw=1.2, label="cold start")
    kax = a2.twinx()
    kax.plot(grid, cal["kl_objective"], "--", color=RED, lw=2)
    kax.set_ylabel("KL to target (saturates)", color=RED)
    kax.tick_params(axis="y", labelcolor=RED)
    a2.set_title("B. Calibration: KL flat where transport descends", fontsize=9.0, loc="left")
    a2.legend(fontsize=7, loc="upper center")

    fig.suptitle("Optimal transport: drift detection and calibration where KL fails (E56)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "transport.png", dpi=200)
    plt.close(fig)


def table_transport(e56):
    cal = e56["calibration"]
    lines = ["\\begin{tabular}{lr}", "\\toprule", "Quantity & Value \\\\", "\\midrule",
             f"True regime shifts & {', '.join(map(str, e56['true_changes']))} \\\\",
             f"Wasserstein localized & {', '.join(map(str, e56['detected_changes']))} \\\\",
             f"Target mean $\\mu^*$ & {cal['target_mean']:.1f} \\\\",
             f"Recovered $\\hat\\mu$ (transport) & {cal['wasserstein_mu_hat']:.1f} \\\\",
             f"Far-region slope (transport) & {cal['w_far_rel_slope']:.3f} \\\\",
             f"Far-region slope (KL) & {cal['kl_far_rel_slope']:.3f} \\\\",
             "\\bottomrule", "\\end{tabular}"]
    (TABLES / "transport.tex").write_text("\n".join(lines) + "\n")


def fig_path_integral(e49):
    """E49: path integral over learning trajectories. Per-agent least-action vs
    unplanned baselines; path-integral marginals concentrating with beta; free
    energy approaching the least-action cost; trajectories-vs-DP tractability."""
    pa = e49["per_agent"]
    fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(8.4, 6.2))

    # A: per-agent least-action vs baselines
    roles = list(pa)
    series = [("least action", "least_action_cost", BLUE),
              ("greedy", None, TEAL), ("random", None, ORANGE), ("eager", None, RED)]
    x = range(len(roles))
    w = 0.2
    for k, (lab, key, col) in enumerate(series):
        if key:
            vals = [pa[r][key] for r in roles]
        else:
            bk = {"greedy": "greedy", "random": "random_mean", "eager": "eager"}[lab]
            vals = [pa[r]["baselines"][bk] for r in roles]
        a.bar([i + (k - 1.5) * w for i in x], vals, w, color=col, label=lab)
    a.axhline(e49["transfer"]["from_scratch"], color=SLATE, lw=1.2, ls=":",
              label="from scratch")
    a.set_xticks(list(x)); a.set_xticklabels(roles, fontsize=8.5)
    a.set_ylabel("learning cost (action)")
    a.set_title("A. Agent spec → least-action curriculum", fontsize=9.5, loc="left")
    a.legend(fontsize=7, ncol=2)

    # B: path-integral marginals, low vs high beta (concentration)
    mlo = e49["marginals_by_beta"]["0.2"]
    mhi = e49["marginals_by_beta"]["10.0"]
    order = sorted(mhi, key=lambda k: -mhi[k])
    xs = range(len(order))
    b.bar([i - 0.2 for i in xs], [mlo[k] for k in order], 0.4, color="#94A3B8",
          label=r"$\beta$=0.2 (explore)")
    b.bar([i + 0.2 for i in xs], [mhi[k] for k in order], 0.4, color=PURPLE,
          label=r"$\beta$=10 (exploit)")
    b.set_xticks(list(xs))
    b.set_xticklabels([k.replace("_", "\n") for k in order], fontsize=5.5, rotation=0)
    b.set_ylabel("path-integral marginal")
    b.set_title("B. Which worlds to learn (forward×backward)", fontsize=9.5, loc="left")
    b.legend(fontsize=7.5)

    # C: free energy -> least action as beta grows
    betas = e49["betas"]
    fe = [e49["free_energy_by_beta"][str(bb)] for bb in betas]
    c.plot(betas, fe, "-o", color=BLUE, lw=2, markersize=4, label=r"$-\frac{1}{\beta}\log Z$")
    c.axhline(e49["transfer"]["least_action_cost"], color=RED, lw=1.5, ls="--",
              label="least action")
    c.set_xscale("log")
    c.set_xlabel(r"inverse temperature $\beta$"); c.set_ylabel("free energy")
    c.set_title("C. Path integral → least action", fontsize=9.5, loc="left")
    c.legend(fontsize=8, loc="lower right")

    # D: tractability — trajectories summed vs DP states
    t = e49["tractability"]
    bars = d.bar(["trajectories\n(summed)", "DP states\n(computed)"],
                 [t["n_trajectories"], t["n_dp_states"]], color=[ORANGE, BLUE])
    for bar, v in zip(bars, [t["n_trajectories"], t["n_dp_states"]]):
        d.text(bar.get_x() + bar.get_width() / 2, v * 1.1, f"{v:,}", ha="center", fontsize=8)
    d.set_yscale("log")
    d.set_ylabel("count (log)")
    d.set_title("D. Infinite trajectories, summed without enumerating", fontsize=9.0, loc="left")

    fig.suptitle("Path integrals over composite-world learning trajectories (E49)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "path_integral.png", dpi=200)
    plt.close(fig)


def table_path_integral(e49):
    """E49: per-agent least-action curriculum cost vs unplanned baselines."""
    pa = e49["per_agent"]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Agent spec & Least action & Greedy & Random & Eager \\\\",
             "\\midrule"]
    for role, d in pa.items():
        bl = d["baselines"]
        lines.append(f"{role.replace('_', ' ')} & "
                     f"\\textbf{{{d['least_action_cost']:.0f}}} & {bl['greedy']:.0f} & "
                     f"{bl['random_mean']:.0f} & {bl['eager']:.0f} \\\\")
    sp = e49["transfer"]
    lines += ["\\midrule",
              f"\\multicolumn{{5}}{{l}}{{\\emph{{Goal from scratch (no transfer): "
              f"{sp['from_scratch']:.0f} — least-action path is {sp['speedup']:.1f}$\\times$ "
              f"cheaper}}}} \\\\",
              "\\bottomrule", "\\end{tabular}"]
    (TABLES / "path_integral.tex").write_text("\n".join(lines) + "\n")


def fig_relativity(e47):
    """E47: relativity as a verified world model. Time dilation (exact vs
    learned/Newtonian, OOD near c), velocity addition saturating at c, the twin
    paradox worldline, and real atomic-clock validation (Hafele-Keating)."""
    td, va = e47["time_dilation"], e47["velocity_addition"]
    tw, hk, gp = e47["twin_paradox"], e47["hafele_keating"], e47["gps"]
    fig = plt.figure(figsize=(8.2, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 1.0], hspace=0.55,
                          wspace=0.28)
    sch = fig.add_subplot(gs[0, :])
    a = fig.add_subplot(gs[1, 0]); b = fig.add_subplot(gs[1, 1])
    c = fig.add_subplot(gs[2, 0]); d = fig.add_subplot(gs[2, 1])

    # --- top: the relativity world as a composition --------------------------
    sch.set_xlim(0, 10); sch.set_ylim(0, 2.85); sch.axis("off")
    _panel(sch, 0.25, 0.2, 9.5, 2.45, "lab frame  ·  observer (reference)", SLATE,
           alpha_fill=0.03)
    _chip(sch, 8.35, 2.42, "_agg: τ = ∫ dt/γ", SLATE, fontsize=6.8)
    _card(sch, 0.95, 0.72, 2.7, 1.25, "frame A — at rest",
          ["agent: stay-at-home", "atomic clock  τ_A", "rate 1/γ = 1.00"],
          bold_edge=TEAL)
    _card(sch, 6.35, 0.72, 2.7, 1.25, "frame B — moving (0.8c)",
          ["agent: traveler", "atomic clock  τ_B", "rate 1/γ = 0.60"],
          bold_edge=PURPLE)
    # bridge: the Lorentz transform couples the two frames (each sees the other slow)
    sch.add_patch(FancyArrowPatch((3.72, 1.5), (6.28, 1.5),
                                  connectionstyle="arc3,rad=-0.32",
                                  arrowstyle="<|-|>", mutation_scale=13,
                                  color=TEAL, lw=2.4, zorder=5))
    _chip(sch, 5.0, 2.18, "bridge: Lorentz transform · γ=1.67", TEAL, fontsize=7)
    # route: the agent changes reference (the turnaround) -> ages less
    sch.add_patch(FancyArrowPatch((2.3, 0.6), (7.7, 0.6),
                                  connectionstyle="arc3,rad=0.22",
                                  arrowstyle="<|-|>", mutation_scale=11,
                                  color=ORANGE, lw=1.9, ls=(0, (5, 2)), zorder=4))
    sch.plot([5.0], [0.3], "o", color=PURPLE, markersize=8, zorder=6)
    sch.text(5.0, 0.06, "route: agent changes reference (turnaround) → ages less",
             ha="center", fontsize=6.8, color=ORANGE)
    sch.set_title("The composite world: reference frames coupled by the Lorentz "
                  "transform; agents carry atomic clocks and cross frames",
                  fontsize=8.6, loc="left", color="#334155")

    # A: time dilation
    full_x = td["in_frac"] + td["ood_frac"]
    full_y = td["in_truth"] + td["truth"]
    a.plot(full_x, full_y, "-", color=BLUE, lw=2.2, label="symbolic (exact)")
    a.plot(td["ood_frac"], td["learned"], "--", color=ORANGE, lw=2, label="learned (fit $v\\leq0.3c$)")
    a.axhline(1.0, color=SLATE, lw=1.5, ls=":", label="Newtonian (no dilation)")
    a.axvspan(0.3, 1.0, color="0.92", zorder=0)
    a.text(0.62, 0.9, "OOD", color=SLATE, fontsize=8)
    a.set_xlabel("speed $v/c$"); a.set_ylabel("clock rate ($1/\\gamma$)")
    a.set_ylim(0, 1.08)
    a.set_title("A. Time dilation: exact vs approximate", fontsize=9.5, loc="left")
    a.legend(fontsize=7.5, loc="lower left")

    # B: velocity addition
    b.plot(va["fracs"], va["rel_over_c"], "-", color=BLUE, lw=2.2,
           label="relativistic $(u{+}v)/(1{+}uv/c^2)$")
    b.plot(va["fracs"], va["gal_over_c"], "--", color=RED, lw=2, label="Galilean $u{+}v$")
    b.axhline(1.0, color=SLATE, lw=1.5, ls=":", label="$c$")
    b.set_xlabel("each input speed ($/c$)"); b.set_ylabel("combined speed ($/c$)")
    b.set_title("B. Velocity addition stays below $c$", fontsize=9.5, loc="left")
    b.legend(fontsize=7.5, loc="upper left")

    # C: twin paradox worldlines
    n = len(tw["traj_stay"])
    lab = [10.0 * i / n for i in range(n)]
    c.plot(lab, tw["traj_stay"], "-", color=TEAL, lw=2.2, label="stay-at-home")
    c.plot(lab, tw["traj_travel"], "-", color=PURPLE, lw=2.2, label="traveler (0.8c)")
    c.annotate(f"{tw['symbolic']['diff_years']:.0f} yr younger",
               xy=(10, tw["symbolic"]["travel_years"]),
               xytext=(6.2, tw["symbolic"]["travel_years"] - 2.2), fontsize=8, color=PURPLE,
               arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1))
    c.set_xlabel("lab-frame time (yr)"); c.set_ylabel("clock / proper time (yr)")
    c.set_title("C. Twin paradox (changing reference)", fontsize=9.5, loc="left")
    c.legend(fontsize=8, loc="upper left")

    # D: Hafele-Keating model vs measured
    x = [0, 1]
    w = 0.36
    model = [hk["model_east_ns"], hk["model_west_ns"]]
    obs = [hk["pub_obs_east_ns"], hk["pub_obs_west_ns"]]
    err = [hk["pub_obs_east_err"], hk["pub_obs_west_err"]]
    d.bar([i - w / 2 for i in x], model, w, color=BLUE, label="symbolic model")
    d.bar([i + w / 2 for i in x], obs, w, yerr=err, color=SLATE, capsize=4,
          label="measured (1971)")
    d.axhline(0, color="k", lw=0.8)
    d.set_xticks(x); d.set_xticklabels(["eastward", "westward"], fontsize=9)
    d.set_ylabel("clock shift vs ground (ns)")
    d.set_title(f"D. Atomic clocks: GPS {gp['net_us_per_day']:+.0f}µs/day; Hafele–Keating",
                fontsize=9.0, loc="left")
    d.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("Relativity as a verified world model: reference frames and atomic clocks (E47)",
                 fontsize=10.5, x=0.02, ha="left", y=0.995)
    fig.subplots_adjust(top=0.945, bottom=0.055, left=0.09, right=0.97)
    fig.savefig(FIGS / "relativity.png", dpi=200)
    plt.close(fig)


def table_relativity(e47):
    """E47: real atomic-clock validation, symbolic model vs measurement."""
    gp, hk = e47["gps"], e47["hafele_keating"]
    lines = ["\\begin{tabular}{llrr}", "\\toprule",
             "Test & Effect & Symbolic model & Measured / documented \\\\",
             "\\midrule",
             f"GPS & SR (orbital $v$) & {gp['sr_us_per_day']:.1f} $\\mu$s/day & \\\\",
             f"GPS & GR (altitude) & {gp['gr_us_per_day']:.1f} $\\mu$s/day & \\\\",
             f"GPS & \\textbf{{net}} & \\textbf{{{gp['net_us_per_day']:+.1f} $\\mu$s/day}} "
             f"& $\\sim$+38 $\\mu$s/day \\\\",
             "\\midrule",
             f"Hafele--Keating & eastward & {hk['model_east_ns']:.0f} ns & "
             f"${hk['pub_obs_east_ns']}\\pm{hk['pub_obs_east_err']}$ ns \\\\",
             f"Hafele--Keating & westward & {hk['model_west_ns']:.0f} ns & "
             f"${hk['pub_obs_west_ns']}\\pm{hk['pub_obs_west_err']}$ ns \\\\",
             f"Hafele--Keating & Newtonian & {hk['newtonian_east_ns']:.0f} / "
             f"{hk['newtonian_west_ns']:.0f} ns & (off by $10^2$ ns) \\\\",
             "\\bottomrule", "\\end{tabular}"]
    (TABLES / "relativity.tex").write_text("\n".join(lines) + "\n")


def fig_next_token(e45):
    """E45: next-char accuracy vs sequence length. The induced symbolic program
    stays exact at every length; fixed-memory neural models and the same LLM
    predicting directly decay - the length-generalization story on the LLM's
    home turf."""
    tasks = e45["per_task"]
    methods = ["symbolic", "ngram", "window_mlp", "llm_direct"]
    labels = {"symbolic": "symbolic (ours)", "ngram": "n-gram",
              "window_mlp": "window-MLP", "llm_direct": "LLM-direct (same model)"}
    colors = {"symbolic": BLUE, "ngram": ORANGE, "window_mlp": PURPLE, "llm_direct": RED}
    styles = {"symbolic": "-o", "ngram": "--s", "window_mlp": "--^", "llm_direct": ":D"}
    # mean accuracy across tasks at each evaluated length, per method
    sizes = sorted({int(s) for t in tasks for s in t["curves"]["symbolic"]})
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(8.2, 3.4),
                                  gridspec_kw={"width_ratios": [1.5, 1]})

    def mean_at(method, size):
        vals = [t["curves"][method].get(str(size), t["curves"][method].get(size))
                for t in tasks if str(size) in t["curves"][method]
                or size in t["curves"][method]]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    for m in methods:
        xs, ys = [], []
        for s in sizes:
            v = mean_at(m, s)
            if v is not None:
                xs.append(s); ys.append(v)
        ax.plot(xs, ys, styles[m], color=colors[m], lw=2, markersize=5, label=labels[m])
    ax.axvline(e45["l_train"], color=SLATE, lw=1, ls=":")
    ax.text(e45["l_train"] * 1.05, 0.05, "train→ | ←OOD", fontsize=7.5, color=SLATE)
    ax.set_xscale("log")
    ax.set_xlabel("Sequence length"); ax.set_ylabel("Next-char exact accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Length generalization (mean over tasks)", fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, loc="center left")

    s = e45["summary"]
    bars = axb.bar([labels[m].split(" ")[0] for m in methods],
                   [s[m]["ood"] for m in methods], color=[colors[m] for m in methods])
    for b, m in zip(bars, methods):
        axb.text(b.get_x() + b.get_width() / 2, s[m]["ood"] + 0.02,
                 f"{s[m]['ood']:.2f}", ha="center", fontsize=8)
    axb.set_ylabel("Mean OOD accuracy"); axb.set_ylim(0, 1.05)
    axb.set_title("Out-of-distribution (longer)", fontsize=9.5, loc="left")
    axb.tick_params(axis="x", labelsize=7.5)

    fig.suptitle("Next-token world models: synthesize the rule, don't be the rule (E45)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "next_token.png", dpi=200)
    plt.close(fig)


def table_next_token(e45):
    """E45: per-task in-dist vs OOD next-char accuracy by method."""
    methods = ["symbolic", "ngram", "window_mlp", "llm_direct"]
    head = ["\\begin{tabular}{ll" + "c" * len(methods) + "}", "\\toprule",
            "Task & Split & Symbolic & n-gram & window-MLP & LLM-direct \\\\",
            "\\midrule"]
    rows = []
    for t in e45["per_task"]:
        for split in ("in_dist", "ood"):
            cells = " & ".join(
                "--" if t["split"][m][split] is None else f"{t['split'][m][split]:.2f}"
                for m in methods)
            lab = t["task"] if split == "in_dist" else ""
            sp = "in-dist" if split == "in_dist" else "OOD"
            rows.append(f"{lab} & {sp} & {cells} \\\\")
    s = e45["summary"]
    foot = ["\\midrule",
            "\\textbf{Mean} & OOD & " + " & ".join(f"{s[m]['ood']:.2f}" for m in methods)
            + " \\\\", "\\bottomrule", "\\end{tabular}"]
    (TABLES / "next_token.tex").write_text("\n".join(head + rows + foot) + "\n")


def fig_many_worlds(e46):
    """E46: a factored store holds an exact version space over world spaces too
    large to enumerate. Left: update+query time vs world count (factored grows
    ~N^(1/#params); enumeration ~N and stops). Right: graceful degradation as a
    mechanism couples w parameters (factor ~ d^w)."""
    scale = e46["scale"]
    coup = e46["coupling"]
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(8.0, 3.4))

    n = [r["n_worlds"] for r in scale]
    fct = [r["factored_ms"] for r in scale]
    en_n = [r["n_worlds"] for r in scale if r["enum_ms"] is not None]
    en = [r["enum_ms"] for r in scale if r["enum_ms"] is not None]
    ax.plot(n, fct, "-o", color=BLUE, lw=2, markersize=4, label="factored store (ours)")
    ax.plot(en_n, en, "--s", color=RED, lw=2, markersize=4, label="enumerated (E43-style)")
    ax.annotate("enumeration\ninfeasible", xy=(en_n[-1], en[-1]),
                xytext=(en_n[-1] * 50, en[-1] * 4), fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("World-space size (number of worlds)")
    ax.set_ylabel("Update + query time (ms)")
    ax.set_title("Exact version space, far past enumeration", fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="upper left")

    w = [r["w"] for r in coup]
    fsize = [r["factor_size"] for r in coup]
    ideal = [r["ideal_factored"] for r in coup]
    axc.plot(w, fsize, "-o", color=ORANGE, lw=2, markersize=4, label="coupled factor ($d^w$)")
    axc.plot(w, ideal, "--^", color=TEAL, lw=2, markersize=4, label="separable ideal ($wd$)")
    axc.set_yscale("log")
    axc.set_xlabel("Coupling width $w$ (params per mechanism)")
    axc.set_ylabel("Factor size (entries)")
    axc.set_title("Boundary: coupling costs (the #P analogue)", fontsize=9.5, loc="left")
    axc.set_xticks(w)
    axc.legend(fontsize=8, loc="upper left")

    fig.suptitle("A database for many worlds: factored, semiring-annotated store (E46)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "many_worlds.png", dpi=200)
    plt.close(fig)


def table_many_worlds(e46):
    """E46: factored vs enumerated cost across world-space sizes."""
    lines = ["\\begin{tabular}{rrrr}", "\\toprule",
             "Worlds & Consistent & Factored (ms) & Enumerated (ms) \\\\",
             "\\midrule"]
    for r in e46["scale"]:
        en = "infeasible" if r["enum_ms"] is None else f"{r['enum_ms']:.0f}"
        mant, exp = f"{r['n_worlds']:.1e}".split("e")
        lines.append(f"${mant}\\times10^{{{int(exp)}}}$ & {r['consistent']:,.0f} & "
                     f"{r['factored_ms']:.2f} & {en} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "many_worlds.tex").write_text("\n".join(lines) + "\n")


def table_induction(e37):
    """E37: equal-information induction (traces only) vs the rule-text anchor."""
    anchor = e37["summary"]["code_from_rules"]
    big = max(e37["summary"]["by_k"], key=lambda r: r["k"])  # representative K

    def row(label, info, indist, ood):
        return f"{label} & {info} & {indist:.2f} & {ood:.2f} \\\\"

    lines = [
        "\\begin{tabular}{llcc}", "\\toprule",
        "Method & Information given & In-dist. & $10\\times$ OOD \\\\",
        "\\midrule",
        row("Code synthesis (rules)", "rule text, 0 transitions",
            anchor["probe_in_dist"], anchor["probe_ood_10x"]),
        "\\midrule",
        "\\multicolumn{4}{l}{\\emph{Equal information: %d random-policy transitions, no rule text}} \\\\"
        % big["k"],
        row("\\textbf{Code induction}", "traces only",
            big["code_from_traces_in_dist"], big["code_from_traces_ood"]),
        row("MLP", "traces only", big["mlp_in_dist"], big["mlp_ood"]),
        row("1-NN memorizer", "traces only", big["knn1_in_dist"], big["knn1_ood"]),
        "\\bottomrule", "\\end{tabular}",
    ]
    (TABLES / "induction.tex").write_text("\n".join(lines) + "\n")


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


def table_repairbench(e28, e29):
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
    (TABLES / "repairbench.tex").write_text(
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


def table_representations(e36):
    """E36 generalization zoo: exact accuracy on novel combinations by K."""
    gen = {r["k"]: r for r in e36["leg_generalization"]}
    ks = sorted(gen)
    rows = [  # (display label, json key)
        ("Linear / ridge", "ridge"),
        ("1-NN memorizer", "knn1"),
        ("$k$-NN ($k{=}5$)", "knn5"),
        ("Kernel SVR", "svr"),
        ("Gaussian process", "gp"),
        ("Random forest", "random_forest"),
        ("Hist.\\ gradient boosting", "hist_grad_boost"),
        ("Koopman / EDMD (deg.\\ 2)", "koopman"),
        ("MLP (2 hidden)", "monolith"),
    ]

    def cells(key):
        out = []
        for k in ks:
            v = gen[k].get(key)
            out.append("--" if not v or v.get("acc") is None else f"{v['acc']:.2f}")
        return " & ".join(out)
    lines = ["\\begin{tabular}{l" + "c" * len(ks) + "}", "\\toprule",
             "Representation & " + " & ".join(f"$K{{=}}{k}$" for k in ks) + " \\\\",
             "\\midrule",
             "\\multicolumn{%d}{l}{\\emph{Monolithic learner over the joint state}} \\\\"
             % (len(ks) + 1)]
    for label, key in rows:
        lines.append(f"{label} & {cells(key)} \\\\")
    lines += ["\\midrule",
              "\\multicolumn{%d}{l}{\\emph{Composite of small worlds (ours)}} \\\\"
              % (len(ks) + 1),
              f"Composite-learned (MLP) & {cells('composite_learned')} \\\\"]
    if all(gen[k].get("composite_hgb") and gen[k]["composite_hgb"]["acc"] is not None
           for k in ks):
        lines.append(f"Composite-learned (trees) & {cells('composite_hgb')} \\\\")
    lines += [f"\\textbf{{Composite-symbolic}} & {cells('composite_symbolic')} \\\\",
              "\\bottomrule", "\\end{tabular}"]
    (TABLES / "representations.tex").write_text("\n".join(lines) + "\n")


def fig_agent_traversal(e42):
    """E42: agent-belief accuracy + interference across hops between two worlds."""
    s = e42["summary"]
    order = ["oracle", "symbolic_per_world", "per_world_window", "shared_online"]
    labels = {"oracle": "oracle", "symbolic_per_world": "symbolic per-world (ours)",
              "per_world_window": "per-world window", "shared_online": "shared online"}
    colors = {"oracle": SLATE, "symbolic_per_world": TEAL,
              "per_world_window": BLUE, "shared_online": RED}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4),
                                   gridspec_kw={"width_ratios": [2, 3]})

    # A. overall prediction accuracy across the hop schedule
    accs = [s[m]["accuracy"] for m in order]
    ax1.barh(range(len(order)), accs, color=[colors[m] for m in order])
    for i, a in enumerate(accs):
        ax1.text(a + 0.01, i, f"{a:.2f}", va="center", fontsize=8)
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels([labels[m] for m in order], fontsize=7.5)
    ax1.set_xlim(0, 1.12)
    ax1.invert_yaxis()
    ax1.set_xlabel("Prediction accuracy")
    ax1.set_title("A. Tracking rules across hops", fontsize=9.5, loc="left")
    ax1.grid(alpha=0.25, axis="x")

    # B. recovery lag on arrival: unchanged vs silently-changed-while-away
    methods = ["symbolic_per_world", "per_world_window", "shared_online"]
    miss = max((s[m]["recovery_silently_changed_return"] or 0) for m in methods) + 2
    width = 0.36
    for mi, m in enumerate(methods):
        u = s[m]["recovery_unchanged_return"]
        c = s[m]["recovery_silently_changed_return"]
        uval = miss if u is None else u
        cval = miss if c is None else c
        ax2.bar(mi - width / 2, uval, width, color=colors[m], alpha=0.55,
                hatch="//" if u is None else None,
                label="returns to unchanged world" if mi == 0 else None)
        ax2.bar(mi + width / 2, cval, width, color=colors[m],
                hatch="//" if c is None else None,
                label="returns to silently-changed world" if mi == 0 else None)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(["symbolic\n(ours)", "per-world\nwindow", "shared\nonline"], fontsize=7.5)
    ax2.set_ylabel("Recovery lag on arrival (steps)")
    ax2.set_title("B. Cost of a hop (interference vs change-detection)", fontsize=9.5, loc="left")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(FIGS / "agent_traversal.png", dpi=200)
    plt.close(fig)


def fig_nonstationary(e41):
    """E41: adaptation to sudden unannounced rule changes."""
    tl = e41["timeline"]
    cp = e41["perfect_perception"]
    warmup, changes = e41["warmup"], e41["changes"]
    order = ["oracle_switch", "symbolic_refit", "window_1nn", "static_frozen"]
    labels = {"oracle_switch": "oracle (knows change)",
              "symbolic_refit": "symbolic monitor+refit (ours)",
              "window_1nn": "sliding-window 1-NN", "static_frozen": "static frozen"}
    colors = {"oracle_switch": SLATE, "symbolic_refit": TEAL,
              "window_1nn": BLUE, "static_frozen": RED}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4),
                                   gridspec_kw={"width_ratios": [3, 2]})

    # A. cumulative errors over time (regret); change markers
    xs = list(range(warmup, warmup + len(tl["symbolic_refit"])))
    for m in order:
        cum, run = [], 0
        for f in tl[m]:
            run += (1 - f)
            cum.append(run)
        ax1.plot(xs, cum, color=colors[m], lw=2, label=labels[m])
    for c in changes:
        ax1.axvline(c, color="#9CA3AF", ls=(0, (3, 3)), lw=1)
    ax1.text(changes[0] + 1, 1, "rule changes", fontsize=7, color="#6B7280")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cumulative wrong predictions")
    ax1.set_title("A. Regret over time (sudden changes marked)", fontsize=9.5, loc="left")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(alpha=0.25)

    # B. recovery lag after each change (None -> "no recovery", drawn tall)
    methods = ["symbolic_refit", "window_1nn", "static_frozen"]
    width = 0.36
    cap = max(len(tl["symbolic_refit"]), 1)
    for ci, c in enumerate(changes):
        for mi, m in enumerate(methods):
            lag = cp[m]["recovery_lag"][str(c)]
            val = cap if lag is None else lag
            x = mi + (ci - 0.5) * width
            ax2.bar(x, val, width, color=colors[m],
                    hatch="//" if lag is None else None,
                    label=labels[m] if ci == 0 else None)
            if lag is None:
                ax2.text(x, val * 0.5, "none", rotation=90, ha="center",
                         va="center", fontsize=6, color="white")
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(["symbolic\n(ours)", "window\n1-NN", "static\nfrozen"], fontsize=7)
    ax2.set_ylabel("Recovery lag (steps)")
    ax2.set_title("B. Steps to recover per change", fontsize=9.5, loc="left")
    ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(FIGS / "nonstationary.png", dpi=200)
    plt.close(fig)


def fig_perception(e40):
    """E40: perceive-then-forecast - multimodal inputs to a verified world model."""
    fc = e40["forecast_exact"]
    degr = e40["degradation"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.3))

    # A. forecast skill: world model vs end-to-end MLP, in-dist and 10x OOD
    groups = ["in-distribution", "10× population (OOD)"]
    x = [0, 1]
    w = 0.36
    ax1.bar([i - w / 2 for i in x],
            [fc["world_model_in_dist"], fc["world_model_ood_10x"]], w,
            color=TEAL, label="perceived world model (ours)")
    ax1.bar([i + w / 2 for i in x],
            [fc["mlp_in_dist"], fc["mlp_ood_10x"]], w,
            color=RED, label="end-to-end MLP")
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=8)
    ax1.set_ylabel(f"Exact day-{e40['horizon']} forecast accuracy")
    ax1.set_ylim(0, 1.08)
    ax1.set_title("A. Multi-step forecast skill", fontsize=9.5, loc="left")
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.grid(alpha=0.25, axis="y")

    # B. graceful degradation: forecast accuracy tracks perception accuracy
    pa = [d["perception_accuracy"] for d in degr]
    fa = [d["forecast_accuracy"] for d in degr]
    ax2.plot([0, 1], [0, 1], ls=(0, (3, 3)), color=SLATE, lw=1,
             label="y = x (dynamics add zero)")
    ax2.plot(pa, fa, "-o", color=TEAL, lw=2, markersize=5,
             label="perceive → world model")
    ax2.set_xlabel("Perception accuracy")
    ax2.set_ylabel("End-to-end forecast accuracy")
    ax2.set_xlim(0.4, 1.03)
    ax2.set_ylim(0.4, 1.03)
    ax2.set_title("B. Error decomposition", fontsize=9.5, loc="left")
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGS / "perception.png", dpi=200)
    plt.close(fig)


def fig_representations(e36):
    """E36: composition vs monolithic learners on three representation tests."""
    gen = e36["leg_generalization"]
    intf = e36["leg_interference"]
    se = e36["leg_sample_efficiency"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3.3))

    ks = [r["k"] for r in gen]
    mono_arms = ["monolith", "ridge", "knn1", "knn5", "svr", "gp",
                 "random_forest", "hist_grad_boost", "koopman"]
    lo = [min(r[a]["acc"] for a in mono_arms if r.get(a) and r[a]["acc"] is not None)
          for r in gen]
    hi = [max(r[a]["acc"] for a in mono_arms if r.get(a) and r[a]["acc"] is not None)
          for r in gen]
    ax1.fill_between(ks, lo, hi, color=RED, alpha=0.15,
                     label="monolithic learners (9 families, range)")
    ax1.plot(ks, [r["hist_grad_boost"]["acc"] for r in gen], "-s", color=RED,
             lw=1.6, markersize=4, label="best monolith (boosted trees)")
    if all(r.get("composite_hgb") and r["composite_hgb"]["acc"] is not None for r in gen):
        ax1.plot(ks, [r["composite_hgb"]["acc"] for r in gen], "-^", color=PURPLE,
                 lw=2, markersize=4, label="composite-learned (trees)")
    ax1.plot(ks, [r["composite_learned"]["acc"] for r in gen], "-o", color=BLUE,
             lw=2, markersize=4, label="composite-learned (MLP)")
    ax1.plot(ks, [r["composite_symbolic"]["acc"] for r in gen], "-D", color=TEAL,
             lw=2, markersize=4, label="composite-symbolic (ours)")
    ax1.set_xlabel("Number of parts (K)")
    ax1.set_ylabel("Exact accuracy on novel combinations")
    ax1.set_xticks(ks)
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_title("A. Compositional generalization", fontsize=9.5, loc="left")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=6.0, loc="center left")

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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

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


def fig_sprint_ladder(e35, e34):
    """E35: the sprint allocation experiment across a model-capability ladder."""
    n = e34["conditions"][0]["n_tasks"]
    e34_by = {c["condition"]: c["solved"] for c in e34["conditions"]}
    anchor = {"fixed": e34_by["fixed"], "round_robin": e34_by["round_robin"],
              "greedy": e34_by["greedy"]}
    ladder = {m: {c["condition"]: c["solved"] for c in cell["conditions"]}
              for m, cell in e35["ladder"].items()}
    models = ["qwen2.5:7b", "deepseek-r1:14b", "gpt-oss:20b", "qwen3-coder:30b"]
    solved = {"qwen2.5:7b": anchor, **ladder}
    conds = [("fixed", SLATE, "fixed 4/task"),
             ("round_robin", BLUE, "round-robin"),
             ("greedy", RED, "greedy min-failing")]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = range(len(models))
    width = 0.26
    for ci, (cond, color, label) in enumerate(conds):
        vals = [solved[m].get(cond, 0) for m in models]
        xs = [i + (ci - 1) * width for i in x]
        ax.bar(xs, vals, width, color=color, label=label)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.3, str(v), ha="center", fontsize=7)
    ax.axhline(n, color="#9CA3AF", lw=0.8, ls=(0, (1, 3)))
    ax.text(0, n + 0.3, f"all {n}", fontsize=7, color="#9CA3AF")
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.replace(":", "\n") for m in models], fontsize=7.5)
    ax.set_ylabel(f"Tasks solved (of {n})")
    ax.set_ylim(0, n + 2)
    ax.set_xlabel("Repair model (increasing capability →)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "sprint_ladder.png", dpi=200)
    plt.close(fig)


def fig_sprint(e34):
    """E34: solved-vs-budget curves per allocation condition on owrb-atomic."""
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

    _rm = repo_metrics()
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
        macro("NumExperiments", str(len(EXPERIMENTS))),
        # E65 MiniGrid head-to-head (trained vs verified on a shared benchmark)
        macro("MGOpenWorldPlan", str(d["e65_minigrid_bench"]["openworld"]["plan_length"])),
        macro("MGDreamerFirstSolve",
              f"{d['e65_minigrid_bench']['dreamerv3']['steps_to_first_solve']:,}".replace(",", "{,}")),
        macro("MGDreamerFinalSolve", pct(d["e65_minigrid_bench"]["dreamerv3"]["final_solve_rate"])),
        macro("MGVjepaCosine", f"{d['e65_minigrid_bench']['vjepa2']['value']:.2f}"),
        # Codebase metrics, counted from the live repo (see repo_metrics()).
        macro("LibModules", str(_rm["core_modules"])),
        macro("LibLOC", f"{_rm['core_loc']:,}".replace(",", "{,}")),
        macro("LibTests", str(_rm["test_functions"])),
        macro("NumModels", str(len(_rm["models"]))),
        macro("ModelList", ", ".join("\\texttt{%s}" % m.replace("_", "\\_")
                                     for m in _rm["models"])),
        # E11 multi-world fidelity
        macro("MultiCodeExact", f"{code_total['exact_rollouts']}/{code_total['n']}"),
        macro("MultiWorldRollouts", str(code_total["n"])),
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
    nr = e17["new_rounds_stats"]            # the independent replication seeds alone
    nrm = nr["mcnemar_judge_vs_random"]
    lines += [
        macro("NewRounds", str(nr["n_rounds"])),
        macro("NewJudge", pct(nr["strategies"]["judge"]["rate"])),
        macro("NewRandom", pct(nr["strategies"]["random"]["rate"])),
        macro("NewMcNemarP", f"{nrm['p']:.3f}"),
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
    e28, e29 = d["e28_repairbench_ablation"], d["e29_repairbench_staged"]
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
    # E34 (sprint composite: attempt allocation on owrb-atomic)
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
    # E35 (sprint allocation across a model-capability ladder)
    e35 = d["e35_sprint_ladder"]["ladder"]
    n35 = d["e35_sprint_ladder"]["conditions"][0]["n_tasks"] if "conditions" in d["e35_sprint_ladder"] else 20

    def solved35(model, cond):
        cell = {c["condition"]: c["solved"] for c in e35[model]["conditions"]}
        return cell.get(cond)

    lines += [
        macro("LadderDeepseekFixed", str(solved35("deepseek-r1:14b", "fixed"))),
        macro("LadderGptossFixed", str(solved35("gpt-oss:20b", "fixed"))),
        macro("LadderQwenCoderFixed", str(solved35("qwen3-coder:30b", "fixed"))),
        macro("LadderGptossGreedy", str(solved35("gpt-oss:20b", "greedy"))),
        macro("LadderQwenCoderGreedy", str(solved35("qwen3-coder:30b", "greedy"))),
    ]
    # E36 (representations: composition vs monolithic learners)
    e36 = d["e36_representations"]
    g36 = {r["k"]: r for r in e36["leg_generalization"]}
    i36 = e36["leg_interference"]
    se36 = {r["n_cap"]: r for r in e36["leg_sample_efficiency"]["rows"]}

    def acc(cell):
        return f"{cell:.2f}"

    mono_arms36 = ["monolith", "ridge", "knn1", "knn5", "svr", "gp",
                   "random_forest", "hist_grad_boost", "koopman"]
    n_mono = len([a for a in mono_arms36 if g36[5].get(a)])

    def best_mono(k):
        return max(g36[k][a]["acc"] for a in mono_arms36
                   if g36[k].get(a) and g36[k][a]["acc"] is not None)

    lines += [
        macro("RepNumMono", str(n_mono)),
        macro("RepBestMonoLo", acc(best_mono(2))),
        macro("RepBestMonoHi", acc(best_mono(5))),
        macro("RepCompGenLo", acc(g36[2]["composite_learned"]["acc"])),
        macro("RepCompGenHi", acc(g36[5]["composite_learned"]["acc"])),
        macro("RepCompHgbHi", acc(g36[5]["composite_hgb"]["acc"])
              if g36[5].get("composite_hgb") and g36[5]["composite_hgb"]["acc"] is not None
              else "n/a"),
        macro("RepIntfMono", acc(i36["monolith_sequential_retained"])),
        macro("RepIntfComp", acc(i36["composite_learned_retained"])),
        macro("RepSampMonoLo", acc(se36[100]["monolith_acc"])),
        macro("RepSampCompLo", acc(se36[100]["composite_learned_acc"])),
        macro("RepMonoParams", str(g36[5]["monolith"]["n_params"])),
        macro("RepCompParams", str(g36[5]["composite_learned"]["n_params"])),
    ]
    # E37 (equal-information induction from traces)
    e37 = d["e37_induction"]
    anc37 = e37["summary"]["code_from_rules"]
    big37 = max(e37["summary"]["by_k"], key=lambda r: r["k"])
    lines += [
        macro("IndRulesIn", acc(anc37["probe_in_dist"])),
        macro("IndRulesOod", acc(anc37["probe_ood_10x"])),
        macro("IndTracesIn", acc(big37["code_from_traces_in_dist"])),
        macro("IndTracesOod", acc(big37["code_from_traces_ood"])),
        macro("IndMlpIn", acc(big37["mlp_in_dist"])),
        macro("IndMlpOod", acc(big37["mlp_ood"])),
        macro("IndKnnOod", acc(big37["knn1_ood"])),
    ]
    # E38 (induction across the generator ladder)
    L = {m["model"]: m for m in d["e38_induction_scale"]["ladder"]}
    lines += [
        macro("ScaleQwenSmall", acc(L["qwen2.5:7b"]["mean_in_dist_bigK"])),
        macro("ScaleQwenCoder", acc(L["qwen3-coder:30b"]["mean_in_dist_bigK"])),
        macro("ScaleGptoss", acc(L["gpt-oss:20b"]["mean_in_dist_bigK"])),
        macro("ScaleNumModels", str(len(L))),
    ]
    # E40 (perceive-then-forecast) + E39 (perception fidelity / decomposition)
    fc = d["e40_perceive_forecast"]["forecast_exact"]
    e39 = d["e39_perception_fidelity"]
    lines += [
        macro("PercHorizon", str(d["e40_perceive_forecast"]["horizon"])),
        macro("PercWorldIn", acc(fc["world_model_in_dist"])),
        macro("PercWorldOod", acc(fc["world_model_ood_10x"])),
        macro("PercMlpIn", acc(fc["mlp_in_dist"])),
        macro("PercMlpOod", acc(fc["mlp_ood_10x"])),
        macro("PercMlpInTol", acc(fc["mlp_in_dist_tol2"])),
        macro("PercDecompHolds", "yes" if e39["decomposition_holds"] else "no"),
    ]
    # E41 (non-stationary: adapting to sudden unannounced rule changes)
    e41 = d["e41_nonstationary"]["perfect_perception"]
    chg = [str(c) for c in d["e41_nonstationary"]["changes"]]

    def lag(method):
        vals = e41[method]["recovery_lag"]
        out = [vals[c] for c in chg]
        return "/".join("never" if v is None else str(v) for v in out)

    lines += [
        macro("NonStatSymLag", lag("symbolic_refit")),
        macro("NonStatWinLag", lag("window_1nn")),
        macro("NonStatSymAcc", acc(e41["symbolic_refit"]["accuracy"])),
        macro("NonStatWinAcc", acc(e41["window_1nn"]["accuracy"])),
        macro("NonStatFrozenAcc", acc(e41["static_frozen"]["accuracy"])),
        macro("NonStatChanges", str(len(chg))),
    ]
    # E42 (agent traversal across connected non-stationary worlds)
    e42 = d["e42_agent_traversal"]["summary"]

    def lag(method, key):
        v = e42[method][key]
        return "never" if v is None else (f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}")

    lines += [
        macro("AgentSymAcc", acc(e42["symbolic_per_world"]["accuracy"])),
        macro("AgentSharedAcc", acc(e42["shared_online"]["accuracy"])),
        macro("AgentWinAcc", acc(e42["per_world_window"]["accuracy"])),
        macro("AgentSymUnchg", lag("symbolic_per_world", "recovery_unchanged_return")),
        macro("AgentSharedUnchg", lag("shared_online", "recovery_unchanged_return")),
        macro("AgentSymChanged", lag("symbolic_per_world", "recovery_silently_changed_return")),
        macro("AgentTolls", str(d["e42_agent_traversal"]["tolls_paid"])),
        macro("AgentHops", str(d["e42_agent_traversal"]["n_arrivals"])),
        macro("AgentDwell", str(d["e42_agent_traversal"]["dwell"])),
    ]
    # E43 (active world-model induction: acting beats passive observation)
    e43 = d["e43_active_induction"]["summary"]

    def steps(x):
        return f"{x:.1f}" if x is not None else "n/a"

    lines += [
        macro("ActiveCands", str(e43["n_candidates"])),
        macro("ActiveRules", str(e43["n_rules"])),
        macro("ActiveBudget", str(e43["budget"])),
        macro("ActiveMeanSteps", steps(e43["active_mean_steps"])),
        macro("ActivePassiveSteps", steps(e43["passive_mean_steps"])),
        macro("ActiveClairSteps", steps(e43["clairvoyant_mean_steps"])),
        macro("ActivePassiveUnresolved", str(e43["passive_unresolved"])),
    ]
    # E44 (emergent economy: macro phenomena from composed verified rules)
    e44 = d["e44_emergent_economy"]
    c1, c2 = e44["claim1_price_formation"], e44["claim2_inflation"]
    c3, c4 = e44["claim3_inequality"], e44["claim4_dial"]

    def num(x):
        return f"{x:.0f}" if float(x).is_integer() else f"{x:.2f}"

    lines += [
        macro("EconAgents", str(e44["n_agents"])),
        macro("EconTicks", str(e44["horizon"])),
        macro("EconScarcePrice", f"{round(c1['scarce_supply_price'])}"),
        macro("EconAbundantPrice", f"{round(c1['abundant_supply_price'])}"),
        macro("EconInflOffSlope", num(c2["burn_off_price_slope"])),
        macro("EconInflOnSlope", num(c2["burn_on_price_slope"])),
        macro("EconGiniOff", f"{c3['redist_off_gini']:.2f}"),
        macro("EconGiniOn", f"{c3['redist_on_gini']:.3f}"),
        macro("EconSelfishWelfare", f"{round(c4['selfish_welfare']):,}"),
        macro("EconCoopWelfare", f"{round(c4['cooperative_welfare']):,}"),
        macro("EconSelfishMax", f"{round(c4['selfish_max_gold']):,}"),
        macro("EconCoopMax", f"{round(c4['cooperative_max_gold']):,}"),
        macro("EconCoopWelfareGain",
              f"{100 * (c4['cooperative_welfare'] / c4['selfish_welfare'] - 1):.1f}"),
    ]
    # E46 (factored many-worlds store)
    e46 = d["e46_many_worlds"]["summary"]
    big = d["e46_many_worlds"]["scale"][-1]
    coup = d["e46_many_worlds"]["coupling"]

    def sci(x):
        m, e = f"{x:.0e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}"

    lines += [
        macro("ManyWorldsMax", f"${sci(e46['max_worlds'])}$"),
        macro("ManyWorldsMaxMs", f"{e46['max_worlds_factored_ms']:.0f}"),
        macro("ManyWorldsEnumMax", f"${sci(e46['enum_max_worlds'])}$"),
        macro("ManyWorldsConsistent", f"{big['consistent']:,}"),
        macro("ManyWorldsCoupleWidth", str(coup[-1]["w"])),
        macro("ManyWorldsCoupleFactor", str(coup[-1]["factor_size"])),
        macro("ManyWorldsCoupleIdeal", str(coup[-1]["ideal_factored"])),
    ]
    # E45 (next-token world models: length generalization)
    e45 = d["e45_next_token"]
    s45 = e45["summary"]
    n_exact = sum(1 for t in e45["per_task"] if t["split"]["symbolic"]["ood"] == 1.0)
    lines += [
        macro("NextTokTasks", str(len(e45["per_task"]))),
        macro("NextTokExactTasks", str(n_exact)),
        macro("NextTokMaxLen", str(max(e45["eval_sizes"]))),
        macro("NextTokTrainLen", str(e45["l_train"])),
        macro("NextTokSymOod", acc(s45["symbolic"]["ood"])),
        macro("NextTokNgramOod", acc(s45["ngram"]["ood"])),
        macro("NextTokMlpOod", acc(s45["window_mlp"]["ood"])),
        macro("NextTokDirectIn", acc(s45["llm_direct"]["in_dist"])),
        macro("NextTokDirectOod", acc(s45["llm_direct"]["ood"])),
    ]
    # E47 (relativity as a verified world model)
    e47 = d["e47_relativity"]
    rtd, rtw, rg, rhk = (e47["time_dilation"], e47["twin_paradox"],
                         e47["gps"], e47["hafele_keating"])
    lines += [
        macro("RelNewtErr", f"{rtd['newtonian_err_near_c']:.2f}"),
        macro("RelLearnErr", f"{rtd['learned_err_near_c']:.2f}"),
        macro("RelTwinStay", f"{rtw['symbolic']['stay_years']:.0f}"),
        macro("RelTwinTravel", f"{rtw['symbolic']['travel_years']:.0f}"),
        macro("RelTwinDiff", f"{rtw['symbolic']['diff_years']:.0f}"),
        macro("RelGpsSR", f"{rg['sr_us_per_day']:.1f}"),
        macro("RelGpsGR", f"{rg['gr_us_per_day']:.1f}"),
        macro("RelGpsNet", f"{rg['net_us_per_day']:+.1f}"),
        macro("RelHKModelEast", f"{rhk['model_east_ns']:.0f}"),
        macro("RelHKModelWest", f"{rhk['model_west_ns']:.0f}"),
        macro("RelHKObsEast", f"{rhk['pub_obs_east_ns']}\\pm{rhk['pub_obs_east_err']}"),
        macro("RelHKObsWest", f"{rhk['pub_obs_west_ns']}\\pm{rhk['pub_obs_west_err']}"),
    ]
    # E49 (path integrals over learning trajectories)
    e49 = d["e49_path_integral"]
    pa49 = e49["per_agent"]

    def cur(role):
        return " $\\to$ ".join(s.replace("_", " ") for s in pa49[role]["curriculum"])

    lines += [
        macro("PathGoal", e49["goal"].replace("_", " ")),
        macro("PathSweCost", f"{pa49['senior_swe']['least_action_cost']:.0f}"),
        macro("PathDirCost", f"{pa49['director']['least_action_cost']:.0f}"),
        macro("PathCeoCost", f"{pa49['ceo']['least_action_cost']:.0f}"),
        macro("PathScratch", f"{e49['transfer']['from_scratch']:.0f}"),
        macro("PathSpeedup", f"{e49['transfer']['speedup']:.1f}"),
        macro("PathRandom", f"{pa49['senior_swe']['baselines']['random_mean']:.0f}"),
        macro("PathTraj", f"{e49['tractability']['n_trajectories']:,}"),
        macro("PathStates", str(e49["tractability"]["n_dp_states"])),
    ]
    # E48 (composite corporate world)
    e48 = d["e48_corporate_world"]
    cpar, cvoa = e48["pareto"], e48["value_of_action"]
    cisig = e48["perception"]["individual_signal"]
    cal, csel = cpar[0], cpar[-1]
    lines += [
        macro("CorpDivisions", str(len(e48["divisions"]))),
        macro("CorpRevenue", str(e48["total_revenue"])),
        macro("CorpAlignedGrowth", f"{cal['company_growth'] * 100:.0f}"),
        macro("CorpSelfishGrowth", f"{csel['company_growth'] * 100:.0f}"),
        macro("CorpAlignedGini", f"{cal['promo_gini']:.2f}"),
        macro("CorpSelfishGini", f"{csel['promo_gini']:.2f}"),
        macro("CorpCeoPct", f"{cvoa['ceo_pct']:.0f}"),
        macro("CorpDirPct", f"{cvoa['director_pct']:.0f}"),
        macro("CorpIcPct", f"{cvoa['ic_pct']:.0f}"),
        macro("CorpTranscripts", str(e48["perception"]["corpus"]["n_records"])),
        macro("CorpIndivOneOnOne", f"{cisig['recover_from_one_on_one'] * 100:.0f}"),
        macro("CorpIndivAllHands", f"{cisig['recover_from_all_hands'] * 100:.0f}"),
    ]
    # E50 (same-day trading world model)
    e50 = d["e50_trading"]
    tr = e50["real"]

    def tpct(x):
        return f"{x * 100:+.1f}"

    lines += [
        macro("TradeUniverse", str(e50["universe"])),
        macro("TradeYears", str(round((int(e50["date_range"][1][:4]) - int(e50["date_range"][0][:4])) or 5))),
        macro("TradeHonestAnn", tpct(tr["honest_oos"]["annualized"])),
        macro("TradeHonestSharpe", f"{tr['honest_oos']['sharpe']:.2f}"),
        macro("TradeNoCostSharpe", f"{tr['honest_no_cost']['sharpe']:.2f}"),
        macro("TradeLookaheadSharpe", f"{tr['lookahead']['sharpe']:.2f}"),
        macro("TradeSpySharpe", f"{tr['spy_buy_hold']['sharpe']:.2f}"),
        macro("TradeSpyAnn", tpct(tr["spy_buy_hold"]["annualized"])),
        macro("TradeHonestHit", f"{tr['honest_oos']['hit_rate'] * 100:.0f}"),
        macro("TradeCostBreakeven", "20"),
        macro("TradeSynSharpe", f"{e50['synthetic']['strategy']['sharpe']:.2f}"),
        macro("TradeSynRandom", f"{e50['synthetic']['random']['sharpe']:.2f}"),
    ]
    # E51 (startup growth world model)
    e51 = d["e51_startups"]
    v51, pw51 = e51["value_of_factor"], e51["power_law"]
    cf51, pr51 = e51["counterfactual"], e51["predictability"]
    lines += [
        macro("StartupN", str(e51["n"])),
        macro("StartupPmfDelta", f"{v51['pmf']['delta_value_pct']:+.0f}"),
        macro("StartupCapitalDelta", f"{v51['capital']['delta_value_pct']:+.0f}"),
        macro("StartupSurvival", f"{pw51['survival_rate'] * 100:.0f}"),
        macro("StartupTopDecile", f"{pw51['top_decile_share'] * 100:.0f}"),
        macro("StartupGini", f"{pw51['value_gini']:.2f}"),
        macro("StartupNoPmf", f"{cf51['no_pmf_value_pct_of_base']:.0f}"),
        macro("StartupCapLowPmf", f"{cf51['double_capital_lowpmf_pct_of_base']:.0f}"),
        macro("StartupSpearman", f"{pr51['spearman_m6_vs_final']:.2f}"),
        macro("StartupTopOverlap", f"{pr51['top_decile_overlap'] * 100:.0f}"),
    ]
    # E52 (wavelet denoising)
    e52 = d["e52_denoise"]
    ed52, sm52 = e52["edges_delta"], e52["smooth_delta"]
    lines += [
        macro("DenoiseSparsity", f"{e52['noisy_sparsity'] * 100:.0f}"),
        macro("DenoisePR", f"{e52['pr_max_error']:.0e}"),
        macro("DenoiseEdgeWavelet", f"{ed52['wavelet']:+.1f}"),
        macro("DenoiseEdgeTuned", f"{ed52['tuned_lowpass']:+.1f}"),
        macro("DenoiseEdgeNaive", f"{ed52['lowpass']:+.1f}"),
        macro("DenoiseSmoothWavelet", f"{sm52['wavelet']:+.1f}"),
        macro("DenoiseSmoothTuned", f"{sm52['tuned_lowpass']:+.1f}"),
        macro("DenoiseNaiveMean", f"{e52['naive_mean']:+.1f}"),
        macro("DenoiseTunedMean", f"{e52['tuned_mean']:+.1f}"),
        macro("DenoiseSpeech", f"{e52['speech_gain']:+.1f}"),
    ]
    # E53 (sheaf consistency)
    e53 = d["e53_sheaf"]
    r3 = e53["rows"]
    one_fault = next(r for r in r3 if r["n_faults"] == 1)
    three = next(r for r in r3 if r["n_faults"] == 3)
    lines += [
        macro("SheafSensors", str(e53["sensors"])),
        macro("SheafLocations", str(e53["locations"])),
        macro("SheafBetti", str(e53["nerve_betti1"])),
        macro("SheafGlueErr", f"{e53['glue_exact_error']:.0e}"),
        macro("SheafOneLocalize", f"{one_fault['localize_acc'] * 100:.0f}"),
        macro("SheafMajRmse", f"{three['majority_rmse']:.3f}"),
        macro("SheafAvgRmse", f"{three['average_rmse']:.3f}"),
    ]
    # E54 (abstract interpretation / bounds)
    e54 = d["e54_bounds"]
    longest = max(e54["rows"], key=lambda r: r["T"])
    lines += [
        macro("BoundsHorizon", str(longest["T"])),
        macro("BoundsAffineWidth", f"{longest['affine_width']:.1f}"),
        macro("BoundsIntervalWidth", f"{longest['interval_width']:.1f}"),
        macro("BoundsTrueWidth", f"{longest['true_width']:.1f}"),
        macro("BoundsMcMiss", f"{max(r['mc_misses_hi'] for r in e54['rows']):.2f}"),
    ]
    # E55 (information geometry)
    e55 = d["e55_infogeom"]
    s55 = e55["summary"]
    lines += [
        macro("InfoGeomWorlds", str(e55["n_worlds"])),
        macro("InfoGeomProbes", str(e55["n_probes"])),
        macro("InfoGeomEig", f"{s55['eig']['mean_steps']:.1f}"),
        macro("InfoGeomHeuristic", f"{s55['heuristic']['mean_steps']:.1f}"),
        macro("InfoGeomRandom", f"{s55['random']['mean_steps']:.1f}"),
    ]
    # E56 (optimal transport)
    e56 = d["e56_transport"]
    c56 = e56["calibration"]
    lines += [
        macro("TransportTrueShifts", ", ".join(map(str, e56["true_changes"]))),
        macro("TransportDetected", ", ".join(map(str, e56["detected_changes"]))),
        macro("TransportMuHat", f"{c56['wasserstein_mu_hat']:.1f}"),
        macro("TransportWSlope", f"{c56['w_far_rel_slope']:.3f}"),
        macro("TransportKlSlope", f"{c56['kl_far_rel_slope']:.3f}"),
    ]
    # E57 (world-model specs + cards)
    e57 = d["e57_world_specs"]
    nexact = sum(1 for r in e57["rows"] if r["round_trip_exact"])
    lines += [
        macro("SpecsNumWorlds", str(e57["n_worlds"])),
        macro("SpecsRoundTrip", f"{nexact}/{e57['n_worlds']}"),
        macro("SpecsComponents", "9"),
    ]
    # E58 (brain simulator)
    e58 = d["e58_brain"]
    lines += [
        macro("BrainNodes", str(e58["env"]["nodes"])),
        macro("BrainOptimal", str(e58["env"]["optimal_len"])),
        macro("BrainFinal", str(e58["panelA_brain_final"])),
        macro("BrainNoMem", f"{e58['panelA_means']['no_memory']:.0f}"),
        macro("BrainRandom", f"{e58['panelA_means']['random']:.0f}"),
        macro("BrainPlateau", str(e58["plateau_depth"])),
    ]
    # E59 (brain architecture search)
    e59 = d["e59_brain_arch"]
    lines += [
        macro("ArchBare", f"{e59['arm_accuracy']['bare LLM']:.2f}"),
        macro("ArchMem", f"{e59['arm_accuracy']['+ memory (retrieval)']:.2f}"),
        macro("ArchOpt", f"{e59['arm_accuracy']['optimized brain']:.2f}"),
        macro("ArchLift", f"{e59['lift_over_bare']:.2f}"),
        macro("ArchConfigs", str(e59["n_configs"])),
    ]
    # E60 (the perceive -> world -> emit -> act boundary)
    e60 = d["e60_io_boundary"]
    lines += [
        macro("IOReca", f"{e60['routing']['trigram'] * 100:.0f}"),
        macro("IOJac", f"{e60['routing']['jaccard'] * 100:.0f}"),
        macro("IOExact", f"{e60['routing']['exact_key'] * 100:.0f}"),
        macro("IORes", f"{e60['resolution_rate'] * 100:.0f}"),
        macro("IOTickets", str(e60["n_tickets"])),
        macro("IOGateIn", str(e60["gates"]["caught_input"])),
        macro("IOGateBad", str(e60["gates"]["bad_inputs"])),
        macro("IOComponents", str(len(e60["components_exercised"]))),
    ]
    # E61 (trained world model vs verified code on downstream control)
    e61 = d["e61_trained_wm_control"]
    best_k = max(e61["k_budgets"])
    lines += [
        macro("TwmVerified", f"{e61['verified_code_return']:.1f}"),
        macro("TwmReactive", f"{e61['reactive_return']:.1f}"),
        macro("TwmRandom", f"{e61['random_return']:.1f}"),
        macro("TwmBestTrained", f"{e61['best_trained_mean']:.1f}"),
        macro("TwmRegret", f"{e61['regret_best_trained']:.1f}"),
        macro("TwmHarm", f"{e61['harm_fraction'] * 100:.0f}"),
        macro("TwmMaxK", f"{best_k:,}".replace(",", "{,}")),
        macro("TwmMlpSmall", f"{e61['trained']['mlp'][str(min(e61['k_budgets']))]['mean']:.1f}"),
        macro("TwmSeeds", str(len(e61["seeds"]))),
    ]
    # E62 (branch-covering acceptance gate)
    e62 = d["e62_branch_gate"]
    lines += [
        macro("BranchSingleFA", f"{e62['single_state_false_accept_rate'] * 100:.0f}"),
        macro("BranchCovFA", f"{e62['branch_covering_false_accept_rate'] * 100:.0f}"),
        macro("BranchFaults", str(e62["n_branch_faults"])),
        macro("BranchProbes", str(e62["n_probe_states"])),
    ]
    # E63 (world-model bake-off)
    e63 = d["e63_world_model_bakeoff"]
    learned = [m for m in e63["sprint_control"] if m != "verified code (CWM)"]
    ood_max = max(e63["fidelity"][dom][m]["probe_ood"]
                  for dom in e63["domains"] for m in learned)
    lines += [
        macro("BakeMethods", str(e63["n_methods_runnable"])),
        macro("BakeDomains", str(len(e63["domains"]))),
        macro("BakeCodeReturn", f"{e63['sprint_control']['verified code (CWM)']:.1f}"),
        macro("BakeLearnedOodMax", f"{ood_max:.2f}"),
        macro("BakeCodeSpeed", f"{e63['sprint_steps_per_sec']['verified code (CWM)']:,}".replace(",", "{,}")),
        macro("BakePerceptual", str(len(e63["perceptual_world_models_compared_on_properties"]))),
    ]
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n")


def fig_world_specs(e57):
    """E57: portable world-model specs. Lossless round-trip + spec size per world,
    and the set of world-model components the spec captures."""
    rows = e57["rows"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))

    names = [r["name"] for r in rows]
    kb = [r["spec_bytes"] / 1024 for r in rows]
    cols = [TEAL if r["kind"] == "composite" else BLUE for r in rows]
    a.barh(range(len(rows)), kb, color=cols)
    a.set_yticks(range(len(rows)))
    a.set_yticklabels(names, fontsize=8)
    a.invert_yaxis()
    for i, r in enumerate(rows):
        if r["round_trip_exact"]:
            a.text(kb[i] + max(kb) * 0.02, i, "exact", va="center", fontsize=7.5,
                   color=TEAL, fontweight="bold")
    a.set_xlim(0, max(kb) * 1.25)
    a.set_xlabel("spec size (KB)")
    a.set_title("A. Portable spec, lossless round-trip", fontsize=9.5, loc="left")
    handles = [Patch(color=BLUE, label="leaf"), Patch(color=TEAL, label="composite")]
    a.legend(handles=handles, fontsize=7.5, loc="upper right")

    comps = ["state schema (+ values)", "actions", "rules (declared contract)",
             "dynamics (verified code)", "perception (perceive)", "emit (outputs)",
             "objectives", "metrics", "composition (recursive)"]
    b.axis("off")
    b.set_title("B. World-model components the spec captures", fontsize=9.5, loc="left")
    for i, name in enumerate(comps):
        y = 0.96 - i * 0.108
        b.scatter([0.04], [y], s=42, color=TEAL, transform=b.transAxes, zorder=3)
        b.text(0.10, y, name, fontsize=9, transform=b.transAxes, va="center")

    fig.suptitle("World-model specs: a portable, lossless, complete artifact (E57)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "world_specs.png", dpi=200)
    plt.close(fig)


def table_world_specs(e57):
    lines = ["\\begin{tabular}{llrcr}", "\\toprule",
             "World & Kind & Children & Round-trip & Spec (KB) \\\\", "\\midrule"]
    for r in e57["rows"]:
        rt = "exact" if r["round_trip_exact"] else "lossy"
        lines.append(f"{r['name']} & {r['kind']} & {r['n_children']} & {rt} & "
                     f"{r['spec_bytes'] / 1024:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "world_specs.tex").write_text("\n".join(lines) + "\n")


def fig_brain(e58):
    """E58: a brain simulator. Memory-augmented tree-of-thoughts ReAct learns to
    the optimal path while memoryless/random do not; and lookahead depth helps
    only up to the needed horizon."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    cur = e58["panelA_curves"]
    L = e58["env"]["optimal_len"]
    ep = range(1, e58["episodes"] + 1)
    a.plot(ep, cur["brain"], "-o", color=BLUE, lw=2, markersize=4,
           label="brain (persistent memory)")
    a.plot(ep, cur["no_memory"], "-s", color=ORANGE, lw=2, markersize=4,
           label="no memory (wiped each episode)")
    a.plot(ep, cur["random"], "-^", color=SLATE, lw=1.8, markersize=4, label="random")
    a.axhline(L, color=TEAL, ls="--", lw=1.5, label=f"optimal (L={L})")
    a.set_xlabel("episode"); a.set_ylabel("steps to goal")
    a.set_title("A. Memory wins: the brain learns to optimal", fontsize=9.3, loc="left")
    a.legend(fontsize=7)

    d = e58["depths"]
    b.plot(d, e58["panelB_steps"], "-o", color=BLUE, lw=2, markersize=4)
    b.set_xlabel("planning depth $D$"); b.set_ylabel("mean steps to goal", color=BLUE)
    b.tick_params(axis="y", labelcolor=BLUE)
    b.axvline(e58["plateau_depth"], color=SLATE, ls=":", lw=1.2)
    b.text(e58["plateau_depth"] + 0.1, max(e58["panelB_steps"]) * 0.8,
           f"plateau\nD={e58['plateau_depth']}", fontsize=7.5, color=SLATE)
    bb = b.twinx()
    bb.plot(d, e58["panelB_success"], "-^", color=TEAL, lw=2, markersize=4)
    bb.set_ylabel("success rate", color=TEAL); bb.tick_params(axis="y", labelcolor=TEAL)
    bb.set_ylim(0, 1.08)
    b.set_title("B. How much to think: depth helps, then plateaus", fontsize=9.3, loc="left")

    fig.suptitle("Brain simulator: tree-of-thoughts ReAct with real memory (E58)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "brain.png", dpi=200)
    plt.close(fig)


def table_brain(e58):
    m = e58["panelA_means"]
    cur = e58["panelA_curves"]
    lines = ["\\begin{tabular}{lrr}", "\\toprule",
             "Agent & Mean steps & Final episode \\\\", "\\midrule",
             f"brain (persistent memory) & {m['brain']:.1f} & {cur['brain'][-1]} \\\\",
             f"no memory (wiped) & {m['no_memory']:.1f} & {cur['no_memory'][-1]} \\\\",
             f"random & {m['random']:.1f} & {cur['random'][-1]} \\\\",
             "\\bottomrule", "\\end{tabular}"]
    (TABLES / "brain.tex").write_text("\n".join(lines) + "\n")


def fig_brain_arch(e59):
    """E59: optimizing the brain architecture with the LLM held constant. Accuracy
    by architecture, and a search that recovers the best of all configs."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    names = list(e59["arm_accuracy"])
    accs = [e59["arm_accuracy"][n] for n in names]
    cols = [SLATE, ORANGE, TEAL, BLUE][:len(names)]
    a.bar(range(len(names)), accs, color=cols)
    a.set_xticks(range(len(names)))
    a.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7.3)
    a.set_ylabel("task accuracy"); a.set_ylim(0, 1.1)
    for i, v in enumerate(accs):
        a.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5)
    a.set_title("A. Architecture matters (one fixed LLM)", fontsize=9.3, loc="left")

    c = e59["search_curve"]
    b.plot(range(1, len(c) + 1), c, "-o", color=BLUE, lw=2, markersize=3.5)
    b.axhline(e59["best_acc"], color=TEAL, ls="--", lw=1.4,
              label=f"best ({e59['best_acc']:.2f})")
    b.set_xlabel("architectures tried"); b.set_ylabel("best accuracy so far")
    b.set_ylim(0, 1.1)
    b.set_title(f"B. Search over {e59['n_configs']} architectures", fontsize=9.3, loc="left")
    b.legend(fontsize=8, loc="lower right")

    fig.suptitle("Optimizing the brain architecture for the task, LLM held constant (E59)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "brain_arch.png", dpi=200)
    plt.close(fig)


def table_brain_arch(e59):
    arms, acc = e59["arms"], e59["arm_accuracy"]
    lines = ["\\begin{tabular}{lll r}", "\\toprule",
             "Architecture & Memory & Tree+verify & Accuracy \\\\", "\\midrule"]
    for name, cfg in arms.items():
        mem = "yes" if cfg[0] == "longterm" else "--"
        tv = f"w{cfg[1]}+verify" if cfg[2] else (f"w{cfg[1]}" if cfg[1] > 1 else "--")
        lines.append(f"{name} & {mem} & {tv} & {acc[name]:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "brain_arch.tex").write_text("\n".join(lines) + "\n")


def fig_io_boundary(e60):
    """E60: the perceive -> world -> emit -> act boundary. Content-addressable recall
    routes realistic paraphrased tickets, beating a fair lexical baseline and naive
    exact-key lookup; the assembled world acts via real tool calls; contract gates
    reject every malformed input/output."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    labels = ["exact-key\nlookup", "token-overlap\n(Jaccard)", "content-addr.\n(MemoryStore)",
              "end-to-end\nresolution"]
    vals = [e60["routing"]["exact_key"], e60["routing"]["jaccard"],
            e60["routing"]["trigram"], e60["resolution_rate"]]
    cols = [SLATE, ORANGE, TEAL, BLUE]
    a.bar(range(len(vals)), vals, color=cols)
    a.set_xticks(range(len(vals))); a.set_xticklabels(labels, fontsize=7.6)
    a.set_ylabel("routing accuracy"); a.set_ylim(0, 1.12)
    for i, v in enumerate(vals):
        a.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5)
    a.set_title(f"A. Routing & acting ({e60['n_tickets']} paraphrased tickets)",
                fontsize=9.3, loc="left")

    g = e60["gates"]
    gl = ["malformed\ninput", "out-of-contract\noutput"]
    seen = [g["bad_inputs"], 1]
    caught = [g["caught_input"], g["caught_output"]]
    x = range(len(gl))
    b.bar([i - 0.2 for i in x], seen, width=0.4, color=SLATE, label="injected")
    b.bar([i + 0.2 for i in x], caught, width=0.4, color=ORANGE, label="caught by gate")
    b.set_xticks(list(x)); b.set_xticklabels(gl, fontsize=8)
    b.set_ylabel("count"); b.set_ylim(0, max(seen) + 1.2)
    for i in x:
        b.text(i + 0.2, caught[i] + 0.05, str(caught[i]), ha="center", fontsize=8.5)
    b.set_title("B. Contract gates catch every fault", fontsize=9.3, loc="left")
    b.legend(fontsize=8, loc="upper right")

    fig.suptitle("The perceive → world → emit → act boundary, validated end to end (E60)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "io_boundary.png", dpi=200)
    plt.close(fig)


def table_io_boundary(e60):
    """One row per I/O component, with the role it plays and whether the E60 world
    exercises it as a working, serializable piece."""
    rows = [
        ("JSONPerceptor", "perceive", "ingest a structured ticket payload"),
        ("RegexPerceptor", "perceive", "pull fields from free text"),
        ("PerceptionGate", "perceive", "reject out-of-contract input"),
        ("MemoryStore", "recall", "semantic recall of past resolutions"),
        ("CodeEmitter", "emit", "deterministic verified-code report"),
        ("ToolEmitter", "act", "choose + execute a registered tool"),
        ("ToolRegistry", "act", "typed, validated tool dispatch"),
        ("EmissionGate", "emit", "reject out-of-contract output"),
    ]
    cov = e60["components_exercised"]
    lines = ["\\begin{tabular}{lll c}", "\\toprule",
             "Component & Boundary & Role & Exercised \\\\", "\\midrule"]
    for name, boundary, role in rows:
        mark = "\\checkmark" if cov.get(name) else "--"
        lines.append(f"{name} & {boundary} & {role} & {mark} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "io_boundary.tex").write_text("\n".join(lines) + "\n")


def fig_trained_wm_control(e61):
    """E61: downstream control return vs training samples K. Verified code (0 samples)
    is the optimal upper bound; trained world models are sample-inefficient and
    planning through them can fall below model-free control."""
    ks = e61["k_budgets"]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.axhline(e61["verified_code_return"], color=TEAL, lw=2.2,
               label=f"verified code, 0 samples ({e61['verified_code_return']:.1f})")
    ax.axhline(e61["reactive_return"], color=SLATE, ls="--", lw=1.4,
               label=f"reactive (model-free, {e61['reactive_return']:.1f})")
    ax.axhline(e61["random_return"], color="#9ca3af", ls=":", lw=1.4,
               label=f"random ({e61['random_return']:.1f})")
    for kind, col, lab in (("mlp", BLUE, "trained MLP world model"),
                           ("nn", ORANGE, "trained 1-NN world model")):
        means = [e61["trained"][kind][str(k)]["mean"] for k in ks]
        sds = [e61["trained"][kind][str(k)]["sd"] for k in ks]
        ax.errorbar(ks, means, yerr=sds, marker="o", color=col, lw=1.8,
                    capsize=3, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("training transitions $K$ (log scale)")
    ax.set_ylabel("downstream return (12-step episode)")
    ax.set_title(f"Verified code vs trained world models on control "
                 f"(E61, {len(e61['seeds'])} seeds)", fontsize=10, loc="left")
    ax.legend(fontsize=7.6, loc="lower right")
    ax.axhspan(e61["random_return"] - 6, e61["reactive_return"], color="#fee2e2", alpha=0.35,
               zorder=0)
    fig.tight_layout()
    fig.savefig(FIGS / "trained_wm_control.png", dpi=200)
    plt.close(fig)


def table_bakeoff(e63):
    """E63: every world model we can run, side by side, averaged over domains
    (probe in-dist / 10x OOD / rollout) with sprint control return + auditability."""
    doms = e63["domains"]
    order = ["verified code (CWM)"] + [m for m in e63["sprint_control"]
                                       if m != "verified code (CWM)"]
    train = {"verified code (CWM)": "0 (rules)"}
    for m in order[1:]:
        train[m] = f"{e63['k_trained']:,}"

    def avg(model, metric):
        vals = [e63["fidelity"][dom][model][metric] for dom in doms]
        return sum(vals) / len(vals)

    rows = []
    for m in order:
        audit = "\\checkmark" if m == "verified code (CWM)" else "--"
        rows.append(f"{m} & {train[m]} & {avg(m,'probe_in'):.2f} & {avg(m,'probe_ood'):.2f} & "
                    f"{avg(m,'rollout'):.2f} & {e63['sprint_control'][m]:+.1f} & {audit} \\\\")
    llm = e63.get("llm_proxy", {})
    rows.append(f"LLM next-state$^\\dagger$ & rules & -- & -- & "
                f"{llm.get('sprint_rollout', 0):.2f} & {llm.get('sprint_control', 0):+.1f} & -- \\\\")
    (TABLES / "bakeoff.tex").write_text(
        "\\begin{tabular}{llccccc}\n\\toprule\n"
        "World model & Train data & Probe & Probe & Rollout & Control & Audit. \\\\\n"
        "& (transitions) & in-dist & 10$\\times$OOD & exact & return & \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n")


def table_minigrid_bench(e65):
    """Trained-vs-verified head-to-head on a shared benchmark, MiniGrid DoorKey-6x6
    (E65): one row per world-model species, with what it costs to reach the solution.
    V-JEPA is perceptual (no symbolic policy), so it has no solve rate -- it reports a
    representation-drift metric, clearly marked as a different species."""
    ow, dv, vj = e65["openworld"], e65["dreamerv3"], e65["vjepa2"]
    fs = f"{dv['steps_to_first_solve']:,}".replace(",", "{,}")
    lines = ["\\begin{tabular}{llccl}", "\\toprule",
             "World model & Species & Train data & Solve & Cost to solution \\\\",
             "\\midrule",
             f"OpenWorld & verified code & 0 & {pct(ow['success'])} & "
             f"0-shot plan, length {ow['plan_length']} \\\\",
             f"DreamerV3 & learned (pixels) & {fs} steps & {pct(dv['final_solve_rate'])} & "
             f"first solve @ {fs} steps \\\\",
             f"V-JEPA-2 & perceptual & pretrained & -- & "
             f"cos-drift {vj['value']:.2f} (not a solve rate) \\\\",
             "\\bottomrule", "\\end{tabular}"]
    (TABLES / "minigrid_bench.tex").write_text("\n".join(lines) + "\n")


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
    table_induction(data["e37_induction"])
    fig_induction_scale(data["e38_induction_scale"])
    table_repair(data["e18_repair_loop"])
    fig_complexity(data["e20_complexity"])
    fig_composition(data["e31_nested_fidelity"])
    fig_composition_cliff(data["e20_complexity"], data["e30_composition"])
    fig_traversal(data["e31_nested_fidelity"])
    fig_dynamic_traversal(data["e33_dynamic_traversal"])
    fig_sprint(data["e34_composite_swe"])
    fig_sprint_ladder(data["e35_sprint_ladder"], data["e34_composite_swe"])
    fig_representations(data["e36_representations"])
    fig_perception(data["e40_perceive_forecast"])
    fig_nonstationary(data["e41_nonstationary"])
    fig_agent_traversal(data["e42_agent_traversal"])
    fig_active_induction(data["e43_active_induction"])
    fig_emergent_economy(data["e44_emergent_economy"])
    fig_many_worlds(data["e46_many_worlds"])
    fig_next_token(data["e45_next_token"])
    table_next_token(data["e45_next_token"])
    fig_relativity(data["e47_relativity"])
    table_relativity(data["e47_relativity"])
    fig_path_integral(data["e49_path_integral"])
    table_path_integral(data["e49_path_integral"])
    fig_corporate_world(data["e48_corporate_world"])
    fig_trading(data["e50_trading"])
    table_trading(data["e50_trading"])
    fig_startups(data["e51_startups"])
    table_startups(data["e51_startups"])
    fig_denoise(data["e52_denoise"])
    table_denoise(data["e52_denoise"])
    fig_sheaf(data["e53_sheaf"])
    table_sheaf(data["e53_sheaf"])
    fig_bounds(data["e54_bounds"])
    table_bounds(data["e54_bounds"])
    fig_infogeom(data["e55_infogeom"])
    table_infogeom(data["e55_infogeom"])
    fig_transport(data["e56_transport"])
    table_transport(data["e56_transport"])
    fig_world_specs(data["e57_world_specs"])
    table_world_specs(data["e57_world_specs"])
    fig_brain(data["e58_brain"])
    table_brain(data["e58_brain"])
    fig_brain_arch(data["e59_brain_arch"])
    table_brain_arch(data["e59_brain_arch"])
    fig_io_boundary(data["e60_io_boundary"])
    table_io_boundary(data["e60_io_boundary"])
    fig_trained_wm_control(data["e61_trained_wm_control"])
    table_bakeoff(data["e63_world_model_bakeoff"])
    table_minigrid_bench(data["e65_minigrid_bench"])
    table_corporate_world(data["e48_corporate_world"])
    table_many_worlds(data["e46_many_worlds"])
    table_representations(data["e36_representations"])
    table_planning(data["e22_planning"])
    table_repairbench(data["e28_repairbench_ablation"], data["e29_repairbench_staged"])
    table_composition(data["e30_composition"], data["e32_regime_switch"])
    numbers_tex(data)
    print("assets written to paper/figs, paper/tables, paper/numbers.tex")


if __name__ == "__main__":
    main()
