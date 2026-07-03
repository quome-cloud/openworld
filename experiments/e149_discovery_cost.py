"""E149 -- Discovery cost: the axis RHAE does NOT measure.

RHAE (E120) scores only the length of the *banked solution's replay path* vs a human baseline --
it is a solution-PATH economy metric, not a cost-to-SOLVE metric. It excludes everything spent
DISCOVERING the win: exploration env-steps, LLM sampling/tokens, and the build+search of the code
world model. For an approach that wins by synthesizing and searching a verified CodeTransition world
model, that omission is large. This experiment reports the missing axis from the capture we already
have (per-session model_config + transcript stats in arc3_traces/meta): the LLM discovery effort
(USD, tokens, turns, tool calls) aggregated per game and per model, paired with the outcome.

RHAE answers "is the found path efficient?"  E149 answers "was finding it cheap?"  They are
orthogonal and should be reported side by side, not conflated.

Scope / honesty:
  - opus and fable (Claude arms, stream-json capture) have cost_usd/tokens/turns; CODEX does not --
    its gzipped-plaintext capture never populated token stats, so codex discovery cost is UNKNOWN
    here (reported as such, not as zero).
  - Exploration env-steps happen inside agent-run python and are not individually logged; the LLM
    turns/cost are the captured proxy for discovery effort. Deterministic read of the capture; no LLM.

  python experiments/e149_discovery_cost.py
"""
import os, sys, json, glob, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACES = os.path.join(ROOT, "experiments/results/arc3_traces")
RESULTS = os.path.join(ROOT, "experiments/results")
MODEL_OF = {"claude-opus-4-8": "opus", "gpt-5.5": "codex", "claude-fable-5": "fable"}
ARCHIVE = {"opus": "arc3_fullgame_sourcefree.json", "codex": "arc3_fullgame_sourcefree_codex.json",
           "fable": "arc3_fullgame_sourcefree_fable.json"}
BLUE, OCHRE, TEAL, INK = "#1d4ed8", "#b45309", "#0f766e", "#16202e"
COL = {"opus": BLUE, "codex": OCHRE, "fable": TEAL}


def arm_of(m):
    a = MODEL_OF.get((m.get("model_config") or {}).get("resolved_model") or m.get("model"))
    if a:
        return a
    return "codex" if "agent-codex" in (m.get("run_id") or "") else None


def solved_full(arm):
    p = os.path.join(RESULTS, ARCHIVE[arm])
    if not os.path.exists(p):
        return {}
    pg = json.load(open(p))["per_game"]
    return {g: (v.get("levels", 0), v.get("win", 0)) for g, v in pg.items()}


def main():
    # aggregate discovery cost per (arm, game) from the capture
    agg = collections.defaultdict(lambda: dict(n=0, cost=0.0, tokens=0, turns=0, tools=0, have_cost=0))
    for f in glob.glob(os.path.join(TRACES, "meta", "*.json")):
        try:
            m = json.load(open(f))
        except Exception:
            continue
        arm = arm_of(m)
        if arm is None:
            continue
        game = m.get("game") or (m.get("run_id") or "").split("__")[0]
        tr = m.get("transcript") or {}
        a = agg[(arm, game)]
        a["n"] += 1
        c = tr.get("cost_usd")
        if isinstance(c, (int, float)) and c > 0:
            a["cost"] += float(c); a["have_cost"] += 1
        tok = tr.get("tokens") or {}
        a["tokens"] += int(tok.get("total") or 0) if isinstance(tok, dict) else 0
        a["turns"] += int(tr.get("num_turns") or 0)
        a["tools"] += int(tr.get("n_tool_calls") or 0)

    # per-arm rollups + per solved-game cost
    full = {arm: solved_full(arm) for arm in ARCHIVE}
    per_arm = {}
    per_game_cost = collections.defaultdict(dict)      # game -> {arm: cost}
    for (arm, game), a in agg.items():
        per_game_cost[game][arm] = round(a["cost"], 2)
    for arm in ("opus", "codex", "fable"):
        keys = [(arm, g) for (arm2, g) in agg if arm2 == arm]
        keys = [k for k in agg if k[0] == arm]
        tot_cost = sum(agg[k]["cost"] for k in keys)
        tot_tok = sum(agg[k]["tokens"] for k in keys)
        tot_turns = sum(agg[k]["turns"] for k in keys)
        n_sess = sum(agg[k]["n"] for k in keys)
        have_cost = sum(agg[k]["have_cost"] for k in keys)
        solved = [g for g, (lv, w) in full[arm].items() if w and lv >= w]
        cost_solved = sum(agg[(arm, g)]["cost"] for g in solved if (arm, g) in agg)
        per_arm[arm] = {
            "sessions": n_sess, "sessions_with_cost": have_cost,
            "total_cost_usd": round(tot_cost, 2), "total_tokens": tot_tok, "total_turns": tot_turns,
            "n_full_solved": len(solved),
            "cost_usd_over_solved_games": round(cost_solved, 2),
            "mean_cost_usd_per_solved_game": round(cost_solved / len(solved), 2) if solved else None,
            "cost_captured": have_cost > 0,
        }

    _plot(per_arm, per_game_cost, full)

    payload = {
        "description": "Discovery cost -- the LLM effort (USD/tokens/turns) to FIND each solve, the axis "
                       "RHAE (solution-path economy) does not measure. Companion to E120/E141 RHAE.",
        "relation_to_rhae": "RHAE = min((human_baseline/replay_actions)^2*100,115) per level -> scores ONLY "
                            "the banked path length. E149 reports the orthogonal cost-to-discover.",
        "codex_note": "codex (gpt-5.5) discovery cost is NOT captured (plaintext-log capture lacks token "
                      "stats); reported as unknown, not zero.",
        "per_arm": per_arm,
        "per_game_cost_usd": {g: v for g, v in sorted(per_game_cost.items())},
    }
    save_results("e149_discovery_cost", payload)

    print("E149 OK -- discovery cost (the axis RHAE omits)")
    for arm in ("opus", "codex", "fable"):
        p = per_arm[arm]
        if p["cost_captured"]:
            print(f"  {arm:5}: ${p['total_cost_usd']:.0f} total over {p['sessions']} sessions | "
                  f"${p['mean_cost_usd_per_solved_game']}/solved game | {p['total_tokens']/1e6:.1f}M tokens, "
                  f"{p['total_turns']} turns | {p['n_full_solved']} full solves")
        else:
            print(f"  {arm:5}: discovery cost NOT captured ({p['sessions']} sessions) -- unknown, not zero")


def _plot(per_arm, per_game_cost, full):
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white",
                         "axes.edgecolor": "#c7ccd4"})
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})
    # A: mean discovery $ per solved game, by arm
    arms = [a for a in ("opus", "fable") if per_arm[a]["cost_captured"]]
    vals = [per_arm[a]["mean_cost_usd_per_solved_game"] or 0 for a in arms]
    ax[0].bar(arms, vals, color=[COL[a] for a in arms], width=0.55)
    for i, v in enumerate(vals):
        ax[0].text(i, v, f"${v:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax[0].set_ylabel("USD / solved game", fontsize=9)
    ax[0].set_title("A. Discovery cost per solved game\n(codex not captured)", fontsize=11, loc="left", fontweight="bold")
    for s in ("top", "right"): ax[0].spines[s].set_visible(False)
    # B: per-game discovery $ (the games are not equally cheap to discover)
    games = sorted(per_game_cost, key=lambda g: -max(per_game_cost[g].get("opus", 0), per_game_cost[g].get("fable", 0)))[:14]
    x = np.arange(len(games)); w = 0.4
    for i, a in enumerate(("opus", "fable")):
        ax[1].bar(x + (i - 0.5) * w, [per_game_cost[g].get(a, 0) for g in games], w, color=COL[a], label=a)
    ax[1].set_xticks(x); ax[1].set_xticklabels(games, rotation=45, ha="right", fontsize=8)
    ax[1].set_ylabel("USD to discover", fontsize=9); ax[1].legend(frameon=False, fontsize=9)
    ax[1].set_title("B. Per-game discovery cost (RHAE is blind to all of this)", fontsize=11, loc="left", fontweight="bold")
    for s in ("top", "right"): ax[1].spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "e149_discovery_cost.png"), dpi=170); plt.close(fig)


if __name__ == "__main__":
    main()
