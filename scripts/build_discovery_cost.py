"""Token-based discovery cost per ARC-AGI-3 game (Fig. arc3_discovery_cost).

The dollar cost (\\texttt{total_cost_usd}) is only in a session's final \\texttt{result} event, which the
overnight kill-and-resume harness usually prevented sessions from reaching -- so USD covers only a subset
of runs. TOKEN usage, by contrast, is reconstructed from per-message usage for every session
(\\texttt{transcript.tokens.total}), so a token-based discovery cost covers ALL 25 games for both the
primary (Claude Opus 4.8) and Claude Fable arms. Codex (GPT-5.5) ran via \\texttt{codex exec} plaintext
logs that record no token stats, so it is reported as not-captured (not zero).

Aggregates experiments/results/arc3_traces/meta/*.json by (game, arm) and writes
  experiments/results/e149_discovery_cost.json  and  papers/assets/figs/arc3_discovery_cost.png.
Run: python scripts/build_discovery_cost.py
"""
import json, glob, os
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "experiments" / "results" / "arc3_traces" / "meta"
FIGS = ROOT / "papers" / "assets" / "figs"
GAMES = ("ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 "
         "sp80 su15 tn36 tr87 tu93 vc33 wa30").split()
BLUE, TEAL, OCHRE = "#1f4e79", "#2a8a7f", "#c8881f"
ARMS = ["opus", "fable"]
ARM_LABEL = {"opus": "Claude Opus 4.8 (primary)", "fable": "Claude Fable"}
ARM_COLOR = {"opus": BLUE, "fable": TEAL}


def arm_of(audit_dir):
    b = os.path.basename(audit_dir or "")
    if b.startswith("sbfable_"):
        return "fable"
    if b.startswith("sbcodex_"):
        return "codex"
    if b.startswith("sb_"):
        return "opus"
    return None


def main():
    agg = {}                                             # (game, arm) -> {tok, usd, n}
    for f in glob.glob(str(META / "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        g = d.get("game"); a = arm_of(d.get("audit_dir")); tr = d.get("transcript") or {}
        if g not in GAMES or a is None or not isinstance(tr, dict):
            continue
        tok = (tr.get("tokens") or {}).get("total") or 0
        usd = tr.get("cost_usd") or tr.get("cost_usd_estimated") or 0
        e = agg.setdefault((g, a), {"tok": 0, "usd": 0.0, "n": 0})
        e["tok"] += tok; e["usd"] += usd; e["n"] += (1 if tok > 0 else 0)

    per_game = {g: {a: agg.get((g, a), {"tok": 0, "usd": 0.0, "n": 0}) for a in ARMS} for g in GAMES}
    out = {
        "description": "Token-based discovery cost -- LLM token effort to FIND each solve, reconstructed "
                       "from session transcripts so it covers all 25 games (opus + Fable). Companion to the "
                       "RHAE path-economy metric, which is blind to discovery cost.",
        "per_game_tokens": {g: {a: per_game[g][a]["tok"] for a in ARMS} for g in GAMES},
        "per_game_cost_usd_est": {g: {a: round(per_game[g][a]["usd"], 2) for a in ARMS} for g in GAMES},
        "per_arm": {a: {"total_tokens": sum(per_game[g][a]["tok"] for g in GAMES),
                        "total_cost_usd_est": round(sum(per_game[g][a]["usd"] for g in GAMES), 2),
                        # per game (both arms attempted all 25); kept under this key for make_paper_assets
                        "mean_cost_usd_per_solved_game": round(sum(per_game[g][a]["usd"] for g in GAMES) / len(GAMES), 2),
                        "mean_tokens_per_game": round(sum(per_game[g][a]["tok"] for g in GAMES) / len(GAMES)),
                        "games_covered": sum(1 for g in GAMES if per_game[g][a]["tok"] > 0),
                        "sessions": sum(per_game[g][a]["n"] for g in GAMES)} for a in ARMS},
        "codex_note": "codex (gpt-5.5) ran via codex exec plaintext logs -> no token stats captured (not zero)",
        "cost_basis": "tokens reconstructed_from_messages; USD estimated_from_tokens where the result block was absent",
    }
    (ROOT / "experiments" / "results" / "e149_discovery_cost.json").write_text(json.dumps(out, indent=1, sort_keys=True))

    # ---- figure ----
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1, 2.3]})
    # A: mean tokens per solved game, per arm
    means = [sum(per_game[g][a]["tok"] for g in GAMES) / 1e6 / 25 for a in ARMS]
    est = [out["per_arm"][a]["total_cost_usd_est"] / 25 for a in ARMS]
    bars = axa.bar(range(len(ARMS)), means, color=[ARM_COLOR[a] for a in ARMS], width=0.62)
    for i, (m, e) in enumerate(zip(means, est)):
        axa.text(i, m + 1, f"{m:.0f}M\n(~${e:.0f})", ha="center", fontsize=9, fontweight="bold")
    axa.set_xticks(range(len(ARMS))); axa.set_xticklabels([a for a in ARMS])
    axa.set_ylabel("mean tokens / game (millions)")
    axa.set_title("A. Discovery cost per game\n(token effort; codex not captured)", fontsize=10, fontweight="bold")
    axa.set_ylim(0, max(means) * 1.25); axa.grid(axis="y", alpha=0.25)
    # B: per-game tokens, opus vs fable, all 25 (sorted by opus tokens desc)
    order = sorted(GAMES, key=lambda g: -per_game[g]["opus"]["tok"])
    x = np.arange(len(order)); w = 0.4
    for i, a in enumerate(ARMS):
        vals = [per_game[g][a]["tok"] / 1e6 for g in order]
        axb.bar(x + (i - 0.5) * w, vals, width=w, color=ARM_COLOR[a], label=ARM_LABEL[a])
    axb.set_xticks(x); axb.set_xticklabels(order, rotation=90, fontsize=7, family="monospace")
    axb.set_ylabel("tokens to discover (millions)")
    axb.set_title("B. Per-game discovery tokens, all 25 games (RHAE is blind to all of this)", fontsize=10, fontweight="bold")
    axb.legend(fontsize=8, frameon=False); axb.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "arc3_discovery_cost.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    for a in ARMS:
        pa = out["per_arm"][a]
        print(f"  {a}: {pa['games_covered']}/25 games, {pa['total_tokens']/1e6:.0f}M tokens, ~${pa['total_cost_usd_est']:.0f} est")
    print("wrote e149_discovery_cost.json + arc3_discovery_cost.png")


if __name__ == "__main__":
    main()
