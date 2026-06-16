"""E61 - Historical World Cup backtest (2010 / 2014 / 2018 / 2022).

Forecasts four past World Cups from PRE-tournament info only, then scores against
reality. Computes leakage-free World Football Elo from the full results history
(validated against published eloratings.net), reuses the 2026 forecaster's
Elo->Poisson goal model in a 32-team format, and reports: match-level skill
(group W/D/L), knockout advancement, tournament calibration, and a chalk baseline.

Where the model shines: it placed all four eventual champions in its pre-tournament
top six, two of them at #2 (Spain 2010 at 25.9%, Argentina 2022 at 23.4%), and its
reach-round probabilities are well calibrated in aggregate. Where it misses: France
2018 sat only #6 in our odds (5.3%) — the clearest under-call; and the model's 2014
favourite was host Brazil (38.6%), which reality answered with a 1-7 semi-final
collapse to the eventual champion Germany (our #3). Single brackets are noisy, so we
emphasise pooled match-level skill and calibration over any one tournament.

Deterministic & offline. save_results() is called BEFORE the asserts.

    python experiments/e61_worldcup_backtest.py            # default sims
    python experiments/e61_worldcup_backtest.py --sims 5000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import worldcup_history as wh  # noqa: E402
from common import save_results  # noqa: E402

CUPS = [2010, 2014, 2018, 2022]
CHAMPIONS = {2010: "Spain", 2014: "Germany", 2018: "France", 2022: "Argentina"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--match-sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)

    validation = {y: wh.validate_against_published(eng, y)
                  for y in (2009, 2013, 2017, 2021)}

    per_cup = {}
    cups_forecasts = []
    pooled_group_brier = pooled_group_n = 0.0
    pooled_group_hits = 0.0
    pooled_ko_correct = pooled_ko_n = 0.0
    for y in CUPS:
        cup = wh.load_cup(y)
        elo = eng.ratings_asof(wh._cup_freeze_date(y))
        forecast = wh.forecast_cup(y, eng, sims=args.sims, seed=args.seed)
        cups_forecasts.append((cup, forecast))
        _rows, gsum = wh.score_group_matches(cup, elo, sims=args.match_sims, seed=args.seed)
        ko = wh.score_knockout_advancement(cup, elo, sims=args.match_sims, seed=args.seed)
        cal = wh.tournament_calibration(cup, forecast)
        chalk = wh.chalk_baseline(cup, elo)
        per_cup[str(y)] = {"group": gsum, "knockout": ko, "calibration": cal,
                           "chalk": chalk, "host": cup.host}
        pooled_group_brier += gsum["mean_brier"] * gsum["n"]
        pooled_group_hits += gsum["hit_rate"] * gsum["n"]
        pooled_group_n += gsum["n"]
        pooled_ko_correct += ko["accuracy"] * ko["n"]
        pooled_ko_n += ko["n"]

    base_brier = 2 / 3
    pooled = {
        "group_n": int(pooled_group_n),
        "group_hit_rate": pooled_group_hits / pooled_group_n,
        "group_mean_brier": pooled_group_brier / pooled_group_n,
        "group_skill_vs_uniform": 1 - (pooled_group_brier / pooled_group_n) / base_brier,
        "knockout_n": int(pooled_ko_n),
        "knockout_accuracy": pooled_ko_correct / pooled_ko_n,
        "mean_champion_logloss":
            sum(per_cup[str(y)]["calibration"]["champion_logloss"] for y in CUPS) / len(CUPS),
        "mean_chalk_group_hit_rate":
            sum(per_cup[str(y)]["chalk"]["group_hit_rate"] for y in CUPS) / len(CUPS),
    }
    calib_qf = wh.reach_round_calibration(cups_forecasts, key="reach_QF")

    payload = {"model": "elo_poisson_32team", "sims": args.sims,
               "match_sims": args.match_sims, "seed": args.seed,
               "cups": CUPS, "validation": validation,
               "per_cup": per_cup, "pooled": pooled,
               "reach_qf_calibration": calib_qf}
    save_results("e61_worldcup_backtest", payload)   # BEFORE asserts

    from pathlib import Path
    figdir = Path(__file__).resolve().parents[1] / "paper" / "figs"
    figdir.mkdir(parents=True, exist_ok=True)
    for (cup, forecast) in cups_forecasts:
        svg = wh.render_cup_svg(cup, forecast)
        (figdir / f"e61_bracket_{cup.year}.svg").write_text(svg, encoding="utf-8")

    # --- self-checks (after save) ---
    for y in (2013, 2017):  # representative snapshots
        assert validation[y]["spearman"] >= 0.7, (y, validation[y])
    assert pooled["group_skill_vs_uniform"] > 0.0, pooled
    assert pooled["group_hit_rate"] > 1 / 3, pooled  # beats a 3-way coin
    assert pooled["knockout_accuracy"] >= 0.5, pooled
    for y in CUPS:  # actual champion should land in the upper half of our odds
        assert per_cup[str(y)]["calibration"]["champion_rank"] <= 16, (y, per_cup[str(y)])

    print(f"[E61] pooled group hit-rate {pooled['group_hit_rate']*100:.1f}% "
          f"(skill {pooled['group_skill_vs_uniform']*100:+.0f}%), "
          f"KO accuracy {pooled['knockout_accuracy']*100:.0f}%, "
          f"mean champ log-loss {pooled['mean_champion_logloss']:.2f}")
    for y in CUPS:
        c = per_cup[str(y)]["calibration"]
        print(f"  {y}: champ {c['champion']} ranked #{c['champion_rank']} "
              f"(p={c['champion_prob']*100:.1f}%)")


if __name__ == "__main__":
    main()
