"""E62 - Benchmark the World Cup forecaster against other models.

Scores six match-level W/D/L forecasters (uniform, Elo-logistic/Davidson, our
Elo->Poisson frozen, our walk-forward, Maher Poisson team-strength, and 538 SPI)
with RPS/Brier on 2010-2022 (538 head-to-head on 2018/2022). The honest question is
where our model ranks relative to the field. save_results() precedes the asserts.

    python experiments/e62_worldcup_benchmark.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import worldcup_benchmark as wb  # noqa: E402
from common import save_results  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    res = wb.run_benchmark(sims=args.sims, seed=args.seed)
    save_results("e62_worldcup_benchmark", res)   # BEFORE asserts

    pm = res["per_model"]
    u = pm["uniform"]["pooled"]["rps"]
    for name in ("elo_logistic", "ours_frozen", "ours_walk_forward", "maher"):
        assert pm[name]["pooled"]["rps"] < u, (name, pm[name]["pooled"]["rps"], u)
    assert res["head_to_head_538"]["n"] == 128
    h2h = res["head_to_head_538"]["per_model"]
    assert h2h["five_thirty_eight"]["rps"] < u  # 538 also beats the floor
    assert h2h["ours_walk_forward"]["n"] == 128

    print("[E62] pooled RPS ranking (lower is better):")
    for name, r in res["ranking"]:
        print(f"  {name:<20} {r:.4f}")
    print("\n538 head-to-head (2018+2022, 128 matches):")
    for name in ("five_thirty_eight", "ours_walk_forward", "ours_frozen",
                 "maher", "elo_logistic", "uniform"):
        s = h2h[name]
        print(f"  {name:<20} RPS {s['rps']:.4f}  Brier {s['brier']:.3f}  "
              f"hit {s['hit_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
