"""Pilot runner: 4 conditions × 10 games (no counter-alliance), default personas.

Outputs results/catan_pilot.csv with one row per game.
Gating checks (per design doc §6 pilot phase):
  - Games end within 80 turns
  - Alliance win rate in condition (a) between 20% and 80%
  - LLM token cost N/A (condition d uses stub)
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List

from .adversarial import make_adversarial_runner
from .conditions import condition_b_runner, run_turn_condition_c, run_turn_condition_d
from .game import run_game, run_turn_greedy
from .personas import ALLIANCE_PLAYERS, DEFAULT_PERSONAS

RESULTS_DIR = Path(__file__).parent / "results"
PILOT_CSV = RESULTS_DIR / "catan_pilot.csv"
PILOT_GAMES_PER_CONDITION = 10
CONDITIONS = {
    "a": lambda: run_turn_greedy,
    "b": condition_b_runner,
    "c": lambda: run_turn_condition_c,
    "d": lambda: (lambda s, p, pers, rng: run_turn_condition_d(s, p, pers, rng, llm_call=None)),
}


def run_pilot(n_games: int = PILOT_GAMES_PER_CONDITION, adversarial: bool = False) -> List[dict]:
    """Run the pilot sweep.  Returns list of result dicts (also written to CSV)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []

    for cond_name, runner_factory in CONDITIONS.items():
        for seed in range(n_games):
            rng = random.Random(seed + hash(cond_name) % 1000)
            runner = runner_factory()
            if adversarial:
                runner = make_adversarial_runner(runner)

            state = run_game(DEFAULT_PERSONAS, rng, turn_runner=runner)
            alliance_win = state.winner in ALLIANCE_PLAYERS

            rows.append({
                "condition": cond_name,
                "persona_config": "default",
                "adversarial": adversarial,
                "seed": seed,
                "winner": state.winner,
                "alliance_win": alliance_win,
                "turns": state.turn,
                "vp_p1": state.vp["P1"],
                "vp_p2": state.vp["P2"],
                "vp_p3": state.vp["P3"],
                "vp_p4": state.vp["P4"],
            })

    fieldnames = ["condition", "persona_config", "adversarial", "seed", "winner",
                  "alliance_win", "turns", "vp_p1", "vp_p2", "vp_p3", "vp_p4"]
    with open(PILOT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def _check_gating(rows: List[dict]) -> None:
    """Print pilot gating results per design doc §6."""
    for cond in CONDITIONS:
        cond_rows = [r for r in rows if r["condition"] == cond]
        turns = [r["turns"] for r in cond_rows]
        wins = [r for r in cond_rows if r["alliance_win"]]
        win_rate = len(wins) / len(cond_rows) if cond_rows else 0
        max_turns = max(turns) if turns else 0
        print(f"Condition {cond}: alliance_win_rate={win_rate:.2f}  max_turns={max_turns}  "
              f"gate_turns={'PASS' if max_turns <= 80 else 'FAIL'}")
    cond_a_rows = [r for r in rows if r["condition"] == "a"]
    wr_a = sum(1 for r in cond_a_rows if r["alliance_win"]) / len(cond_a_rows)
    print(f"Condition (a) win rate gate (0.2–0.8): {'PASS' if 0.2 <= wr_a <= 0.8 else 'FAIL (adjust P3/P4 strength)'}")


if __name__ == "__main__":
    print("Running pilot (40 games)...")
    rows = run_pilot()
    _check_gating(rows)
    print(f"Results written to {PILOT_CSV}")
