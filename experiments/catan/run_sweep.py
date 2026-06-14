"""Full sweep: 4 conditions × 2 adversarial modes × 3 persona configs × 30 games = 720 games.

Outputs:
  results/catan_sweep.csv   — one row per game
  results/catan_win_rate.svg — bar chart of alliance win rate by condition × adversarial
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List

from .adversarial import make_adversarial_runner
from .conditions import condition_b_runner, run_turn_condition_c, run_turn_condition_d
from .game import run_game, run_turn_greedy
from .personas import ALLIANCE_PLAYERS, PERSONA_CONFIGS

RESULTS_DIR = Path(__file__).parent / "results"
SWEEP_CSV = RESULTS_DIR / "catan_sweep.csv"
SVG_PATH = RESULTS_DIR / "catan_win_rate.svg"

GAMES_PER_CELL = 30
CONDITIONS = {
    "a": lambda: run_turn_greedy,
    "b": condition_b_runner,
    "c": lambda: run_turn_condition_c,
    "d": lambda: (lambda s, p, pers, rng: run_turn_condition_d(s, p, pers, rng, llm_call=None)),
}
ADVERSARIAL_MODES = [False, True]


def run_sweep(games_per_cell: int = GAMES_PER_CELL) -> List[dict]:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    cell = 0
    total = len(CONDITIONS) * len(ADVERSARIAL_MODES) * len(PERSONA_CONFIGS) * games_per_cell

    for cond_name, runner_factory in CONDITIONS.items():
        for adv in ADVERSARIAL_MODES:
            for persona_name, personas in PERSONA_CONFIGS.items():
                for seed in range(games_per_cell):
                    rng = random.Random(seed + cell * 1000)
                    runner = runner_factory()
                    if adv:
                        runner = make_adversarial_runner(runner)

                    state = run_game(personas, rng, turn_runner=runner)
                    alliance_win = state.winner in ALLIANCE_PLAYERS

                    rows.append({
                        "condition": cond_name,
                        "persona_config": persona_name,
                        "adversarial": adv,
                        "seed": seed,
                        "winner": state.winner,
                        "alliance_win": alliance_win,
                        "turns": state.turn,
                        "vp_p1": state.vp["P1"],
                        "vp_p2": state.vp["P2"],
                        "vp_p3": state.vp["P3"],
                        "vp_p4": state.vp["P4"],
                    })
                cell += 1

    fieldnames = ["condition", "persona_config", "adversarial", "seed", "winner",
                  "alliance_win", "turns", "vp_p1", "vp_p2", "vp_p3", "vp_p4"]
    with open(SWEEP_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def _win_rate(rows, cond, adv):
    subset = [r for r in rows if r["condition"] == cond and r["adversarial"] == adv]
    if not subset:
        return 0.0
    return sum(1 for r in subset if r["alliance_win"]) / len(subset)


def generate_svg(rows: List[dict]) -> str:
    conds = list(CONDITIONS.keys())
    bar_w, gap, grp_gap = 40, 6, 20
    x0, y0, chart_h = 80, 20, 200
    colors = {"no_adv": "#4a90d9", "adv": "#e07b54"}
    n_groups = len(conds)
    group_w = 2 * bar_w + gap + grp_gap
    total_w = x0 + n_groups * group_w + 40

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{y0 + chart_h + 80}">',
        '<style>text{font-family:monospace;font-size:11px}</style>',
        f'<rect width="{total_w}" height="{y0+chart_h+80}" fill="#0a0a0a"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+chart_h}" stroke="#444" stroke-width="1"/>',
        f'<line x1="{x0}" y1="{y0+chart_h}" x2="{total_w-20}" y2="{y0+chart_h}" stroke="#444" stroke-width="1"/>',
    ]
    # Y-axis labels
    for pct in [0, 25, 50, 75, 100]:
        y = y0 + chart_h - int(pct / 100 * chart_h)
        lines.append(f'<text x="{x0-5}" y="{y+4}" fill="#888" text-anchor="end">{pct}%</text>')
        lines.append(f'<line x1="{x0}" y1="{y}" x2="{total_w-20}" y2="{y}" stroke="#222" stroke-width="1"/>')

    for i, cond in enumerate(conds):
        gx = x0 + i * group_w + grp_gap // 2
        for j, (adv, color_key) in enumerate([(False, "no_adv"), (True, "adv")]):
            wr = _win_rate(rows, cond, adv)
            bh = int(wr * chart_h)
            bx = gx + j * (bar_w + gap)
            by = y0 + chart_h - bh
            color = colors[color_key]
            lines.append(f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" fill="{color}" opacity="0.85"/>')
            lines.append(f'<text x="{bx+bar_w//2}" y="{by-4}" fill="#ccc" text-anchor="middle">{wr:.0%}</text>')
        # Condition label
        lx = gx + bar_w + gap // 2
        lines.append(f'<text x="{lx}" y="{y0+chart_h+18}" fill="#aaa" text-anchor="middle">({cond})</text>')

    # Legend
    lx = x0
    ly = y0 + chart_h + 38
    lines += [
        f'<rect x="{lx}" y="{ly}" width="14" height="10" fill="{colors["no_adv"]}"/>',
        f'<text x="{lx+18}" y="{ly+9}" fill="#aaa">No counter-alliance</text>',
        f'<rect x="{lx+160}" y="{ly}" width="14" height="10" fill="{colors["adv"]}"/>',
        f'<text x="{lx+178}" y="{ly+9}" fill="#aaa">Counter-alliance</text>',
        f'<text x="{total_w//2}" y="{ly+28}" fill="#666" text-anchor="middle">'
        f'Alliance (P1+P2) win rate by coordination condition</text>',
    ]
    lines.append('</svg>')
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Running full sweep ({GAMES_PER_CELL * len(CONDITIONS) * 2 * len(PERSONA_CONFIGS)} games)...")
    rows = run_sweep()
    print("Win rates (no adversarial):", {c: f"{_win_rate(rows,c,False):.0%}" for c in CONDITIONS})
    print("Win rates (adversarial):   ", {c: f"{_win_rate(rows,c,True):.0%}" for c in CONDITIONS})
    svg = generate_svg(rows)
    SVG_PATH.write_text(svg)
    print(f"CSV → {SWEEP_CSV}")
    print(f"SVG → {SVG_PATH}")
