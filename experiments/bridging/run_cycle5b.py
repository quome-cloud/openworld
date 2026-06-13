"""Cycle 5b sensitivity check: fully-random slate (no archetype seeds).

Tests whether the bridging advantage over majority vote is robust to slate
composition, or is conditional on archetype availability.

Run from the openworld repo root:
    python -m experiments.bridging.run_cycle5b

Key difference from run_cycle5.py:
  - Slate is K=7 fully-random bundles per trial (no fixed archetypes).
  - The oracle bundle is almost never in the slate; gap_fraction denominators
    still use table.g_oracle (same reference point).

Outputs:
    experiments/bridging/results/cycle5b_results.csv
    experiments/bridging/results/cycle5b_summary.txt
"""

from __future__ import annotations

import csv
import pathlib
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .conditions_c import condition_c_community_notes, condition_c_polarity_product
from .personas import ISSUES, generate_personas
from .policy import STANCES, PayoffTable, PolicyBundle, build_payoff_table, enumerate_bundles
from .simulation import (
    _bundle_index,
    condition_a,
    condition_z,
)

# ── Config ────────────────────────────────────────────────────────────────────

N_PERSONAS = 20
PERSONA_SEED = 0
N_TRIALS = 50
BASE_SEED = 0
K_SLATE = 7
CONDITIONS = ("Z", "A", "C_CN", "C_PP", "D")
SPILLOVER_CONFIGS = ("centrist", "off_axis")

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


# ── Fully-random slate ────────────────────────────────────────────────────────

def _random_slate(trial_seed: int) -> List[PolicyBundle]:
    """K=7 bundles drawn uniformly at random — no archetypes."""
    rng = random.Random(trial_seed)
    return [
        PolicyBundle(stances={issue: rng.choice(STANCES) for issue in ISSUES})
        for _ in range(K_SLATE)
    ]


# ── Trial runner (inline — avoids modifying simulation.py) ───────────────────

@dataclass
class SensitivityResult:
    spillover_config: str
    condition: str
    trial: int
    gap_fraction: float
    G_achieved: float
    G_random: float
    G_oracle: float


def _run_trial(
    trial_idx: int,
    personas,
    table: PayoffTable,
    spillover_config: str,
) -> List[SensitivityResult]:
    trial_seed = BASE_SEED + trial_idx
    rng = random.Random(trial_seed)
    slate = _random_slate(trial_seed)

    g_random = table.g_random
    g_oracle = table.g_oracle

    results = []
    for cond in CONDITIONS:
        if cond == "Z":
            winner = condition_z(slate, rng)
        elif cond == "A":
            winner = condition_a(slate, personas)
        elif cond == "C_CN":
            winner = condition_c_community_notes(slate, personas)
        elif cond == "C_PP":
            winner = condition_c_polarity_product(slate, personas)
        elif cond == "D":
            # Oracle always wins — note it may not be in the slate
            # gap_fraction = 1.0 by construction
            results.append(SensitivityResult(
                spillover_config=spillover_config,
                condition="D",
                trial=trial_idx,
                gap_fraction=1.0,
                G_achieved=g_oracle,
                G_random=g_random,
                G_oracle=g_oracle,
            ))
            continue
        else:
            raise ValueError(cond)

        g_achieved = table.g_values[_bundle_index(winner)]
        gap = (g_achieved - g_random) / (g_oracle - g_random) if g_oracle != g_random else 0.0
        results.append(SensitivityResult(
            spillover_config=spillover_config,
            condition=cond,
            trial=trial_idx,
            gap_fraction=gap,
            G_achieved=g_achieved,
            G_random=g_random,
            G_oracle=g_oracle,
        ))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Sensitivity check: K={K_SLATE} fully-random slates, {N_TRIALS} trials")
    print(f"Generating {N_PERSONAS} personas (seed={PERSONA_SEED})...")
    personas = generate_personas(n=N_PERSONAS, seed=PERSONA_SEED)

    print("Enumerating 5^8 bundles...")
    bundles = enumerate_bundles()

    all_results: List[SensitivityResult] = []

    for cfg in SPILLOVER_CONFIGS:
        print(f"\nLoading oracle table for '{cfg}' (cache)...")
        table = build_payoff_table(personas, cfg, bundles=bundles, cache=True)
        print(f"  G_oracle={table.g_oracle:.4f}  G_random={table.g_random:.4f}")

        for trial_idx in range(N_TRIALS):
            results = _run_trial(trial_idx, personas, table, cfg)
            all_results.extend(results)

        if trial_idx % 10 == 9 or trial_idx == N_TRIALS - 1:
            last = {r.condition: r.gap_fraction for r in results}
            print(f"  trial {N_TRIALS}/{N_TRIALS} "
                  + "  ".join(f"{c}={last[c]:.3f}" for c in CONDITIONS))

    # Write CSV
    csv_path = RESULTS_DIR / "cycle5b_results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "spillover_config", "condition", "trial", "gap_fraction",
            "G_achieved", "G_random", "G_oracle",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "spillover_config": r.spillover_config,
                "condition": r.condition,
                "trial": r.trial,
                "gap_fraction": f"{r.gap_fraction:.6f}",
                "G_achieved": f"{r.G_achieved:.6f}",
                "G_random": f"{r.G_random:.6f}",
                "G_oracle": f"{r.G_oracle:.6f}",
            })
    print(f"\nWrote {len(all_results)} rows → {csv_path}")

    # Summary table
    lines = []
    lines.append("Cycle 5b: Fully-Random Slate Sensitivity Check")
    lines.append(f"K={K_SLATE} random bundles/trial, N={N_PERSONAS} personas, {N_TRIALS} trials")
    lines.append("")
    lines.append(f"{'Condition':<10}" + "".join(f"  {'med':>6} {'mean':>6} {'IQR':>14}  " for _ in SPILLOVER_CONFIGS))
    lines.append(f"{'':10}" + "".join(f"  {c:<28}" for c in SPILLOVER_CONFIGS))
    lines.append("-" * 80)

    for cond in CONDITIONS:
        row = f"{cond:<10}"
        for cfg in SPILLOVER_CONFIGS:
            vals = [r.gap_fraction for r in all_results
                    if r.spillover_config == cfg and r.condition == cond]
            s = sorted(vals)
            n = len(s)
            med = statistics.median(s)
            mean = statistics.mean(s)
            q1 = statistics.median(s[: n // 2])
            q3 = statistics.median(s[(n + 1) // 2 :])
            row += f"  med={med:.3f} mean={mean:.3f} IQR=[{q1:.3f},{q3:.3f}]  "
        lines.append(row)

    lines.append("")

    # C vs A comparison
    lines.append("C_CN vs A delta (C_CN_med - A_med):")
    for cfg in SPILLOVER_CONFIGS:
        a_med = statistics.median([r.gap_fraction for r in all_results
                                   if r.spillover_config == cfg and r.condition == "A"])
        c_med = statistics.median([r.gap_fraction for r in all_results
                                   if r.spillover_config == cfg and r.condition == "C_CN"])
        verdict = "C>>A (robust)" if c_med - a_med > 0.05 else "C≈A (fragile)"
        lines.append(f"  {cfg}: C_CN={c_med:.3f}  A={a_med:.3f}  delta={c_med-a_med:+.3f}  → {verdict}")

    summary = "\n".join(lines)
    print("\n" + summary)

    txt_path = RESULTS_DIR / "cycle5b_summary.txt"
    txt_path.write_text(summary + "\n")
    print(f"\nWrote summary → {txt_path}")


if __name__ == "__main__":
    main()
