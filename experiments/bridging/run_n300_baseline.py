"""N=300 headline baseline replication.

Replicates the original K=7 archetype-seeded experiment at N=300 personas
(Prism's specified population size for the paper's headline claim).
P2 bimodal_sym distribution, both spillover configs, 50 trials/cell.

Run from the openworld repo root:
    python -m experiments.bridging.run_n300_baseline

Outputs:
    experiments/bridging/results/n300_baseline_results.csv
    experiments/bridging/results/n300_baseline_summary.txt
"""

from __future__ import annotations

import csv
import pathlib
import random
import statistics
import time
from dataclasses import dataclass
from typing import List

from .conditions_c import condition_c_community_notes, condition_c_polarity_product
from .personas import ISSUES
from .policy import enumerate_bundles
from .run_axis1 import generate_personas_custom
from .run_axis2 import build_oracle_fast, generate_slate
from .simulation import _bundle_index, condition_a, condition_z

# ── Config ────────────────────────────────────────────────────────────────────

N_PERSONAS = 300
PERSONA_SEED = 42
K = 7
SLATE_TYPE = "archetype"
SPILLOVER_CFGS = ("centrist", "off_axis")
N_TRIALS = 50
BASE_SEED = 3000
CONDITIONS = ("Z", "A", "C_CN", "C_PP", "D")

P2_COMPONENTS = [(0.40, -0.45, 0.25), (0.40, 0.45, 0.25), (0.20, 0.00, 0.15)]

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


@dataclass
class BaselineResult:
    spillover_config: str
    condition: str
    trial: int
    gap_fraction: float
    G_achieved: float
    G_random: float
    G_oracle: float


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"N=300 Headline Baseline — K={K} archetype-seeded, P2, {N_TRIALS} trials/cell")

    print(f"Generating {N_PERSONAS} personas (P2 bimodal_sym, seed={PERSONA_SEED})...")
    personas = generate_personas_custom(P2_COMPONENTS, n=N_PERSONAS, seed=PERSONA_SEED)
    mean_ideo = statistics.mean(p.latent_ideology for p in personas)
    std_ideo = statistics.stdev(p.latent_ideology for p in personas)
    print(f"  Ideology: mean={mean_ideo:+.3f}  std={std_ideo:.3f}")

    print("Enumerating 5^8 bundles (390,625)...")
    bundles = enumerate_bundles()

    all_results: List[BaselineResult] = []

    for cfg in SPILLOVER_CFGS:
        print(f"\nBuilding oracle (N={N_PERSONAS}, '{cfg}', vectorized)...")
        t0 = time.time()
        table = build_oracle_fast(personas, cfg, bundles, cache=True)
        elapsed = time.time() - t0
        print(f"  {elapsed:.1f}s  G_oracle={table.g_oracle:.4f}  G_random={table.g_random:.4f}")

        for trial_idx in range(N_TRIALS):
            trial_seed = BASE_SEED + trial_idx
            rng = random.Random(trial_seed)
            slate = generate_slate(K, SLATE_TYPE, trial_seed)

            g_random = table.g_random
            g_oracle = table.g_oracle

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
                    all_results.append(BaselineResult(
                        spillover_config=cfg, condition="D", trial=trial_idx,
                        gap_fraction=1.0, G_achieved=g_oracle,
                        G_random=g_random, G_oracle=g_oracle,
                    ))
                    continue
                else:
                    raise ValueError(cond)

                g_achieved = table.g_values[_bundle_index(winner)]
                gap = table.gap_fraction(g_achieved)
                all_results.append(BaselineResult(
                    spillover_config=cfg, condition=cond, trial=trial_idx,
                    gap_fraction=gap, G_achieved=g_achieved,
                    G_random=g_random, G_oracle=g_oracle,
                ))

    # Summary table
    lines = [
        f"N={N_PERSONAS} Headline Baseline Summary",
        f"K={K} archetype-seeded, P2 bimodal_sym, {N_TRIALS} trials/cell",
        "",
        f"{'Condition':<10} {'centrist (med)':>14} {'off_axis (med)':>15}",
        "-" * 42,
    ]
    for cond in CONDITIONS:
        c_med = statistics.median([r.gap_fraction for r in all_results
            if r.spillover_config == "centrist" and r.condition == cond])
        o_med = statistics.median([r.gap_fraction for r in all_results
            if r.spillover_config == "off_axis" and r.condition == cond])
        lines.append(f"{cond:<10} {c_med:>14.3f} {o_med:>15.3f}")

    lines += [
        "",
        "C_CN vs A delta:",
    ]
    for cfg in SPILLOVER_CFGS:
        c_med = statistics.median([r.gap_fraction for r in all_results
            if r.spillover_config == cfg and r.condition == "C_CN"])
        a_med = statistics.median([r.gap_fraction for r in all_results
            if r.spillover_config == cfg and r.condition == "A"])
        lines.append(f"  {cfg}: C_CN={c_med:.3f}  A={a_med:.3f}  delta={c_med-a_med:+.3f}")

    summary = "\n".join(lines)
    print(f"\n{summary}")

    # Write files
    csv_path = RESULTS_DIR / "n300_baseline_results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "spillover_config", "condition", "trial",
            "gap_fraction", "G_achieved", "G_random", "G_oracle",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "spillover_config": r.spillover_config, "condition": r.condition,
                "trial": r.trial, "gap_fraction": f"{r.gap_fraction:.6f}",
                "G_achieved": f"{r.G_achieved:.6f}", "G_random": f"{r.G_random:.6f}",
                "G_oracle": f"{r.G_oracle:.6f}",
            })
    print(f"\nWrote {len(all_results)} rows → {csv_path}")

    txt_path = RESULTS_DIR / "n300_baseline_summary.txt"
    txt_path.write_text(summary + "\n")
    print(f"Wrote summary → {txt_path}")


if __name__ == "__main__":
    main()
