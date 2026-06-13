"""Axis 4a: Adversarial coalition voting.

One cluster (R-cluster: ideology > 0) adopts a strategic voting rule:
for any bundle endorsed by ≥50% of the L-cluster (ideology < 0), the
R-cluster sets its endorsement to 0 regardless of true preference.

This tests whether C_CN/C_PP are robust to coordinated strategic downvoting.
If C survives, cross-partisan endorsement signal is present even after attack.
If C degrades toward A, the matrix factorization is deceived by the attack.

Condition A (majority vote) is UNAFFECTED — it uses individual welfare, not
endorsement matrices.

Run from the openworld repo root:
    python -m experiments.bridging.run_adversarial

Outputs:
    experiments/bridging/results/adversarial_results.csv
    experiments/bridging/results/adversarial_summary.txt
"""

from __future__ import annotations

import csv
import pathlib
import random
import statistics
import time
from dataclasses import dataclass
from typing import List, Set

import numpy as np

from .conditions_c import (
    build_endorsement_matrix,
    condition_c_community_notes,
    condition_c_polarity_product,
    fit_community_notes,
)
from .personas import ISSUES, Persona
from .policy import PolicyBundle, enumerate_bundles
from .run_axis1 import generate_personas_custom
from .run_axis2 import build_oracle_fast, generate_slate
from .simulation import _bundle_index, condition_a, condition_z

# ── Config ────────────────────────────────────────────────────────────────────

N_PERSONAS = 100
PERSONA_SEED = 42
K = 7
SLATE_TYPE = "archetype"
SPILLOVER_CFGS = ("centrist", "off_axis")
N_TRIALS = 50
BASE_SEED = 4000
CONDITIONS_ADVERSARIAL = ("Z", "A", "C_CN", "C_CN_ADV", "C_PP", "C_PP_ADV", "D")

P2_COMPONENTS = [(0.40, -0.45, 0.25), (0.40, 0.45, 0.25), (0.20, 0.00, 0.15)]

# L-cluster: ideology < threshold; R-cluster: ideology > threshold
L_THRESHOLD = -0.05
R_THRESHOLD = +0.05
L_ENDORSEMENT_THRESHOLD = 0.50   # fraction of L-cluster that must endorse for attack to trigger

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


# ── Adversarial endorsement matrix ────────────────────────────────────────────

def build_adversarial_endorsement_matrix(
    personas: List[Persona],
    slate: List[PolicyBundle],
) -> np.ndarray:
    """Build endorsement matrix with R-cluster strategic downvoting.

    R-cluster personas set endorsement to 0 for any bundle endorsed by
    ≥50% of L-cluster personas (by count, not rate, to avoid edge effects).

    Returns float64 (N x M) matrix.
    """
    # Clean matrix (no attack)
    R = build_endorsement_matrix(personas, slate)

    M = len(slate)
    N = len(personas)

    l_indices = [i for i, p in enumerate(personas) if p.latent_ideology < L_THRESHOLD]
    r_indices = [i for i, p in enumerate(personas) if p.latent_ideology > R_THRESHOLD]

    if not l_indices or not r_indices:
        return R

    # Per-bundle L-cluster endorsement count
    l_endorsement_counts = R[l_indices, :].sum(axis=0)     # (M,)
    l_endorsement_rates = l_endorsement_counts / len(l_indices)

    # Bundles targeted by attack: endorsed by ≥50% of L-cluster
    attacked_bundles = np.where(l_endorsement_rates >= L_ENDORSEMENT_THRESHOLD)[0]

    if len(attacked_bundles) == 0:
        return R

    # R-cluster zeroes out endorsement for attacked bundles
    R_adv = R.copy()
    for j in attacked_bundles:
        R_adv[r_indices, j] = 0.0

    return R_adv


def condition_c_cn_adversarial(
    slate: List[PolicyBundle],
    personas: List[Persona],
) -> PolicyBundle:
    """C_CN under adversarial R-cluster downvoting."""
    R_adv = build_adversarial_endorsement_matrix(personas, slate)
    i_n = fit_community_notes(R_adv)
    best_idx = int(np.argmax(i_n))
    return slate[best_idx]


def condition_c_pp_adversarial(
    slate: List[PolicyBundle],
    personas: List[Persona],
) -> PolicyBundle:
    """C_PP (polarity-product) under adversarial R-cluster downvoting.

    Uses the adversarial endorsement matrix to compute per-community
    endorsement rates. R-cluster endorsement of L-supported bundles is 0.
    """
    R_adv = build_adversarial_endorsement_matrix(personas, slate)
    M = len(slate)
    N = len(personas)

    communities = sorted(set(p.network_community for p in personas))
    if len(communities) < 2:
        # Fallback: use ideology split as two communities
        community_map = {
            p.persona_id: (0 if p.latent_ideology < 0 else 1) for p in personas
        }
    else:
        community_map = {p.persona_id: p.network_community for p in personas}

    # Build community membership arrays
    community_rows: dict = {}
    for i, p in enumerate(personas):
        c = community_map[p.persona_id]
        community_rows.setdefault(c, []).append(i)

    scores = np.ones(M, dtype=np.float64)
    for c, rows in community_rows.items():
        if not rows:
            continue
        rates = R_adv[rows, :].mean(axis=0)   # (M,)
        clipped = np.clip(rates, 1e-8, 1.0)
        scores *= clipped

    best_idx = int(np.argmax(scores))
    return slate[best_idx]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AdversarialResult:
    spillover_config: str
    condition: str
    trial: int
    gap_fraction: float
    G_achieved: float
    G_random: float
    G_oracle: float
    n_attacked_bundles: int   # how many bundles were attacked in this trial


def run_trial_adversarial(
    trial_idx: int,
    personas: List[Persona],
    table,
    spillover_config: str,
) -> List[AdversarialResult]:
    trial_seed = BASE_SEED + trial_idx
    rng = random.Random(trial_seed)
    slate = generate_slate(K, SLATE_TYPE, trial_seed)

    g_random = table.g_random
    g_oracle = table.g_oracle

    # Count attacked bundles for this slate
    R = build_endorsement_matrix(personas, slate)
    l_indices = [i for i, p in enumerate(personas) if p.latent_ideology < L_THRESHOLD]
    if l_indices:
        l_rates = R[l_indices, :].mean(axis=0)
        n_attacked = int((l_rates >= L_ENDORSEMENT_THRESHOLD).sum())
    else:
        n_attacked = 0

    results = []
    for cond in CONDITIONS_ADVERSARIAL:
        if cond == "Z":
            winner = condition_z(slate, rng)
        elif cond == "A":
            winner = condition_a(slate, personas)
        elif cond == "C_CN":
            winner = condition_c_community_notes(slate, personas)
        elif cond == "C_CN_ADV":
            winner = condition_c_cn_adversarial(slate, personas)
        elif cond == "C_PP":
            winner = condition_c_polarity_product(slate, personas)
        elif cond == "C_PP_ADV":
            winner = condition_c_pp_adversarial(slate, personas)
        elif cond == "D":
            results.append(AdversarialResult(
                spillover_config=spillover_config, condition="D", trial=trial_idx,
                gap_fraction=1.0, G_achieved=g_oracle,
                G_random=g_random, G_oracle=g_oracle,
                n_attacked_bundles=n_attacked,
            ))
            continue
        else:
            raise ValueError(cond)

        g_achieved = table.g_values[_bundle_index(winner)]
        gap = table.gap_fraction(g_achieved)
        results.append(AdversarialResult(
            spillover_config=spillover_config, condition=cond, trial=trial_idx,
            gap_fraction=gap, G_achieved=g_achieved,
            G_random=g_random, G_oracle=g_oracle,
            n_attacked_bundles=n_attacked,
        ))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Axis 4a: Adversarial coalition — N={N_PERSONAS}, K={K} archetype-seeded, "
          f"P2, {N_TRIALS} trials/cell")
    print(f"R-cluster (ideology>{R_THRESHOLD}) downvotes bundles endorsed by "
          f"≥{L_ENDORSEMENT_THRESHOLD*100:.0f}% of L-cluster (ideology<{L_THRESHOLD})")

    personas = generate_personas_custom(P2_COMPONENTS, n=N_PERSONAS, seed=PERSONA_SEED)
    l_count = sum(1 for p in personas if p.latent_ideology < L_THRESHOLD)
    r_count = sum(1 for p in personas if p.latent_ideology > R_THRESHOLD)
    print(f"  L-cluster: {l_count} personas  R-cluster: {r_count} personas  "
          f"centrist: {N_PERSONAS - l_count - r_count} personas")

    print("Enumerating 5^8 bundles (390,625)...")
    bundles = enumerate_bundles()

    all_results: List[AdversarialResult] = []

    for cfg in SPILLOVER_CFGS:
        print(f"\n{'='*60}")
        print(f"Spillover config: {cfg}")

        t0 = time.time()
        table = build_oracle_fast(personas, cfg, bundles, cache=True)
        elapsed = time.time() - t0
        print(f"  Oracle: {elapsed:.1f}s  G_oracle={table.g_oracle:.4f}  "
              f"G_random={table.g_random:.4f}")

        cell_results = []
        for trial_idx in range(N_TRIALS):
            res = run_trial_adversarial(trial_idx, personas, table, cfg)
            cell_results.extend(res)
        all_results.extend(cell_results)

        # Cell summary
        for cond in ("A", "C_CN", "C_CN_ADV", "C_PP", "C_PP_ADV"):
            vals = [r.gap_fraction for r in cell_results if r.condition == cond]
            med = statistics.median(vals)
            print(f"  {cond:<12}: med gap={med:.3f}")

        avg_attacked = statistics.mean(
            r.n_attacked_bundles for r in cell_results if r.condition == "A"
        )
        print(f"  Avg attacked bundles/trial: {avg_attacked:.1f}/{K}")

    # Summary table
    lines = [
        f"Adversarial Coalition Summary (N={N_PERSONAS}, K={K} archetype-seeded, P2)",
        f"L-cluster: ideology<{L_THRESHOLD}  R-cluster: ideology>{R_THRESHOLD}",
        f"Attack: R zeroes endorsement for bundles with L-rate ≥ {L_ENDORSEMENT_THRESHOLD*100:.0f}%",
        "",
        f"{'Condition':<14} {'centrist (med)':>14} {'off_axis (med)':>15}",
        "-" * 46,
    ]
    for cond in ("Z", "A", "C_CN", "C_CN_ADV", "C_PP", "C_PP_ADV", "D"):
        c_med = statistics.median([r.gap_fraction for r in all_results
            if r.spillover_config == "centrist" and r.condition == cond])
        o_med = statistics.median([r.gap_fraction for r in all_results
            if r.spillover_config == "off_axis" and r.condition == cond])
        lines.append(f"{cond:<14} {c_med:>14.3f} {o_med:>15.3f}")

    lines += ["", "Adversarial degradation (C_CN vs C_CN_ADV):"]
    for cfg in SPILLOVER_CFGS:
        clean = statistics.median([r.gap_fraction for r in all_results
            if r.spillover_config == cfg and r.condition == "C_CN"])
        adv = statistics.median([r.gap_fraction for r in all_results
            if r.spillover_config == cfg and r.condition == "C_CN_ADV"])
        lines.append(f"  {cfg}: clean={clean:.3f}  adversarial={adv:.3f}  "
                     f"degradation={clean-adv:+.3f}")

    summary = "\n".join(lines)
    print(f"\n{summary}")

    # Write CSV
    csv_path = RESULTS_DIR / "adversarial_results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "spillover_config", "condition", "trial",
            "gap_fraction", "G_achieved", "G_random", "G_oracle",
            "n_attacked_bundles",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "spillover_config": r.spillover_config, "condition": r.condition,
                "trial": r.trial, "gap_fraction": f"{r.gap_fraction:.6f}",
                "G_achieved": f"{r.G_achieved:.6f}", "G_random": f"{r.G_random:.6f}",
                "G_oracle": f"{r.G_oracle:.6f}",
                "n_attacked_bundles": r.n_attacked_bundles,
            })
    print(f"\nWrote {len(all_results)} rows → {csv_path}")

    txt_path = RESULTS_DIR / "adversarial_summary.txt"
    txt_path.write_text(summary + "\n")
    print(f"Wrote summary → {txt_path}")


if __name__ == "__main__":
    main()
