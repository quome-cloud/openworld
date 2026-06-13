"""World 1 simulation runner: candidate slate, conditions Z / A / D, trial CSV output.

Conditions implemented here:
  Z — Random selection        (utility floor anchor)
  A — Majority vote           (plurality over candidate slate)
  D — Oracle                  (precomputed argmax G — the ceiling)

Condition C (bridging-ranked) is in conditions_c.py; Condition B (deliberative
democracy via LLM) is deferred to a follow-up PR per T344 scope.

Output: trial-level CSV with one row per (condition, spillover_config, trial).
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .personas import ISSUES, Persona, generate_personas
from .policy import (
    STANCES,
    PayoffTable,
    PolicyBundle,
    build_payoff_table,
    enumerate_bundles,
)

# ── Candidate slate ───────────────────────────────────────────────────────────

_ARCHETYPE_VALUES = (-2, -1, 0, 1, 2)


def _archetype_bundle(stance: int) -> PolicyBundle:
    """All-issues bundle at a single stance level."""
    return PolicyBundle(stances={issue: stance for issue in ISSUES})


def generate_candidate_slate(
    personas: List[Persona],
    trial_seed: int,
    n_random: int = 2,
) -> List[PolicyBundle]:
    """Generate a K=7 candidate slate for one trial.

    Slate composition:
      - 5 archetype bundles: all-issues at each of {-2, -1, 0, +1, +2}
      - n_random random bundles seeded by trial (to sample the space)

    The archetype bundles ensure all conditions see polarised + centrist options.
    Random per-trial bundles add variance and occasionally capture spillovers,
    giving bridging algorithms a chance to distinguish themselves from majority vote.

    The same slate is used for ALL conditions in a given trial, so algorithm
    comparisons are fair.
    """
    rng = random.Random(trial_seed)
    slate: List[PolicyBundle] = [_archetype_bundle(s) for s in _ARCHETYPE_VALUES]
    for _ in range(n_random):
        stances = {issue: rng.choice(STANCES) for issue in ISSUES}
        slate.append(PolicyBundle(stances=stances))
    return slate


# ── Conditions ────────────────────────────────────────────────────────────────

def condition_z(slate: List[PolicyBundle], rng: random.Random) -> PolicyBundle:
    """Condition Z: uniformly random bundle from the candidate slate."""
    return rng.choice(slate)


def condition_a(slate: List[PolicyBundle], personas: List[Persona]) -> PolicyBundle:
    """Condition A: majority vote (plurality, binary endorse).

    Each persona votes for the slate bundle maximising their individual welfare.
    The bundle with the most votes wins.
    Ties broken lexicographically on bundle tuple (deterministic across seeds).
    """
    votes: Dict[PolicyBundle, int] = {b: 0 for b in slate}
    for p in personas:
        best = max(slate, key=lambda b: p.welfare(b.stances))
        votes[best] += 1
    max_votes = max(votes.values())
    tied = [b for b, v in votes.items() if v == max_votes]
    # Lexicographic tie-break ensures reproducibility regardless of slate order.
    return min(tied, key=lambda b: b.to_tuple())


def condition_d(table: PayoffTable) -> PolicyBundle:
    """Condition D: oracle — the precomputed G-maximising bundle."""
    return table.oracle


# ── Trial result ──────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    """One row of the output CSV (one condition × one trial)."""

    world: str
    condition: str              # "Z", "A", "C", "D"
    spillover_config: str       # "centrist" or "off_axis"
    trial: int
    G_achieved: float
    G_random: float             # G(random bundle) for this trial's Z run
    G_oracle: float             # G(oracle) from PayoffTable
    gap_fraction: float         # (G_achieved - G_random) / (G_oracle - G_random)
    cost_usd: float = 0.0       # LLM cost (Condition B only)
    cost_flag: bool = False     # True if cost_usd > $0.05 (Condition B only)


# ── Trial runner ──────────────────────────────────────────────────────────────

def run_trial(
    trial_idx: int,
    personas: List[Persona],
    table: PayoffTable,
    spillover_config: str,
    conditions: Sequence[str] = ("Z", "A", "D"),
    n_random_slate: int = 2,
    base_seed: int = 0,
) -> List[TrialResult]:
    """Run one trial: all requested conditions on the same persona population.

    Each trial gets a unique seed (base_seed + trial_idx) so the random slate
    and Condition Z draw vary across trials while remaining reproducible.

    Args:
        trial_idx:        Trial index (0-based).
        personas:         Shared persona population for this trial.
        table:            Precomputed PayoffTable for this (personas, spillover_config).
        spillover_config: "centrist" or "off_axis".
        conditions:       Which conditions to run (subset of Z, A, D).
        n_random_slate:   Number of random bundles added to the archetype slate.
        base_seed:        Base random seed; trial seed = base_seed + trial_idx.

    Returns:
        List of TrialResult, one per condition.
    """
    trial_seed = base_seed + trial_idx
    rng = random.Random(trial_seed)

    slate = generate_candidate_slate(personas, trial_seed, n_random=n_random_slate)

    # Condition Z provides the per-trial G_random reference point.
    # For the normalised gap_fraction we use the PayoffTable's g_random (mean G
    # over all 390k bundles) rather than a single Z draw, since the latter is
    # noisy and the spec says gap_fraction = (X - G_random) / (G_oracle - G_random).
    g_random = table.g_random
    g_oracle = table.g_oracle

    results: List[TrialResult] = []

    for cond in conditions:
        if cond == "Z":
            winner = condition_z(slate, rng)
        elif cond == "A":
            winner = condition_a(slate, personas)
        elif cond == "D":
            winner = condition_d(table)
        else:
            raise ValueError(f"Unknown condition: {cond!r} (supported: Z, A, D)")

        g_achieved = table.g_values[_bundle_index(winner)]

        results.append(TrialResult(
            world="world1_political",
            condition=cond,
            spillover_config=spillover_config,
            trial=trial_idx,
            G_achieved=g_achieved,
            G_random=g_random,
            G_oracle=g_oracle,
            gap_fraction=table.gap_fraction(g_achieved),
        ))

    return results


def _bundle_index(bundle: PolicyBundle) -> int:
    """Compute the index of bundle in the enumerate_bundles() lexicographic ordering.

    Bundles are produced by itertools.product(STANCES, repeat=8) in order, so
    the index is the standard mixed-radix conversion: treat each issue's stance_idx
    as a digit in base-5, most-significant first (issue order = ISSUES order).
    O(n_issues) — no table lookup needed.
    """
    idx = 0
    for issue in ISSUES:
        stance_idx = STANCES.index(bundle.stances[issue])
        idx = idx * len(STANCES) + stance_idx
    return idx


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_experiment(
    n_trials: int = 50,
    n_personas: int = 300,
    persona_seed: int = 42,
    spillover_configs: Sequence[str] = ("centrist", "off_axis"),
    conditions: Sequence[str] = ("Z", "A", "D"),
    base_seed: int = 0,
    output_path: Optional[Path] = None,
    cache: bool = True,
    verbose: bool = True,
) -> List[TrialResult]:
    """Run the full World 1 experiment.

    Generates N_trials × |conditions| × |spillover_configs| rows.
    Writes a CSV to output_path if provided.

    Args:
        n_trials:          Number of trials per (condition, spillover_config).
        n_personas:        Persona population size (200–500 recommended).
        persona_seed:      Seed for persona generation (fixed across all trials).
        spillover_configs: Spillover configurations to evaluate.
        conditions:        Algorithm conditions to run.
        base_seed:         Base seed for trial-level randomness.
        output_path:       If set, write results to this CSV path.
        cache:             Enable oracle PayoffTable disk cache.
        verbose:           Print per-trial progress.

    Returns:
        All TrialResult objects (flat list).
    """
    if verbose:
        print(f"Generating {n_personas} personas (seed={persona_seed})...")
    personas = generate_personas(n=n_personas, seed=persona_seed)

    # Pre-generate all bundles once; share across spillover configs to save memory.
    if verbose:
        print("Enumerating 5^8 = 390,625 candidate bundles...")
    all_bundles = enumerate_bundles()

    all_results: List[TrialResult] = []

    for spillover_config in spillover_configs:
        if verbose:
            print(f"\nBuilding oracle table (spillover_config={spillover_config!r})...")
        table = build_payoff_table(
            personas, spillover_config, bundles=all_bundles, cache=cache
        )
        if verbose:
            print(
                f"  G_oracle={table.g_oracle:.4f}  G_random={table.g_random:.4f}  "
                f"oracle={table.oracle.to_tuple()}"
            )

        for trial_idx in range(n_trials):
            trial_results = run_trial(
                trial_idx=trial_idx,
                personas=personas,
                table=table,
                spillover_config=spillover_config,
                conditions=conditions,
                base_seed=base_seed,
            )
            all_results.extend(trial_results)

            if verbose and (trial_idx % 10 == 0 or trial_idx == n_trials - 1):
                gap_a = next(
                    (r.gap_fraction for r in trial_results if r.condition == "A"), None
                )
                print(
                    f"  trial {trial_idx:3d}/{n_trials}  "
                    + (f"gap(A)={gap_a:.3f}" if gap_a is not None else "")
                )

    if output_path is not None:
        _write_csv(all_results, output_path)
        if verbose:
            print(f"\nWrote {len(all_results)} rows → {output_path}")

    return all_results


def _write_csv(results: List[TrialResult], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "world", "condition", "spillover_config", "trial",
        "G_achieved", "G_random", "G_oracle", "gap_fraction",
        "cost_usd", "cost_flag",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "world": r.world,
                "condition": r.condition,
                "spillover_config": r.spillover_config,
                "trial": r.trial,
                "G_achieved": f"{r.G_achieved:.6f}",
                "G_random": f"{r.G_random:.6f}",
                "G_oracle": f"{r.G_oracle:.6f}",
                "gap_fraction": f"{r.gap_fraction:.6f}",
                "cost_usd": f"{r.cost_usd:.6f}",
                "cost_flag": str(r.cost_flag),
            })
