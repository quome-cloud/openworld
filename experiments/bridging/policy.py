"""PolicyBundle, Global Utility, and Oracle for the US Two-Party Political Simulator.

The policy space is 8 issues × 5 stances = 5^8 = 390,625 possible bundles.
The oracle is the argmax of G(bundle) over all possible bundles, precomputed
once per persona population + spillover configuration.

Spillover configurations:
  "centrist"  — positive-sum bundles cluster near ideological center (default)
  "off_axis"  — positive-sum bundles are NOT centrist (stress-test for bridging)
"""

from __future__ import annotations

import hashlib
import itertools
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .personas import ISSUES, Persona

_CACHE_DIR = Path(__file__).parent / ".cache"

# ── Stance scale ──────────────────────────────────────────────────────────────
# -2 = strong progressive, -1 = lean progressive, 0 = centrist,
# +1 = lean conservative, +2 = strong conservative

STANCES = (-2, -1, 0, 1, 2)


@dataclass(frozen=True)
class PolicyBundle:
    """A concrete set of policy positions, one per issue dimension.

    stances: dict mapping each of the 8 ISSUES to an int in {-2,-1,0,+1,+2}.
    Frozen so instances are hashable and usable as dict keys.
    """

    stances: Dict[str, int]

    def __post_init__(self):
        for issue in ISSUES:
            if issue not in self.stances:
                raise ValueError(f"PolicyBundle missing stance for issue: {issue!r}")
            if self.stances[issue] not in STANCES:
                raise ValueError(
                    f"PolicyBundle stance for {issue!r} must be in {STANCES}, "
                    f"got {self.stances[issue]!r}"
                )

    def __hash__(self):
        return hash(tuple(self.stances[i] for i in ISSUES))

    def __eq__(self, other):
        if not isinstance(other, PolicyBundle):
            return NotImplemented
        return all(self.stances[i] == other.stances[i] for i in ISSUES)

    @classmethod
    def from_tuple(cls, values: Tuple[int, ...]) -> "PolicyBundle":
        """Construct from an ordered tuple aligned with ISSUES."""
        return cls(stances=dict(zip(ISSUES, values)))

    def to_tuple(self) -> Tuple[int, ...]:
        """Return stances as an ordered tuple aligned with ISSUES."""
        return tuple(self.stances[i] for i in ISSUES)


# ── Spillover configurations ──────────────────────────────────────────────────

# Each spillover is a (condition_dict, welfare_bonus) pair.
# condition_dict maps issue → required stance; a bundle matches if all
# specified stances are satisfied.

SPILLOVERS_CENTRIST: List[Tuple[Dict[str, int], float]] = [
    # (healthcare=0, fiscal=0) → bipartisan cost-containment dividend
    ({"healthcare": 0, "fiscal": 0}, 0.05),
    # (climate=-1, foreign_policy=0) → clean-energy export boost
    ({"climate": -1, "foreign_policy": 0}, 0.03),
    # (criminal_justice=-1, education=0) → long-run crime reduction
    ({"criminal_justice": -1, "education": 0}, 0.04),
]

SPILLOVERS_OFF_AXIS: List[Tuple[Dict[str, int], float]] = [
    # (climate=-2, fiscal=+1) → aggressive climate progressivism + fiscal conservatism
    ({"climate": -2, "fiscal": 1}, 0.05),
    # (civil_rights=+1, criminal_justice=-1) → public safety dividend
    ({"civil_rights": 1, "criminal_justice": -1}, 0.03),
    # (education=+2, healthcare=-1) → human-capital dividend
    ({"education": 2, "healthcare": -1}, 0.04),
]

SPILLOVER_CONFIGS = {
    "centrist": SPILLOVERS_CENTRIST,
    "off_axis": SPILLOVERS_OFF_AXIS,
}


def compute_spillover(
    bundle: PolicyBundle,
    spillover_config: str = "centrist",
) -> float:
    """Return total spillover welfare bonus for a bundle under the given config."""
    spillovers = SPILLOVER_CONFIGS[spillover_config]
    bonus = 0.0
    for condition, amount in spillovers:
        if all(bundle.stances.get(issue) == stance for issue, stance in condition.items()):
            bonus += amount
    return bonus


def global_utility(
    bundle: PolicyBundle,
    personas: List[Persona],
    spillover_config: str = "centrist",
) -> float:
    """Compute G(bundle) = mean individual welfare + spillover bonuses.

    G ∈ [0, 1+max_spillover] approximately; exact range depends on persona distribution.
    """
    mean_welfare = sum(p.welfare(bundle.stances) for p in personas) / len(personas)
    spillover = compute_spillover(bundle, spillover_config)
    return mean_welfare + spillover


# ── All-bundles enumeration ───────────────────────────────────────────────────

def enumerate_bundles() -> List[PolicyBundle]:
    """Return all 5^8 = 390,625 possible PolicyBundles in lexicographic order."""
    return [
        PolicyBundle.from_tuple(combo)
        for combo in itertools.product(STANCES, repeat=len(ISSUES))
    ]


# ── Precomputed payoff table ──────────────────────────────────────────────────

@dataclass
class PayoffTable:
    """Precomputed G values for every possible bundle.

    Constructed once per (personas, spillover_config) pair; reused across
    all experimental conditions to avoid redundant computation.

    Attributes:
        bundles:          All 5^8 PolicyBundles.
        g_values:         Parallel list of G(bundle) scores.
        oracle:           Bundle with maximum G.
        g_oracle:         G(oracle) — the ceiling.
        g_random:         Mean G over all bundles — the random-selection floor.
        oracle_index:     Index into bundles/g_values of the oracle bundle.
    """

    bundles: List[PolicyBundle]
    g_values: List[float]
    oracle: PolicyBundle
    g_oracle: float
    g_random: float
    oracle_index: int

    def gap_fraction(self, g_achieved: float) -> float:
        """Fraction of achievable improvement gap captured.

        gap_fraction = (G_achieved - G_random) / (G_oracle - G_random)
        Clamped to [0, 1] to handle floating-point edge cases.
        """
        denom = self.g_oracle - self.g_random
        if denom < 1e-12:
            return 1.0  # degenerate: all bundles equivalent
        raw = (g_achieved - self.g_random) / denom
        return max(0.0, min(1.0, raw))


def _config_hash(personas: List[Persona], spillover_config: str) -> str:
    """Stable hash of (persona population, spillover_config) for cache keying.

    Incorporates persona count, all ideology values + community assignments
    (fully characterises the population without storing the full object), and
    the spillover config name.
    """
    digest_input = json.dumps({
        "spillover_config": spillover_config,
        "n": len(personas),
        "ideologies": [round(p.latent_ideology, 8) for p in personas],
        "communities": [p.network_community for p in personas],
    }, sort_keys=True)
    return hashlib.sha256(digest_input.encode()).hexdigest()[:16]


def build_payoff_table(
    personas: List[Persona],
    spillover_config: str = "centrist",
    bundles: Optional[List[PolicyBundle]] = None,
    cache: bool = True,
) -> PayoffTable:
    """Precompute G for every bundle and return a PayoffTable.

    This is O(5^8 × N_personas × N_issues) — about 390k × 300 × 8 ≈ 940M
    multiplications. With Python floats it runs in ~8–15 seconds for N=300.
    Results are stable across runs given the same personas + spillover_config.

    Args:
        personas:         List of Persona objects.
        spillover_config: "centrist" or "off_axis".
        bundles:          Pre-generated bundle list (pass to reuse across configs).
        cache:            If True, read/write a pickle cache keyed by config hash.
                          Cache lives at experiments/bridging/.cache/oracle_<hash>.pkl.
                          Invalidated when persona population or spillover_config changes.

    Returns:
        PayoffTable with oracle, g_oracle, g_random, and all g_values.
    """
    if bundles is None:
        bundles = enumerate_bundles()

    if cache:
        _CACHE_DIR.mkdir(exist_ok=True)
        cache_key = _config_hash(personas, spillover_config)
        cache_path = _CACHE_DIR / f"oracle_{cache_key}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as fh:
                table = pickle.load(fh)
            # Restore the (possibly shared) bundles reference if same length
            if len(table.bundles) == len(bundles):
                table = PayoffTable(
                    bundles=bundles,
                    g_values=table.g_values,
                    oracle=table.oracle,
                    g_oracle=table.g_oracle,
                    g_random=table.g_random,
                    oracle_index=table.oracle_index,
                )
            return table

    # Precompute per-persona ideal-stance arrays for vectorized inner loop
    n_personas = len(personas)
    n_issues = len(ISSUES)
    max_distance = 4.0

    # issue_weights[p][i] and ideal_stances[p][i]
    weights = [
        [p.issue_weights[issue] for issue in ISSUES]
        for p in personas
    ]
    ideals = [
        [p.ideal_stances[issue] for issue in ISSUES]
        for p in personas
    ]

    g_values: List[float] = []
    best_g = float("-inf")
    best_idx = 0
    total_g = 0.0

    for idx, bundle in enumerate(bundles):
        stances_arr = bundle.to_tuple()

        # Mean individual welfare across all personas
        mean_w = 0.0
        for p_idx in range(n_personas):
            persona_w = 0.0
            for i_idx in range(n_issues):
                stance = stances_arr[i_idx]
                ideal = ideals[p_idx][i_idx]
                loss = ((abs(stance - ideal)) / max_distance) ** 2
                persona_w += weights[p_idx][i_idx] * (1.0 - loss)
            mean_w += persona_w
        mean_w /= n_personas

        # Spillover bonus
        spillover = compute_spillover(bundle, spillover_config)

        g = mean_w + spillover
        g_values.append(g)
        total_g += g

        if g > best_g:
            best_g = g
            best_idx = idx

    g_random = total_g / len(bundles)

    table = PayoffTable(
        bundles=bundles,
        g_values=g_values,
        oracle=bundles[best_idx],
        g_oracle=best_g,
        g_random=g_random,
        oracle_index=best_idx,
    )

    if cache:
        with open(cache_path, "wb") as fh:  # cache_path defined above when cache=True
            pickle.dump(table, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return table
