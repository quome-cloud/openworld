"""Persona generator for the US Two-Party Political Simulator (World 1).

Generates N personas drawn from an approximate 2024–2026 USA political
distribution: bimodal ideology mixture + Dirichlet issue weights +
Stochastic Block Model network communities.

All randomness is seeded for reproducibility.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# The 8 policy issue dimensions (ANES-aligned)
ISSUES = [
    "immigration",
    "healthcare",
    "climate",
    "fiscal",
    "foreign_policy",
    "civil_rights",
    "education",
    "criminal_justice",
]

# Stochastic Block Model parameters
# 6 communities: 2 large ideological (~40% each), 4 small bridgeable (~5% each)
_SBM_COMMUNITY_FRACTIONS = [0.40, 0.40, 0.05, 0.05, 0.05, 0.05]
_SBM_P_IN = 0.30    # within-community edge probability
_SBM_P_OUT = 0.02   # cross-community edge probability

# Bimodal ideology mixture components: (weight, mean, std)
_IDEOLOGY_COMPONENTS = [
    (0.40, -0.45, 0.25),   # left-leaning
    (0.40,  0.45, 0.25),   # right-leaning
    (0.20,  0.00, 0.15),   # moderate center
]


@dataclass
class Persona:
    """A simulated US voter persona."""

    persona_id: int
    latent_ideology: float          # ∈ [-1, 1]; negative=progressive, positive=conservative
    issue_weights: Dict[str, float] # per-issue salience, sums to 1.0
    ideal_stances: Dict[str, float] # ideal position per issue ∈ [-2, 2]
    network_community: int          # community index ∈ {0..K-1}
    # Network adjacency stored externally; this is just the community assignment.

    def welfare(self, bundle_stances: Dict[str, int]) -> float:
        """Compute this persona's welfare given a policy bundle.

        Uses quadratic loss from ideal point, weighted by issue salience.
        welfare ∈ [0, 1] per issue before weighting.
        """
        max_distance = 4.0  # range is [-2, 2], max |stance - ideal| = 4
        total = 0.0
        for issue in ISSUES:
            stance = bundle_stances.get(issue, 0)
            ideal = self.ideal_stances[issue]
            loss = (abs(stance - ideal) / max_distance) ** 2
            total += self.issue_weights[issue] * (1.0 - loss)
        return total


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _dirichlet(alpha: List[float], rng: random.Random) -> List[float]:
    """Sample from a Dirichlet distribution using gamma sampling."""
    gammas = [_sample_gamma(a, rng) for a in alpha]
    total = sum(gammas)
    if total == 0:
        return [1.0 / len(alpha)] * len(alpha)
    return [g / total for g in gammas]


def _sample_gamma(alpha: float, rng: random.Random) -> float:
    """Sample Gamma(alpha, 1) using Johnk's method for alpha < 1,
    Marsaglia-Tsang for alpha >= 1."""
    if alpha >= 1.0:
        # Marsaglia-Tsang method
        d = alpha - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = rng.gauss(0, 1)
            v = (1.0 + c * x) ** 3
            if v > 0:
                u = rng.random()
                if u < 1.0 - 0.0331 * (x * x) ** 2:
                    return d * v
                if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
                    return d * v
    else:
        # Johnk's method for alpha < 1: Gamma(alpha) = Gamma(1+alpha) * U^(1/alpha)
        return _sample_gamma(1.0 + alpha, rng) * (rng.random() ** (1.0 / alpha))


def _sample_ideology(rng: random.Random) -> float:
    """Sample from the bimodal mixture of Gaussians."""
    weights = [c[0] for c in _IDEOLOGY_COMPONENTS]
    cumulative = []
    total = 0.0
    for w in weights:
        total += w
        cumulative.append(total)
    u = rng.random() * total
    for i, c in enumerate(cumulative):
        if u <= c:
            mean, std = _IDEOLOGY_COMPONENTS[i][1], _IDEOLOGY_COMPONENTS[i][2]
            return _clamp(rng.gauss(mean, std), -1.0, 1.0)
    mean, std = _IDEOLOGY_COMPONENTS[-1][1], _IDEOLOGY_COMPONENTS[-1][2]
    return _clamp(rng.gauss(mean, std), -1.0, 1.0)


def _assign_community(n_personas: int, rng: random.Random) -> List[int]:
    """Assign each persona to a community using the SBM fractions."""
    k = len(_SBM_COMMUNITY_FRACTIONS)
    assignments = []
    counts = [max(1, round(f * n_personas)) for f in _SBM_COMMUNITY_FRACTIONS]
    # Adjust to exactly n_personas
    diff = n_personas - sum(counts)
    counts[0] += diff
    for community_id, count in enumerate(counts):
        assignments.extend([community_id] * count)
    rng.shuffle(assignments)
    return assignments[:n_personas]


def generate_personas(
    n: int = 300,
    seed: int = 42,
    ideology_noise: float = 0.1,
) -> List[Persona]:
    """Generate N personas for the US Two-Party Political Simulator.

    Args:
        n: Number of personas (200-500 recommended).
        seed: Random seed for reproducibility.
        ideology_noise: Std of per-issue ideal point noise around ideology.

    Returns:
        List of Persona objects with stable IDs 0..N-1.
    """
    rng = random.Random(seed)

    community_assignments = _assign_community(n, rng)
    # Re-shuffle with same rng state to get stable assignments

    personas = []
    for i in range(n):
        ideology = _sample_ideology(rng)

        # Issue weights: Dirichlet(alpha=0.5 per issue) — "single-issue voter" skewed
        alpha = [0.5] * len(ISSUES)
        weights_raw = _dirichlet(alpha, rng)
        issue_weights = {issue: weights_raw[j] for j, issue in enumerate(ISSUES)}

        # Ideal stances: derived from latent_ideology + idiosyncratic noise
        # Ideology maps to ideal stance on each issue:
        # latent_ideology ∈ [-1, 1] → ideal_stance ∈ [-2, 2]
        ideal_stances = {}
        for issue in ISSUES:
            base = ideology * 2.0  # scale to [-2, 2]
            noise = rng.gauss(0, ideology_noise * 2.0)
            ideal_stances[issue] = _clamp(base + noise, -2.0, 2.0)

        personas.append(Persona(
            persona_id=i,
            latent_ideology=ideology,
            issue_weights=issue_weights,
            ideal_stances=ideal_stances,
            network_community=community_assignments[i],
        ))

    return personas


def generate_sbm_edges(
    personas: List[Persona],
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """Generate edges for a Stochastic Block Model network.

    Within-community p_in=0.30, cross-community p_out=0.02.
    Returns list of (persona_id_a, persona_id_b) undirected edges.
    """
    rng = random.Random(seed + 1000)  # separate seed from persona generation
    edges = []
    n = len(personas)
    for i in range(n):
        for j in range(i + 1, n):
            p = _SBM_P_IN if personas[i].network_community == personas[j].network_community else _SBM_P_OUT
            if rng.random() < p:
                edges.append((i, j))
    return edges
