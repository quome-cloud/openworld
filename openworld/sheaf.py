"""Sheaf gluing & consistency for composite / multi-agent worlds.

Semirings let us parse sub-worlds; SHEAVES let us check whether the local views
of those sub-worlds glue into a consistent global one. A sheaf assigns data to
each local region (open set) of a cover and requires: local sections that AGREE
on overlaps glue to a unique global section. When they disagree, there is no
global section - the obstruction is a (Cech) cohomology class, and it localizes
the inconsistency.

This is exactly the multi-agent observation problem: several agents/meetings/
sensors observe overlapping parts of a world; do their reports describe ONE
consistent global state? If not, where is the fault? A sheaf answers both, and -
with redundant overlap - can CORRECT a faulty report by majority over the agents
that observe each variable, where naive averaging is silently corrupted by it.

Numpy-only, deterministic. Additive module.

  cover    : {agent: [variables it observes]}
  sections : {agent: {variable: reported value}}
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np


def overlaps(cover: Dict[str, List[str]]) -> List[Tuple[str, str, List[str]]]:
    """Pairs of agents and the variables they share (the nerve's edges)."""
    out = []
    for a, b in combinations(cover, 2):
        shared = sorted(set(cover[a]) & set(cover[b]))
        if shared:
            out.append((a, b, shared))
    return out


def disagreements(cover, sections, tol=1e-9):
    """Per overlapping pair/variable, the absolute disagreement (the Cech
    1-cochain delta0(s))."""
    out = {}
    for a, b, shared in overlaps(cover):
        for v in shared:
            d = abs(sections[a][v] - sections[b][v])
            if d > tol:
                out[(a, b, v)] = d
    return out


def is_consistent(cover, sections, tol=1e-9) -> bool:
    """Do the local sections glue? (Sheaf axiom: agree on all overlaps.)"""
    return len(disagreements(cover, sections, tol)) == 0


def obstruction_norm(cover, sections) -> float:
    """Total magnitude of the gluing obstruction (0 iff a global section exists)."""
    return float(sum(disagreements(cover, sections).values()))


def glue(cover, sections, tol=1e-9) -> Dict[str, float]:
    """The unique global section, if local sections are consistent."""
    if not is_consistent(cover, sections, tol):
        raise ValueError("no global section: local sections disagree on overlaps")
    g = {}
    for a, vs in cover.items():
        for v in vs:
            g[v] = sections[a][v]
    return g


def localize_fault(cover, sections) -> Tuple[str, float]:
    """The agent implicated in the most disagreement (the likely faulty source)."""
    blame: Dict[str, float] = {a: 0.0 for a in cover}
    for (a, b, _), d in disagreements(cover, sections).items():
        blame[a] += d
        blame[b] += d
    worst = max(blame, key=blame.get)
    return worst, blame[worst]


def majority_glue(cover, sections, tol=1e-9) -> Dict[str, float]:
    """Robust gluing: for each variable, the consensus value across the agents
    that observe it (mode within tolerance) - recovers the truth despite a
    minority faulty report, where averaging is corrupted by it."""
    obs: Dict[str, List[float]] = {}
    for a, vs in cover.items():
        for v in vs:
            obs.setdefault(v, []).append(sections[a][v])
    g = {}
    for v, vals in obs.items():
        # cluster equal-within-tol values; pick the largest cluster's mean
        clusters: List[List[float]] = []
        for x in vals:
            for c in clusters:
                if abs(c[0] - x) <= tol * max(1, abs(c[0])) + 1e-9:
                    c.append(x)
                    break
            else:
                clusters.append([x])
        best = max(clusters, key=len)
        g[v] = float(np.mean(best))
    return g


def nerve_betti1(cover) -> int:
    """First Betti number of the nerve (edges - vertices + components): the
    structural cycle-capacity of the cover - how much redundancy there is to
    detect/correct with."""
    verts = list(cover)
    edges = [(a, b) for a, b, _ in overlaps(cover)]
    parent = {v: v for v in verts}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    comps = len({find(v) for v in verts})
    return len(edges) - len(verts) + comps
