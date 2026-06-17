"""The many-worlds causal-assumption-testing demo: deterministic claims.

Mirrors test_minigrid_world.py: import the example module and assert the exact
numbers it reports, so the demo cannot silently drift.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from manyworlds_assumption_test import (  # noqa: E402
    MECHANISMS, PARAMS, TRUTH, cohort, soft_posterior,
)
from openworld import COUNTING, WorldStore  # noqa: E402


def test_exact_version_space_recovers_truth_and_surfaces_identifiability():
    store = WorldStore(PARAMS, MECHANISMS, semiring=COUNTING)
    assert store.total_worlds() == 16
    for state, action, nxt in cohort(n=200, flip=0.0):
        store.observe(state, action, nxt)
    # three edges are pinned to truth; the direct stress->bayley edge is not
    assert store.count() == 2
    for p in ("A_income_stress", "A_stress_preterm", "A_preterm_bayley"):
        m = store.marginal(p)
        assert m[TRUTH[p]] == 1.0
    direct = store.marginal("A_stress_direct")
    assert direct["yes"] == 0.5 and direct["no"] == 0.5   # unidentifiable -> 50/50


def test_soft_posterior_is_noise_robust_where_hard_pruning_fails():
    noisy = cohort(n=400, flip=0.12, seed=99)
    # hard version-space over-prunes under noise: the truth is eliminated
    hard = WorldStore(PARAMS, MECHANISMS, semiring=COUNTING)
    for state, action, nxt in noisy:
        hard.observe(state, action, nxt)
    assert hard.count() == 0
    # the soft posterior still concentrates on the true assumptions
    marg = soft_posterior(noisy, eps=0.12)
    for p in ("A_income_stress", "A_stress_preterm", "A_preterm_bayley"):
        assert marg(p)[TRUTH[p]] > 0.9
    direct = marg("A_stress_direct")
    assert abs(direct["yes"] - 0.5) < 0.05               # still unidentifiable
