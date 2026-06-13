"""Tests for PolicyBundle, Global Utility, and Oracle (World 1)."""

import itertools
import pytest

from experiments.bridging.personas import ISSUES, generate_personas
from experiments.bridging.policy import (
    SPILLOVER_CONFIGS,
    SPILLOVERS_CENTRIST,
    SPILLOVERS_OFF_AXIS,
    STANCES,
    PayoffTable,
    PolicyBundle,
    build_payoff_table,
    compute_spillover,
    enumerate_bundles,
    global_utility,
)


# ── PolicyBundle ──────────────────────────────────────────────────────────────

def _centrist_bundle():
    return PolicyBundle(stances={issue: 0 for issue in ISSUES})


def _conservative_bundle():
    return PolicyBundle(stances={issue: 2 for issue in ISSUES})


def test_policy_bundle_valid_construction():
    b = _centrist_bundle()
    assert all(b.stances[issue] == 0 for issue in ISSUES)


def test_policy_bundle_rejects_missing_issue():
    stances = {issue: 0 for issue in ISSUES}
    del stances["immigration"]
    with pytest.raises(ValueError, match="immigration"):
        PolicyBundle(stances=stances)


def test_policy_bundle_rejects_invalid_stance():
    stances = {issue: 0 for issue in ISSUES}
    stances["immigration"] = 3  # not in STANCES
    with pytest.raises(ValueError, match="immigration"):
        PolicyBundle(stances=stances)


def test_policy_bundle_hashable():
    b1 = _centrist_bundle()
    b2 = _centrist_bundle()
    assert hash(b1) == hash(b2)
    d = {b1: "value"}
    assert d[b2] == "value"


def test_policy_bundle_equality():
    b1 = _centrist_bundle()
    b2 = _centrist_bundle()
    b3 = _conservative_bundle()
    assert b1 == b2
    assert b1 != b3


def test_policy_bundle_from_tuple_round_trip():
    t = (0, 1, -1, 2, -2, 0, 1, -1)
    b = PolicyBundle.from_tuple(t)
    assert b.to_tuple() == t


# ── Enumerate bundles ─────────────────────────────────────────────────────────

def test_enumerate_bundles_count():
    bundles = enumerate_bundles()
    assert len(bundles) == 5 ** 8  # 390,625


def test_enumerate_bundles_all_stances_covered():
    bundles = enumerate_bundles()
    # Each issue should appear with each of the 5 stances the same number of times
    for issue in ISSUES:
        stance_counts = {s: 0 for s in STANCES}
        for b in bundles:
            stance_counts[b.stances[issue]] += 1
        expected = 5 ** 7  # all others free
        assert all(v == expected for v in stance_counts.values())


# ── Spillovers ────────────────────────────────────────────────────────────────

def test_spillover_centrist_only_first_at_all_zeros():
    """All-zeros bundle only triggers (healthcare=0, fiscal=0) → 0.05.
    The other two centrist spillovers need climate=-1 and criminal_justice=-1."""
    b = PolicyBundle(stances={issue: 0 for issue in ISSUES})
    spillover = compute_spillover(b, "centrist")
    assert abs(spillover - 0.05) < 1e-9


def test_spillover_centrist_all_three_triggered():
    """A bundle satisfying all three centrist conditions yields full 0.12 bonus."""
    # (healthcare=0, fiscal=0) + (climate=-1, foreign_policy=0) + (criminal_justice=-1, education=0)
    stances = {issue: 0 for issue in ISSUES}
    stances["climate"] = -1
    stances["criminal_justice"] = -1
    b = PolicyBundle(stances=stances)
    spillover = compute_spillover(b, "centrist")
    assert abs(spillover - (0.05 + 0.03 + 0.04)) < 1e-9


def test_spillover_off_axis_none_at_centrist_bundle():
    b = _centrist_bundle()
    spillover = compute_spillover(b, "off_axis")
    assert spillover == 0.0


def test_spillover_off_axis_first_condition():
    # (climate=-2, fiscal=+1) → 0.05
    stances = {issue: 0 for issue in ISSUES}
    stances["climate"] = -2
    stances["fiscal"] = 1
    b = PolicyBundle(stances=stances)
    spillover = compute_spillover(b, "off_axis")
    assert abs(spillover - 0.05) < 1e-9


def test_spillover_invalid_config():
    b = _centrist_bundle()
    with pytest.raises(KeyError):
        compute_spillover(b, "nonexistent")


# ── Global utility ────────────────────────────────────────────────────────────

def test_global_utility_centrist_above_random():
    """Centrist bundle should score above average for a mixed-ideology population."""
    personas = generate_personas(n=100, seed=10)
    centrist = _centrist_bundle()
    conservative = _conservative_bundle()
    g_centrist = global_utility(centrist, personas, "centrist")
    g_conservative = global_utility(conservative, personas, "centrist")
    # Centrist should beat strong-conservative for a bimodal population
    assert g_centrist > g_conservative


def test_global_utility_returns_float():
    personas = generate_personas(n=10, seed=11)
    b = _centrist_bundle()
    g = global_utility(b, personas)
    assert isinstance(g, float)
    assert 0.0 < g < 2.0  # reasonable range


# ── PayoffTable / Oracle ──────────────────────────────────────────────────────

def test_build_payoff_table_small():
    """Run with tiny persona set for speed."""
    personas = generate_personas(n=20, seed=0)
    table = build_payoff_table(personas, "centrist")

    assert len(table.bundles) == 5 ** 8
    assert len(table.g_values) == 5 ** 8
    assert table.g_oracle >= table.g_random
    assert table.oracle_index == table.g_values.index(max(table.g_values))
    assert table.g_oracle == table.g_values[table.oracle_index]


def test_oracle_is_maximum():
    personas = generate_personas(n=20, seed=1)
    table = build_payoff_table(personas, "centrist")
    assert all(table.g_oracle >= g for g in table.g_values)


def test_gap_fraction_oracle_is_one():
    personas = generate_personas(n=20, seed=2)
    table = build_payoff_table(personas, "centrist")
    assert abs(table.gap_fraction(table.g_oracle) - 1.0) < 1e-9


def test_gap_fraction_random_is_zero():
    personas = generate_personas(n=20, seed=3)
    table = build_payoff_table(personas, "centrist")
    assert abs(table.gap_fraction(table.g_random) - 0.0) < 1e-6


def test_gap_fraction_clamps_to_unit_interval():
    personas = generate_personas(n=20, seed=4)
    table = build_payoff_table(personas, "centrist")
    assert table.gap_fraction(table.g_oracle + 100) == 1.0
    assert table.gap_fraction(table.g_random - 100) == 0.0


def test_payoff_table_off_axis_differs_from_centrist():
    personas = generate_personas(n=20, seed=5)
    table_c = build_payoff_table(personas, "centrist")
    table_o = build_payoff_table(personas, "off_axis")
    # Different spillover configs → different oracles (very likely)
    assert table_c.oracle != table_o.oracle or abs(table_c.g_oracle - table_o.g_oracle) > 1e-9


def test_payoff_table_reproducible():
    personas = generate_personas(n=10, seed=0)
    t1 = build_payoff_table(personas, "centrist")
    t2 = build_payoff_table(personas, "centrist")
    assert t1.g_oracle == t2.g_oracle
    assert t1.oracle == t2.oracle
    assert t1.g_values == t2.g_values


def test_payoff_table_bundles_reuse():
    """Passing pre-generated bundles avoids redundant enumeration."""
    bundles = enumerate_bundles()
    personas = generate_personas(n=10, seed=0)
    t1 = build_payoff_table(personas, "centrist", bundles=bundles)
    t2 = build_payoff_table(personas, "off_axis", bundles=bundles)
    # Same bundle objects shared
    assert t1.bundles is t2.bundles
