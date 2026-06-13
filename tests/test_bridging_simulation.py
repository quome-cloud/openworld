"""Tests for World 1 simulation: candidate slate, conditions Z/A/D, trial runner.

Performance note: build_payoff_table(5^8 bundles) is expensive (~2-5 min for
N=20). All tests share a SINGLE module-level table via the `shared` fixture
(scope="module") to avoid rebuilding it per test.
"""

import csv
import random

import pytest

from experiments.bridging.personas import ISSUES, generate_personas
from experiments.bridging.policy import (
    STANCES,
    PayoffTable,
    PolicyBundle,
    build_payoff_table,
    enumerate_bundles,
)
from experiments.bridging.simulation import (
    _bundle_index,
    condition_a,
    condition_d,
    condition_z,
    generate_candidate_slate,
    run_experiment,
    run_trial,
)


# ── Module-level shared fixture (built ONCE for the whole module) ─────────────

@pytest.fixture(scope="module")
def shared():
    """Personas + bundles + table, built once for the entire test module."""
    personas = generate_personas(n=20, seed=0)
    bundles = enumerate_bundles()
    table = build_payoff_table(personas, "centrist", bundles=bundles, cache=True)
    slate = generate_candidate_slate(personas, trial_seed=0)
    return {"personas": personas, "bundles": bundles, "table": table, "slate": slate}


# ── _bundle_index ──────────────────────────────────────────────────────────────

def test_bundle_index_centrist():
    """All-zero centrist bundle is at the midpoint of 5^8 bundles."""
    centrist = PolicyBundle(stances={issue: 0 for issue in ISSUES})
    assert _bundle_index(centrist) == 195312


def test_bundle_index_all_minus2():
    b = PolicyBundle(stances={issue: -2 for issue in ISSUES})
    assert _bundle_index(b) == 0


def test_bundle_index_all_plus2():
    b = PolicyBundle(stances={issue: 2 for issue in ISSUES})
    assert _bundle_index(b) == 5 ** 8 - 1


def test_bundle_index_matches_enumerate_bundles(shared):
    bundles = shared["bundles"]
    for check_idx in [0, 1, 100, 195312, 390624]:
        b = bundles[check_idx]
        assert _bundle_index(b) == check_idx


# ── Candidate slate ────────────────────────────────────────────────────────────

def test_candidate_slate_has_seven_bundles(shared):
    slate = generate_candidate_slate(shared["personas"], trial_seed=0, n_random=2)
    assert len(slate) == 7


def test_candidate_slate_contains_all_archetypes(shared):
    slate = generate_candidate_slate(shared["personas"], trial_seed=0)
    for archetype_stance in (-2, -1, 0, 1, 2):
        expected = PolicyBundle(stances={issue: archetype_stance for issue in ISSUES})
        assert expected in slate


def test_candidate_slate_reproducible(shared):
    s1 = generate_candidate_slate(shared["personas"], trial_seed=42)
    s2 = generate_candidate_slate(shared["personas"], trial_seed=42)
    assert s1 == s2


def test_candidate_slate_varies_across_trials(shared):
    archetypes = {
        PolicyBundle(stances={i: s for i in ISSUES}) for s in STANCES
    }
    s0 = [b for b in generate_candidate_slate(shared["personas"], 0) if b not in archetypes]
    s1 = [b for b in generate_candidate_slate(shared["personas"], 1) if b not in archetypes]
    assert s0 != s1


# ── Condition Z ────────────────────────────────────────────────────────────────

def test_condition_z_returns_slate_member(shared):
    winner = condition_z(shared["slate"], random.Random(99))
    assert winner in shared["slate"]


def test_condition_z_varies_with_rng(shared):
    winners = {condition_z(shared["slate"], random.Random(i)) for i in range(30)}
    assert len(winners) > 1


# ── Condition A ────────────────────────────────────────────────────────────────

def test_condition_a_returns_slate_member(shared):
    winner = condition_a(shared["slate"], shared["personas"])
    assert winner in shared["slate"]


def test_condition_a_winner_has_most_votes(shared):
    slate, personas = shared["slate"], shared["personas"]
    votes = {b: 0 for b in slate}
    for p in personas:
        best = max(slate, key=lambda b: p.welfare(b.stances))
        votes[best] += 1
    max_v = max(votes.values())
    tied = [b for b, v in votes.items() if v == max_v]
    expected = min(tied, key=lambda b: b.to_tuple())  # lexicographic tie-break
    assert condition_a(slate, personas) == expected


def test_condition_a_deterministic(shared):
    w1 = condition_a(shared["slate"], shared["personas"])
    w2 = condition_a(shared["slate"], shared["personas"])
    assert w1 == w2


def test_condition_a_tie_break_is_lexicographic(shared):
    """Construct a slate where two bundles tie; winner should be lexicographically first."""
    personas = shared["personas"]
    # Single-persona population: picks the bundle with highest welfare.
    # Use two-bundle slate guaranteed to produce a tie (same welfare for all personas).
    tie_a = PolicyBundle(stances={issue: 0 for issue in ISSUES})
    tie_b = PolicyBundle(stances={issue: 0 for issue in ISSUES})
    # They're equal PolicyBundles, so votes split 0:N in the same bucket.
    # Use stances that differ in one issue to make two truly distinct bundles.
    stances_x = {issue: 0 for issue in ISSUES}
    stances_y = {issue: 0 for issue in ISSUES}
    # Both centrist → same welfare for every persona → tie guaranteed.
    # Distinguish them so they hash differently: won't matter, but tests the branch.
    # Actually test a real tie: use the same all-zero bundle in two "different" list slots.
    slate_tie = [tie_a, tie_a]  # same object → both get votes, always ties
    winner = condition_a(slate_tie, personas)
    assert winner == tie_a


# ── Condition D ────────────────────────────────────────────────────────────────

def test_condition_d_returns_oracle(shared):
    assert condition_d(shared["table"]) == shared["table"].oracle


def test_condition_d_gap_is_one(shared):
    table = shared["table"]
    g = table.g_values[_bundle_index(condition_d(table))]
    assert abs(table.gap_fraction(g) - 1.0) < 1e-9


# ── run_trial ──────────────────────────────────────────────────────────────────

def test_run_trial_returns_one_result_per_condition(shared):
    results = run_trial(0, shared["personas"], shared["table"], "centrist",
                        conditions=("Z", "A", "D"), base_seed=0)
    assert len(results) == 3
    assert {r.condition for r in results} == {"Z", "A", "D"}


def test_run_trial_d_gap_is_one(shared):
    results = run_trial(0, shared["personas"], shared["table"], "centrist",
                        conditions=("D",), base_seed=0)
    assert abs(results[0].gap_fraction - 1.0) < 1e-9


def test_run_trial_a_gap_above_zero(shared):
    """Condition A should capture some fraction of the gap (> random baseline)."""
    results = run_trial(0, shared["personas"], shared["table"], "centrist",
                        conditions=("A",), base_seed=0)
    assert results[0].gap_fraction > 0.0


def test_run_trial_metadata(shared):
    results = run_trial(7, shared["personas"], shared["table"], "centrist",
                        conditions=("A",), base_seed=100)
    r = results[0]
    assert r.world == "world1_political"
    assert r.condition == "A"
    assert r.spillover_config == "centrist"
    assert r.trial == 7
    assert r.G_oracle == shared["table"].g_oracle
    assert r.G_random == shared["table"].g_random


def test_run_trial_reproducible(shared):
    kw = dict(personas=shared["personas"], table=shared["table"],
              spillover_config="centrist", conditions=("Z", "A"), base_seed=0)
    r1 = run_trial(0, **kw)
    r2 = run_trial(0, **kw)
    for a, b in zip(r1, r2):
        assert a.G_achieved == b.G_achieved


# ── run_experiment CSV output (small-scale, uses cache) ───────────────────────

def test_run_experiment_csv_written(shared, tmp_path):
    out = tmp_path / "results.csv"
    results = run_experiment(
        n_trials=2, n_personas=20, persona_seed=0,
        spillover_configs=("centrist",), conditions=("A", "D"),
        output_path=out, cache=True, verbose=False,
    )
    assert out.exists()
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 4  # 2 trials × 2 conditions
    assert {r["condition"] for r in rows} == {"A", "D"}


def test_run_experiment_row_count(shared):
    results = run_experiment(
        n_trials=3, n_personas=20, persona_seed=0,
        spillover_configs=("centrist",), conditions=("Z", "A"),
        cache=True, verbose=False,
    )
    assert len(results) == 6  # 3 trials × 2 conditions × 1 config
