"""Tests for Condition C: Community Notes and polarity-product bridging."""

import numpy as np
import pytest

from experiments.bridging.conditions_c import (
    build_endorsement_matrix,
    compute_bridge_scores_polarity,
    condition_c_community_notes,
    condition_c_polarity_product,
    fit_community_notes,
)
from experiments.bridging.personas import ISSUES, generate_personas
from experiments.bridging.policy import PolicyBundle
from experiments.bridging.simulation import generate_candidate_slate


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def setup():
    personas = generate_personas(n=60, seed=7)
    slate = generate_candidate_slate(personas, trial_seed=0)
    R = build_endorsement_matrix(personas, slate)
    return {"personas": personas, "slate": slate, "R": R}


# ── Endorsement matrix ────────────────────────────────────────────────────────

def test_endorsement_matrix_shape(setup):
    R = setup["R"]
    assert R.shape == (60, 7)


def test_endorsement_matrix_binary(setup):
    R = setup["R"]
    unique = set(R.flatten().tolist())
    assert unique <= {0.0, 1.0}


def test_endorsement_matrix_relative_threshold(setup):
    """Each persona endorses ≥1 bundle (mean-threshold means at least the top half)."""
    R = setup["R"]
    endorsements_per_persona = R.sum(axis=1)
    assert (endorsements_per_persona >= 1).all()


def test_endorsement_matrix_no_all_zeros_row():
    """A persona always endorses at least one bundle (relative threshold guarantees this)."""
    personas = generate_personas(n=20, seed=99)
    slate = generate_candidate_slate(personas, trial_seed=0)
    R = build_endorsement_matrix(personas, slate)
    assert (R.sum(axis=1) > 0).all()


# ── fit_community_notes ────────────────────────────────────────────────────────

def test_fit_community_notes_output_shape(setup):
    i_n = fit_community_notes(setup["R"])
    assert i_n.shape == (7,)


def test_fit_community_notes_intercepts_vary(setup):
    """Intercepts should differ — not all-zero — after fitting."""
    i_n = fit_community_notes(setup["R"])
    assert np.std(i_n) > 1e-4


def test_fit_community_notes_centrist_scores_high(setup):
    """The centrist (all-zeros) bundle should score highest or near-highest.

    With a bimodal population, the centrist bundle gets cross-cluster endorsement
    (both left and right personas include it in their top half), so its intercept
    should be relatively high after absorbing within-cluster bias into factors.
    """
    slate = setup["slate"]
    centrist_idx = next(
        i for i, b in enumerate(slate)
        if all(b.stances[issue] == 0 for issue in ISSUES)
    )
    i_n = fit_community_notes(setup["R"])
    centrist_score = i_n[centrist_idx]
    # Centrist should be in the top half by intercept
    assert centrist_score >= np.median(i_n)


def test_fit_community_notes_reproducible(setup):
    R = setup["R"]
    i_n1 = fit_community_notes(R, seed=0)
    i_n2 = fit_community_notes(R, seed=0)
    np.testing.assert_array_equal(i_n1, i_n2)


def test_fit_community_notes_different_seeds_converge(setup):
    """Different seeds should converge to similar (not necessarily identical) rankings."""
    R = setup["R"]
    i_n1 = fit_community_notes(R, seed=0)
    i_n2 = fit_community_notes(R, seed=42)
    # Rankings should agree on the top bundle
    assert np.argmax(i_n1) == np.argmax(i_n2)


# ── condition_c_community_notes ───────────────────────────────────────────────

def test_condition_c_cn_returns_slate_member(setup):
    winner = condition_c_community_notes(setup["slate"], setup["personas"])
    assert winner in setup["slate"]


def test_condition_c_cn_deterministic(setup):
    w1 = condition_c_community_notes(setup["slate"], setup["personas"])
    w2 = condition_c_community_notes(setup["slate"], setup["personas"])
    assert w1 == w2


def test_condition_c_cn_beats_majority_vote_on_average():
    """C_CN gap_fraction should exceed A gap_fraction averaged over trials.

    Uses n=20, seed=0 so the PayoffTable hits the disk cache from
    test_bridging_simulation.py's run — no redundant 5^8 build.
    """
    from experiments.bridging.policy import build_payoff_table, enumerate_bundles
    from experiments.bridging.simulation import run_trial

    personas = generate_personas(n=20, seed=0)
    bundles = enumerate_bundles()
    table = build_payoff_table(personas, "centrist", bundles=bundles, cache=True)

    c_gaps, a_gaps = [], []
    for trial in range(5):
        results = run_trial(trial, personas, table, "centrist",
                            conditions=("A", "C_CN"), base_seed=0)
        a_gaps.append(next(r.gap_fraction for r in results if r.condition == "A"))
        c_gaps.append(next(r.gap_fraction for r in results if r.condition == "C_CN"))

    assert sum(c_gaps) / len(c_gaps) > sum(a_gaps) / len(a_gaps), (
        f"C_CN avg gap {sum(c_gaps)/len(c_gaps):.3f} should exceed "
        f"A avg gap {sum(a_gaps)/len(a_gaps):.3f}"
    )


# ── compute_bridge_scores_polarity ────────────────────────────────────────────

def test_polarity_scores_shape(setup):
    scores = compute_bridge_scores_polarity(setup["personas"], setup["slate"])
    assert scores.shape == (7,)


def test_polarity_scores_non_negative(setup):
    scores = compute_bridge_scores_polarity(setup["personas"], setup["slate"])
    assert (scores >= 0).all()


def test_polarity_scores_at_most_one(setup):
    scores = compute_bridge_scores_polarity(setup["personas"], setup["slate"])
    assert (scores <= 1.0 + 1e-9).all()


def test_polarity_product_centrist_scores_high(setup):
    """The centrist bundle should score high on polarity-product bridging."""
    slate = setup["slate"]
    centrist_idx = next(
        i for i, b in enumerate(slate)
        if all(b.stances[issue] == 0 for issue in ISSUES)
    )
    scores = compute_bridge_scores_polarity(setup["personas"], slate)
    assert scores[centrist_idx] >= np.median(scores)


# ── condition_c_polarity_product ──────────────────────────────────────────────

def test_condition_c_pp_returns_slate_member(setup):
    winner = condition_c_polarity_product(setup["slate"], setup["personas"])
    assert winner in setup["slate"]


def test_condition_c_pp_deterministic(setup):
    w1 = condition_c_polarity_product(setup["slate"], setup["personas"])
    w2 = condition_c_polarity_product(setup["slate"], setup["personas"])
    assert w1 == w2


def test_condition_c_pp_two_cluster_vs_network_communities(setup):
    """Both community-labeling modes should return a slate member."""
    w_net = condition_c_polarity_product(setup["slate"], setup["personas"],
                                         use_network_communities=True)
    w_ideo = condition_c_polarity_product(setup["slate"], setup["personas"],
                                          use_network_communities=False)
    assert w_net in setup["slate"]
    assert w_ideo in setup["slate"]
