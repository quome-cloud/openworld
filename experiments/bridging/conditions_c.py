"""Condition C: bridging-ranked algorithm variants for World 1.

Implements two variants from the bridging literature survey (T340 §3):

  (a) Community Notes matrix factorization — rank bundles by the note-intercept
      i_n from the L2-regularized model r̂_un = μ + i_u + i_n + f_u·f_n.
      λ_i=0.15, λ_f=0.03, d=1 (one latent factor), all parameters from the
      production Community Notes spec (Wojcik et al. 2022, arXiv:2210.15723).

  (b) Polarity-product bridge score — geometric mean of per-cluster endorsement
      rates; variant that requires only explicit community labels (no SVD).
      Serves as the cheap ablation: if (b) ≈ (a), latent factorization is overkill.

The Polis-style GAC variant (c) is not implemented here — it requires PCA +
K-means on the endorsement matrix, which is ill-conditioned at K=7 items.
The polarity-product (b) already tests the core multi-cluster insight.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np

from .personas import ISSUES, Persona
from .policy import PolicyBundle

# ── Endorsement matrix ────────────────────────────────────────────────────────

def build_endorsement_matrix(
    personas: List[Persona],
    slate: List[PolicyBundle],
) -> np.ndarray:
    """Compute (N_personas × N_bundles) binary endorsement matrix.

    Persona p endorses bundle b iff welfare(p, b) ≥ p's mean welfare across
    the slate. This relative threshold gives each persona equal weight regardless
    of absolute welfare level — consistent with the Community Notes design
    (every rater's contributions are normalised to have the same influence).

    Returns float array with dtype float64, values in {0.0, 1.0}.
    """
    N, M = len(personas), len(slate)
    W = np.array([
        [p.welfare(b.stances) for b in slate]
        for p in personas
    ], dtype=np.float64)  # (N, M)

    # Per-persona mean welfare across slate
    thresholds = W.mean(axis=1, keepdims=True)  # (N, 1)
    return (W >= thresholds).astype(np.float64)


# ── Variant (a): Community Notes matrix factorization ─────────────────────────

def fit_community_notes(
    R: np.ndarray,
    lambda_i: float = 0.15,
    lambda_f: float = 0.03,
    d: int = 1,
    n_iter: int = 200,
    lr: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Fit the Community Notes model r̂_un = μ + i_u + i_n + f_u·f_n.

    Args:
        R:         Binary endorsement matrix (N_personas × N_bundles), float64.
        lambda_i:  L2 penalty on intercepts (default 0.15, per production spec).
        lambda_f:  L2 penalty on latent factors (default 0.03, per production spec).
        d:         Latent factor dimension (default 1, per production spec).
        n_iter:    Gradient-descent iterations.
        lr:        Learning rate.
        seed:      RNG seed for parameter initialization.

    Returns:
        i_n (np.ndarray, shape (N_bundles,)): bundle-level intercepts.
        Higher i_n = stronger cross-faction endorsement signal.
    """
    rng = np.random.default_rng(seed)
    N, M = R.shape

    mu = float(R.mean())
    i_u = np.zeros(N, dtype=np.float64)
    i_n = np.zeros(M, dtype=np.float64)
    f_u = rng.standard_normal((N, d)) * 0.01
    f_n = rng.standard_normal((M, d)) * 0.01

    for _ in range(n_iter):
        R_hat = mu + i_u[:, None] + i_n[None, :] + (f_u @ f_n.T)  # (N, M)
        E = R_hat - R  # residuals (N, M)

        grad_mu = E.mean()
        grad_i_u = E.mean(axis=1) + lambda_i * i_u
        grad_i_n = E.mean(axis=0) + lambda_i * i_n
        grad_f_u = (E @ f_n) / M + lambda_f * f_u
        grad_f_n = (E.T @ f_u) / N + lambda_f * f_n

        mu -= lr * grad_mu
        i_u -= lr * grad_i_u
        i_n -= lr * grad_i_n
        f_u -= lr * grad_f_u
        f_n -= lr * grad_f_n

    return i_n


def condition_c_community_notes(
    slate: List[PolicyBundle],
    personas: List[Persona],
    lambda_i: float = 0.15,
    lambda_f: float = 0.03,
) -> PolicyBundle:
    """Condition C (variant a): Community Notes bridging.

    Ranks slate bundles by i_n (note intercept) from the fitted matrix
    factorization model. Ties broken lexicographically on bundle tuple.
    """
    R = build_endorsement_matrix(personas, slate)
    i_n = fit_community_notes(R, lambda_i=lambda_i, lambda_f=lambda_f)

    max_score = float(i_n.max())
    tied = [
        slate[j] for j, score in enumerate(i_n)
        if abs(score - max_score) < 1e-9
    ]
    if len(tied) == 1:
        return tied[0]
    return min(tied, key=lambda b: b.to_tuple())


# ── Variant (b): Polarity-product bridge score ────────────────────────────────

def compute_bridge_scores_polarity(
    personas: List[Persona],
    slate: List[PolicyBundle],
    use_network_communities: bool = True,
) -> np.ndarray:
    """Compute per-bundle polarity-product bridge scores (variant b).

    bridge_score(b) = (∏_{k∈communities} endorsement_rate(b, community_k))^{1/K}

    Where endorsement_rate(b, k) = fraction of community k personas who endorse b.
    The geometric mean rewards cross-community agreement; any community with zero
    endorsement kills the score entirely.

    Args:
        personas:               List of Persona objects.
        slate:                  Candidate policy bundles.
        use_network_communities: If True, use persona.network_community labels.
                                 If False, binarize by latent_ideology sign (left/right).

    Returns:
        bridge_scores: float array (N_bundles,).
    """
    M = len(slate)
    R = build_endorsement_matrix(personas, slate)  # (N, M)

    if use_network_communities:
        community_ids = [p.network_community for p in personas]
    else:
        # Simple two-cluster split by ideology sign
        community_ids = [0 if p.latent_ideology <= 0 else 1 for p in personas]

    unique_communities = sorted(set(community_ids))
    K = len(unique_communities)

    bridge_scores = np.zeros(M, dtype=np.float64)
    for j in range(M):
        log_score = 0.0
        for k in unique_communities:
            mask = [i for i, c in enumerate(community_ids) if c == k]
            if not mask:
                continue
            rate = R[mask, j].mean()
            # Use a small floor to avoid log(0); mirrors Polis's handling of silent groups
            log_score += math.log(max(rate, 1e-6))
        bridge_scores[j] = math.exp(log_score / K)

    return bridge_scores


def condition_c_polarity_product(
    slate: List[PolicyBundle],
    personas: List[Persona],
    use_network_communities: bool = True,
) -> PolicyBundle:
    """Condition C (variant b): polarity-product bridging.

    Ranks slate bundles by geometric mean of per-community endorsement rate.
    Ties broken lexicographically on bundle tuple.
    """
    scores = compute_bridge_scores_polarity(
        personas, slate, use_network_communities=use_network_communities
    )
    max_score = float(scores.max())
    tied = [slate[j] for j, s in enumerate(scores) if abs(s - max_score) < 1e-9]
    if len(tied) == 1:
        return tied[0]
    return min(tied, key=lambda b: b.to_tuple())
