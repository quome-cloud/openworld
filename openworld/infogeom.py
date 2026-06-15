"""Information geometry for world identification: probe by expected information.

The candidate worlds form a statistical manifold; the right way to choose the next
probe is the one whose outcome carries the most information about which world is
true. That is the expected information gain (mutual information between the probe's
outcome and the world identity under the current posterior) - the information-
geometric / optimal-experiment-design criterion. It adapts: as the posterior
concentrates, it picks probes that separate the worlds still in contention, rather
than re-confirming what is already known.

This sharpens the active-induction story (E43) and the many-worlds posterior
(E46): instead of a heuristic (split the version space by count), choose probes by
information, and update the posterior by Bayes. Numpy-only, deterministic.

  Q[probe] : a [n_worlds, n_outcomes] matrix of outcome distributions.
"""

from __future__ import annotations

import numpy as np


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return float(np.sum(p[m] * np.log2(p[m] / np.maximum(q[m], 1e-300))))


def expected_info_gain(posterior: np.ndarray, Q: np.ndarray) -> float:
    """Mutual information I(outcome; world) for a probe under the posterior:
    H(predictive) - E_world[H(outcome | world)]."""
    posterior = np.asarray(posterior, float)
    predictive = posterior @ Q                       # marginal outcome distribution
    h_cond = sum(posterior[t] * entropy(Q[t]) for t in range(len(posterior)))
    return entropy(predictive) - h_cond


def bayes_update(posterior: np.ndarray, Q: np.ndarray, outcome: int) -> np.ndarray:
    post = np.asarray(posterior, float) * Q[:, outcome]
    s = post.sum()
    return post / s if s > 0 else posterior


def fisher_information(Q: np.ndarray, posterior: np.ndarray) -> float:
    """A scalar Fisher-style sensitivity of a probe: posterior-weighted spread of
    its outcome distributions across worlds (how sharply outcomes change with the
    world). Higher = more locally informative."""
    mean = posterior @ Q
    return float(sum(posterior[t] * np.sum((Q[t] - mean) ** 2 / np.maximum(mean, 1e-9))
                     for t in range(len(posterior))))
