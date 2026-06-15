"""Tests for information-geometric world identification."""

import numpy as np

from openworld import bayes_update, expected_info_gain, fisher_information
from openworld.infogeom import entropy, kl


def test_entropy_and_kl_basics():
    assert abs(entropy([0.5, 0.5]) - 1.0) < 1e-9          # one bit
    assert abs(entropy([1.0, 0.0])) < 1e-9                # certainty -> 0
    assert kl([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert kl([0.5, 0.5], [0.9, 0.1]) > 0.0


def test_eig_prefers_separating_probe():
    posterior = np.array([0.5, 0.5])
    # an uninformative probe: both worlds give the same outcome distribution
    flat = np.array([[0.5, 0.5], [0.5, 0.5]])
    # an informative probe: the worlds give opposite outcomes
    split = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_info_gain(posterior, flat) < 1e-9
    assert expected_info_gain(posterior, split) > 0.9


def test_bayes_update_concentrates_on_truth():
    posterior = np.full(3, 1 / 3)
    Q = np.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]])
    # repeatedly observe outcome 0, which world 0 produces most often
    for _ in range(10):
        posterior = bayes_update(posterior, Q, 0)
    assert int(np.argmax(posterior)) == 0
    assert posterior[0] > 0.9


def test_fisher_information_nonnegative_and_discriminates():
    posterior = np.array([0.5, 0.5])
    flat = np.array([[0.5, 0.5], [0.5, 0.5]])
    split = np.array([[0.9, 0.1], [0.1, 0.9]])
    assert fisher_information(flat, posterior) >= 0.0
    assert fisher_information(split, posterior) > fisher_information(flat, posterior)
