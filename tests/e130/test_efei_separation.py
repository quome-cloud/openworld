import numpy as np
from experiments.e130 import efei


def test_expert_error_decreases_as_one_over_sqrt_N():
    rng = np.random.default_rng(1)
    theta = rng.normal(size=8)
    e1   = np.mean([efei.expert_error(theta, 1,   n=50, tau=1.0, beta=0.0, d=8, rng=rng) for _ in range(40)])
    e100 = np.mean([efei.expert_error(theta, 100, n=50, tau=1.0, beta=0.0, d=8, rng=rng) for _ in range(40)])
    assert e100 < e1 / 5.0                       # ~1/sqrt(100) = 1/10 the error


def test_amateur_cost_is_theta_of_M():
    rng = np.random.default_rng(2)
    for M in (10, 40, 160):
        mean_trials = np.mean([efei.amateur_trials(M, rng) for _ in range(400)])
        assert 0.6 * M <= mean_trials <= 1.5 * M  # geometric mean = M


def test_separation_expert_consultations_independent_of_M():
    # Expert reaches accuracy delta in N independent of pool size M; amateur needs ~M.
    N = efei.expert_consultations_for(delta=0.2, n=50, tau=1.0, beta=0.0, d=8)
    assert N < 80
    rng = np.random.default_rng(3)
    for M in (200, 2000):                          # amateur cost grows with M, expert N does not
        amateur = np.mean([efei.amateur_trials(M, rng) for _ in range(200)])
        assert amateur > 3 * N
