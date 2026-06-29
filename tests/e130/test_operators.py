import numpy as np
from experiments.e130 import operators as op


def test_tension_is_norm_of_difference():
    a = np.array([1.0, 0.0]); b = np.array([0.0, 0.0])
    assert abs(op.tension(a, b) - 1.0) < 1e-9
    assert op.tension(a, a) == 0.0


def test_cycle_converges_geometrically_to_theta_star():
    # Theorem 4.6: iterating the cycle map contracts tension to 0 at rate rho<1,
    # and the fixed point is (theta_star, theta_star).
    rng = np.random.default_rng(0)
    d = 5
    theta = rng.normal(size=d)
    sI = rng.normal(size=d); sE = rng.normal(size=d)
    alpha, gamma = 0.5, 0.5
    r = op.rho(alpha, gamma)
    assert r < 1.0
    prev_err = np.linalg.norm(sI - theta) + np.linalg.norm(sE - theta)
    for _ in range(200):
        sI, sE = op.cycle_map(sI, sE, theta, alpha, gamma)
        err = np.linalg.norm(sI - theta) + np.linalg.norm(sE - theta)
        # each step shrinks the error by at most rho (+ small slack for the
        # block-triangular off-diagonal coupling)
        assert err <= prev_err * (r + 0.25) + 1e-9
        prev_err = err
    assert prev_err < 1e-6           # converged to the diagonal at theta_star
    assert op.tension(sI, sE) < 1e-6


def test_I_gamma_reduces_tension_each_application():
    # consolidating the reading into the model strictly reduces tension when gamma in (0,1)
    sI = np.array([2.0, 0.0]); sE = np.array([0.0, 0.0])
    t0 = op.tension(sI, sE)
    sI2 = op.I_gamma(sI, sE, 0.5)
    assert op.tension(sI2, sE) < t0
