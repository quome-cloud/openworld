import math
import numpy as np
from experiments.e127 import certify, engine
from experiments.e127.safe_exec import compile_engine
from tests.e127.toy import ToyGame, TOY_ENGINE_SRC, TOY_WRONG_SRC

def _acts(seq):
    return [(a, None, None) for a in seq]

def test_betai_known_values():
    # I_x(1,1) = x ; symmetry I_x(a,b) = 1 - I_{1-x}(b,a)
    assert abs(certify.betai(1, 1, 0.37) - 0.37) < 1e-9
    assert abs(certify.betai(2, 3, 0.5) - (1 - certify.betai(3, 2, 0.5))) < 1e-9

def test_clopper_pearson_closed_form_k_equals_n():
    # For k=n successes, CP lower bound = delta**(1/n)
    for n, delta in [(10, 0.05), (50, 0.05), (300, 0.05)]:
        assert abs(certify.clopper_pearson_lower(n, n, delta) - delta ** (1.0 / n)) < 1e-6

def test_clopper_pearson_monotone():
    a = certify.clopper_pearson_lower(95, 100, 0.05)
    b = certify.clopper_pearson_lower(99, 100, 0.05)
    assert 0.0 < a < b < 1.0

def _holdout(n_eps=40, seed=1):
    rng = np.random.default_rng(seed)
    eps = []
    for _ in range(n_eps):
        g = ToyGame()
        k = int(rng.integers(6, 20))
        seq = [int(rng.choice([1, 2, 3, 4, 5, 7])) for _ in range(k)]
        eps.append(engine.play(g, _acts(seq)))
    return eps

def test_faithful_engine_certifies():
    factory = compile_engine(TOY_ENGINE_SRC)
    cert = certify.certify_engine(factory, _holdout(), n_levels=1, eps=0.01, delta=0.05, coverage_target=0.0)
    assert cert["errored"] is False
    assert cert["exact"] == cert["n"]          # perfect reproduction
    assert cert["acc_lower"] >= 0.99
    assert cert["pass"] is True

def test_wrong_engine_fails_certificate():
    factory = compile_engine(TOY_WRONG_SRC)
    cert = certify.certify_engine(factory, _holdout(), n_levels=1, eps=0.01, delta=0.05, coverage_target=0.0)
    assert cert["pass"] is False
    assert cert["acc"] < 1.0
