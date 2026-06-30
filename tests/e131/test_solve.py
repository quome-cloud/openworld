import numpy as np
from experiments.e131.lookahead import solve_lookahead
from experiments.e130 import perception as P
from tests.e131.test_lookahead import TwoStepGame


def test_solve_chains_to_the_win():
    g = TwoStepGame()
    res = solve_lookahead(g, lambda fr: P.extrospect(fr, avail=[1, 2, 3]),
                          seed_actions=[], win=1, depth=2, beam=4, budget=50)
    assert res.best_levels == 1
    assert res.best_actions[:2] == [[1], [2]]      # discovered + committed the ordered pair
