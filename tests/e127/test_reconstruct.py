"""Tests for the differential-CEGIS reconstruction loop (T9 keystone).

The REAL env is the convergence oracle; the second model is a diversity source, NEVER the
acceptance gate. Three directional tests (from the brief) plus a CLICK-game convergence test prove
the loop is all-modality (works on directional AND mouse games)."""
import numpy as np
from experiments.e127 import reconstruct
from tests.e127.toy import toy_factory, TOY_ENGINE_SRC, TOY_WRONG_SRC, ACTION_API
from tests.e127.toy_click import toy_click_factory, TOY_CLICK_ENGINE_SRC, CLICK_ACTION_API


# A PLAUSIBLE-BUT-WRONG click reconstruction: renders the 3 buttons in their unpressed colors but
# NEVER advances phase on a press (analogous to TOY_WRONG_SRC). Diverges from the real env the moment
# the correct next button is pressed (real recolors it to 3 and bumps the status bar).
TOY_CLICK_WRONG_SRC = '''
_BTN = [((2, 2), 5), ((4, 5), 6), ((6, 1), 7)]
class Engine:
    def __init__(self):
        self.state = {"levels": 0, "phase": 0, "t": 0, "done": False}
    def _draw(self):
        f = np.zeros((8, 8), dtype=int)
        for i, ((ry, cx), col) in enumerate(_BTN):
            f[ry, cx] = col
        f[0, 0] = (self.state["t"] % 15) + 1
        return f
    def reset(self):
        self.state = {"levels": 0, "phase": 0, "t": 0, "done": False}
        return self._draw()
    def step(self, action):
        return self._draw()
    def is_win(self, prev_frame):
        return False
'''


def test_converges_to_certified_engine_with_fake_runners():
    # Model A: wrong at round 0, then authors the faithful engine once it sees counterexamples.
    def runner_a(prompt, round_idx):
        return TOY_WRONG_SRC if round_idx == 0 else TOY_ENGINE_SRC
    # Model B: persistently authors the wrong engine (a diversity source that stays wrong).
    def runner_b(prompt, round_idx):
        return TOY_WRONG_SRC
    res = reconstruct.reconstruct(toy_factory, ACTION_API, n_levels=1,
                                  max_rounds=4, _runners=[runner_a, runner_b], seed=0)
    assert res["certificate"]["pass"] is True            # champion certified against the REAL env
    assert res["champion_acc"] >= 0.99
    assert res["engine_src"] is not None
    # The champion is the faithful engine (A == reality). On the COMMON holdout, ab_agreement(A,B)
    # identically equals acc_B_vs_real, so the gap is exactly 0 (to float tolerance) -- the clean
    # baseline: no shared-prior bias beyond reality.
    assert abs(res["ab_vs_real_gap"]) < 0.05
    assert res["real_steps"] > 0


def test_no_false_unity_when_both_models_share_a_wrong_engine():
    # Both models agree on the SAME wrong engine (folie a deux). Agreement is high, but the
    # certificate must FAIL because neither matches the real env.
    def wrong_runner(prompt, round_idx):
        return TOY_WRONG_SRC
    res = reconstruct.reconstruct(toy_factory, ACTION_API, n_levels=1,
                                  max_rounds=3, _runners=[wrong_runner, wrong_runner], seed=0)
    assert res["certificate"]["pass"] is False           # agreement != correctness
    assert res["ab_agreement"] >= 0.99                    # they DO agree with each other
    assert res["ab_vs_real_gap"] > 0.0                    # ... but not with reality -> positive gap


def test_budget_is_respected():
    def runner(prompt, round_idx):
        return TOY_WRONG_SRC
    res = reconstruct.reconstruct(toy_factory, ACTION_API, n_levels=1, max_rounds=3,
                                  budget={"limit": 50, "used": 0}, _runners=[runner, runner], seed=0)
    assert res["real_steps"] <= 50 + 8                    # small batch overshoot tolerance


def test_converges_on_click_game():
    # ALL-MODALITY: the SAME loop must reconstruct a MOUSE game. A: wrong then faithful click engine.
    def runner_a(prompt, round_idx):
        return TOY_CLICK_WRONG_SRC if round_idx == 0 else TOY_CLICK_ENGINE_SRC
    # B: persistently the wrong click engine (renders buttons, never advances on a press).
    def runner_b(prompt, round_idx):
        return TOY_CLICK_WRONG_SRC
    res = reconstruct.reconstruct(toy_click_factory, CLICK_ACTION_API, n_levels=1,
                                  max_rounds=4, _runners=[runner_a, runner_b], seed=0)
    assert res["certificate"]["pass"] is True             # champion = faithful click engine, vs REAL
    assert res["champion_acc"] >= 0.99
    assert res["engine_src"] is not None
    # Faithful click champion (A == reality) -> common-holdout gap is ~0, the clean baseline.
    assert abs(res["ab_vs_real_gap"]) < 0.05
    assert res["real_steps"] > 0
