# tests/e127/test_toy_click.py
import numpy as np
from tests.e127.toy_click import ToyClickGame, toy_click_factory, TOY_CLICK_ENGINE_SRC
from experiments.e127.safe_exec import compile_engine
from experiments.e127 import engine, perception as P

def test_reset_layout_click_only():
    g = ToyClickGame(); f = g.reset()
    assert f.shape == (8, 8)
    assert f[2, 2] == 5 and f[4, 5] == 6 and f[6, 1] == 7
    assert f[0, 0] == 1 and g.avail == [6] and g.levels == 0 and g.done is False

def test_invalid_click_is_noop():
    g = ToyClickGame(); g.reset()
    before = g.frame.copy()
    g.step(6, 0, 0)                      # empty cell -> no-op (status bar unchanged too)
    assert np.array_equal(g.frame, before)

def test_wrong_order_click_is_noop():
    g = ToyClickGame(); g.reset()
    before = g.frame.copy()
    g.step(6, 5, 4)                      # clicking B (x=5,y=4) before A -> no-op
    assert np.array_equal(g.frame, before)

def test_ordered_protocol_wins():
    g = ToyClickGame(); g.reset()
    g.step(6, 2, 2)                      # press A (x=col=2,y=row=2)
    assert g.frame[2, 2] == 3 and g.levels == 0
    g.step(6, 5, 4)                      # press B
    assert g.frame[4, 5] == 3 and g.levels == 0
    g.step(6, 1, 6)                      # press C -> level up
    assert g.levels == 1 and g.done is True

def test_targets_inferred_from_pixels():
    g = ToyClickGame(); f = g.reset()
    targets = P.infer_click_targets(f)
    assert (2, 2) in targets and (4, 5) in targets and (6, 1) in targets   # the 3 buttons (+status cell)

def test_faithful_click_engine_matches():
    factory = compile_engine(TOY_CLICK_ENGINE_SRC); assert factory is not None
    e = factory(); g = ToyClickGame()
    ef = e.reset(); gf = g.reset()
    assert np.array_equal(ef, gf)
    for (x, y) in [(0, 0), (2, 2), (3, 3), (5, 4), (5, 4), (1, 6)]:   # mix of valid/invalid clicks
        ef = e.step((6, x, y)); gf = g.step(6, x, y)
        assert np.array_equal(ef, gf), f"mismatch after click ({x},{y})"
