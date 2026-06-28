import numpy as np
from tests.e127.toy import ToyGame, toy_factory, TOY_ENGINE_SRC, TOY_WRONG_SRC, TOY_ACTIONS
from experiments.e127.safe_exec import compile_engine

def test_reset_layout():
    g = ToyGame(); f = g.reset()
    assert f.shape == (8, 8)
    assert f[4, 4] == 8                       # cursor
    assert f[1, 1] == 4 and f[1, 6] == 4 and f[6, 3] == 4   # gems
    assert f[0, 0] == 1                        # status bar t=0 -> (0%15)+1
    assert g.levels == 0 and g.win == 1 and g.done is False

def test_status_bar_changes_every_step():
    g = ToyGame(); g.reset()
    g.step(7); assert g.frame[0, 0] == 2
    g.step(7); assert g.frame[0, 0] == 3

def test_cursor_moves_and_clamps():
    g = ToyGame(); g.reset()
    g.step(1); assert g.frame[4, 4] == 0 and g.frame[3, 4] == 8   # moved up
    for _ in range(10):
        g.step(1)
    assert np.argwhere(g.frame == 8)[0][0] == 1                   # clamped at row 1

def test_collection_and_levelup_is_procedural():
    g = ToyGame(); g.reset()
    # path cursor (4,4) -> collect (1,1): up x3 to row1, left x3 to col1
    for a in (1, 1, 1, 3, 3, 3):
        g.step(a)
    assert g.levels == 0                       # only 1 gem collected, no level yet
    # collect (1,6): right x5 to col6
    for a in (4, 4, 4, 4, 4):
        g.step(a)
    # collect (6,3): down x5 to row6, then to col3 (already? col6->col3 left x3)
    for a in (2, 2, 2, 2, 2, 3, 3, 3):
        g.step(a)
    assert g.levels == 1 and g.done is True    # third gem -> level up, game done

def test_determinism():
    g1, g2 = ToyGame(), ToyGame()
    g1.reset(); g2.reset()
    seq = [1, 3, 7, 4, 2, 5]
    for a in seq:
        g1.step(a); g2.step(a)
    assert np.array_equal(g1.frame, g2.frame)

def test_faithful_engine_matches_toygame():
    # The reference reconstruction reproduces ToyGame frame-for-frame over a long sequence.
    factory = compile_engine(TOY_ENGINE_SRC); assert factory is not None
    e = factory(); g = ToyGame()
    ef = e.reset(); gf = g.reset()
    assert np.array_equal(ef, gf)
    rng = np.random.default_rng(0)
    for _ in range(60):
        a = int(rng.choice(TOY_ACTIONS))
        ef = e.step((a, None, None)); gf = g.step(a)
        assert np.array_equal(ef, gf), f"mismatch at action {a}"

def test_wrong_engine_diverges_from_toygame():
    factory = compile_engine(TOY_WRONG_SRC); assert factory is not None
    e = factory(); g = ToyGame()
    e.reset(); g.reset()
    diverged = False
    for a in (1, 1, 1, 3, 3, 3, 7, 7):
        ef = e.step((a, None, None)); gf = g.step(a)
        if not np.array_equal(ef, gf):
            diverged = True
    assert diverged
