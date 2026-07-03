# tests/e127/test_engine.py
from experiments.e127 import engine
from experiments.e127.safe_exec import compile_engine
from tests.e127.toy import ToyGame, TOY_ENGINE_SRC, TOY_WRONG_SRC

def _acts(seq):
    return [(a, None, None) for a in seq]

def test_play_then_score_faithful_is_perfect():
    g = ToyGame()
    ep = engine.play(g, _acts([1, 1, 1, 3, 3, 3, 4, 4]))
    assert ep[0]["action"] is None and ep[0]["frame"].shape == (8, 8)
    factory = compile_engine(TOY_ENGINE_SRC)
    s = engine.score_rollout(factory, ep)
    assert s["errored"] is False
    assert s["transitions"] == 8
    assert s["exact"] == 8 and s["cell_acc"] == 1.0

def test_score_wrong_engine_imperfect():
    g = ToyGame()
    ep = engine.play(g, _acts([1, 1, 1, 3, 3, 3, 7, 7]))   # collects gem (1,1); wrong engine won't
    factory = compile_engine(TOY_WRONG_SRC)
    s = engine.score_rollout(factory, ep)
    assert s["exact"] < s["transitions"]                    # diverges once gem is collected

def test_levelup_accounting():
    g = ToyGame()
    seq = [1, 1, 1, 3, 3, 3, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 3, 3, 3]   # full clear -> 1 levelup
    ep = engine.play(g, _acts(seq))
    assert ep[-1]["levels"] == 1
    faithful = compile_engine(TOY_ENGINE_SRC)
    s = engine.score_rollout(faithful, ep)
    assert s["levelup_total"] == 1 and s["levelup_match"] == 1

def test_rollout_runtime_fault_is_errored_not_raised_in_score():
    bad = "class Engine:\n    def reset(self): return np.zeros((8,8),dtype=int)\n    def step(self, a): raise ValueError('boom')\n    def is_win(self, p): return False\n"
    factory = compile_engine(bad)
    g = ToyGame(); ep = engine.play(g, _acts([7, 7]))
    s = engine.score_rollout(factory, ep)
    assert s["errored"] is True and s["exact"] == 0

def test_identity_mask_flags_status_bar_only():
    g = ToyGame()
    ep = engine.play(g, _acts([7] * 30))     # noop: only the status bar (0,0) changes every step
    mask = engine.identity_mask([ep], thr=0.95)
    assert mask.shape == (8, 8)
    assert mask[0, 0] == True
    assert mask.sum() == 1                    # nothing else changes under pure noop

def test_first_disagreement():
    a = compile_engine(TOY_ENGINE_SRC); b = compile_engine(TOY_WRONG_SRC)
    acts = _acts([1, 1, 1, 3, 3, 3, 7])       # gem collected at the 6th action (index 5)
    idx = engine.first_disagreement(a, b, acts)
    assert idx is not None and idx == 5
    same = engine.first_disagreement(a, a, acts)
    assert same is None

def test_lookup_table_heuristic():
    # A genuine memorized frame-table dump (tens of thousands of int literals) is flagged...
    big = "class Engine:\n    TABLE = {" + ",".join(f"{i}:{i}" for i in range(16000)) + "}\n"
    assert engine.looks_like_lookup_table(big) is True
    # ...but a legitimate engine -- even one embedding a full literal 64x64 board (~4096 ints) -- is NOT
    # (this is the dc22 false-reject the old threshold caused).
    board = "class Engine:\n    B = [" + ",".join(str(i % 16) for i in range(4096)) + "]\n"
    assert engine.looks_like_lookup_table(board) is False
    assert engine.looks_like_lookup_table(TOY_ENGINE_SRC) is False
