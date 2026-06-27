# tests/test_e124_search.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e124 import search, codex_goalc

class ToyGame:
    """Level-up only after the exact 3-step sequence [1],[1],[2]. Single-step BFS to depth 3 over 3 actions
    is 27 nodes; with budget 5 it cannot reach it, but the macro [[1],[1],[2]] solves in one option."""
    WIN = [(1,), (1,), (2,)]
    def __init__(self): self.reset()
    def reset(self): self.seq = []; self.levels = 0; self.done = False; self.frame = np.zeros((64,64),dtype=int)
    def step(self, a, x=None, y=None):
        self.seq.append((a,) if x is None else (6,x,y))
        if self.seq == self.WIN: self.levels = 1; self.done = True
        if len(self.seq) > 6: self.done = True
    def clone_actions(self): return list(self.seq)

def _cands(frame): return [[1],[2],[3]]

def test_macro_solves_what_blind_cannot_in_budget():
    g = ToyGame()
    macro_goal = codex_goalc.Goal([], [[[1],[1],[2]]], None, "", False, [])
    assert search.run(ToyGame(), codex_goalc.Goal([],[],None,"",False,[]), budget=5,
                      rung="blind", candidates_fn=_cands, mask=None) is None
    out = search.run(ToyGame(), macro_goal, budget=5, rung="blind_macros", candidates_fn=_cands, mask=None)
    assert out == [[1],[1],[2]]
