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


def test_ladder_blind_fails_macros_solves():
    """Offline ladder-dispatch test: run_one with an injected macro goal.
    blind (no macros, budget 5) cannot reach the 3-step win; blind_macros solves it."""
    import e124_autonomous_search as e
    goal = codex_goalc.Goal([], [[[1],[1],[2]]], None, "", False, [])
    res = e.run_one(ToyGame, _cands, None, 5, goal)
    assert res["blind"] is None
    assert res["blind_macros"] == 3


class ToyGame2:
    """A DEEP procedure: level-up only after [1,1,1,2,2,2] (depth 6). frame[0,0] records the count of leading
    1s, so a subgoal predicate `frame[0,0] >= 3` fires at the half-way point. Single-step BFS over {1,2} to
    depth 6 needs ~126 nodes; with budget 30 it cannot reach it. But ORDERED subgoals split it into two
    depth-3 searches (reach the half-way subgoal, then the win) -- well within budget."""
    WIN = [(1,), (1,), (1,), (2,), (2,), (2,)]
    def __init__(self): self.reset()
    def reset(self):
        self.seq = []; self.levels = 0; self.done = False; self.frame = np.zeros((64, 64), dtype=int)
    def step(self, a, x=None, y=None):
        self.seq.append((a,) if x is None else (6, x, y))
        c = 0
        for s in self.seq:
            if s == (1,):
                c += 1
            else:
                break
        self.frame = np.zeros((64, 64), dtype=int); self.frame[0, 0] = c
        if self.seq == self.WIN:
            self.levels = 1; self.done = True
        if len(self.seq) > 8:
            self.done = True

def _cands2(frame): return [[1], [2]]

def test_subgoals_solve_what_blind_cannot():
    """The core Task-6b claim: codex's ordered subgoals collapse a deep procedure that blind BFS cannot
    crack in the same budget."""
    half = "def predicate(frame):\n    return frame[0, 0] >= 3"
    sub_goal = codex_goalc.Goal([("half", half)], [], None, "", False, [])
    blind = search.run(ToyGame2(), codex_goalc.Goal([], [], None, "", True, []),
                       budget=30, rung="blind", candidates_fn=_cands2, mask=None)
    sub = search.run(ToyGame2(), sub_goal, budget=30, rung="subgoals", candidates_fn=_cands2, mask=None)
    assert blind is None
    assert sub == [[1], [1], [1], [2], [2], [2]]


def test_interactive_receding_horizon_solves():
    """The interactive MPC loop: a planner that proposes a 3-step maneuver per round solves ToyGame2's
    depth-6 procedure in 2 grounded rounds, where myopic single-step greedy could stall."""
    from e124 import interactive
    def planner(frame, history, rnd, horizon):
        # round 0: reach the half-way subgoal; round 1: finish. (A mock; live uses codex.plan_ahead.)
        return [[[1],[1],[1]]] if rnd == 0 else [[[2],[2],[2]]]
    sol = interactive.solve_interactive(ToyGame2, planner, None, max_rounds=5)
    assert sol == [[1],[1],[1],[2],[2],[2]]

def test_interactive_returns_none_when_stuck():
    from e124 import interactive
    def planner(frame, history, rnd, horizon):
        return [[[2]]]   # never makes progress toward the [1,1,1,...] win; no novel state after first
    sol = interactive.solve_interactive(ToyGame2, planner, None, max_rounds=4)
    assert sol is None
