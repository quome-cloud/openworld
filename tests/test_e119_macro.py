import json, random, numpy as np
from openworld import MockLLM
from e119 import macro, slm


OJ = {"bg": 0, "objects": [{"id": 0, "color": 5, "centroid": (10, 20)},
                           {"id": 1, "color": 3, "centroid": (40, 8)}], "relations": []}


def test_compile_directional_and_repeat():
    assert macro.compile_macro(["a1", "a3 x2"], OJ, [1, 2, 3, 4]) == [(1,), (3,), (3,)]


def test_compile_click_resolves_object_centroid():
    # centroid is (row, col) = (y, x); click tuple is (6, x=col, y=row)
    assert macro.compile_macro(["click #0"], OJ, [6]) == [(6, 20, 10)]


def test_compile_drops_unresolvable_ops():
    # a5 not in avail -> dropped; click when 6 not in avail -> dropped; missing obj -> dropped
    assert macro.compile_macro(["a5", "a2", "click #9"], OJ, [1, 2, 3, 4]) == [(2,)]
    assert macro.compile_macro(["click #0"], OJ, [1, 2]) == []


def test_compile_caps_repeat_at_four():
    assert macro.compile_macro(["a1 x99"], OJ, [1]) == [(1,), (1,), (1,), (1,)]


class StepGame:
    """Each action 7 advances pos by 1 (deterministic). frame[0,pos]=4. No reward path here.
    Mirrors the Game/_PrefixGame surface propose_macros replays against."""
    def __init__(self): self.win = 1; self.gid = "step"; self.reset()
    def reset(self): self.pos = 0; self.levels = 0; self.done = False; self.avail = [7, 1]; self._r(); return self.frame
    def _r(self): g = np.zeros((64, 64), int); g[0, self.pos] = 4; self.frame = g
    def step(self, a, x=None, y=None):
        if a == 7 and self.pos < 63: self.pos += 1
        if a == 1 and self.pos > 0: self.pos -= 1
        self._r(); return self.frame


def _key(f): return int(np.asarray(f).reshape(64, 64)[0].argmax())


def test_propose_macros_returns_consensus_macro():
    # 4 of 6 samples agree on ["a7","a7"] (-> pos 2); they cluster, clear tau=0.5, and survive.
    replies = [json.dumps(["a7", "a7"])] * 4 + [json.dumps(["a1"]), json.dumps(["a7", "a7", "a7"])]
    llm = MockLLM(replies)
    macros = macro.propose_macros(llm, StepGame(), [], {"objects": []}, [], None,
                                  avail=[7, 1], key_fn=_key, k_max=8, n=6, tau=0.5)
    assert [(7,), (7,)] in macros            # the consensus macro survived
    assert macros[0] == [(7,), (7,)]         # ranked first by cluster mass


def test_propose_macros_abstains_on_disagreement():
    replies = [json.dumps(["a7"]), json.dumps(["a7", "a7"]), json.dumps(["a1"]),
               json.dumps(["a7", "a7", "a7"]), json.dumps(["a1", "a1"]), json.dumps(["a7", "a1"])]
    llm = MockLLM(replies)
    macros = macro.propose_macros(llm, StepGame(), [], {"objects": []}, [], None,
                                  avail=[7, 1], key_fn=_key, k_max=8, n=6, tau=0.6)
    assert macros == []                       # no cluster clears tau -> abstain


def test_rank_macros_subgoal_satisfier_first():
    g = StepGame()
    # subgoal: reach color 5. StepGame never produces color 5, so make a game variant:
    class ColorAtThree(StepGame):
        def _r(self):
            v = 5 if self.pos == 3 else 4
            x = np.zeros((64, 64), int); x[0, self.pos] = v; self.frame = x
    sub = {"type": "reach", "color": 5}
    m_far = [(7,), (7,), (7,)]    # reaches pos 3 -> color 5 -> satisfies subgoal
    m_near = [(7,)]               # reaches pos 1 -> color 4 -> does not
    ranked = macro.rank_macros([m_near, m_far], ColorAtThree(), [], sub, _key, seen=set())
    assert ranked[0] == m_far     # subgoal-satisfier ranked first


def test_random_macros_seeded_and_bounded():
    rng1 = random.Random(0); rng2 = random.Random(0)
    a = macro.propose_random_macros([1, 2, 3, 4], {"objects": []}, k_max=8, count=5, rng=rng1)
    b = macro.propose_random_macros([1, 2, 3, 4], {"objects": []}, k_max=8, count=5, rng=rng2)
    assert a == b                                 # same seed -> identical
    assert len(a) == 5
    assert all(2 <= len(m) <= 8 for m in a)       # length bounds
    assert all(act[0] in (1, 2, 3, 4) for m in a for act in m)   # only avail directional actions
