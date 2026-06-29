import json, numpy as np
from e119 import slm
from openworld import MockLLM   # framework's scripted BaseLLM — bind to OpenWorld, no bespoke stub


def test_llm_options_pins_gemma_differently_from_qwen():
    q = slm.llm_options("qwen2.5-coder:7b")
    gm = slm.llm_options("gemma2:9b")
    assert q["temperature"] == 0.7 and q["top_k"] == 20
    assert gm["temperature"] == 1.0 and gm["top_k"] == 64   # Gemma defaults differ


def test_compile_and_satisfiable_reach_color():
    pred = {"type": "reach", "color": 5}
    f_no = np.zeros((64, 64), int)
    f_yes = np.zeros((64, 64), int); f_yes[10, 10] = 5
    assert slm.satisfiable(pred, [f_no, f_yes]) is True
    assert slm.satisfiable(pred, [f_no]) is False


def test_propose_subgoal_votes_and_returns_predicate():
    frames = [np.zeros((64, 64), int)]
    frames[0][2, 2] = 5
    oj = {"objects": [{"id": 0, "color": 5}], "relations": []}
    replies = [json.dumps({"type": "reach", "color": 5})] * 4
    llm = MockLLM(replies)
    pred = slm.propose_subgoal(llm, oj, frames, n=4, tau=0.5)
    assert pred == {"type": "reach", "color": 5}


def test_propose_subgoal_abstains_on_disagreement():
    frames = [np.zeros((64, 64), int)]
    oj = {"objects": [], "relations": []}
    replies = [json.dumps({"type": "reach", "color": c}) for c in (1, 2, 3, 4)]
    llm = MockLLM(replies)
    pred = slm.propose_subgoal(llm, oj, frames, n=4, tau=0.6)
    assert pred is None
