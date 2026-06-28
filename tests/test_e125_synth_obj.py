import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from e125 import synth

S0 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 1}]}
S1 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 2}]}
def _t(s, a, ns, lu): return {"state": s, "action": a, "next_state": ns, "level_up": lu}
TR = [_t(S0, [4], S1, False)]

def test_render_obj_transitions_shows_objects_and_action():
    out = synth.render_obj_transitions(TR)
    assert "action=[4]" in out and "c3" in out and "x1" in out and "x2" in out

def test_obj_prompt_states_object_contract_and_goal():
    p = synth._obj_prompt(TR, "actions=[1,2,3,4]")
    assert "predict(state, action)" in p and "next_state" in p
    assert "goal_score" in p and "{predict_src, goal_score_src, rationale}" in p

def test_obj_diff_lists_mispredicted_objects():
    bad_next = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 9}]}   # wrong x
    d = synth._obj_diff([(TR[0], bad_next)])
    assert "action=[4]" in d and "you->" in d and "real->" in d

def test_obj_diff_level_up_flag_wrong():
    # predicted next_state equals real next_state (same objects), so it's a level_up mismatch
    d = synth._obj_diff([(_t(S0, [4], S1, True), S1)])
    assert "level_up" in d

def test_obj_funsearch_prompt_kshot_and_failed_block():
    samples = [{"src": "def predict(state, action):\n    return state, False", "score": 0, "fails": []},
               {"src": "def predict(state, action):\n    return state, True", "score": 1, "fails": []}]
    p = synth._obj_funsearch_prompt(samples, "actions=[4]", failed=["tried Z -> scored 0"])
    assert "predict_v0" in p and "predict_v1" in p and "predict_v2" in p
    assert "tried Z -> scored 0" in p and "do not repeat" in p.lower()
