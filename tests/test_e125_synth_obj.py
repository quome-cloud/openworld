import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from e125 import synth, verify

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
    # predicted next_state equals real next_state (same bg + same objects) -> level_up mismatch
    d = synth._obj_diff([(_t(S0, [4], S1, True), S1)])
    assert "level_up" in d

def test_obj_diff_bg_mismatch_not_level_up_message():
    # bg differs but objects match -> full state_key differs; must NOT emit the "level_up flag" message
    real_next = {"bg": 1, "objects": [{"color": 3, "size": 1, "y": 1, "x": 2}]}
    predicted_ns = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 2}]}
    d = synth._obj_diff([(_t(S0, [4], real_next, False), predicted_ns)])
    assert "level_up flag" not in d, "bg-only mismatch should not print level_up message"
    assert "you->" in d and "real->" in d

def test_obj_funsearch_prompt_kshot_and_failed_block():
    samples = [{"src": "def predict(state, action):\n    return state, False", "score": 0, "fails": []},
               {"src": "def predict(state, action):\n    return state, True", "score": 1, "fails": []}]
    p = synth._obj_funsearch_prompt(samples, "actions=[4]", failed=["tried Z -> scored 0"])
    assert "predict_v0" in p and "predict_v1" in p and "predict_v2" in p
    assert "tried Z -> scored 0" in p and "do not repeat" in p.lower()

def test_obj_funsearch_prompt_includes_goal_src():
    samples = [{"src": "def predict(state, action):\n    return state, False", "score": 0, "fails": []}]
    goal_src = "def goal_score(state):\n    return float(5 - state['objects'][0]['x'])"
    p = synth._obj_funsearch_prompt(samples, "actions=[4]", goal_src=goal_src)
    assert "Current goal_score():" in p
    assert goal_src in p

def test_obj_funsearch_prompt_no_goal_src_omits_block():
    samples = [{"src": "def predict(state, action):\n    return state, False", "score": 0, "fails": []}]
    p = synth._obj_funsearch_prompt(samples, "actions=[4]")
    assert "Current goal_score():" not in p


# --- synthesize_obj tests (Task 3) ---

GOODO = ("def predict(state, action):\n"
         "    ns={'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}\n"
         "    if action==[4]:\n        [o.__setitem__('x', o['x']+1) for o in ns['objects']]\n"
         "    return ns, False")
GOALO = "def goal_score(state):\n    o=state['objects'][0]\n    return float(5 - o['x'])"
S2 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 3}]}
TR2 = [_t(S0, [4], S1, False), _t(S1, [4], S2, False)]

def _runner(src, goal):
    def run(prompt, schema, model, game, **kw):
        return {"final": {"predict_src": src, "goal_score_src": goal, "rationale": "x"},
                "events": [], "tainted": False, "raw": "", "model_version": ""}
    return run

def test_synthesize_obj_accepts_and_returns_ensemble(tmp_path):
    src, fn, goal, ens = synth.synthesize_obj(TR2, "actions=[4]", "g", n_retries=1,
                                              traces_dir=str(tmp_path), _runner=_runner(GOODO, GOALO))
    assert fn is not None and callable(goal)
    ns, lu = fn(dict(S0), [4]); assert ns["objects"][0]["x"] == 2
    assert isinstance(ens, list) and len(ens) >= 1 and all(callable(f) for f in ens)

def test_synthesize_obj_rejects_numpy_predict(tmp_path):
    npp = "def predict(state, action):\n    import numpy as np\n    return state, False"
    src, fn, goal, ens = synth.synthesize_obj(TR2, "actions=[4]", "g", n_retries=1,
                                              traces_dir=str(tmp_path), _runner=_runner(npp, GOALO))
    assert fn is None and ens == []      # gate env == sandbox: a numpy predict cannot pass

def test_synthesize_obj_returns_none_when_never_passes(tmp_path):
    bad = "def predict(state, action):\n    return state, False"   # never moves -> mispredicts
    src, fn, goal, ens = synth.synthesize_obj(TR2, "actions=[4]", "g", n_retries=2,
                                              traces_dir=str(tmp_path), _runner=_runner(bad, GOALO))
    assert fn is None and ens == []


# --- ensemble verified-only test (plan fix #1) ---

# 4 transitions => split=2, held=[t2,t3] so len(held)==2
_S3 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 4}]}
_S4 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 5}]}
TR4 = [_t(S0, [4], S1, False), _t(S1, [4], S2, False),
       _t(S2, [4], _S3, False), _t(_S3, [4], _S4, False)]

# PARTIAL: moves only when x < 4 -> scores 1/2 on held (matches t2 but not t3)
_PARTIAL_SRC = (
    "def predict(state, action):\n"
    "    ns = {'bg': state['bg'], 'objects': [dict(o) for o in state['objects']]}\n"
    "    if action == [4] and ns['objects'][0]['x'] < 4:\n"
    "        ns['objects'][0]['x'] += 1\n"
    "    return ns, False"
)
# FULL: always moves -> scores 2/2 on held
_FULL_SRC = (
    "def predict(state, action):\n"
    "    ns = {'bg': state['bg'], 'objects': [dict(o) for o in state['objects']]}\n"
    "    if action == [4]:\n"
    "        ns['objects'][0]['x'] += 1\n"
    "    return ns, False"
)

def test_ensemble_verified_only():
    """After accept, every fn in the returned ensemble must pass the full held-set gate
    (score == len(held)). Sub-gate partial programs must not leak in."""
    calls = [0]
    def stateful_runner(prompt, schema, model, game, **kw):
        src = _PARTIAL_SRC if calls[0] == 0 else _FULL_SRC
        calls[0] += 1
        return {"final": {"predict_src": src, "goal_score_src": GOALO, "rationale": "r"},
                "events": [], "tainted": False, "raw": "", "model_version": ""}

    _, fn, _, ens = synth.synthesize_obj(TR4, "actions=[4]", "g", n_retries=3,
                                         _runner=stateful_runner)
    assert fn is not None, "full program should be accepted"
    # held is TR4[split:] where split = max(1, min(3, int(4*0.7))) = 2
    held = TR4[2:]
    assert len(held) == 2
    # partial must score < full on held (sanity-check the fixture)
    partial_fn = verify.compile_obj_predict(_PARTIAL_SRC)
    partial_score, _ = verify.score_obj(partial_fn, held)
    assert 0 < partial_score < len(held), f"fixture broken: partial scored {partial_score}/{len(held)}"
    # CRITICAL: no sub-gate member leaks into the ensemble
    for f in ens:
        score, _ = verify.score_obj(f, held)
        assert score == len(held), (
            f"sub-gate program leaked into ensemble (score {score}/{len(held)})"
        )
