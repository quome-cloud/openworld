import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from e125 import simworld, verify

# object world: action [4] moves the single object +1 in x (clamped 10); win when x==10.
PRED = ("def predict(state, action):\n"
        "    ns={'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}\n"
        "    if action==[4]:\n        o=ns['objects'][0]; o['x']=min(10,o['x']+1)\n"
        "    return ns, bool(ns['objects'][0]['x']==10)")
GOAL = "def goal_score(state):\n    return float(10 - state['objects'][0]['x'])"
S0 = {"bg":0, "objects":[{"color":3,"size":1,"y":1,"x":1}]}
fn = verify.compile_predict(PRED); goal = verify.compile_goal(GOAL)

def test_plan_obj_finds_win_via_energy_descent():
    plan = simworld.plan_obj(fn, S0, lambda s:[[4],[2],[3]], budget=200, goal_fn=goal)
    assert plan == [[4]]*9

def test_plan_obj_dedups_noops_and_returns_none_when_unreachable():
    stay = verify.compile_predict("def predict(state, action):\n    return {'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}, False")
    assert simworld.plan_obj(stay, S0, lambda s:[[2],[3]], budget=50, goal_fn=goal) is None
