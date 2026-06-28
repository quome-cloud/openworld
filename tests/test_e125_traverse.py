import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import traverse, verify, objstate

class RealObjGame:
    def __init__(self): self.reset()
    def reset(self): self.x=1; self.levels=0; self.done=False; self._draw()
    def _draw(self):
        self.frame=np.zeros((8,8),dtype=int); self.frame[1,self.x]=3
    def step(self,a,x=None,y=None):
        if a==4: self.x=min(7,self.x+1)
        self._draw()
        if self.x==4: self.levels=1; self.done=True

PRED=("def predict(state, action):\n    ns={'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}\n"
      "    if action==[4]:\n        o=ns['objects'][0]; o['x']=o['x']+1\n    return ns, bool(ns['objects'][0]['x']==4)")
GOAL="def goal_score(state):\n    return float(4 - state['objects'][0]['x'])"
fn=verify.compile_predict(PRED); goal=verify.compile_goal(GOAL)
WM={"predict_src":PRED,"predict_fn":fn,"goal_src":GOAL,"goal_fn":goal,"ensemble":[fn]}
perc=lambda f: objstate.object_state(f)

def test_traverse_solves_via_imagination_plan():
    # ensemble agrees (single fn), plan_obj finds [4]*3 to x==4 -> executes verified plan -> real level-up
    r = traverse.traverse_level(RealObjGame, lambda s:[[4],[2]], WM, "actions=[2,4]", "g",
                                perceive=perc, budget_plan=200)
    assert r["solved"] is True and r["actions"] == [[4],[4],[4]] and r["macros_used"] == 0

def test_traverse_uses_macro_fallback_when_no_plan():
    # goal_fn=None + a predict whose level_up NEVER fires -> no imagination plan -> macro fallback solves
    nowin=verify.compile_predict("def predict(state, action):\n    ns={'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}\n    if action==[4]:\n        ns['objects'][0]['x']=ns['objects'][0]['x']+1\n    return ns, False")
    wm2={"predict_src":"src","predict_fn":nowin,"goal_src":None,"goal_fn":None,"ensemble":[nowin]}
    def macro_runner(prompt, schema, model, game, **kw):
        return {"final":{"macro":[[4],[4],[4]],"rationale":"go right","goal_note":"x->4"},
                "events":[],"tainted":False,"raw":"","model_version":""}
    r = traverse.traverse_level(RealObjGame, lambda s:[[4],[2]], wm2, "actions=[2,4]", "g",
                                macro_runner=macro_runner, perceive=perc, budget_plan=50)
    assert r["solved"] is True and r["macros_used"] >= 1

def test_traverse_abandons_without_banked_answers():
    nowin=verify.compile_predict("def predict(state, action):\n    return {'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}, False")
    wm2={"predict_src":"s","predict_fn":nowin,"goal_src":None,"goal_fn":None,"ensemble":[nowin]}
    def macro_runner(prompt, schema, model, game, **kw):
        return {"final":{"macro":[[2]],"rationale":"x","goal_note":"x"},"events":[],"tainted":False,"raw":"","model_version":""}
    r = traverse.traverse_level(RealObjGame, lambda s:[[2]], wm2, "actions=[2]", "g",
                                macro_runner=macro_runner, perceive=perc, budget_plan=20, max_macros=3)
    assert r["solved"] is False and "macros_used" in r
