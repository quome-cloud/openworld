import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import simworld

# model: action [1] increments frame[0,0]; level_up when it reaches 5 (depth 5 -> blind-real would be slow)
PRED = "def predict(frame, action):\n    nf=frame.copy()\n    if action==[1]: nf[0,0]=frame[0,0]+1\n    return nf, bool(nf[0,0]==5)"
from e125 import verify
fn = verify.compile_predict(PRED)

def test_simgame_steps_via_predict():
    g = simworld.SimGame(fn, np.zeros((64,64),dtype=int)); g.reset()
    g.step(1); assert g.frame[0,0]==1 and g.levels==0
    for _ in range(4): g.step(1)
    assert g.levels==1 and g.done

def test_plan_finds_winning_trajectory_in_sim():
    plan = simworld.plan(fn, np.zeros((64,64),dtype=int), lambda fr:[[1],[2]], budget=2000)
    assert plan == [[1],[1],[1],[1],[1]]


# --- best-first energy descent guided by a goal_score, + the per-node-frame perf fix ---
# counter c=frame[0,0]; [1]=+1, [2]=-1, [3]/[4]=noise on other cells (blow up a blind frontier); win at c==10.
ENERGY = ("def predict(frame, action):\n"
          "    nf=frame.copy(); c=int(frame[0,0])\n"
          "    if action==[1]: c=min(10,c+1)\n"
          "    elif action==[2]: c=max(0,c-1)\n"
          "    elif action==[3]: nf[5,5]=(int(frame[5,5])+1)%9; return nf,False\n"
          "    elif action==[4]: nf[6,6]=(int(frame[6,6])+1)%9; return nf,False\n"
          "    nf[0,0]=c\n    return nf, bool(c==10)")
GOALE = "def goal_score(frame):\n    return float(10 - int(frame[0,0]))"

def _counting(fn):
    calls = {"n": 0}
    def wrapped(frame, action):
        calls["n"] += 1
        return fn(frame, action)
    wrapped.calls = calls
    return wrapped

def test_plan_energy_descent_reaches_deep_goal_blind_bfs_cannot():
    fn2 = verify.compile_predict(ENERGY); goal = verify.compile_goal(GOALE)
    cands = lambda fr: [[1],[2],[3],[4]]
    init = np.zeros((64,64),dtype=int)
    # a blind frontier under this tight budget can't reach depth 10 with branching 4...
    assert simworld.plan(fn2, init, cands, budget=80) is None
    # ...but energy descent walks straight down goal_score to the win.
    assert simworld.plan(fn2, init, cands, budget=80, goal_fn=goal) == [[1]]*10

def test_plan_stores_frame_per_node_no_prefix_replay():
    line = ("def predict(frame, action):\n    nf=frame.copy(); nf[0,0]=min(10,int(frame[0,0])+1)\n"
            "    return nf, bool(nf[0,0]==10)")
    counting = _counting(verify.compile_predict(line))
    plan = simworld.plan(counting, np.zeros((64,64),dtype=int), lambda fr:[[1]], budget=100)
    assert plan == [[1]]*10
    # one predict() call per node expansion (~10), NOT prefix-replay (which would re-run ~55 steps for depth 10)
    assert counting.calls["n"] <= 12
