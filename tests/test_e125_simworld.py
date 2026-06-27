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
