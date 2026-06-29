"""E98 -- goal-RECOGNIZING solver: close the verified loop with the induced reward (E97).

Uses openworld.CodeObjective (the verified reward from E97) so the agent recognizes a level-completion
from its OWN synthesized objective -- not the privileged env signal. (1) Loop-closure: replay the
verified level-1 solution and confirm the verified reward fires EXACTLY when the env completes a level
(perception+dynamics+reward all synthesized/verified). (2) Honest L2: greedy/search using the verified
reward as the goal signal.

  python3 e98_goal_solver.py --game sp80 --prefix 5,2,... --reward /tmp/e97_sp80.json
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import arc_agi
from arcengine import GameAction
import openworld as O
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]
def grid(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--game",default="sp80"); ap.add_argument("--prefix",required=True)
    ap.add_argument("--reward",required=True); ap.add_argument("--l2_attempts",type=int,default=2000); ap.add_argument("--out",default="")
    a=ap.parse_args(); prefix=[int(x) for x in a.prefix.split(",") if x.strip()]
    obj=O.CodeObjective(json.load(open(a.reward))["reward_code"])  # the verified reward (foundational primitive)
    arc=arc_agi.Arcade(); env=arc.make(a.game)
    # (1) loop-closure: verified reward vs env truth along the level-1 solution
    o=env.reset(); g=grid(o); base=o.levels_completed; agree=match=n=0
    for act in prefix:
        no=env.step(ACTS[act-1])
        if no is None or getattr(no,"frame",None) is None: break
        ng=grid(no)
        r=obj.score({"frame":g.tolist()},{"name":str(act)},{"frame":ng.tolist()})   # agent's OWN reward
        env_win=1.0 if no.levels_completed>base else 0.0                              # ground truth
        n+=1; agree+=int((r>0)==(env_win>0)); match+=int(r==env_win)
        base=no.levels_completed; g=ng
    loop_acc=round(agree/n,4) if n else 0.0
    print(f"[e98/{a.game}] loop-closure: verified reward matches env on {agree}/{n} steps (acc {loop_acc}) -- agent recognizes its own win",flush=True)
    # (2) honest L2: search using the verified reward as the goal signal
    def replay():
        oo=env.reset()
        for x in prefix:
            oo=env.step(ACTS[x-1])
            if oo is None or getattr(oo,"frame",None) is None: return None
        return oo
    rng=random.Random(0); inter=[5,6,7]; avail=list(env.reset().available_actions)
    interact=[x for x in avail if x in inter] or avail; move=[x for x in avail if x not in inter] or avail
    best2=False
    o=replay(); g=grid(o) if o else None
    for at in range(a.l2_attempts):
        if replay() is None: continue
        g=grid(env.reset()); 
        oo=replay(); g=grid(oo)
        for _ in range(60):
            act=rng.choice(interact) if rng.random()<0.45 else rng.choice(move)
            no=env.step(ACTS[act-1])
            if no is None or getattr(no,"frame",None) is None: break
            ng=grid(no)
            if obj.score({"frame":g.tolist()},{"name":str(act)},{"frame":ng.tolist()})>0:  # verified reward recognizes a win
                best2=True; print(f"[e98/{a.game}] verified reward fired in L2 search (attempt {at})!",flush=True); break
            g=ng
            if str(no.state)!="GameState.NOT_FINISHED": break
        if best2: break
    res={"game":a.game,"loop_closure_acc":loop_acc,"loop_steps":n,
         "verified_reward_recognizes_win":loop_acc>=0.99,"level2_reward_fired":best2}
    print(f"[e98/{a.game}] loop closed (acc {loop_acc}) | level2 reward fired: {best2}",flush=True)
    out=Path(a.out) if a.out else Path("results")/f"e98_goal_{a.game}.json"; out.write_text(json.dumps(res,indent=2)); print("wrote",out)

if __name__=="__main__": main()
