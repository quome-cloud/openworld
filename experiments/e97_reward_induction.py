"""E97 -- apply the framework's verified-reward induction (openworld.induce_reward) to ARC-AGI-3.

Demonstrates the new foundational primitive on a real game: collect (frame,action,next,reward=Δlevel)
transitions for sp80 (including the captured level-1 WIN via replaying the verified solution), then
induce a VERIFIED CodeObjective that predicts level-completion -- the objective becomes a first-class
verified artifact, symmetric to the synthesized dynamics.

  python3 e97_reward_induction.py --game sp80 --prefix 5,2,...
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import arc_agi
from arcengine import GameAction
import openworld as O

ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]
import e86_arc3 as E

def grid(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)

def collect_reward_examples(game, prefix, n_extra, seed):
    arc=arc_agi.Arcade(); env=arc.make(game)
    def replay():
        o=env.reset()
        for a in prefix:
            o=env.step(ACTS[a-1])
            if o is None or getattr(o,"frame",None) is None: return None
        return o
    ex=[]; 
    # WIN examples: replay the verified solution; the last action yields reward=1
    for _ in range(6):
        o=env.reset(); lvl=o.levels_completed; g=grid(o)
        for a in prefix:
            no=env.step(ACTS[a-1])
            if no is None or getattr(no,"frame",None) is None: break
            ng=grid(no); r=1.0 if no.levels_completed>lvl else 0.0
            ex.append({"state":{"frame":g.tolist()},"action":{"name":str(a)},"next_state":{"frame":ng.tolist()},"reward":r})
            lvl=no.levels_completed; g=ng
            if str(no.state)!="GameState.NOT_FINISHED": break
    # extra reward-0 transitions (post-win level-2 exploration)
    rng=random.Random(seed); o=replay(); 
    if o is not None:
        g=grid(o); avail=list(o.available_actions); base=o.levels_completed
        for _ in range(n_extra):
            a=rng.choice(avail); no=env.step(ACTS[a-1])
            if no is None or getattr(no,"frame",None) is None: o=replay(); g=grid(o) if o else g; continue
            ng=grid(no); r=1.0 if no.levels_completed>base else 0.0
            ex.append({"state":{"frame":g.tolist()},"action":{"name":str(a)},"next_state":{"frame":ng.tolist()},"reward":r}); g=ng
            if no.levels_completed!=base or str(no.state)!="GameState.NOT_FINISHED": o=replay(); g=grid(o) if o else g
    return ex

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--game",default="sp80"); ap.add_argument("--prefix",required=True)
    ap.add_argument("--extra",type=int,default=120); ap.add_argument("--seed",type=int,default=0); ap.add_argument("--out",default="")
    a=ap.parse_args(); prefix=[int(x) for x in a.prefix.split(",") if x.strip()]
    ex=collect_reward_examples(a.game, prefix, a.extra, a.seed)
    pos=sum(e["reward"]>0 for e in ex)
    print(f"[e97/{a.game}] {len(ex)} reward examples ({pos} positive); inducing verified reward via openworld.induce_reward...",flush=True)
    # numpy-free prompt: the reward code must be pure-python (sandbox). Help Claude reason over frames as lists.
    def prompt_fn(train):
        wins=[e for e in train if e["reward"]>0][:2]; zeros=[e for e in train if e["reward"]==0][:4]
        def changed(e):
            f=e["state"]["frame"]; n=e["next_state"]["frame"]
            return sum(1 for i in range(64) for j in range(64) if f[i][j]!=n[i][j])
        hint="\n".join(f"- reward={e['reward']} action={e['action']['name']} cells_changed={changed(e)}" for e in wins+zeros)
        return ("Induce the win/reward of an ARC-AGI-3 game as PURE PYTHON (no imports). Write exactly:\n\n"
                "    def reward(state, action, next_state):  # state['frame'],next_state['frame'] are 64x64 lists of ints\n"
                "        # return 1.0 when this transition COMPLETES a level, else 0.0\n\n"
                "A level-completion reloads the board (a large fraction of cells change at once), unlike normal moves "
                "(a few cells). Use that. Return ONLY a ```python block.\n\nExamples (reward, action, cells_changed):\n"+hint)
    obj,acc=O.induce_reward(ex, lambda p: E.claude_cli(p,timeout=600), prompt_fn=prompt_fn, rounds=4)
    res={"game":a.game,"examples":len(ex),"positive":pos,"induced_reward_verified_acc":round(acc,4),
         "is_code_objective":isinstance(obj,O.CodeObjective),"reward_code":obj.code if obj else None}
    print(f"[e97/{a.game}] induced verified reward acc={acc:.3f} (CodeObjective={res['is_code_objective']})",flush=True)
    out=Path(a.out) if a.out else Path("results")/f"e97_reward_{a.game}.json"; out.write_text(json.dumps(res,indent=2)); print("wrote",out)

if __name__=="__main__": main()
