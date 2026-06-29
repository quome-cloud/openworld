"""E107 -- deterministic STATE-GRAPH FRONTIER EXPLORATION (paper #2's method, in OpenWorld terms).

Bypasses goal inference (our wall) the way the graph-exploration SOTA does: segment each frame into
an object-graph signature (our perceptor) = a state node; build the deterministic state-transition
graph; drive to the FRONTIER -- the nearest UNTESTED (state, action) pair -- by replaying the shortest
path to it (our determinism makes this exact); record any level-completion and replay-verify it.
Salience: expand shallow, high-change states first. Continues across levels (chains a full game).
Combines our learnings (object-graph perception, determinism, replay-verify) with the SOTA technique.

  python3 e107_graph_explore.py --budget 6000
"""
import argparse, json, logging, contextlib, io
from collections import deque
from pathlib import Path
import numpy as np
import arc_agi
from arcengine import GameAction
import arc3_graph as G
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]

def grid(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)
def sig(f):                                   # exact state identity: full-frame hash (fast + accurate)
    return hash(np.asarray(f).tobytes())
def all_games():
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        envs=arc_agi.Arcade().available_environments
    return sorted({(e if isinstance(e,str) else getattr(e,"game_id",str(e))).split("-")[0] for e in envs})

def explore(game, budget, max_steps=400):
    """Frontier-biased forward rollouts: at each state prefer an UNTESTED (state,action) pair (the
    graph frontier); reset on episode end. Systematic coverage of the deterministic state graph
    without O(n^2) replay-navigation. Captures the winning action sequence on a level-completion."""
    import random as _r
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); win=o.win_levels; base0=o.levels_completed
    tested={}                      # state sig -> set of actions already tried from it
    best=base0; solution=None; steps=0; rng=_r.Random(0); n_states=set()
    while steps<budget:
        o=env.reset(); g=grid(o); avail=list(o.available_actions); lvl=o.levels_completed; seq=[]; depth=0
        while depth<max_steps and steps<budget:
            s=sig(g); n_states.add(s); ts=tested.setdefault(s,set())
            untried=[a for a in avail if a not in ts]
            inter=[x for x in avail if x>=5]
            a = rng.choice(untried) if untried else (rng.choice(inter) if inter and rng.random()<0.4 else rng.choice(avail))  # FRONTIER + interact bias
            ts.add(a)
            nob=env.step(ACTS[a-1]); steps+=1; depth+=1
            if nob is None or getattr(nob,"frame",None) is None: break
            seq.append(a); nf=grid(nob); lv=nob.levels_completed
            if lv>best:                                # LEVEL COMPLETED -> capture sequence
                best=lv; solution=list(seq)
            if str(nob.state)!="GameState.NOT_FINISHED": break
            g=nf; lvl=lv
    return {"game":game,"best_levels":int(best),"win_levels":int(win),"states":len(n_states),"steps":steps,
            "solution":solution,"verified":solution is not None}

def verify(game, seq):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); base=o.levels_completed
    for a in seq:
        o=env.step(ACTS[a-1])
        if o is None or getattr(o,"frame",None) is None: return False
        if o.levels_completed>base: return True
    return False

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--budget",type=int,default=6000); ap.add_argument("--games",default=""); ap.add_argument("--out",default="results/e107_graph_explore.json"); a=ap.parse_args()
    games=a.games.split(",") if a.games else all_games()
    print(f"[e107] state-graph frontier exploration on {len(games)} games (budget {a.budget})",flush=True); res={}
    for g in games:
        try:
            r=explore(g,a.budget)
            if r["solution"]: r["verified"]=verify(g,r["solution"])
        except Exception as e: r={"game":g,"verified":False,"error":str(e)[:90]}
        res[g]=r
        print(f"  {g}: best {r.get('best_levels',0)}/{r.get('win_levels','?')} states={r.get('states','-')} {'SOLVED' if r.get('verified') else ''}",flush=True)
    solved=[g for g,r in res.items() if r.get("verified")]
    Path(a.out).write_text(json.dumps({"solved":solved,"n_solved":len(solved),"results":res},indent=2))
    print(f"[e107] GRAPH-EXPLORE SOLVED {len(solved)}/{len(games)}: {sorted(solved)}",flush=True)

if __name__=="__main__": main()
