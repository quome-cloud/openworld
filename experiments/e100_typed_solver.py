"""E100 -- typed-perception-guided directed solver (framework-leveraged, for games random can't crack).

Random search only reaches games whose win it stumbles into. This uses the typed perceptor: (1) probe
to find the AGENT (the object that moves) and its action->movement map; (2) identify TARGET objects;
(3) deliberately NAVIGATE the agent to each target (greedy distance reduction via the movement map)
and try INTERACT actions there -- the common "bring agent to target and act" win pattern. Capture +
deterministic replay-verify. CPU-only.

  python3 e100_typed_solver.py --game ka59
"""
import argparse, json, random, logging, contextlib, io
from pathlib import Path
import numpy as np
import arc_agi
from arcengine import GameAction
import arc3_graph as G
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]

def grid(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)

def all_games():
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        envs=arc_agi.Arcade().available_environments
    return sorted({(e if isinstance(e,str) else getattr(e,"game_id",str(e))).split("-")[0] for e in envs})

def centroid(g, color):
    ys,xs=np.where(g==color)
    return (ys.mean(),xs.mean()) if len(ys) else None

def probe(game, steps, seed):
    """Find agent color (moves most) + per-action movement (dr,dc) of the agent."""
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); avail=list(o.available_actions)
    g=grid(o); rng=random.Random(seed)
    trans=[]
    for _ in range(steps):
        a=rng.choice(avail); no=env.step(ACTS[a-1])
        if no is None or getattr(no,"frame",None) is None: o=env.reset(); g=grid(o); continue
        ng=grid(no); trans.append({"frame":g.tolist(),"next":ng.tolist(),"action":a}); g=ng
        if str(no.state)!="GameState.NOT_FINISHED": o=env.reset(); g=grid(o)
    roles=G.infer_typed_objects(trans) if trans else {}
    agents=[c for c,i in roles.items() if i["role"]=="agent"]
    agent=max(agents, key=lambda c: roles[c]["moves"]) if agents else None
    targets=[c for c,i in roles.items() if i["role"]=="target"]
    # movement map: avg agent centroid delta per action
    from collections import defaultdict
    deltas=defaultdict(list)
    for t in trans:
        if agent is None: break
        ca=centroid(np.asarray(t["frame"]),agent); cb=centroid(np.asarray(t["next"]),agent)
        if ca and cb: deltas[t["action"]].append((cb[0]-ca[0],cb[1]-ca[1]))
    mmap={a:(float(np.mean([d[0] for d in v])),float(np.mean([d[1] for d in v]))) for a,v in deltas.items() if v}
    return agent, targets, mmap, avail

def replay_verify(game, seq):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); base=o.levels_completed
    for a in seq:
        o=env.step(ACTS[a-1])
        if o is None or getattr(o,"frame",None) is None: return False
        if o.levels_completed>base: return True
    return False

def solve(game, seed=0, probe_steps=120, budget=4000):
    agent,targets,mmap,avail=probe(game,probe_steps,seed)
    if agent is None or not mmap:
        return {"game":game,"agent":agent,"verified":False,"note":"no agent/movement map"}
    inter=[a for a in avail if a>=5]; rng=random.Random(seed)
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); g=grid(o); lvl=o.levels_completed; recent=[]
    movers=[a for a in mmap if abs(mmap[a][0])+abs(mmap[a][1])>0.3]
    for step in range(budget):
        ac=centroid(g,agent)
        tgt=None
        tcs=[centroid(g,t) for t in targets if centroid(g,t) is not None]
        if ac and tcs:
            tgt=min(tcs,key=lambda tc: abs(tc[0]-ac[0])+abs(tc[1]-ac[1]))
        # choose action: 30% interact, else move toward nearest target (greedy via mmap), else random mover
        if inter and rng.random()<0.3:
            a=rng.choice(inter)
        elif tgt and ac and movers:
            a=min(movers,key=lambda m: abs((ac[0]+mmap[m][0])-tgt[0])+abs((ac[1]+mmap[m][1])-tgt[1]))
        else:
            a=rng.choice(avail)
        no=env.step(ACTS[a-1])
        if no is None or getattr(no,"frame",None) is None:
            o=env.reset(); g=grid(o); lvl=o.levels_completed; recent=[]; continue
        recent.append(a)
        if no.levels_completed>lvl:
            seq=list(recent); ok=replay_verify(game,seq)
            return {"game":game,"agent":agent,"targets":targets,"verified":ok,"level":int(no.levels_completed),
                    "seq_len":len(seq),"sequence":seq if ok else None}
        lvl=no.levels_completed; g=grid(no)
        if str(no.state)!="GameState.NOT_FINISHED": o=env.reset(); g=grid(o); lvl=o.levels_completed; recent=[]
    return {"game":game,"agent":agent,"targets":targets,"verified":False,"level":0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--game",default=""); ap.add_argument("--exclude",default="")
    ap.add_argument("--budget",type=int,default=5000); ap.add_argument("--seeds",type=int,default=3); ap.add_argument("--out",default="results/e100_typed_solver.json"); a=ap.parse_args()
    games=[a.game] if a.game else [g for g in all_games() if g not in a.exclude.split(",")]
    print(f"[e100] typed-guided solve on {len(games)} games",flush=True); res={}
    for g in games:
        best={"verified":False,"level":0}
        for sd in range(a.seeds):
            try: r=solve(g,sd,budget=a.budget)
            except Exception as e: r={"game":g,"verified":False,"error":str(e)[:80]}
            if r.get("verified") or r.get("level",0)>best.get("level",0): best=r
            if best.get("verified"): break
        res[g]=best
        print(f"  {g}: {'SOLVED' if best.get('verified') else 'no'} agent={best.get('agent')} targets={best.get('targets')} level={best.get('level',0)}",flush=True)
    solved=[g for g,r in res.items() if r.get("verified")]
    Path(a.out).write_text(json.dumps({"solved":solved,"n":len(solved),"results":res},indent=2))
    print(f"[e100] typed-guided SOLVED {len(solved)}: {solved}",flush=True)

if __name__=="__main__": main()
