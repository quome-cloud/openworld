"""E101 -- MODEL-BASED solver: plan navigation THROUGH the synthesized code world model.

Unlike E99 (real-env random search) and E100 (real-env greedy nav), this puts the code world model in
the loop: load the synthesized verified predict() (the CodeTransition), and at each step do
model-predictive control -- simulate candidate action sequences IN THE MODEL (imagination), score the
predicted frame by how close the typed AGENT gets to a typed TARGET, execute the best first action in
the real env, intersperse interact actions, and recognize the win with the verified reward. The world
model drives planning; the real env supplies the win-trigger it can't simulate. CPU-only (reuses
existing e86b models -- no LLM).

  python3 e101_model_solver.py --game ka59
"""
import argparse, json, random, glob, logging, contextlib, io
from pathlib import Path
import numpy as np
import arc_agi
from arcengine import GameAction
import arc3_graph as G
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]
MODELS={Path(f).stem:f for f in glob.glob(str(Path(__file__).resolve().parent/"results"/"arc3_e86b_claude"/"*.json"))}

def grid(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)

def load_predict(game):
    d=json.load(open(MODELS[game])); code=d.get("code")
    if not code: return None,0.0
    ns={"np":np,"numpy":np}
    try: exec(compile(code,"<m>","exec"),ns); return ns["predict"], d.get("verified_exact",0.0)
    except Exception: return None,0.0

def centroid(g,color):
    ys,xs=np.where(g==color); return (ys.mean(),xs.mean()) if len(ys) else None

def typed(game, seed):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); avail=list(o.available_actions); g=grid(o); rng=random.Random(seed); tr=[]
    for _ in range(120):
        a=rng.choice(avail); no=env.step(ACTS[a-1])
        if no is None or getattr(no,"frame",None) is None: o=env.reset(); g=grid(o); continue
        ng=grid(no); tr.append({"frame":g.tolist(),"next":ng.tolist(),"action":a}); g=ng
        if str(no.state)!="GameState.NOT_FINISHED": o=env.reset(); g=grid(o)
    roles=G.infer_typed_objects(tr) if tr else {}
    agents=[c for c,i in roles.items() if i["role"]=="agent"]
    agent=max(agents,key=lambda c:roles[c]["moves"]) if agents else None
    targets=[c for c,i in roles.items() if i["role"]=="target"]
    return agent,targets,avail

def plan_action(predict, frame, agent, targets, avail, depth=4, beam=6):
    """MPC IN THE MODEL: beam over action seqs via predict(); minimize agent->nearest-target distance."""
    def dist(f):
        ac=centroid(f,agent); 
        if ac is None: return 1e9
        ds=[abs(centroid(f,t)[0]-ac[0])+abs(centroid(f,t)[1]-ac[1]) for t in targets if centroid(f,t) is not None]
        return min(ds) if ds else 1e9
    beams=[(dist(frame),np.asarray(frame),None)]
    for _ in range(depth):
        nxt=[]
        for _,st,fa in beams:
            for a in avail:
                try: ns=np.asarray(predict(st,a))
                except Exception: continue
                if ns.shape!=(64,64): continue
                nxt.append((dist(ns),ns,fa if fa is not None else a))
        if not nxt: return None
        nxt.sort(key=lambda x:x[0]); beams=nxt[:beam]
    return beams[0][2]

def replay_verify(game, seq):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); base=o.levels_completed
    for a in seq:
        o=env.step(ACTS[a-1])
        if o is None or getattr(o,"frame",None) is None: return False
        if o.levels_completed>base: return True
    return False

def solve(game, seed=0, budget=4000):
    predict,fid=load_predict(game)
    if predict is None: return {"game":game,"verified":False,"note":"no model"}
    agent,targets,avail=typed(game,seed)
    if agent is None: return {"game":game,"verified":False,"note":"no agent","model_fidelity":fid}
    inter=[a for a in avail if a>=5]; rng=random.Random(seed)
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); g=grid(o); lvl=o.levels_completed; recent=[]
    for step in range(budget):
        if inter and rng.random()<0.3: a=rng.choice(inter)
        else:
            a=plan_action(predict,g,agent,targets,avail) if targets else rng.choice(avail)   # model-based nav
            if a is None: a=rng.choice(avail)
        no=env.step(ACTS[a-1])
        if no is None or getattr(no,"frame",None) is None:
            o=env.reset(); g=grid(o); lvl=o.levels_completed; recent=[]; continue
        recent.append(a)
        if no.levels_completed>lvl:
            seq=list(recent); ok=replay_verify(game,seq)
            return {"game":game,"verified":ok,"level":int(no.levels_completed),"seq_len":len(seq),
                    "model_fidelity":fid,"agent":agent,"targets":targets,"sequence":seq if ok else None}
        lvl=no.levels_completed; g=grid(no)
        if str(no.state)!="GameState.NOT_FINISHED": o=env.reset(); g=grid(o); lvl=o.levels_completed; recent=[]
    return {"game":game,"verified":False,"level":0,"model_fidelity":fid,"agent":agent}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--game",default=""); ap.add_argument("--budget",type=int,default=5000); ap.add_argument("--seeds",type=int,default=3); ap.add_argument("--out",default="results/e101_model_solver.json"); a=ap.parse_args()
    games=[a.game] if a.game else sorted(MODELS)
    print(f"[e101] model-based solve on {len(games)} games (using synthesized code world models)",flush=True); res={}
    for g in games:
        best={"verified":False,"level":0}
        for sd in range(a.seeds):
            try: r=solve(g,sd,a.budget)
            except Exception as e: r={"game":g,"verified":False,"error":str(e)[:80]}
            if r.get("verified") or r.get("level",0)>best.get("level",0): best=r
            if best.get("verified"): break
        res[g]=best
        print(f"  {g}: {'SOLVED' if best.get('verified') else 'no'} (model_fid={best.get('model_fidelity')}, agent={best.get('agent')}, level={best.get('level',0)})",flush=True)
    solved=[g for g,r in res.items() if r.get("verified")]
    Path(a.out).write_text(json.dumps({"solved":solved,"n":len(solved),"results":res},indent=2))
    print(f"[e101] MODEL-BASED SOLVED {len(solved)}: {solved}",flush=True)

if __name__=="__main__": main()
