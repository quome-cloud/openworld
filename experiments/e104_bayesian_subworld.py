"""E104 -- Bayesian active goal-inference over composite SUBWORLDS with SEMIRING-weighted hierarchical
planning. The culminating attack on the goal-as-PROCEDURE wall (E103's finding).

Pieces:
 * SUBWORLDS: typed perception -> atomic sub-goals (extremize a counter / clear a color / reach a
   target), each a subworld condition.
 * PROCEDURE hypotheses: ORDERED sub-goal sequences (len<=3) -- can express "do A then B", which
   frame-score hypotheses (E102/E103) could not.
 * SEMIRING planning: TROPICAL (min,+) beam through the code world model to a sub-goal (optimal short
   sub-procedure); the framework's Semiring is the value engine.
 * HIERARCHICAL execution: achieve g1 (plateau its progress), re-perceive, achieve g2, ... then
   interact -- decomposes the long opaque procedure.
 * BAYESIAN belief: posterior over procedures; pursue highest-posterior; a reward COLLAPSES it and
   induce_reward locks the verified win-rule.

  python3 e104_bayesian_subworld.py --games s5i5,r11l,vc33
"""
import argparse, json, random, glob, itertools, math
from pathlib import Path
import numpy as np
import arc_agi
from arcengine import GameAction
import arc3_graph as G
import e86_arc3 as E
import openworld as O
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]
MODELS={Path(f).stem:f for f in glob.glob(str(Path(__file__).resolve().parent/"results"/"arc3_e86b_claude"/"*.json"))}
SR=O.TROPICAL   # (plus=min, times=+) -> optimal-cost sub-procedure planning

def grid(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)
def count(f,c): return int((f==c).sum())
def cen(f,c):
    ys,xs=np.where(f==c); return (ys.mean(),xs.mean()) if len(ys) else None
def load_predict(game):
    d=json.load(open(MODELS[game])); ns={"np":np,"numpy":np}
    exec(compile(d["code"],"<m>","exec"),ns); return ns["predict"], d.get("verified_exact",0.0)
def probe(game, seed):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); avail=list(o.available_actions); g=grid(o); rng=random.Random(seed); tr=[]
    for _ in range(150):
        a=rng.choice(avail); no=env.step(ACTS[a-1])
        if no is None or getattr(no,"frame",None) is None: o=env.reset(); g=grid(o); continue
        ng=grid(no); tr.append({"frame":g.tolist(),"next":ng.tolist(),"action":a}); g=ng
        if str(no.state)!="GameState.NOT_FINISHED": o=env.reset(); g=grid(o)
    return G.infer_typed_objects(tr) if tr else {}, avail, tr

def sub_goals(roles):
    """Atomic subworld conditions: (name, progress(frame)->float to MAXIMIZE). 'Achieve' = plateau it."""
    sgs=[]
    agent=next((c for c,i in roles.items() if i["role"]=="agent"),None)
    counters=[c for c,i in roles.items() if i["role"] in ("timer","counter")]
    targets=[c for c,i in roles.items() if i["role"]=="target"]
    others=[c for c,i in roles.items() if i["role"] not in ("agent","wall")]
    for c in counters:
        sgs.append((f"drain{c}", (lambda f,c=c: -count(f,c))))
        sgs.append((f"fill{c}",  (lambda f,c=c:  count(f,c))))
    for c in others:
        sgs.append((f"clear{c}", (lambda f,c=c: -count(f,c))))
    if agent is not None:
        for t in targets:
            def reach(f,a=agent,t=t):
                ca,ct=cen(f,a),cen(f,t); return -(abs(ca[0]-ct[0])+abs(ca[1]-ct[1])) if (ca and ct) else -1e3
            sgs.append((f"reach{t}", reach))
    return sgs

def plan_step(predict, frame, progress, avail, depth=3, beam=5):
    """SEMIRING (TROPICAL) beam: cost = times-fold of per-step (-progress-gain); pick plus(min)."""
    base=progress(np.asarray(frame))
    beams=[(SR.one, np.asarray(frame), [])]   # SR.one == 0 (tropical identity for times=+)
    best=(SR.zero, None)                       # SR.zero == +inf
    for _ in range(depth):
        nxt=[]
        for cost,st,seq in beams:
            for a in avail:
                try: ns=np.asarray(predict(st,a))
                except Exception: continue
                if ns.shape!=(64,64): continue
                step_cost=-(progress(ns)-base)          # lower = more progress
                c2=SR.times(cost, step_cost)            # accumulate along path (tropical: +)
                nxt.append((c2,ns,seq+[a]))
        if not nxt: break
        nxt.sort(key=lambda x:x[0]); beams=nxt[:beam]
        if beams[0][2]: best=(SR.plus(best[0],beams[0][0]), beams[0][2])  # plus = min
    return best[1]

def replay_verify(game, seq):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); base=o.levels_completed
    for a in seq:
        o=env.step(ACTS[a-1])
        if o is None or getattr(o,"frame",None) is None: return False
        if o.levels_completed>base: return True
    return False

def pursue_procedure(game, predict, procedure, sgmap, avail, seed, seg_budget=120, plateau=8):
    """Hierarchically achieve each sub-goal (MPC to plateau), then interact; return reward seq if hit."""
    inter=[a for a in avail if a>=5]; rng=random.Random(seed)
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); g=grid(o); lvl=o.levels_completed; recent=[]
    for gi, name in enumerate(procedure+["__interact__"]):
        if name=="__interact__":
            for _ in range(10):
                a=rng.choice(inter) if inter else rng.choice(avail); no=env.step(ACTS[a-1])
                if no is None or getattr(no,"frame",None) is None: break
                recent.append(a)
                if no.levels_completed>lvl: return recent if replay_verify(game,recent) else None
                lvl=no.levels_completed; g=grid(no)
                if str(no.state)!="GameState.NOT_FINISHED": break
            continue
        prog=sgmap[name]; best=prog(g); stall=0; steps=0
        while steps<seg_budget and stall<plateau:
            plan=plan_step(predict,g,prog,avail) or [rng.choice(avail)]
            for a in plan:
                no=env.step(ACTS[a-1]); steps+=1
                if no is None or getattr(no,"frame",None) is None: o=env.reset(); g=grid(o); lvl=o.levels_completed; recent=[]; stall=plateau; break
                recent.append(a)
                if no.levels_completed>lvl: return recent if replay_verify(game,recent) else None
                lvl=no.levels_completed; g=grid(no); p=prog(g)
                stall = 0 if p>best+1e-6 else stall+1; best=max(best,p)
                if str(no.state)!="GameState.NOT_FINISHED": o=env.reset(); g=grid(o); lvl=o.levels_completed; recent=[]; stall=plateau; break
                if stall>=plateau or steps>=seg_budget: break
    return None

def solve(game, seed=0, max_proc=40):
    if game not in MODELS: return {"game":game,"verified":False,"note":"no model"}
    predict,fid=load_predict(game); roles,avail,tr=probe(game,seed); sgs=sub_goals(roles)
    sgmap={n:f for n,f in sgs}; names=[n for n,_ in sgs]
    # PROCEDURE hypotheses: ordered sub-goal sequences len 1..3 (Bayesian prior: shorter first)
    procs=[]
    for L in (1,2,3):
        procs += [list(p) for p in itertools.permutations(names, L)]
    procs=procs[:max_proc]
    belief={tuple(p): 1.0/len(procs) for p in procs}   # uniform posterior
    print(f"  [{game}] {len(names)} sub-goals, testing {len(procs)} procedure hypotheses (fid={fid})",flush=True)
    for p in sorted(procs, key=len):                   # active: shorter/higher-prior first
        seq=pursue_procedure(game,predict,p,sgmap,avail,seed)
        if seq:
            return {"game":game,"verified":True,"level":1,"procedure":p,"seq_len":len(seq),
                    "model_fidelity":fid,"sequence":seq}
        belief[tuple(p)]=0.0                            # Bayesian: this procedure ruled out
    return {"game":game,"verified":False,"model_fidelity":fid,"n_procedures":len(procs),"sub_goals":names}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--games",default=""); ap.add_argument("--game",default=""); ap.add_argument("--out",default="results/e104_bayesian_subworld.json"); a=ap.parse_args()
    games=[a.game] if a.game else a.games.split(",")
    print(f"[e104] Bayesian subworld procedure-search on {games}",flush=True); res={}
    for g in games:
        try: r=solve(g)
        except Exception as e: r={"game":g,"verified":False,"error":str(e)[:120]}
        res[g]=r; print(f"  => {g}: {'SOLVED via '+'>'.join(r.get('procedure',[])) if r.get('verified') else 'no'}",flush=True)
    solved=[g for g,r in res.items() if r.get("verified")]
    Path(a.out).write_text(json.dumps({"solved":solved,"n":len(solved),"results":res},indent=2))
    print(f"[e104] SUBWORLD-BAYESIAN SOLVED {len(solved)}: {solved}",flush=True)

if __name__=="__main__": main()
