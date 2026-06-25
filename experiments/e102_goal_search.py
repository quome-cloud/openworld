"""E102 -- core-knowledge GOAL-DISCOVERY solver (the principled attack on the goal-inference wall).

Instead of guessing the goal (E89) or wandering (E88), we GENERATE a finite, structured space of
candidate goals from core-knowledge priors over the typed perception, then PLAN to each one THROUGH
the synthesized code world model and execute -- the candidate whose pursuit triggers a real reward IS
the goal. Composes: typed perceptor -> candidate CodeObjectives -> goal-conditioned MPC through
CodeTransition -> reward confirms. CPU-only (reuses e86b models, no LLM).

  python3 e102_goal_search.py --game ka59
"""
import argparse, json, random, glob
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
    if game not in MODELS: return None,0.0
    d=json.load(open(MODELS[game])); code=d.get("code")
    if not code: return None,0.0
    ns={"np":np,"numpy":np}
    try: exec(compile(code,"<m>","exec"),ns); return ns.get("predict"), d.get("verified_exact",0.0)
    except Exception: return None,0.0

def count(f,c): return int((f==c).sum())
def centroid(f,c):
    ys,xs=np.where(f==c); return (ys.mean(),xs.mean()) if len(ys) else None

def probe_roles(game, seed):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); avail=list(o.available_actions); g=grid(o); rng=random.Random(seed); tr=[]
    for _ in range(150):
        a=rng.choice(avail); no=env.step(ACTS[a-1])
        if no is None or getattr(no,"frame",None) is None: o=env.reset(); g=grid(o); continue
        ng=grid(no); tr.append({"frame":g.tolist(),"next":ng.tolist(),"action":a}); g=ng
        if str(no.state)!="GameState.NOT_FINISHED": o=env.reset(); g=grid(o)
    return G.infer_typed_objects(tr) if tr else {}, avail

def candidate_goals(roles):
    """Core-knowledge goal templates instantiated over perceived roles -> [(name, score(frame))]."""
    agent=next((c for c,i in roles.items() if i["role"]=="agent"), None)
    targets=[c for c,i in roles.items() if i["role"]=="target"]
    counters=[c for c,i in roles.items() if i["role"] in ("timer","counter")]
    others=[c for c,i in roles.items() if i["role"] not in ("agent","wall")]
    C=[]
    if agent is not None:
        for t in targets:                                   # reach / dock agent onto target
            def reach(f,a=agent,t=t):
                ca,ct=centroid(f,a),centroid(f,t)
                return -(abs(ca[0]-ct[0])+abs(ca[1]-ct[1])) if (ca and ct) else -1e3
            C.append((f"reach_{t}", reach))
    for t in targets:                                       # cover/consume target cells
        C.append((f"cover_{t}", (lambda f,t=t: -count(f,t))))
    for c in others:                                        # collect/remove all of a color
        C.append((f"remove_{c}", (lambda f,c=c: -count(f,c))))
        C.append((f"amass_{c}", (lambda f,c=c: count(f,c))))
    for k in counters:                                      # extremize a counter both ways
        C.append((f"drain_{k}", (lambda f,k=k: -count(f,k))))
        C.append((f"fill_{k}", (lambda f,k=k: count(f,k))))
    return C

def plan_seq(predict, frame, score, avail, depth=3, beam=5):
    """Goal-conditioned MPC IN THE MODEL: return the best depth-D action sequence (maximize score)."""
    beams=[(score(np.asarray(frame)), np.asarray(frame), [])]
    for _ in range(depth):
        nxt=[]
        for sc,st,seq in beams:
            for a in avail:
                try: ns=np.asarray(predict(st,a))
                except Exception: continue
                if ns.shape!=(64,64): continue
                nxt.append((score(ns), ns, seq+[a]))
        if not nxt: break
        nxt.sort(key=lambda x:-x[0]); beams=nxt[:beam]
    return beams[0][2] if beams and beams[0][2] else None

def replay_verify(game, seq):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); base=o.levels_completed
    for a in seq:
        o=env.step(ACTS[a-1])
        if o is None or getattr(o,"frame",None) is None: return False
        if o.levels_completed>base: return True
    return False

def pursue(game, predict, score, avail, budget, seed):
    """Run an episode pursuing one candidate goal (model-MPC + interacts); return reward seq if hit."""
    inter=[a for a in avail if a>=5]; rng=random.Random(seed)
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); g=grid(o); lvl=o.levels_completed; recent=[]; steps=0
    while steps<budget:
        plan=plan_seq(predict,g,score,avail) or [rng.choice(avail)]
        if inter and rng.random()<0.25: plan=plan+[rng.choice(inter)]   # try interacting at the goal config
        for a in plan:
            no=env.step(ACTS[a-1]); steps+=1
            if no is None or getattr(no,"frame",None) is None:
                o=env.reset(); g=grid(o); lvl=o.levels_completed; recent=[]; break
            recent.append(a)
            if no.levels_completed>lvl:
                seq=list(recent); return seq if replay_verify(game,seq) else None
            lvl=no.levels_completed; g=grid(no)
            if str(no.state)!="GameState.NOT_FINISHED": o=env.reset(); g=grid(o); lvl=o.levels_completed; recent=[]; break
            if steps>=budget: break
    return None

def solve(game, seed=0, per_goal=900):
    predict,fid=load_predict(game)
    if predict is None: return {"game":game,"verified":False,"note":"no model"}
    roles,avail=probe_roles(game,seed)
    cands=candidate_goals(roles)
    for name,score in cands:
        seq=pursue(game,predict,score,avail,per_goal,seed)
        if seq:
            return {"game":game,"verified":True,"level":1,"goal":name,"seq_len":len(seq),
                    "model_fidelity":fid,"n_candidates":len(cands),"sequence":seq}
    return {"game":game,"verified":False,"level":0,"model_fidelity":fid,"n_candidates":len(cands),
            "roles":{int(c):i["role"] for c,i in roles.items()}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--game",default=""); ap.add_argument("--games",default="")
    ap.add_argument("--per_goal",type=int,default=900); ap.add_argument("--out",default="results/e102_goal_search.json"); a=ap.parse_args()
    games=[a.game] if a.game else (a.games.split(",") if a.games else sorted(MODELS))
    print(f"[e102] goal-discovery solve on {len(games)} games",flush=True); res={}
    for g in games:
        try: r=solve(g, per_goal=a.per_goal)
        except Exception as e: r={"game":g,"verified":False,"error":str(e)[:100]}
        res[g]=r
        print(f"  {g}: {'SOLVED via '+r.get('goal','?') if r.get('verified') else 'no'} (cands={r.get('n_candidates','-')}, fid={r.get('model_fidelity')})",flush=True)
    solved=[g for g,r in res.items() if r.get("verified")]
    Path(a.out).write_text(json.dumps({"solved":solved,"n":len(solved),"results":res},indent=2))
    print(f"[e102] GOAL-DISCOVERY SOLVED {len(solved)}: {solved}",flush=True)

if __name__=="__main__": main()
