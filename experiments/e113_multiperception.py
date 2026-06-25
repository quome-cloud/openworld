"""E113 -- extended MULTI-PERCEPTION CONSENSUS: add perceptions + goal-directed planning to grow the
union beyond E112's 10/25. New consensus members:
  * RICHER click perception: small-object cells + boundaries + every distinct non-bg color +
    recently-CHANGED cells, re-perceived per state (catches lp85's missed target, ft09 conditional
    sprites).
  * TYPED-NAVIGATION modality (planning help): infer agent (the mover) + per-action movement map +
    typed targets; the rollout selects directional actions GOAL-DIRECTED (move agent toward nearest
    target) + interact -- directed search for reachability-bottlenecked games random misses.
Each modality builds a discovered openworld.World; ConsensusTransition selects the best per game.

  python3 e113_multiperception.py --games bp35,ft09,lp85 --budget 120000
"""
import argparse, json, logging, contextlib, io, random as _r
from pathlib import Path
import numpy as np
import arc_agi, arc3_graph as G
from arcengine import GameAction
import openworld as O
from openworld.transition import FunctionTransition
try: from tqdm import tqdm
except Exception:
    def tqdm(x=None,**k): return x if x is not None else None
SIMPLE={1:GameAction.ACTION1,2:GameAction.ACTION2,3:GameAction.ACTION3,4:GameAction.ACTION4,5:GameAction.ACTION5,7:GameAction.ACTION7}
MAPS=Path("/Users/jim/Desktop/openworld/papers/arc-3/maps")
def g_of(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)
def all_games():
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        envs=arc_agi.Arcade().available_environments
    return sorted({(e if isinstance(e,str) else getattr(e,"game_id",str(e))).split("-")[0] for e in envs})
def parse(a):
    if a[0]=="c": x,y=a[1:].split("_"); return ("c",int(x),int(y))
    return ("s",int(a[1:]))
def step(env,a):
    k=parse(a); return env.step(GameAction.ACTION6,{"x":k[1],"y":k[2]}) if k[0]=="c" else env.step(SIMPLE[k[1]])
def cen(f,c):
    ys,xs=np.where(f==c); return (ys.mean(),xs.mean()) if len(ys) else None

def click_targets(g,bg,prev=None,cap=110):
    """RICHER: small-object cells + bbox corners + a few cells per distinct color + changed cells."""
    objs,_=G.objects(g,bg); cells=[]
    for ob in sorted(objs,key=lambda o:o["size"]):
        r0,c0,r1,c1=ob["bbox"]
        if ob["size"]<=60:
            cells+=[(c,r) for r in range(r0,r1+1) for c in range(c0,c1+1) if g[r,c]!=bg]
        cells+=[(c0,r0),(c1,r1),(c0,r1),(c1,r0)]                # boundary corners (any size)
    vals,cnts=np.unique(g,return_counts=True)
    for v,_ in sorted([(int(v),int(c)) for v,c in zip(vals,cnts) if int(v)!=bg],key=lambda kc:kc[1]):
        ys,xs=np.where(g==v); cells+=list(zip(xs.tolist()[:6],ys.tolist()[:6]))   # a few per color
    if prev is not None:
        ys,xs=np.where(g!=prev); cells+=list(zip(xs.tolist(),ys.tolist()))         # recently changed
    seen=set(); out=[]
    for (x,y) in cells:
        if 0<=x<64 and 0<=y<64 and (x,y) not in seen: seen.add((x,y)); out.append(f"c{x}_{y}")
    return out[:cap]
def action_model(g,bg,avail,mode,prev=None):
    acts=[]
    if 6 in avail and mode in ("click","both"): acts+=click_targets(g,bg,prev)
    if mode in ("dir","both","typednav"): acts+=[f"s{a}" for a in avail if a in SIMPLE]
    return acts

def detect_mask(ENV,avail,probe=14):
    o=ENV.reset(); g=g_of(o); chg=np.zeros((64,64)); n=0
    vals,cnts=np.unique(g,return_counts=True); bg=int(vals[np.argmax(cnts)])
    for _ in range(probe):
        acts=action_model(g,bg,avail,'both')
        if not acts: break
        no=step(ENV,acts[n%len(acts)])
        if no is None or getattr(no,"frame",None) is None: o=ENV.reset(); g=g_of(o); continue
        ng=g_of(no); chg+=(g!=ng); n+=1; g=ng
        if str(no.state)!="GameState.NOT_FINISHED": o=ENV.reset(); g=g_of(o)
    return (chg/max(n,1))>0.95, bg

def typednav_setup(ENV, avail, bg, seed=0):
    """Infer agent (mover), per-action movement map, typed targets -- for goal-directed selection."""
    from collections import defaultdict
    o=ENV.reset(); g=g_of(o); rng=_r.Random(seed); tr=[]; dirs=[a for a in avail if a in SIMPLE]
    if not dirs: return None,{},[]
    for _ in range(150):
        a=rng.choice(dirs); no=step(ENV,f"s{a}")
        if no is None or getattr(no,"frame",None) is None: o=ENV.reset(); g=g_of(o); continue
        ng=g_of(no); tr.append({"frame":g.tolist(),"next":ng.tolist(),"action":a}); g=ng
        if str(no.state)!="GameState.NOT_FINISHED": o=ENV.reset(); g=g_of(o)
    roles=G.infer_typed_objects(tr) if tr else {}
    agents=[c for c,i in roles.items() if i["role"]=="agent"]
    agent=max(agents,key=lambda c:roles[c]["moves"]) if agents else None
    targets=[c for c,i in roles.items() if i["role"]=="target"]
    deltas=defaultdict(list)
    for t in tr:
        if agent is None: break
        ca=cen(np.asarray(t["frame"]),agent); cb=cen(np.asarray(t["next"]),agent)
        if ca and cb: deltas[t["action"]].append((cb[0]-ca[0],cb[1]-ca[1]))
    mmap={a:(float(np.mean([d[0] for d in v])),float(np.mean([d[1] for d in v]))) for a,v in deltas.items() if v}
    return agent, mmap, targets

def solve_game(game, mode, budget=120000, max_depth=400, seed=0):
    arc=arc_agi.Arcade(); ENV=arc.make(game); o=ENV.reset(); avail=list(o.available_actions); win=o.win_levels; base=o.levels_completed
    mask,bg=detect_mask(ENV,avail)
    keep=(~mask).reshape(-1)
    def sig(g): return hash(g.reshape(-1)[keep].tobytes())
    s0=sig(g_of(ENV.reset()))
    agent=mmap=None; targets=[]
    if mode=="typednav":
        agent,mmap,targets=typednav_setup(ENV,avail,bg,seed)
        if agent is None or not mmap: return {"avail":avail,"win":win,"base":base,"level":base,"full":[],"steps":0,"table":{},"s0":s0,"click":6 in avail,"note":"no agent"}
        movers=[a for a in mmap if abs(mmap[a][0])+abs(mmap[a][1])>0.3]; inter=[a for a in avail if a in (5,7)]
    rng=_r.Random(seed); tested={}; table={}; best=base; best_path=[]; steps=0; acache={}; STATE_CAP=40000
    bar=tqdm(total=budget,desc=f"{game}:{mode}",unit="step",leave=False,ncols=80)
    while steps<budget and best<win and len(tested)<STATE_CAP:
        ob=ENV.reset(); g=g_of(ob); seq=[]; d=0; prev=None
        while d<max_depth and steps<budget:
            s=sig(g)
            if mode=="typednav":
                ac=cen(g,agent); tcs=[cen(g,t) for t in targets if cen(g,t) is not None]
                if inter and rng.random()<0.3: a=f"s{rng.choice(inter)}"
                elif ac and tcs and movers:
                    tgt=min(tcs,key=lambda tc:abs(tc[0]-ac[0])+abs(tc[1]-ac[1]))
                    a=f"s{min(movers,key=lambda m:abs((ac[0]+mmap[m][0])-tgt[0])+abs((ac[1]+mmap[m][1])-tgt[1]))}"
                else: a=f"s{rng.choice([x for x in avail if x in SIMPLE] or [1])}"
            else:
                acts=acache.get(s)
                if acts is None: acts=action_model(g,bg,avail,mode,prev); acache[s]=acts
                if not acts: break
                ts=tested.setdefault(s,set()); unt=[x for x in acts if x not in ts]
                a=rng.choice(unt) if unt else rng.choice(acts); ts.add(a)
            no=step(ENV,a); steps+=1; d+=1
            if no is None or getattr(no,"frame",None) is None: break
            seq.append(a); ng=g_of(no); table.setdefault(s,{})[a]=sig(ng)
            if no.levels_completed>best: best=no.levels_completed; best_path=list(seq)
            if str(no.state)!="GameState.NOT_FINISHED": break
            prev=g; g=ng
        if hasattr(bar,"update"): bar.update(steps-bar.n); bar.set_postfix(best=best,states=len(tested))
    if hasattr(bar,"close"): bar.close()
    return {"avail":avail,"win":win,"base":base,"level":best,"full":best_path,"steps":steps,"table":table,"s0":s0,"click":6 in avail}

def mode_transition(table):
    def fn(state,action):
        nm=action.get("name") if isinstance(action,dict) else getattr(action,"name",action)
        return {"sig": table.get(state.get("sig"),{}).get(nm, state.get("sig"))}
    return FunctionTransition(fn)
def build_consensus_world(game, mode_results, s0):
    members=[(mode_transition(tbl),float(fid)) for (m,tbl,fid) in mode_results if tbl] or [(mode_transition({}),0.0)]
    cons=O.ConsensusTransition(members,mode="select")
    acts=sorted({a for (m,tbl,fid) in mode_results for d in tbl.values() for a in d})[:80]
    return O.World(name=f"arc3-map-{game}",description=f"Discovered multi-perception map of {game}",
                   initial_state={"sig":s0},actions=acts or ["noop"],transition=cons), len(members)
def verify(game, full):
    if not full: return 0
    e=arc_agi.Arcade().make(game); o=e.reset(); base=o.levels_completed; mx=base
    for a in full:
        o=step(e,a)
        if o is None or getattr(o,"frame",None) is None: break
        mx=max(mx,o.levels_completed)
    return int(mx-base)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--games",default=""); ap.add_argument("--budget",type=int,default=120000); ap.add_argument("--out",default="results/e113_multiperception.json"); a=ap.parse_args()
    games=a.games.split(",") if a.games else all_games()
    print(f"[e113] extended multi-perception on {len(games)} games",flush=True); res={}
    for g in tqdm(games,desc="games",ncols=80):
        try:
            avail0=list(arc_agi.Arcade().make(g).reset().available_actions)
            modes=(["click"] if avail0==[6] else (["dir","typednav"] if 6 not in avail0 else ["dir","typednav","click"]))
            best=None; lv=0; s0=None; mode_results=[]; bm=None
            for m in modes:
                rm=solve_game(g,m,a.budget); lvm=verify(g,rm["full"]); s0=rm["s0"]
                mode_results.append((m,rm["table"],lvm))
                if best is None or lvm>lv: best=rm; lv=lvm; bm=m
                if lv>0: break                              # cheap-first: stop once solved
            world,nmem=build_consensus_world(g,mode_results,s0); nodes=-1
            try:
                spec=O.to_spec(world,preview_steps=12); nodes=len(spec.get("preview",{}).get("graph",{}).get("nodes",[]))
                MAPS.mkdir(parents=True,exist_ok=True); O.render_card(world,str(MAPS/f"{g}.svg"))
            except Exception: pass
            res[g]={"game":g,"levels_solved":lv,"win_levels":int(best["win"]),"winning_mode":bm if lv>0 else None,
                    "modes":[m for m,_,_ in mode_results],"map_nodes":nodes,"verified":lv>0,"solution":best["full"] if lv>0 else None}
        except Exception as e:
            import traceback; traceback.print_exc(); res[g]={"game":g,"levels_solved":0,"verified":False,"error":str(e)[:90]}
        rr=res[g]; print(f"  {g}: {rr.get('levels_solved',0)}/{rr.get('win_levels','?')} {'via '+str(rr.get('winning_mode')) if rr.get('verified') else 'no'}",flush=True)
    solved=[g for g,r in res.items() if r.get("verified")]
    Path(a.out).write_text(json.dumps({"solved":solved,"n_solved":len(solved),"results":res},indent=2))
    print(f"[e113] MULTI-PERCEPTION+ SOLVED {len(solved)}/{len(games)}: {sorted(solved)}",flush=True)
if __name__=="__main__": main()
