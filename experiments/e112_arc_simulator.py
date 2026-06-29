"""E112 -- OpenWorld ARC-AGI-3 SIMULATOR (world-time compute via a discovered map).

Thesis: spend inference-time compute BUILDING + SEARCHING a discovered world model (the OpenWorld map),
not a trained policy -- more compute -> more levels solved. For each game:
  1. Perceive: detect status cells (change ~every step) -> mask; masked-frame hash = state (perceptor).
  2. Action model: directional (simple actions) AND/OR click targets inferred FROM PIXELS (small
     connected components + rare colors). Hybrid games use both. (Honest: no engine valid-action API.)
  3. Sequential level solving: BFS from the current frontier to the NEXT level-up; chain level by level
     (handles the compositionality cliff). Single env, reset+replay (never arc.make in a loop).
  4. The discovered (state,action)->next table becomes an openworld.World (FunctionTransition); reward =
     levels_completed (ground truth) / induced CodeObjective; to_spec -> preview.graph = the MAP.
  5. Replay-verify the full solution; serialize the map (render_card atlas) per game.

  python3 e112_arc_simulator.py --games vc33,ar25,ls20 --budget 5000
  python3 e112_arc_simulator.py --budget 5000          # all 25
"""
import argparse, json, logging, contextlib, io
from collections import deque
from pathlib import Path
import numpy as np
import arc_agi, arc3_graph as G
from arcengine import GameAction
import openworld as O
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x=None,**k):
        return x if x is not None else None
from openworld.transition import FunctionTransition
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

def click_targets(g,bg,cap=70):
    objs,_=G.objects(g,bg); cells=[]
    for ob in sorted(objs,key=lambda o:o["size"]):
        if ob["size"]<=40:
            r0,c0,r1,c1=ob["bbox"]; cells+=[(c,r) for r in range(r0,r1+1) for c in range(c0,c1+1) if g[r,c]!=bg]
    vals,cnts=np.unique(g,return_counts=True)
    for v,_ in sorted([(int(v),int(c)) for v,c in zip(vals,cnts) if int(v)!=bg],key=lambda kc:kc[1])[:4]:
        ys,xs=np.where(g==v); cells+=list(zip(xs.tolist(),ys.tolist()))
    seen=set(); out=[]
    for (x,y) in cells:
        if (x,y) not in seen: seen.add((x,y)); out.append(f"c{x}_{y}")
    return out[:cap]
def action_model(g,bg,avail,mode="both"):
    acts=[]
    if 6 in avail and mode in ("click","both"): acts+=click_targets(g,bg)
    if mode in ("dir","both"): acts+=[f"s{a}" for a in avail if a in SIMPLE]
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

def solve_game(game, mode, budget=150000, max_depth=400, seed=0):
    """Frontier-biased rollouts (deep + state-dedup) over the correct action model -> solves shallow
    click games AND deep directional games; chains levels naturally. Builds the discovered table (map)."""
    import random as _r
    arc=arc_agi.Arcade(); ENV=arc.make(game); o=ENV.reset(); avail=list(o.available_actions); win=o.win_levels; base=o.levels_completed
    mask,bg=detect_mask(ENV,avail)
    keep=(~mask).reshape(-1)                          # unmasked cells (status bar excluded)
    def sig(g): return hash(g.reshape(-1)[keep].tobytes())   # fast: no full-frame copy
    s0=sig(g_of(ENV.reset()))
    rng=_r.Random(seed); tested={}; table={}; best=base; best_path=[]; steps=0; acache={}; STATE_CAP=40000
    bar=tqdm(total=budget, desc=f"{game}:{mode}", unit="step", leave=False, ncols=80)
    while steps<budget and best<win and len(tested)<STATE_CAP:
        ob=ENV.reset(); g=g_of(ob); seq=[]; d=0
        while d<max_depth and steps<budget:
            s=sig(g)
            acts=acache.get(s)
            if acts is None: acts=action_model(g,bg,avail,mode); acache[s]=acts
            if not acts: break
            ts=tested.setdefault(s,set()); unt=[a for a in acts if a not in ts]
            a=rng.choice(unt) if unt else rng.choice(acts); ts.add(a)        # FRONTIER bias
            no=step(ENV,a); steps+=1; d+=1
            if no is None or getattr(no,"frame",None) is None: break
            seq.append(a); ng=g_of(no); table.setdefault(s,{})[a]=sig(ng)
            if no.levels_completed>best: best=no.levels_completed; best_path=list(seq)
            if str(no.state)!="GameState.NOT_FINISHED": break
            g=ng
        if hasattr(bar,"update"): bar.update(steps-bar.n); bar.set_postfix(best=best, states=len(tested))
    if hasattr(bar,"close"): bar.close()
    return {"avail":avail,"win":win,"base":base,"level":best,"full":best_path,"steps":steps,"table":table,"s0":s0,"click":6 in avail}

def build_world(game, table, s0):
    acts=sorted({a for d in table.values() for a in d})[:80]
    def fn(state, action):
        nm=action.get("name") if isinstance(action,dict) else (action.name if hasattr(action,"name") else action)
        return {"sig": table.get(state.get("sig"),{}).get(nm, state.get("sig"))}
    return O.World(name=f"arc3-map-{game}", description=f"Discovered reachable-state map of ARC-AGI-3 {game}",
                   initial_state={"sig":s0}, actions=acts or ["noop"], transition=FunctionTransition(fn))

def mode_transition(table):
    def fn(state, action):
        nm=action.get("name") if isinstance(action,dict) else (action.name if hasattr(action,"name") else action)
        return {"sig": table.get(state.get("sig"),{}).get(nm, state.get("sig"))}
    return FunctionTransition(fn)

def build_consensus_world(game, mode_results, s0):
    """Hard voting across MODALITIES as openworld.ConsensusTransition: each modality's discovered
    dynamics is a member Transition weighted by its fidelity (levels solved); mode='select' uses the
    highest-fidelity modality (vote available for per-state agreement). The combined World is the
    multi-modal discovered map."""
    members=[(mode_transition(tbl), float(fid)) for (m,tbl,fid) in mode_results if tbl]
    if not members: members=[(mode_transition({}),0.0)]
    cons=O.ConsensusTransition(members, mode="select")
    acts=sorted({a for (m,tbl,fid) in mode_results for d in tbl.values() for a in d})[:80]
    w=O.World(name=f"arc3-map-{game}", description=f"Discovered multi-modal map of ARC-AGI-3 {game} (ConsensusTransition hard-vote over {len(members)} modalities)",
              initial_state={"sig":s0}, actions=acts or ["noop"], transition=cons)
    return w, len(members)

def verify(game, full):
    if not full: return 0
    e=arc_agi.Arcade().make(game); o=e.reset(); base=o.levels_completed; mx=base
    for a in full:
        o=step(e,a)
        if o is None or getattr(o,"frame",None) is None: break
        mx=max(mx,o.levels_completed)
    return int(mx-base)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--games",default=""); ap.add_argument("--budget",type=int,default=150000); ap.add_argument("--out",default="results/e112_arc_simulator.json"); a=ap.parse_args()
    games=a.games.split(",") if a.games else all_games()
    print(f"[e112] OpenWorld ARC-3 simulator on {len(games)} games (budget {a.budget})",flush=True); res={}
    for gi,g in enumerate(tqdm(games, desc='games', ncols=80)):
        try:
            avail0=list(arc_agi.Arcade().make(g).reset().available_actions)
            modes=(["click"] if avail0==[6] else (["dir"] if 6 not in avail0 else ["dir","click"]))
            best=None; lv=0; s0=None; mode_results=[]
            for m in modes:
                rm=solve_game(g,m,a.budget); lvm=verify(g,rm["full"]); s0=rm["s0"]
                mode_results.append((m,rm["table"],lvm))
                if best is None or lvm>lv: best=rm; lv=lvm; best_mode=m
                if lv>0 and m=="dir": break
            # HARD VOTING across modalities as ConsensusTransition -> the combined OpenWorld map
            world,nmem=build_consensus_world(g,mode_results,s0); nodes=-1
            try:
                spec=O.to_spec(world,preview_steps=12); nodes=len(spec.get("preview",{}).get("graph",{}).get("nodes",[]))
                MAPS.mkdir(parents=True,exist_ok=True); O.render_card(world,str(MAPS/f"{g}.svg"))
            except Exception: pass
            res[g]={"game":g,"levels_solved":lv,"win_levels":int(best["win"]),"reached":int(best["level"]),
                    "modes":[m for m,_,_ in mode_results],"winning_mode":best_mode if lv>0 else None,
                    "consensus_members":nmem,"steps":best["steps"],"map_nodes":nodes,"solution_len":len(best["full"]),
                    "verified":lv>0,"solution":best["full"] if lv>0 else None}
        except Exception as e:
            import traceback; traceback.print_exc(); res[g]={"game":g,"levels_solved":0,"verified":False,"error":str(e)[:90]}
        rr=res[g]; print(f"  {g}: {rr.get('levels_solved',0)}/{rr.get('win_levels','?')} levels {'CLICK' if rr.get('click') else 'dir'} steps={rr.get('steps','-')} {'SOLVED' if rr.get('verified') else ''}",flush=True)
    solved=[g for g,r in res.items() if r.get("verified")]
    tot=sum(r.get("levels_solved",0) for r in res.values())
    Path(a.out).write_text(json.dumps({"games_with_a_solve":solved,"n_games":len(solved),"total_levels":tot,"results":res},indent=2))
    print(f"[e112] SIMULATOR: {len(solved)}/{len(games)} games with >=1 level; {tot} total levels solved",flush=True)
if __name__=="__main__": main()
