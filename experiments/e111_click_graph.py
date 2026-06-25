"""E111 -- ARC-AGI-3 solving via a DISCOVERED OPENWORLD MAP (honest pixel-inferred clicks).

Builds the explored state-transition graph as a real openworld.World (the "map"):
 * Perception (Frame Processor): mask status cells (change every step) -> masked-frame signature = state.
 * Click targets INFERRED FROM PIXELS (honest): cells of small connected components (sprites) + rare
   colors -- the Graph Explorer's morphological cues. Non-target clicks are no-ops that dedup away, so
   the graph self-filters to real targets.
 * Exploration: BFS over masked states (deterministic replay nav) -> discovered (state,action)->next
   table + level-up edges.
 * The discovered table becomes an openworld.World via FunctionTransition; to_spec() yields preview.graph
   (the MAP) and render_card() the atlas SVG (viewable in `openworld serve /view`).
 * Solve = captured shortest path to a level-up, replay-verified.

  python3 e111_click_graph.py --games vc33 --budget 6000
"""
import argparse, json, logging, contextlib, io
from collections import deque
from pathlib import Path
import numpy as np
import arc_agi, arc3_graph as G
from arcengine import GameAction
import openworld as O
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

def parse(a):  # action repr -> step
    return ("c",int(a[1:].split("_")[0]),int(a[1:].split("_")[1])) if a[0]=="c" else ("s",int(a[1:]))
def step(env,a):
    k=parse(a)
    return env.step(GameAction.ACTION6,{"x":k[1],"y":k[2]}) if k[0]=="c" else env.step(SIMPLE[k[1]])

def pixel_targets(g, bg, cap=60):
    """HONEST click targets from pixels: cells of small connected components (sprites) + rare colors."""
    objs,_=G.objects(g,bg)
    cells=[]
    for ob in sorted(objs,key=lambda o:o["size"]):
        if ob["size"]<=40:
            r0,c0,r1,c1=ob["bbox"]
            cells += [(c,r) for r in range(r0,r1+1) for c in range(c0,c1+1) if g[r,c]!=bg]
    # add rarest-color cells as fallback
    vals,cnts=np.unique(g,return_counts=True)
    for v,c in sorted([(int(v),int(c)) for v,c in zip(vals,cnts) if int(v)!=bg],key=lambda kc:kc[1])[:3]:
        ys,xs=np.where(g==v); cells+=list(zip(xs.tolist(),ys.tolist()))
    seen=set(); out=[]
    for (x,y) in cells:
        if (x,y) not in seen: seen.add((x,y)); out.append(f"c{x}_{y}")
    return out[:cap]

def candidate_actions(g, bg, avail):
    return pixel_targets(g,bg) if 6 in avail else [f"s{a}" for a in avail if a in SIMPLE]

def status_mask(game, avail, seed=0, probe=14):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); g=g_of(o); chg=np.zeros((64,64)); n=0
    vals,cnts=np.unique(g,return_counts=True); bg=int(vals[np.argmax(cnts)])
    for _ in range(probe):
        acts=candidate_actions(g,bg,avail)
        if not acts: break
        no=step(env,acts[n%len(acts)])
        if no is None or getattr(no,"frame",None) is None: o=env.reset(); g=g_of(o); continue
        ng=g_of(no); chg+=(g!=ng); n+=1; g=ng
        if str(no.state)!="GameState.NOT_FINISHED": o=env.reset(); g=g_of(o)
    return (chg/max(n,1))>0.95, bg

def explore(game, budget=6000, max_depth=40):
    arc=arc_agi.Arcade(); o=arc.make(game).reset(); avail=list(o.available_actions); win=o.win_levels; base0=o.levels_completed
    mask,bg=status_mask(game,avail)
    def sig(g): gg=g.copy(); gg[mask]=0; return hash(gg.tobytes())
    def replay(path):
        e=arc.make(game); ob=e.reset()
        for a in path:
            ob=step(e,a)
            if ob is None or getattr(ob,"frame",None) is None: return None,None
        return e,ob
    ENV=arc.make(game)
    def replay1(path):
        ob=ENV.reset()
        for a in path:
            ob=step(ENV,a)
            if ob is None or getattr(ob,"frame",None) is None: return None
        return ob
    g0=g_of(ENV.reset()); s0=sig(g0)
    table={}; seen={s0}; q=deque([([],s0,g0)]); best=base0; sol=None; steps=0
    while q and steps<budget:
        path,s,g=q.popleft()
        for a in candidate_actions(g,bg,avail):
            if steps>=budget: break
            o2=replay1(path)
            if o2 is None: break
            no=step(ENV,a); steps+=1
            if no is None or getattr(no,"frame",None) is None: continue
            ng=g_of(no); lv=no.levels_completed; ns=sig(ng)
            table.setdefault(s,{})[a]=ns
            if lv>best: best=lv; sol=path+[a]
            if ns not in seen and str(no.state)=="GameState.NOT_FINISHED" and len(path)<max_depth and ns!=s:
                seen.add(ns); q.append((path+[a],ns,ng))
        if best>=win: break
    return {"avail":avail,"win":win,"base0":base0,"best":best,"sol":sol,"steps":steps,
            "table":table,"s0":s0,"states":len(seen)}

def build_world(game, table, s0):
    """The discovered state-transition graph as an openworld.World (the MAP)."""
    acts=sorted({a for d in table.values() for a in d})
    def fn(state, action):
        nm=action.name if hasattr(action,"name") else action
        nxt=table.get(state.get("sig"),{}).get(nm, state.get("sig"))
        return {"sig":nxt}
    return O.World(name=f"arc3-map-{game}", description=f"Discovered reachable-state MAP of ARC-AGI-3 {game}",
                   initial_state={"sig":s0}, actions=acts[:64], transition=FunctionTransition(fn))

def verify(game, sol):
    e=arc_agi.Arcade().make(game); o=e.reset(); base=o.levels_completed
    for a in sol:
        o=step(e,a)
        if o is None or getattr(o,"frame",None) is None: return False
        if o.levels_completed>base: return True
    return False

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--games",default=""); ap.add_argument("--budget",type=int,default=6000); ap.add_argument("--out",default="results/e111_click_graph.json"); a=ap.parse_args()
    games=a.games.split(",") if a.games else all_games()
    print(f"[e111] discovered-OpenWorld-map solving on {len(games)} games",flush=True); res={}
    for g in games:
        try:
            r=explore(g,a.budget)
            ver=verify(g,r["sol"]) if r["sol"] else False
            # build the OpenWorld map + render the atlas card
            world=build_world(g, r["table"], r["s0"]); mappath=None
            try:
                spec=O.to_spec(world, preview_steps=10); nodes=len(spec.get("preview",{}).get("graph",{}).get("nodes",[]))
                MAPS.mkdir(parents=True,exist_ok=True); mappath=str(MAPS/f"{g}.svg"); O.render_card(world, mappath)
            except Exception as ex: nodes=-1; mappath=f"map-err:{str(ex)[:40]}"
            res[g]={"game":g,"best_levels":int(r["best"]),"win_levels":int(r["win"]),"click":6 in r["avail"],
                    "states":r["states"],"steps":r["steps"],"verified":ver,"map_nodes":nodes,"map":mappath,
                    "solution":r["sol"] if ver else None}
        except Exception as e:
            import traceback; traceback.print_exc(); res[g]={"game":g,"verified":False,"error":str(e)[:90]}
        rr=res[g]; print(f"  {g}: best {rr.get('best_levels',0)}/{rr.get('win_levels','?')} {'CLICK' if rr.get('click') else 'dir'} states={rr.get('states','-')} mapnodes={rr.get('map_nodes','-')} {'SOLVED' if rr.get('verified') else ''}",flush=True)
    solved=[g for g,r in res.items() if r.get("verified")]
    Path(a.out).write_text(json.dumps({"solved":solved,"n_solved":len(solved),"results":res},indent=2))
    print(f"[e111] MAP-SOLVED {len(solved)}/{len(games)}: {sorted(solved)}",flush=True)
if __name__=="__main__": main()
