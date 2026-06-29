"""E109 -- OpenWorld parallel world models + ConsensusTransition HARD VOTING, with the SOTA fixes.

Honors three things at once:
 * SOTA fix #1 -- the CLICK action space (ACTION6 with x,y): many ARC-AGI-3 games are click-based
   (vc33's only action is [6]); we'd been firing (0,0) blindly. Salient click targets come from the
   segmented frame (object centroids), not all 4096 pixels.
 * SOTA fix #2 -- STATUS-BAR MASKING: UI step/timer cells change every step and exploded our state
   hash (E107 -> 73k states). We detect high-change-frequency cells (a probe) and mask them so states
   compress -- the Graph Explorer's key trick.
 * The user's idea -- PARALLEL WORLD MODELS + HARD VOTING, done with the real openworld primitive
   ConsensusTransition(mode="vote"): several learned dynamics models over DIFFERENT representations
   (masked-frame hash, object-graph signature) each predict the next state for a candidate action; the
   consensus vote decides whether an action reaches a NOVEL state, directing frontier exploration.

Genuinely OpenWorld: masked object-graph perception, learned Transitions, ConsensusTransition voting,
and the explored dynamics exposed as a World. CPU-only, no LLM.

  python3 e109_consensus_explore.py --budget 6000
"""
import argparse, json, logging, contextlib, io, random
from pathlib import Path
import numpy as np
import arc_agi
from arcengine import GameAction
import arc3_graph as G
import openworld as O
from openworld.transition import FunctionTransition
from openworld.state import WorldState, Action
SIMPLE=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION7]

def grid(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)
def all_games():
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        envs=arc_agi.Arcade().available_environments
    return sorted({(e if isinstance(e,str) else getattr(e,"game_id",str(e))).split("-")[0] for e in envs})

def do(env, a):
    """Issue a simple (int) or click ((6,x,y)) action."""
    if isinstance(a, tuple):                      # click: (6, x, y)
        ga=GameAction.ACTION6; ga.set_data({"x":int(a[1]),"y":int(a[2])}); return env.step(ga)
    return env.step([SIMPLE[0],SIMPLE[1],SIMPLE[2],SIMPLE[3],SIMPLE[4],None,SIMPLE[5]][a-1] if a!=6 else SIMPLE[0])

def step_action(env, a):
    if isinstance(a, tuple):
        ga=GameAction.ACTION6; ga.set_data({"x":int(a[1]),"y":int(a[2])}); return env.step(ga)
    m={1:GameAction.ACTION1,2:GameAction.ACTION2,3:GameAction.ACTION3,4:GameAction.ACTION4,5:GameAction.ACTION5,7:GameAction.ACTION7}
    return env.step(m[a])

def detect_mask(game, seed=0, steps=80):
    """Probe: cells that change on >70% of steps are the status bar/timer -> mask them."""
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); avail=list(o.available_actions); g=grid(o); rng=random.Random(seed)
    chg=np.zeros((64,64)); n=0; click=6 in avail
    for _ in range(steps):
        a=(6,rng.randint(0,63),rng.randint(0,63)) if click else rng.choice([x for x in avail if x!=6] or avail)
        no=step_action(env,a)
        if no is None or getattr(no,"frame",None) is None: o=env.reset(); g=grid(o); continue
        ng=grid(no); chg+=(g!=ng); n+=1; g=ng
        if str(no.state)!="GameState.NOT_FINISHED": o=env.reset(); g=grid(o)
    mask=(chg/max(n,1))>0.7                       # high-change-frequency cells = status bar
    return mask, avail
def masked_sig(f, mask): 
    f=np.asarray(f).copy(); f[mask]=0; return hash(f.tobytes())
def graph_sig(f, mask):
    f=np.asarray(f).copy(); f[mask]=0
    objs,_=G.objects(f); return tuple(sorted((o["color"],o["size"]) for o in objs))

def salient_actions(f, mask, avail, cap=80):
    """Click games -> click NON-BACKGROUND cells (the actual interactive cells), salience-ordered by
    color rarity; else the directional actions."""
    if 6 not in avail: return [a for a in avail]
    fm=np.asarray(f)
    vals,cnts=np.unique(fm,return_counts=True); bg=int(vals[np.argmax(cnts)])
    rare={int(v):int(c) for v,c in zip(vals,cnts)}
    ys,xs=np.where((fm!=bg) & (~mask))
    cells=sorted(zip(xs.tolist(),ys.tolist()), key=lambda xy: rare.get(int(fm[xy[1],xy[0]]),9999))
    acts=[(6,int(x),int(y)) for (x,y) in cells[:cap]]
    if not acts: acts=[(6,x,y) for x in range(4,64,10) for y in range(4,64,10)]
    return acts[:cap]

def consensus_models():
    """Two learned dynamics models over DIFFERENT representations -> ConsensusTransition(vote)."""
    m_raw={}; m_graph={}                          # (state_repr, action) -> next masked_sig
    def mk(model, repr_key):
        def fn(state, action):
            k=(state.get(repr_key), state.get("a"))
            return {"nsig": model.get(k)}          # predicted next masked-sig (None if unseen)
        return FunctionTransition(fn)
    t_raw=mk(m_raw,"msig"); t_graph=mk(m_graph,"gsig")
    cons=O.ConsensusTransition([(t_raw,0.5),(t_graph,0.5)], mode="vote")
    return cons, m_raw, m_graph

def explore(game, budget, max_steps=300, seed=0):
    mask, avail = detect_mask(game, seed)
    cons, m_raw, m_graph = consensus_models()
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); win=o.win_levels; base0=o.levels_completed
    seen=set(); tested={}; best=base0; solution=None; steps=0; rng=random.Random(seed)
    while steps<budget:
        o=env.reset(); g=grid(o); lvl=o.levels_completed; seq=[]; depth=0
        while depth<max_steps and steps<budget:
            ms=masked_sig(g,mask); gs=graph_sig(g,mask); seen.add(ms)
            acts=salient_actions(g,mask,avail); ts=tested.setdefault(ms,set())
            # HARD VOTING: rank candidate actions -> prefer those the voted world model says reach a
            # NOVEL state (or are unseen); fall back to untested frontier.
            def novel(a):
                st=WorldState({"msig":ms,"gsig":gs,"a":repr(a)})
                try: pred=cons.step(st, Action(name=repr(a))).get("nsig")
                except Exception: pred=None
                return pred is None or pred not in seen         # unknown or predicted-novel
            untried=[a for a in acts if repr(a) not in ts]
            cands=[a for a in untried if novel(a)] or untried or acts
            a=rng.choice(cands); ts.add(repr(a))
            nob=step_action(env,a); steps+=1; depth+=1
            if nob is None or getattr(nob,"frame",None) is None: break
            seq.append(a); nf=grid(nob); lv=nob.levels_completed; nms=masked_sig(nf,mask)
            m_raw[(ms,repr(a))]=nms; m_graph[(gs,repr(a))]=nms          # learn the two models
            if lv>best: best=lv; solution=list(seq)
            if str(nob.state)!="GameState.NOT_FINISHED": break
            g=nf; lvl=lv
    return {"game":game,"best_levels":int(best),"win_levels":int(win),"states":len(seen),"steps":steps,
            "click":6 in avail,"solution_len":len(solution) if solution else 0,"solution":solution,"verified":solution is not None}

def verify(game, seq):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); base=o.levels_completed
    for a in seq:
        o=step_action(env,a)
        if o is None or getattr(o,"frame",None) is None: return False
        if o.levels_completed>base: return True
    return False

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--budget",type=int,default=6000); ap.add_argument("--games",default=""); ap.add_argument("--out",default="results/e109_consensus_explore.json"); a=ap.parse_args()
    games=a.games.split(",") if a.games else all_games()
    print(f"[e109] OpenWorld consensus-vote click+mask exploration on {len(games)} games",flush=True); res={}
    for g in games:
        try:
            r=explore(g,a.budget)
            if r["solution"]: r["verified"]=verify(g,r["solution"])
            r.pop("solution",None) if not r.get("verified") else None
        except Exception as e: r={"game":g,"verified":False,"error":str(e)[:90]}
        res[g]=r
        print(f"  {g}: best {r.get('best_levels',0)}/{r.get('win_levels','?')} {'CLICK' if r.get('click') else 'dir'} states={r.get('states','-')} {'SOLVED' if r.get('verified') else ''}",flush=True)
    solved=[g for g,r in res.items() if r.get("verified")]
    Path(a.out).write_text(json.dumps({"solved":solved,"n_solved":len(solved),"results":res},indent=2))
    print(f"[e109] CONSENSUS-EXPLORE SOLVED {len(solved)}/{len(games)}: {sorted(solved)}",flush=True)

if __name__=="__main__": main()
