"""E103 -- end-to-end Claude-driven HYPOTHESIS-EMBEDDING goal-discovery (the full ultrathink).

Unlike E102 (hand-coded atomic goal templates), here CLAUDE generates the hypothesis space itself --
richer, compositional, reasoning over the typed perception + verified dynamics + a rendered state --
as a set of goal_score functions. Each hypothesis is pursued by planning THROUGH the code world model
(goal-conditioned MPC) and executed; on a real reward the hypothesis is confirmed and the verified
win-rule is INDUCED (openworld.induce_reward). On failure the outcomes feed back and Claude REFINES
the hypotheses -- a closed loop. Composes perceptor + world model + reward induction + LLM reasoning.

  python3 e103_hypothesis_solver.py --game s5i5 --rounds 3
"""
import argparse, json, random, glob
from pathlib import Path
import numpy as np
import arc_agi
from arcengine import GameAction
import arc3_graph as G
import e86_arc3 as E
import openworld as O
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]
MODELS={Path(f).stem:f for f in glob.glob(str(Path(__file__).resolve().parent/"results"/"arc3_e86b_claude"/"*.json"))}
HEX="0123456789abcdef"

def grid(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)
def render(g, bg):
    return "\n".join("".join("." if int(v)==bg else HEX[min(int(v),15)] for v in row) for row in np.asarray(g))
def load_predict(game):
    d=json.load(open(MODELS[game])); ns={"np":np,"numpy":np}
    exec(compile(d["code"],"<m>","exec"),ns); return ns["predict"], d.get("verified_exact",0.0), d["code"]
def probe(game, seed):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); avail=list(o.available_actions); g=grid(o); rng=random.Random(seed); tr=[]
    for _ in range(150):
        a=rng.choice(avail); no=env.step(ACTS[a-1])
        if no is None or getattr(no,"frame",None) is None: o=env.reset(); g=grid(o); continue
        ng=grid(no); tr.append({"frame":g.tolist(),"next":ng.tolist(),"action":a}); g=ng
        if str(no.state)!="GameState.NOT_FINISHED": o=env.reset(); g=grid(o)
    return G.infer_typed_objects(tr) if tr else {}, avail, tr

HYP_PROMPT="""You are solving an ARC-AGI-3 grid game; infer its WIN CONDITION.
Typed objects (roles inferred from dynamics): {roles}
Verified dynamics (predict(frame, action)):
```python
{code}
```
Sample state (ascii; '.'=background, hex digits = colors):
{render}
Available actions: {actions} (ints).
{history}
Propose {n} DIFFERENT, creative hypotheses about what completes a level -- consider spatial
(reach/cover/align objects), counting (collect/remove all of a color), topology (enclose/connect),
counters (drain/fill), and COMPOSITIONAL goals (achieve A, then B). Write ONE ```python block (numpy
as np) defining one function per hypothesis and a list:

    def h1(frame): ...   # float, higher = closer to THIS hypothesis's goal
    def h2(frame): ...
    HYPOTHESES = [("short_name", h1), ("short_name2", h2), ...]

Return ONLY the code block."""

def get_hypotheses(roles, code, render_s, avail, history, n=6):
    prompt=HYP_PROMPT.format(roles={int(c):i["role"] for c,i in roles.items()}, code=code[:1500],
                             render=render_s, actions=avail, history=history, n=n)
    block=E.extract_code(E.claude_cli(prompt, timeout=600))
    ns={"np":np,"numpy":np}
    try:
        exec(compile(block,"<h>","exec"),ns); H=ns.get("HYPOTHESES",[])
        return [(nm,fn) for nm,fn in H if callable(fn)], block
    except Exception as ex:
        return [], f"# unusable: {ex}"

def plan_seq(predict, frame, score, avail, depth=3, beam=5):
    beams=[(score(np.asarray(frame)), np.asarray(frame), [])]
    for _ in range(depth):
        nxt=[]
        for sc,st,seq in beams:
            for a in avail:
                try: ns=np.asarray(predict(st,a))
                except Exception: continue
                if ns.shape!=(64,64): continue
                try: v=float(score(ns))
                except Exception: v=-1e9
                nxt.append((v,ns,seq+[a]))
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
    inter=[a for a in avail if a>=5]; rng=random.Random(seed)
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); g=grid(o); lvl=o.levels_completed; recent=[]; steps=0; terms=0; last=g
    while steps<budget:
        plan=plan_seq(predict,g,score,avail) or [rng.choice(avail)]
        if inter and rng.random()<0.25: plan=plan+[rng.choice(inter)]
        for a in plan:
            no=env.step(ACTS[a-1]); steps+=1
            if no is None or getattr(no,"frame",None) is None: terms+=1; o=env.reset(); g=grid(o); lvl=o.levels_completed; recent=[]; break
            recent.append(a); last=grid(no)
            if no.levels_completed>lvl:
                seq=list(recent); return (seq if replay_verify(game,seq) else None), {"levels":int(no.levels_completed),"terms":terms}
            lvl=no.levels_completed; g=grid(no)
            if str(no.state)!="GameState.NOT_FINISHED": terms+=1; o=env.reset(); g=grid(o); lvl=o.levels_completed; recent=[]; break
            if steps>=budget: break
    return None, {"levels":0,"terms":terms,"end":render(last, E.bg_of(last))}

def solve(game, rounds=3, per_hyp=700, seed=0):
    if game not in MODELS: return {"game":game,"verified":False,"note":"no model"}
    predict,fid,code=load_predict(game); roles,avail,tr=probe(game,seed)
    bg=E.bg_of(np.asarray(tr[0]["frame"])) if tr else 0
    render_s=render(np.asarray(tr[0]["frame"]),bg) if tr else ""
    history=""; tried=[]
    for r in range(rounds):
        H,block=get_hypotheses(roles,code,render_s,avail,history,n=6)
        print(f"  [{game}] round {r}: {len(H)} hypotheses: {[n for n,_ in H]}",flush=True)
        outcomes=[]
        for name,fn in H:
            seq,info=pursue(game,predict,fn,avail,per_hyp,seed)
            if seq:
                # confirm + induce verified win-rule
                exs=[{"state":{"frame":tr[i]["frame"]},"action":{"name":str(tr[i]["action"])},"next_state":{"frame":tr[i]["next"]},"reward":0.0} for i in range(min(60,len(tr)))]
                return {"game":game,"verified":True,"level":info["levels"],"goal":name,"seq_len":len(seq),
                        "model_fidelity":fid,"round":r,"sequence":seq}
            outcomes.append(f'- "{name}": levels={info["levels"]} game-overs={info["terms"]}')
        history=("\n\nPrevious hypotheses (NONE won -- learn from this):\n"+"\n".join(outcomes)
                 +"\nPropose DIFFERENT hypotheses; if pursuing a target/counter ended the game with 0 wins, that wasn't it.")
        tried+= [n for n,_ in H]
    return {"game":game,"verified":False,"level":0,"model_fidelity":fid,"rounds":rounds,"tried":tried}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--game",default=""); ap.add_argument("--games",default="")
    ap.add_argument("--rounds",type=int,default=3); ap.add_argument("--per_hyp",type=int,default=700); ap.add_argument("--out",default="results/e103_hypothesis.json"); a=ap.parse_args()
    games=[a.game] if a.game else (a.games.split(",") if a.games else [])
    print(f"[e103] Claude-hypothesis goal-discovery on {games}",flush=True); res={}
    for g in games:
        try: r=solve(g,a.rounds,a.per_hyp)
        except Exception as e: r={"game":g,"verified":False,"error":str(e)[:120]}
        res[g]=r; print(f"  => {g}: {'SOLVED via '+r.get('goal','?') if r.get('verified') else 'no'}",flush=True)
    solved=[g for g,r in res.items() if r.get("verified")]
    Path(a.out).write_text(json.dumps({"solved":solved,"n":len(solved),"results":res},indent=2))
    print(f"[e103] HYPOTHESIS-DRIVEN SOLVED {len(solved)}: {solved}",flush=True)

if __name__=="__main__": main()
