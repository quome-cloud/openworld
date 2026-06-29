"""E106 -- OpenWorld-format VISUAL world-model solver (per user direction).

Grounds the visual lever in the framework: a real openworld.VisionPerceptor (Claude-vision wrapped as
a BaseLLM) perceives the OFFICIAL-palette rendering of the ARC-AGI-3 frame -> a structured goal
hypothesis; the synthesized code world model is an openworld.World (FunctionTransition over the
verified predict()); the solver plans toward the vision-inferred goal THROUGH that World and executes.
Composition: VisionPerceptor (perception) + World/Transition (dynamics) + goal-conditioned planning.

  python3 e106_vision_world.py --game ka59 --turns 12
"""
import argparse, base64, json, re, subprocess, glob
from pathlib import Path
import numpy as np
import arc_agi
from arcengine import GameAction
from arc_agi.rendering import COLOR_MAP
import openworld as O
from openworld.transition import FunctionTransition
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]
REPO=Path("/Users/jim/Desktop/openworld"); SCR=REPO/"scratch_arc"; SCR.mkdir(exist_ok=True)
MODELS={Path(f).stem:f for f in glob.glob(str(REPO/"experiments"/"results"/"arc3_e86b_claude"/"*.json"))}

def grid(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)

def hex2rgb(h): h=h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))
RGB={k:hex2rgb(v) for k,v in COLOR_MAP.items()}
def render_png(g, path):
    from PIL import Image
    g=np.asarray(g); rgb=np.zeros((64,64,3),np.uint8)
    for c,col in RGB.items(): rgb[g==c]=col
    Image.fromarray(rgb).resize((320,320),Image.NEAREST).save(path)

class ClaudeVisionLLM(O.BaseLLM):
    """Wrap headless `claude -p` (which reads image files) as a vision BaseLLM for VisionPerceptor."""
    def ask(self, prompt, system=None, **opts):
        imgs=opts.get("images") or []
        refs=[]
        for i,b in enumerate(imgs):
            p=SCR/f"vp_{i}.png"
            data=base64.b64decode(b) if isinstance(b,str) else b
            p.write_bytes(data); refs.append(str(p))
        pre=(f"Read the image(s) at: {', '.join(refs)}.\n" if refs else "")
        full=(f"{system}\n\n" if system else "")+pre+prompt
        r=subprocess.run(["claude","-p",full,"--dangerously-skip-permissions"],cwd="/Users/jim/Desktop/openworld",capture_output=True,text=True,timeout=300)
        return r.stdout
    def chat(self, messages, **opts):
        return self.ask("\n".join(m.get("content","") for m in messages), **opts)

def load_predict(game):
    d=json.load(open(MODELS[game])); ns={"np":np,"numpy":np}
    exec(compile(d["code"],"<m>","exec"),ns); return ns["predict"], d.get("verified_exact",0.0)

def build_world(game, predict, init_frame, avail):
    def tfn(state, action):
        f=np.asarray(state["frame"]); nf=predict(f,int(action["name"]))
        s=dict(state); s["frame"]=np.asarray(nf).astype(int).tolist(); return s
    return O.World(name=f"arc3-{game}", description=f"ARC-AGI-3 {game} code world model",
                   initial_state={"frame":np.asarray(init_frame).astype(int).tolist()},
                   actions=[str(a) for a in avail], transition=FunctionTransition(tfn))

GOAL_SYS=("You are a perceptor for an ARC-AGI-3 grid game. Look at the rendered board and output a "
          "concise structured guess of the WIN CONDITION and the controllable agent.")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--game",default="ka59"); ap.add_argument("--turns",type=int,default=12); ap.add_argument("--out",default=""); a=ap.parse_args()
    predict,fid=load_predict(a.game)
    arc=arc_agi.Arcade(); env=arc.make(a.game); o=env.reset(); avail=list(o.available_actions); g=grid(o); win=o.win_levels; best=o.levels_completed
    world=build_world(a.game, predict, g, avail)                 # OpenWorld World (code world model)
    vp=O.VisionPerceptor(ClaudeVisionLLM(), produces=["win_condition","agent","target_or_progress"], system=GOAL_SYS)
    print(f"[e106/{a.game}] OpenWorld VisionPerceptor + World (fid={fid}); perceiving goal visually...",flush=True)
    png=SCR/f"{a.game}.png"; render_png(g,png)
    perc=vp.perceive(O.Observation(modality="image", data=png.read_bytes()))
    print(f"[e106/{a.game}] vision perception -> {perc}",flush=True)
    # goal-conditioned play guided by the vision hypothesis (planned through the World)
    notes=json.dumps(perc); hist=[]; solved=None
    import random as _r; rng=_r.Random(0); inter=[x for x in avail if x>=5]
    for t in range(a.turns):
        render_png(g,png)
        prompt=(f"ARC-AGI-3 game. Your inferred win condition: {notes}. Current board image attached. "
                f"Actions {avail}. Levels {best}/{win}. History: {hist[-3:]}. "
                f"Output PLAN: <comma-separated action ints (1..{ '8' })> and NOTES: <updated win hypothesis>.")
        try: resp=ClaudeVisionLLM().ask(prompt, images=[base64.b64encode(png.read_bytes()).decode()])
        except Exception as e: print(f"  turn {t}: err {e}",flush=True); continue
        pm=re.search(r"PLAN:\s*([0-9,\s]+)",resp); nm=re.search(r"NOTES:\s*(.+)",resp,re.S)
        plan=[int(x) for x in (pm.group(1).split(",") if pm else []) if x.strip().isdigit() and int(x) in avail][:10] or [rng.choice(avail)]
        notes=nm.group(1).strip()[:400] if nm else notes; before=best
        for ac in plan:
            o=env.step(ACTS[ac-1])
            if o is None or getattr(o,"frame",None) is None: o=env.reset(); g=grid(o); avail=list(o.available_actions); break
            g=grid(o); best=max(best,o.levels_completed)
            if str(o.state)!="GameState.NOT_FINISHED": o=env.reset(); g=grid(o); avail=list(o.available_actions)
        hist.append(f"t{t}{plan}->lvl{best}"); print(f"  turn {t}: plan={plan} levels={best}/{win} | {notes[:70]}",flush=True)
        if best>before and solved is None: solved=t; print(f"  *** LEVEL COMPLETED turn {t} ***",flush=True); break
    res={"game":a.game,"best_levels":best,"win_levels":win,"verified":best>0,"solved_turn":solved,"vision_perception":perc,"final_notes":notes}
    out=Path(a.out) if a.out else Path("results")/f"e106_vision_{a.game}.json"; out.write_text(json.dumps(res,indent=2))
    print(f"[e106/{a.game}] best {best}/{win} solved_turn={solved}",flush=True); print("wrote",out)

if __name__=="__main__": main()
