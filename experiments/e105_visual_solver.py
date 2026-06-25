"""E105 -- VISUAL interactive solver: the one untried lever. Render ARC-AGI-3 frames in the OFFICIAL
palette and let Claude reason VISUALLY about the goal (its ARC strength), instead of text/structured
representations (E92/E102/E103/E104, all of which failed on the walled games). Closed-loop: render
state -> Claude (vision) proposes a plan + goal hypothesis -> execute -> render -> repeat.

  python3 e105_visual_solver.py --game ka59 --turns 25
"""
import argparse, json, re, subprocess
from pathlib import Path
import numpy as np
import arc_agi
from arcengine import GameAction
from arc_agi.rendering import frame_to_rgb_array
from PIL import Image
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]
REPO=Path("/Users/jim/Desktop/openworld"); SCRATCH=REPO/"scratch_arc"

def grid(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)

def render_png(g, path):
    rgb=np.asarray(frame_to_rgb_array(np.asarray(g)))
    if rgb.dtype!=np.uint8: rgb=(np.clip(rgb,0,1)*255).astype(np.uint8) if rgb.max()<=1 else rgb.astype(np.uint8)
    Image.fromarray(rgb).resize((320,320),Image.NEAREST).save(path)

def claude_vision(prompt, timeout=300):
    r=subprocess.run(["claude","-p",prompt],capture_output=True,text=True,timeout=timeout); return r.stdout

PROMPT="""You are PLAYING an ARC-AGI-3 grid game and must complete a LEVEL. Look at the rendered game.
Current state image: {cur}
{prev}
Available actions (integers): {avail}. Levels completed so far: {lvl} (raising it is the only success).
Recent turns (plan -> level gained): {hist}
Your running goal hypothesis: {notes}

Read the image(s), reason VISUALLY about what completes a level (what object is the agent? what's the
target/goal configuration?), then output EXACTLY:
PLAN: <comma-separated action integers, 1 to {pmax}>
NOTES: <updated visual goal hypothesis to remember>"""

def parse(resp, avail, pmax):
    pm=re.search(r"PLAN:\s*([0-9,\s]+)",resp); nm=re.search(r"NOTES:\s*(.+)",resp,re.S)
    plan=[int(t) for t in (pm.group(1).split(",") if pm else []) if t.strip().isdigit() and int(t) in avail]
    return (plan[:pmax] or [avail[0]]), (nm.group(1).strip()[:500] if nm else "")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--game",default="ka59"); ap.add_argument("--turns",type=int,default=25); ap.add_argument("--plan",type=int,default=8); ap.add_argument("--out",default=""); a=ap.parse_args()
    arc=arc_agi.Arcade(); env=arc.make(a.game); o=env.reset(); avail=list(o.available_actions); g=grid(o); win=o.win_levels; best=o.levels_completed
    notes="(none)"; hist=[]; solved=None; prevp=None
    curp=SCRATCH/f"{a.game}_cur.png"; render_png(g,curp)
    print(f"[e105/{a.game}] VISUAL play, {a.turns} turns (official-palette rendering + Claude vision)",flush=True)
    for t in range(a.turns):
        prev=f"Previous state image (for comparison): {prevp}" if prevp else ""
        prompt=PROMPT.format(cur=curp,prev=prev,avail=avail,lvl=best,hist="; ".join(hist[-4:]) or "(none)",notes=notes,pmax=a.plan)
        try: resp=claude_vision(prompt)
        except Exception as e: print(f"  turn {t}: claude err {e}",flush=True); continue
        plan,notes=parse(resp,avail,a.plan); before=best; prevp=SCRATCH/f"{a.game}_prev.png"; render_png(g,prevp)
        for ac in plan:
            o=env.step(ACTS[ac-1])
            if o is None or getattr(o,"frame",None) is None: o=env.reset(); g=grid(o); avail=list(o.available_actions); break
            g=grid(o); best=max(best,o.levels_completed)
            if str(o.state)!="GameState.NOT_FINISHED": o=env.reset(); g=grid(o); avail=list(o.available_actions)
        render_png(g,curp); hist.append(f"t{t} {plan}->lvl{best}")
        print(f"  turn {t}: plan={plan} levels={best}/{win} | {notes[:75]}",flush=True)
        if best>before and solved is None: solved=t; print(f"  *** LEVEL COMPLETED turn {t} ***",flush=True)
    res={"game":a.game,"best_levels":best,"win_levels":win,"solved_turn":solved,"verified":best>0,"final_notes":notes}
    out=Path(a.out) if a.out else Path("results")/f"e105_visual_{a.game}.json"; out.write_text(json.dumps(res,indent=2))
    print(f"[e105/{a.game}] best {best}/{win} solved_turn={solved}",flush=True); print("wrote",out)

if __name__=="__main__": main()
