"""Fig: a worked example -- ka59 rendered in the official palette, with the agent's verified solution.
Shows what a game and a 'hybrid' solution actually look like. Run with the arc-agi venv."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import arc_agi
from arcengine import GameAction, enums
from arc_agi.rendering import COLOR_MAP
FIG=Path("/Users/jim/Desktop/openworld/papers/arc-3/figs"); FIG.mkdir(parents=True,exist_ok=True)
def hex2rgb(h): h=h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))
RGB={k:hex2rgb(v) for k,v in COLOR_MAP.items()}
def g_of(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)
def render(g):
    rgb=np.zeros((64,64,3),np.uint8)
    for c,col in RGB.items(): rgb[np.asarray(g)==c]=col
    return rgb
SIMPLE={1:GameAction.ACTION1,2:GameAction.ACTION2,3:GameAction.ACTION3,4:GameAction.ACTION4,5:GameAction.ACTION5,7:GameAction.ACTION7}
def step(env,a):
    return env.step(GameAction.ACTION6,{"x":a[1],"y":a[2]}) if a[0]==6 else env.step(SIMPLE[a[0]])
sol=json.load(open("/Users/jim/Desktop/openworld/experiments/results/agent_solves/ka59.json"))["actions"]
env=arc_agi.Arcade().make("ka59"); o=env.reset(); f0=g_of(o)
# capture: initial, after directional nav (before the click), after full solve
click_idx=next(i for i,a in enumerate(sol) if a[0]==6)
o=env.reset()
for a in sol[:click_idx]: o=step(env,a)
f1=g_of(o); base=o.levels_completed
for a in sol[click_idx:]: o=step(env,a)
f2=g_of(o); solved=o.levels_completed>0
fig,ax=plt.subplots(1,3,figsize=(12,4.4))
for a in ax: a.set_xticks([]); a.set_yticks([])
ax[0].imshow(render(f0)); ax[0].set_title("(a) initial frame", fontsize=11, fontweight="bold")
ax[0].text(0.5,-0.08,"two rooms, a purple door, a green\nagent, a target square",transform=ax[0].transAxes,ha="center",fontsize=8.5,color="#334155")
ax[1].imshow(render(f1)); ax[1].set_title(f"(b) after navigation\n{[a[0] for a in sol[:click_idx]]}", fontsize=10.5, fontweight="bold")
ax[1].text(0.5,-0.08,"agent walks to the door\n(directional actions)",transform=ax[1].transAxes,ha="center",fontsize=8.5,color="#334155")
ck=sol[click_idx]
ax[2].imshow(render(f2)); ax[2].scatter([ck[1]],[ck[2]],s=180,facecolors="none",edgecolors="#dc2626",linewidths=2.4)
ax[2].set_title(f"(c) click ({ck[1]},{ck[2]}) → level "+("complete ✓" if solved else "?"), fontsize=10.5, fontweight="bold")
ax[2].text(0.5,-0.08,"a single click operates the door;\nthe level completes (hybrid solution)",transform=ax[2].transAxes,ha="center",fontsize=8.5,color="#334155")
fig.suptitle("Worked example: ka59 — a hybrid (navigate + click) solution no fixed hypothesis space expressed", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout(); plt.savefig(FIG/"arc3_example_ka59.png", dpi=140, bbox_inches="tight"); plt.close()
print("wrote arc3_example_ka59.png | solved=",solved,"levels",base,"->",o.levels_completed)
