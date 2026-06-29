"""Architecture/method figure: how the MULTI-PERCEPTION CONSENSUS world model works.
OpenWorld primitives are highlighted (teal) so the framework contribution is legible at a glance.
Writes papers/arc-3/figs/arc3_architecture.png. Run with a python that has matplotlib."""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
FIG=Path("papers/arc-3/figs"); FIG.mkdir(parents=True,exist_ok=True)
# palette
OW_FILL="#ccfbf1"; OW_EDGE="#0d9488"; OW_DARK="#0f766e"          # OpenWorld primitive (teal)
ENV_FILL="#e2e8f0"; ENV_EDGE="#64748b"                            # environment / data (slate)
RES_FILL="#dcfce7"; RES_EDGE="#16a34a"                            # result (green)
OUT_FILL="#fef3c7"; OUT_EDGE="#d97706"                            # map output (ochre)
fig,ax=plt.subplots(figsize=(13,6.4)); ax.set_xlim(0,13); ax.set_ylim(0,6.6); ax.axis("off")

def box(x,y,w,h,title,sub,fill,edge,tcol="#0f172a",lw=1.6,fs=10,badge=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.10",
                 linewidth=lw,edgecolor=edge,facecolor=fill,mutation_scale=1))
    ax.text(x+w/2,y+h-0.30,title,ha="center",va="top",fontsize=fs,fontweight="bold",color=tcol)
    if sub: ax.text(x+w/2,y+h-0.30-0.34,sub,ha="center",va="top",fontsize=fs-2.4,color="#334155")
    if badge:
        ax.text(x+0.12,y+0.12,"OpenWorld",ha="left",va="bottom",fontsize=6.5,style="italic",color=OW_DARK)
def arrow(x1,y1,x2,y2,col="#475569",lw=1.8,style="-|>"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=14,lw=lw,color=col,
                 shrinkA=2,shrinkB=2,connectionstyle="arc3,rad=0"))

# --- Stage 1: env frame ---
box(0.15,2.5,1.85,1.7,"ARC-AGI-3","64×64×16 grid\n+ levels_completed",ENV_FILL,ENV_EDGE,fs=10)
g=np.array([[0,1,0,2,0],[0,0,3,0,0],[4,0,0,0,5],[0,0,6,0,0],[0,7,0,0,0]])
ax.imshow(g,extent=(0.55,1.6,2.62,3.12),cmap="tab10",vmin=0,vmax=9,aspect="auto",zorder=5)

# --- Stage 2: perception (OpenWorld Perceptor) ---
box(2.5,2.55,2.5,1.6,"Perceptor","status-bar MASK →\nmasked-frame state σ(s);\npixel-inferred targets",OW_FILL,OW_EDGE,badge=True,fs=10)
arrow(1.95,3.35,2.5,3.35)

# --- Stage 3: two modality world models (parallel) ---
box(5.6,4.0,3.0,1.5,"World$_\\mathrm{dir}$  (directional)","actions 1–5,7;  discovered\n(s,a)→s′ map · world-time\ncompute · FunctionTransition",OW_FILL,OW_EDGE,badge=True,fs=9.5)
box(5.6,0.9,3.0,1.5,"World$_\\mathrm{click}$  (click)","ACTION6 (x,y) on sprite\ncells;  discovered map ·\nFunctionTransition",OW_FILL,OW_EDGE,badge=True,fs=9.5)
arrow(5.0,3.55,5.6,4.6); arrow(5.0,3.15,5.6,1.65)
ax.text(4.9,4.75,"multi-\nperception",ha="center",fontsize=8,color=OW_DARK,style="italic")

# --- Stage 4: ConsensusTransition (synthesis) ---
box(9.0,2.45,2.6,1.8,"ConsensusTransition","hard-vote / select\nbest-fidelity modality",OW_FILL,OW_DARK,tcol=OW_DARK,lw=2.4,badge=True,fs=10)
arrow(8.6,4.6,9.0,3.75); arrow(8.6,1.65,9.0,2.95)
ax.text(8.6,3.5,"synthesis",ha="center",fontsize=8,color=OW_DARK,style="italic")

# --- Stage 5: plan -> solve + map ---
box(11.7,3.5,1.25,1.4,"Plan →","solve\nlevel ↑  ✓",RES_FILL,RES_EDGE,tcol="#166534",fs=10)
box(11.6,0.9,1.35,1.5,"Atlas map","to_spec →\npreview.graph\nrender_card /view",OUT_FILL,OUT_EDGE,tcol="#92400e",badge=True,fs=9.5)
arrow(11.6,3.85,11.7,4.1)                       # consensus -> plan
arrow(10.5,2.45,11.6,1.7)                      # consensus -> map
# reward annotation
ax.text(10.3,4.55,"reward: levels_completed /\ninduced CodeObjective",ha="center",fontsize=7.2,color=OW_DARK)

# title + legend
ax.text(6.5,6.35,"Multi-Perception Consensus World Model",ha="center",fontsize=15,fontweight="bold",color="#0f172a")
ax.text(6.5,6.0,"perceive each game in multiple modalities → discover a World map per modality → synthesize by consensus → plan to solve",
        ha="center",fontsize=9.2,color="#475569")
# legend
lx=0.3
for lab,fc,ec in [("OpenWorld primitive",OW_FILL,OW_EDGE),("environment / data",ENV_FILL,ENV_EDGE),("solve",RES_FILL,RES_EDGE),("map output",OUT_FILL,OUT_EDGE)]:
    ax.add_patch(FancyBboxPatch((lx,0.18),0.32,0.26,boxstyle="round,pad=0.01",facecolor=fc,edgecolor=ec,lw=1.4))
    ax.text(lx+0.4,0.31,lab,ha="left",va="center",fontsize=8,color="#334155"); lx+=2.5
plt.tight_layout(); plt.savefig(FIG/"arc3_architecture.png",dpi=150,bbox_inches="tight"); plt.close()
print("wrote arc3_architecture.png")
