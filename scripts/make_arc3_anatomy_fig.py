"""Appendix primer figure: 'Anatomy of a world model'. Clarifies the concept that confuses readers --
the environment is a world, and our *model* of it is a World object with the same anatomy (perception
-> state -> transition -> reward), and worlds nest. Writes papers/arc-3/figs/arc3_anatomy.png."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
FIG=Path("papers/arc-3/figs"); FIG.mkdir(parents=True,exist_ok=True)
OW="#ccfbf1"; OWE="#0d9488"; ENVF="#e2e8f0"; ENVE="#64748b"; RW="#fef3c7"; RWE="#d97706"
fig,ax=plt.subplots(figsize=(12,6.2)); ax.set_xlim(0,13); ax.set_ylim(0,7); ax.axis("off")
def box(x,y,w,h,t,sub,fc,ec,fs=10,tc="#0f172a",lw=1.8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.10",lw=lw,edgecolor=ec,facecolor=fc))
    ax.text(x+w/2,y+h-0.28,t,ha="center",va="top",fontsize=fs,fontweight="bold",color=tc)
    if sub: ax.text(x+w/2,y+h-0.62,sub,ha="center",va="top",fontsize=fs-2.3,color="#334155")
def arr(x1,y1,x2,y2,c="#475569",lw=1.8,lab=None,labdy=0.18):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,lw=lw,color=c))
    if lab: ax.text((x1+x2)/2,(y1+y2)/2+labdy,lab,ha="center",fontsize=8,color=c,style="italic")
# Environment world (left)
box(0.3,2.6,2.6,2.0,"ENVIRONMENT","the game\n(a world)",ENVF,ENVE,fs=11)
# World model (right) -- a big box with the anatomy inside
box(4.0,1.0,8.6,5.2,"WORLD MODEL  =  an openworld.World","same anatomy as the world it models",OW,OWE,fs=12,tc=OWE)
# components inside
box(4.4,3.9,2.4,1.1,"Perceptor","frame → state $\\sigma(s)$",ENVF,ENVE,fs=9.5)
box(7.1,3.9,2.4,1.1,"State $s$","masked grid signature",OW,OWE,fs=9.5)
box(9.8,3.9,2.4,1.1,"Transition $T$","$s' = T(s,a)$  (code)",OW,OWE,fs=9.5)
box(4.4,1.5,2.4,1.1,"Actions $a$","1–5,7 or click$(x,y)$",ENVF,ENVE,fs=9.5)
box(7.1,1.5,2.4,1.1,"Reward","levels_completed",RW,RWE,fs=9.5,tc="#92400e")
# nested subworld (world-in-a-world)
box(9.8,1.5,2.4,1.1,"sub-World","worlds nest\n(CompositeWorld)",OW,OWE,fs=9,tc=OWE)
# internal flow arrows
arr(6.8,4.45,7.1,4.45); arr(9.5,4.45,9.8,4.45)
arr(6.8,2.05,7.1,2.05)
# env <-> model
arr(2.9,3.9,4.0,4.45,c=ENVE,lab="observe (frame)",labdy=0.22)
arr(4.0,2.0,2.9,3.0,c=ENVE,lab="act (a)",labdy=-0.3)
ax.text(6.5,6.75,"Anatomy of a world model: the environment is a world; our model is a $\\mathit{World}$ too",ha="center",fontsize=13,fontweight="bold")
# bottom mapping
ax.text(6.5,0.55,"ARC-AGI-3 mapping:  state = $64{\\times}64$ grid  ·  action = ACTION1–7 / ACTION6$(x,y)$  ·  transition = verified predict() (CodeTransition)",ha="center",fontsize=8.3,color="#334155")
ax.text(6.5,0.2,"perceptor = status-masked frame  ·  reward = levels_completed (or induced CodeObjective)  ·  the discovered state-graph IS this World; to_spec renders its map",ha="center",fontsize=8.3,color="#334155")
plt.tight_layout(); plt.savefig(FIG/"arc3_anatomy.png",dpi=140,bbox_inches="tight"); plt.close()
print("wrote arc3_anatomy.png")
