"""Fig: goal-as-STATE vs goal-as-PROCEDURE -- the central diagnosis, illustrated.
Writes papers/arc-3/figs/arc3_goal_as_procedure.png."""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
FIG=Path("papers/arc-3/figs"); FIG.mkdir(parents=True,exist_ok=True)
fig,(axL,axR)=plt.subplots(1,2,figsize=(11,4.4))
# LEFT: goal-as-state -- a score surface; MPC climbs to the wrong peak; the win is elsewhere
xs=np.linspace(0,10,300)
score=2.6*np.exp(-((xs-3.0)**2)/1.6)+1.1*np.exp(-((xs-7.3)**2)/0.7)
axL.plot(xs,score,color="#475569",lw=2)
axL.fill_between(xs,score,color="#e2e8f0",alpha=0.6)
axL.scatter([3.0],[2.6],s=70,color="#dc2626",zorder=5); axL.text(3.0,2.85,"score peak\n(MPC climbs here)",ha="center",fontsize=8,color="#dc2626")
axL.scatter([7.3],[1.1],s=120,marker="*",color="#16a34a",zorder=5); axL.text(7.3,1.45,"actual WIN",ha="center",fontsize=8,color="#166534",fontweight="bold")
axL.add_patch(FancyArrowPatch((1.2,0.9),(2.7,2.4),arrowstyle="-|>",mutation_scale=14,color="#dc2626",lw=2))
axL.set_title("Goal-as-STATE: a score over one frame", fontsize=11, fontweight="bold")
axL.set_xlabel("state space (schematic)"); axL.set_ylabel("goal score"); axL.set_xticks([]); axL.set_yticks([])
axL.text(5,-0.5,"Optimizing a single-state score (MPC, atomic objectives,\nLLM/Bayesian goal hypotheses) reaches the wrong place.",ha="center",fontsize=8,color="#334155")
# RIGHT: goal-as-procedure -- an ordered path A->B->C; no single-state score ranks it
axR.set_xlim(0,10); axR.set_ylim(0,5); axR.axis("off")
pts=[(1.2,3.6,"A\nreset"),(4.0,1.4,"B\ndrain timer"),(6.5,3.7,"C\nalign blocks"),(8.8,1.8,"WIN\ninteract")]
for i,(x,y,lab) in enumerate(pts):
    c="#16a34a" if i==3 else "#0d9488"
    axR.add_patch(Circle((x,y),0.45,facecolor="#ccfbf1" if i<3 else "#dcfce7",edgecolor=c,lw=2,zorder=4))
    axR.text(x,y,lab,ha="center",va="center",fontsize=8,zorder=5,color="#0f172a")
for i in range(len(pts)-1):
    (x1,y1,_),(x2,y2,_)=pts[i],pts[i+1]
    axR.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=16,color="#0f766e",lw=2.2,connectionstyle="arc3,rad=0.15",shrinkA=18,shrinkB=18))
axR.set_title("Goal-as-PROCEDURE: an ordered sequence", fontsize=11, fontweight="bold")
axR.text(5,0.35,"The win is a specific A→B→C protocol. No score over a single\nstate ranks it — so search stumbles in; planning-to-a-state cannot.",ha="center",fontsize=8,color="#334155")
plt.tight_layout(); plt.savefig(FIG/"arc3_goal_as_procedure.png", dpi=140, bbox_inches="tight"); plt.close()
print("wrote arc3_goal_as_procedure.png")
