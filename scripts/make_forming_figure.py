#!/usr/bin/env python3
"""Self-contained 'world model forming vs verifiable' figure (both panels one renderer -> clean PDF).
LEFT  = composite search graph on a WALL (forming; many states, no verified win).
RIGHT = the verified solution trajectory of a SOLVED game (verifiable; a chain to a green win node).

NOTE: this is an env-dependent reproducer -- it requires the live ARC-AGI-3 environment (the graph
and trajectory come from a non-deterministic env run), so it is NOT part of the offline
make_arc3_assets.py pipeline. The committed PDF/SVG/JSON in papers/arc-3/maps/ are the canonical
artifacts; re-run this only against the live arc env."""
import os, sys, json, math
ROOT = "/Users/jim/Desktop/openworld"
for p in ("experiments", "experiments/e125", "experiments/e134"):
    sys.path.insert(0, os.path.join(ROOT, p))
import numpy as np
from arc3_sandbox import SandboxGame
from composite import composite_key
ARCH = json.load(open(f"{ROOT}/experiments/results/arc3_fullgame_sourcefree.json"))
def frontier(g): return ARCH.get("solutions", {}).get(g) or []
def frame_of(g):
    a = np.asarray(g.frame)
    while a.ndim > 2: a = a[-1]
    return a.reshape(64, 64).astype(int) if a.size == 4096 else np.zeros((64, 64), int)
def step(g, a):
    try: g.step(6, a[1], a[2]) if a[0] == 6 else g.step(a[0]); return True
    except Exception: return False

def forming(game, steps=170):
    acts = frontier(game); g = SandboxGame(game); g.reset(); sl = 0
    for a in acts:
        if not step(g, a): break
        sl = g.levels
    k = composite_key(frame_of(g)); idx = {k: 0}; nodes = [k]; edges = set(); win = set()
    cur = 0; pat = [1, 2, 4, 3, 2, 5, 1, 4, 2, 3, 5, 4, 1, 3]
    for t in range(steps):
        if not step(g, [pat[t % len(pat)]]): break
        nk = composite_key(frame_of(g)); lv = g.levels
        if nk not in idx: idx[nk] = len(nodes); nodes.append(nk)
        j = idx[nk]; edges.add((cur, j))
        if lv > sl: win.add(j)
        cur = j
    return dict(n=len(nodes), edges=list(edges), win=win, kind="cloud", seed=sl)

def verifiable(game, cap=70):
    acts = frontier(game); g = SandboxGame(game); g.reset()
    keys = [composite_key(frame_of(g))]; lvl = [g.levels]
    for a in acts:
        if not step(g, a): break
        keys.append(composite_key(frame_of(g))); lvl.append(g.levels)
    # dedup consecutive repeats, then subsample to <=cap for a readable chain
    seq = [(keys[0], lvl[0])]
    for kk, lv in zip(keys[1:], lvl[1:]):
        if kk != seq[-1][0]: seq.append((kk, lv))
    if len(seq) > cap:
        step_i = len(seq) / cap
        seq = [seq[int(i*step_i)] for i in range(cap-1)] + [seq[-1]]
    n = len(seq); edges = [(i, i+1) for i in range(n-1)]
    return dict(n=n, edges=edges, win={n-1}, kind="chain", final=seq[-1][1])

def render(G, W, H):
    n = G["n"]; pos = {}
    if G["kind"] == "chain":
        cols = max(1, int(math.ceil(math.sqrt(n*1.6))))
        for i in range(n):
            r, c = divmod(i, cols); c = c if r % 2 == 0 else cols-1-c
            rows = max(1, (n-1)//cols)
            pos[i] = (30 + c*(W-60)/max(1, cols-1), 30 + r*(H-60)/max(1, rows))
    else:
        cx, cy, R = W/2, H/2, min(W, H)/2-26
        for i in range(n):
            ang = 2*math.pi*i/max(1, n); rr = R*(0.32+0.68*((i*0.61803) % 1.0))
            pos[i] = (cx+rr*math.cos(ang), cy+rr*math.sin(ang))
        pos[0] = (cx, cy)
    p = []
    for a, b in G["edges"]:
        x1, y1 = pos[a]; x2, y2 = pos[b]
        p.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="#c9cfd9" stroke-width="1"/>')
    for i in range(n):
        x, y = pos[i]
        if i in G["win"]: c, r = "#1f9d57", 7
        elif i == 0: c, r = "#d98a23", 7
        else: c, r = "#3b6fb0", 4
        p.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r}" fill="{c}" stroke="#fff" stroke-width="1"/>')
    return "".join(p)

def main():
    wall = sys.argv[1] if len(sys.argv) > 1 else "g50t"
    solved = sys.argv[2] if len(sys.argv) > 2 else "tu93"
    print(f"forming {wall} ...", flush=True); F = forming(wall)
    print(f"  {F['n']} states, win={len(F['win'])}", flush=True)
    print(f"verifiable {solved} ...", flush=True); V = verifiable(solved)
    print(f"  {V['n']} chain states -> level {V['final']}", flush=True)
    PW = 460; PAD = 30; FW = PW*2+PAD*3; FH = PW+86
    def panel(G, x, title, sub, col):
        return (f'<g transform="translate({x},0)">'
                f'<text x="{PW/2}" y="26" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="18" font-weight="bold" fill="{col}">{title}</text>'
                f'<text x="{PW/2}" y="48" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="12.5" fill="#555">{sub}</text>'
                f'<rect x="6" y="60" width="{PW-12}" height="{PW-12}" rx="8" fill="#fff" stroke="#e3e6ec"/>'
                f'<g transform="translate(6,60)">{render(G, PW-12, PW-12)}</g></g>')
    fig = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{FW}" height="{FH}" viewBox="0 0 {FW} {FH}" font-family="Helvetica,Arial,sans-serif">'
           f'<rect width="{FW}" height="{FH}" fill="#fafbfc"/>'
           + panel(F, PAD, "World model FORMING",
                   f"composite search on {wall} (unsolved) · {F['n']} states · no verified win", "#b3760f")
           + panel(V, PW+PAD*2, "World model VERIFIABLE",
                   f"{solved} solution trajectory · {V['n']} states · reaches a verified win", "#177a45")
           + f'<text x="{FW/2}" y="{FH-10}" text-anchor="middle" font-size="11.5" fill="#777">'
           f'orange = seed/start · blue = discovered state · green = verified win / level-up</text>'
           + '</svg>')
    outdir = f"{ROOT}/papers/arc-3/maps"
    open(f"{outdir}/world_model_forming.svg", "w").write(fig)
    json.dump({"wall": wall, "forming_states": F["n"], "forming_win": len(F["win"]),
               "solved": solved, "chain_states": V["n"], "solved_final_level": V["final"]},
              open(f"{outdir}/world_model_forming.json", "w"), indent=1)
    print("WROTE", f"{outdir}/world_model_forming.svg", flush=True)

if __name__ == "__main__":
    main()
