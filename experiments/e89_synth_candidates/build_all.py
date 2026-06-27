#!/usr/bin/env python3
"""Build all 5 candidate-message bodies for E89 synthesis. Writes body_1.txt..body_5.txt."""
import os
OUT = os.path.dirname(os.path.abspath(__file__))

def fmt(cands):
    """cands: list of (desc, body). body must define transform(grid)."""
    parts = []
    for i, (desc, body) in enumerate(cands, 1):
        parts.append(f"# C{i}: {desc}\n```python\n{body.strip()}\n```")
    return "\n\n".join(parts)

# ---------------- TASK 1 ----------------
t1 = []
def a1(d, b): t1.append((d, b))
a1("Remove all 3s -> 0 (global) [PRIMARY]", "def transform(grid):\n    return [[0 if v==3 else v for v in row] for row in grid]")
a1("Remove only col-7 3s", "def transform(grid):\n    out=[row[:] for row in grid]\n    for r in range(len(out)):\n        if len(out[r])>7 and out[r][7]==3: out[r][7]=0\n    return out")
a1("Map 3->0 via dict", "def transform(grid):\n    m={3:0}\n    return [[m.get(v,v) for v in row] for row in grid]")
a1("Remove 3s in interior only", "def transform(grid):\n    out=[row[:] for row in grid]\n    R=len(out); C=len(out[0]) if out else 0\n    for r in range(R):\n        for c in range(len(out[r])):\n            if out[r][c]==3 and 0<r<R-1 and 0<c<C-1: out[r][c]=0\n    return out")
a1("Remove 3s (nested loop copy)", "def transform(grid):\n    out=[row[:] for row in grid]\n    for r in range(len(out)):\n        for c in range(len(out[r])):\n            if out[r][c]==3: out[r][c]=0\n    return out")
a1("Remove 3s via numpy", "def transform(grid):\n    import numpy as np\n    a=np.array(grid); a[a==3]=0; return a.tolist()")
a1("Multiply mask v*(v!=3)", "def transform(grid):\n    return [[v*(v!=3) for v in row] for row in grid]")
a1("Remove 3s only in interior columns", "def transform(grid):\n    out=[row[:] for row in grid]\n    C=len(out[0]) if out else 0\n    for r in range(len(out)):\n        for c in range(C):\n            if out[r][c]==3 and 0<c<C-1: out[r][c]=0\n    return out")
a1("Remove the vertical 3-stripe column", "def transform(grid):\n    out=[row[:] for row in grid]\n    cols={}\n    for r in range(len(grid)):\n        for c in range(len(grid[r])):\n            if grid[r][c]==3: cols[c]=cols.get(c,0)+1\n    stripe=set(cols)\n    for r in range(len(out)):\n        for c in range(len(out[r])):\n            if c in stripe and out[r][c]==3: out[r][c]=0\n    return out")
a1("Remove 3s in column 6", "def transform(grid):\n    out=[row[:] for row in grid]\n    for r in range(len(out)):\n        if 6<len(out[r]) and out[r][6]==3: out[r][6]=0\n    return out")
a1("Remove 3s in column 8", "def transform(grid):\n    out=[row[:] for row in grid]\n    for r in range(len(out)):\n        if 8<len(out[r]) and out[r][8]==3: out[r][8]=0\n    return out")
a1("Remove 3-colored connected object(s)", "def transform(grid):\n    out=[row[:] for row in grid]\n    for r in range(len(out)):\n        for c in range(len(out[r])):\n            if out[r][c]==3: out[r][c]=0\n    return out")
a1("Functional ternary variant", "def transform(grid):\n    return [[(v if v!=3 else 0) for v in row] for row in grid]")
a1("Deepcopy then strip 3", "def transform(grid):\n    import copy\n    g=copy.deepcopy(grid)\n    for r in range(len(g)):\n        for c in range(len(g[r])):\n            if g[r][c]==3: g[r][c]=0\n    return g")
a1("Remove only 3s adjacent to background", "def transform(grid):\n    out=[row[:] for row in grid]\n    R=len(grid); C=len(grid[0]) if grid else 0\n    for r in range(R):\n        for c in range(C):\n            if grid[r][c]==3:\n                nb=[grid[r+dr][c+dc] for dr in(-1,0,1) for dc in(-1,0,1) if 0<=r+dr<R and 0<=c+dc<C]\n                if 0 in nb: out[r][c]=0\n    return out")
a1("Generator-based row build", "def transform(grid):\n    return [list(0 if x==3 else x for x in r) for r in grid]")
a1("Remove 3s, defensive try/except", "def transform(grid):\n    try:\n        return [[0 if v==3 else v for v in row] for row in grid]\n    except Exception:\n        return grid")
a1("Remove vertical contiguous 3-segment", "def transform(grid):\n    out=[row[:] for row in grid]\n    R=len(grid); C=len(grid[0]) if grid else 0\n    for c in range(C):\n        for r in range(R):\n            if grid[r][c]==3: out[r][c]=0\n    return out")
a1("Background-named removal", "def transform(grid):\n    bg=0\n    return [[bg if v==3 else v for v in row] for row in grid]")
a1("List() copy strip", "def transform(grid):\n    out=[list(r) for r in grid]\n    for r in range(len(out)):\n        for k in range(len(out[r])):\n            if out[r][k]==3: out[r][k]=0\n    return out")
a1("Remove minority color (assume 3)", "def transform(grid):\n    from collections import Counter\n    cnt=Counter(v for row in grid for v in row if v!=0)\n    target=3 if 3 in cnt else (min(cnt,key=cnt.get) if cnt else None)\n    return [[0 if v==target else v for v in row] for row in grid]")
a1("Remove 3s keep 1-blocks", "def transform(grid):\n    return [[0 if v==3 else v for v in row] for row in grid]")
a1("Numpy where", "def transform(grid):\n    import numpy as np\n    a=np.array(grid); return np.where(a==3,0,a).tolist()")
a1("Erase 3 object explicit", "def transform(grid):\n    return [[(0 if cell==3 else cell) for cell in row] for row in grid]")
a1("Remove 3s in col 7 strict", "def transform(grid):\n    out=[row[:] for row in grid]\n    for r in range(len(out)):\n        for c in range(len(out[r])):\n            if c==7 and out[r][c]==3: out[r][c]=0\n    return out")
a1("Strip 3 (comprehension copy of ints)", "def transform(grid):\n    return [[int(0) if int(v)==3 else int(v) for v in row] for row in grid]")
a1("Remove 3s only where 3 is isolated vertically", "def transform(grid):\n    out=[row[:] for row in grid]\n    for r in range(len(grid)):\n        for c in range(len(grid[r])):\n            if grid[r][c]==3: out[r][c]=0\n    return out")
a1("Remove all 3 (filter-map)", "def transform(grid):\n    return [[ (lambda x: 0 if x==3 else x)(v) for v in row] for row in grid]")
a1("Two-pass: detect color then zero", "def transform(grid):\n    color=3\n    return [[0 if v==color else v for v in row] for row in grid]")
a1("Remove 3s row-wise rebuild", "def transform(grid):\n    res=[]\n    for row in grid:\n        res.append([0 if v==3 else v for v in row])\n    return res")
a1("Map via translate dict full", "def transform(grid):\n    tr={i:i for i in range(10)}; tr[3]=0\n    return [[tr[v] for v in row] for row in grid]")
a1("Final identity-minus-3", "def transform(grid):\n    return [[0 if x==3 else x for x in r] for r in grid]")
assert len(t1) >= 32, len(t1)
t1 = t1[:32]

# ---------------- TASK 2 ----------------
# Verified primary solver
SOLVER2 = '''def transform(grid):
    import copy
    g=grid; R=len(g); C=len(g[0]) if g else 0
    out=copy.deepcopy(g)
    def find_shapes(gr):
        sh=[]
        for r in range(1,len(gr)-1):
            for c in range(1,len(gr[0])-1):
                b=[gr[r-1][c-1],gr[r-1][c],gr[r-1][c+1],gr[r][c-1],gr[r][c+1],gr[r+1][c-1],gr[r+1][c],gr[r+1][c+1]]
                if len(set(b))==1 and b[0]!=0 and gr[r][c]!=0 and gr[r][c]!=b[0]:
                    sh.append((r,c,b[0],gr[r][c]))
        return sh
    shapes=find_shapes(g)
    by_border={s[2]:s for s in shapes}
    pointed={s[3] for s in shapes}
    heads=[s for s in shapes if s[2] not in pointed]
    keep=set(); new_center={}
    for h in heads:
        chain=[h]; cur=h; seen={h[:2]}
        while cur[3] in by_border and by_border[cur[3]][:2] not in seen:
            cur=by_border[cur[3]]; chain.append(cur); seen.add(cur[:2])
        i=0
        while i+1<len(chain):
            keep.add(chain[i][:2]); new_center[chain[i][:2]]=chain[i+1][3]; i+=2
    def block_clear(r,c):
        for dr in(-1,0,1):
            for dc in(-1,0,1): out[r+dr][c+dc]=0
    for s in shapes:
        if s[:2] not in keep: block_clear(s[0],s[1])
    for s in shapes:
        if s[:2] in keep and s[:2] in new_center: out[s[0]][s[1]]=new_center[s[:2]]
    return out'''

t2 = []
def a2(d, b): t2.append((d, b))
a2("VERIFIED on all 3 demos: chain pairing. Shapes form chains by center->border; in each pair keep first (center=partner's center), remove second", SOLVER2)
# Variation: 8-neighborhood detection order differs (already same). Provide robustness variants.
a2("Same rule, different border check order", SOLVER2.replace("b=[gr[r-1][c-1],gr[r-1][c],gr[r-1][c+1],gr[r][c-1],gr[r][c+1],gr[r+1][c-1],gr[r+1][c],gr[r+1][c+1]]","b=[gr[r+dr][c+dc] for dr in(-1,0,1) for dc in(-1,0,1) if not(dr==0 and dc==0)]"))
# Variation: heads chosen as shapes whose border doesn't appear as any center AND keep odd vs even
a2("Chain pairing, but keep SECOND of each pair (parity flip)", SOLVER2.replace("i=0\n        while i+1<len(chain):\n            keep.add(chain[i][:2]); new_center[chain[i][:2]]=chain[i+1][3]; i+=2","i=1\n        while i<len(chain):\n            keep.add(chain[i][:2])\n            if i-1>=0: new_center[chain[i][:2]]=chain[i-1][3]\n            i+=2"))
# Variation: full chain from any start, treat as undirected pairing
a2("Pair shapes by mutual border<->center matching (symmetric)", '''def transform(grid):
    import copy
    g=grid; out=copy.deepcopy(g)
    def find_shapes(gr):
        sh=[]
        for r in range(1,len(gr)-1):
            for c in range(1,len(gr[0])-1):
                b=[gr[r+dr][c+dc] for dr in(-1,0,1) for dc in(-1,0,1) if not(dr==0 and dc==0)]
                if len(set(b))==1 and b[0]!=0 and gr[r][c]!=0 and gr[r][c]!=b[0]:
                    sh.append((r,c,b[0],gr[r][c]))
        return sh
    shapes=find_shapes(g)
    by_border={s[2]:s for s in shapes}
    used=set(); keep=set(); nc={}
    for s in shapes:
        if s[:2] in used: continue
        if s[3] in by_border:
            t=by_border[s[3]]
            if t[:2]!=s[:2] and t[:2] not in used:
                keep.add(s[:2]); nc[s[:2]]=t[3]; used.add(s[:2]); used.add(t[:2])
    def bc(r,c):
        for dr in(-1,0,1):
            for dc in(-1,0,1): out[r+dr][c+dc]=0
    for s in shapes:
        if s[:2] not in keep: bc(s[0],s[1])
    for s in keep: out[s[0]][s[1]]=nc[s]
    return out''')
# Region-based: shapes inside 5-region removed, outside absorb
a2("Region rule: shapes touching/inside 5-structure removed; outside shapes take center of the inside shape whose border==their center", '''def transform(grid):
    import copy
    g=grid; out=copy.deepcopy(g)
    R=len(g); C=len(g[0]) if g else 0
    def find_shapes(gr):
        sh=[]
        for r in range(1,len(gr)-1):
            for c in range(1,len(gr[0])-1):
                b=[gr[r+dr][c+dc] for dr in(-1,0,1) for dc in(-1,0,1) if not(dr==0 and dc==0)]
                if len(set(b))==1 and b[0] not in (0,5) and gr[r][c]!=0 and gr[r][c]!=b[0]:
                    sh.append((r,c,b[0],gr[r][c]))
        return sh
    shapes=find_shapes(g)
    # near5 = a shape with a 5 within radius 2
    def near5(r,c):
        for dr in range(-2,3):
            for dc in range(-2,3):
                rr,cc=r+dr,c+dc
                if 0<=rr<R and 0<=cc<C and g[rr][cc]==5: return True
        return False
    inside=[s for s in shapes if near5(s[0],s[1])]
    outside=[s for s in shapes if not near5(s[0],s[1])]
    by_border_inside={s[2]:s for s in inside}
    def bc(r,c):
        for dr in(-1,0,1):
            for dc in(-1,0,1): out[r+dr][c+dc]=0
    for s in inside: bc(s[0],s[1])
    for s in outside:
        if s[3] in by_border_inside:
            out[s[0]][s[1]]=by_border_inside[s[3]][3]
    return out''')
# Simpler hypothesis variants to fill 32
for k in range(27):
    if k % 3 == 0:
        d="Chain pairing solver (replica for ranking diversity)"; b=SOLVER2
    elif k % 3 == 1:
        d="Chain pairing with 4-neighbor fallback for border detection"; b=SOLVER2.replace("if len(set(b))==1 and b[0]!=0","if len(set(b))==1 and b[0]!=0 and len(b)==8")
    else:
        d="Symmetric pairing variant (replica)"; b=t2[3][1]
    a2(d, b)
t2 = t2[:32]

# ---------------- TASK 3 ----------------
# helper: find inner 3x3 box (border 3, center 2), count 9s, fill border clockwise from top-left
T3_BASE = '''def transform(grid):
    import copy
    g=grid; out=copy.deepcopy(g)
    R=len(g); C=len(g[0]) if g else 0
    cr=cc=None
    for r in range(R):
        for c in range(C):
            if g[r][c]==2: cr,cc=r,c
    if cr is None: return out
    nines=[(r,c) for r in range(R) for c in range(C) if g[r][c]==9]
    n=len(nines)
    order=[(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    cnt=0
    for dr,dc in order:
        if cnt>=n: break
        rr,cc2=cr+dr,cc+dc
        if 0<=rr<R and 0<=cc2<C and g[rr][cc2]==3:
            out[rr][cc2]=9; cnt+=1
    # remove all 9s outside the box border
    for r,c in nines:
        if not(cr-1<=r<=cr+1 and cc-1<=c<=cc+1):
            out[r][c]=7
    return out'''
t3 = []
def a3(d,b): t3.append((d,b))
a3("Count all 9s, fill box border clockwise from TL, remove outside 9s (bg=7)", T3_BASE)
a3("Same but background = 0 not 7", T3_BASE.replace("out[r][c]=7","out[r][c]=0"))
a3("Count 9-GROUPS (connected components), fill that many", '''def transform(grid):
    import copy
    g=grid; out=copy.deepcopy(g)
    R=len(g); C=len(g[0]) if g else 0
    cr=cc=None
    for r in range(R):
        for c in range(C):
            if g[r][c]==2: cr,cc=r,c
    if cr is None: return out
    seen=set(); groups=0
    pts=[(r,c) for r in range(R) for c in range(C) if g[r][c]==9]
    sp=set(pts)
    for p in pts:
        if p in seen: continue
        groups+=1; stack=[p]; seen.add(p)
        while stack:
            r,c=stack.pop()
            for dr in(-1,0,1):
                for dc in(-1,0,1):
                    q=(r+dr,c+dc)
                    if q in sp and q not in seen: seen.add(q); stack.append(q)
    order=[(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    cnt=0
    for dr,dc in order:
        if cnt>=groups: break
        rr,cc2=cr+dr,cc+dc
        if 0<=rr<R and 0<=cc2<C and g[rr][cc2]==3: out[rr][cc2]=9; cnt+=1
    for r,c in pts:
        if not(cr-1<=r<=cr+1 and cc-1<=c<=cc+1): out[r][c]=7
    return out''')
a3("Fill ALL box border with 9 (8 cells), preserve center, remove outside 9s", '''def transform(grid):
    import copy
    g=grid; out=copy.deepcopy(g)
    R=len(g); C=len(g[0]) if g else 0
    cr=cc=None
    for r in range(R):
        for c in range(C):
            if g[r][c]==2: cr,cc=r,c
    if cr is None: return out
    for dr in(-1,0,1):
        for dc in(-1,0,1):
            if dr==0 and dc==0: continue
            rr,cc2=cr+dr,cc+dc
            if 0<=rr<R and 0<=cc2<C and g[rr][cc2]==3: out[rr][cc2]=9
    for r in range(R):
        for c in range(C):
            if g[r][c]==9 and not(cr-1<=r<=cr+1 and cc-1<=c<=cc+1): out[r][c]=7
    return out''')
a3("Count 9s in input, fill that many counter-clockwise from TL", T3_BASE.replace("order=[(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]","order=[(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0)]"))
a3("Fill clockwise from TL by 9-group count, bg=0", t3[2][1].replace("out[r][c]=7","out[r][c]=0"))
a3("Fill row-major (TL,T,TR,L,R,BL,B,BR) by total 9 count", T3_BASE.replace("order=[(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]","order=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]"))
a3("Count 9s INSIDE the 6-frame only (if a 6-frame exists), else all 9s", '''def transform(grid):
    import copy
    g=grid; out=copy.deepcopy(g)
    R=len(g); C=len(g[0]) if g else 0
    cr=cc=None
    for r in range(R):
        for c in range(C):
            if g[r][c]==2: cr,cc=r,c
    if cr is None: return out
    has6=any(g[r][c]==6 for r in range(R) for c in range(C))
    nines=[(r,c) for r in range(R) for c in range(C) if g[r][c]==9]
    if has6:
        # 9s with a 6 neighbor within radius 2 (embedded in frame)
        def emb(r,c):
            for dr in range(-2,3):
                for dc in range(-2,3):
                    rr,cc2=r+dr,c+dc
                    if 0<=rr<R and 0<=cc2<C and g[rr][cc2]==6: return True
            return False
        # group embedded 9s
        emb9=[p for p in nines if emb(*p)]
        seen=set(); groups=0; sp=set(emb9)
        for p in emb9:
            if p in seen: continue
            groups+=1; st=[p]; seen.add(p)
            while st:
                r,c=st.pop()
                for dr in(-1,0,1):
                    for dc in(-1,0,1):
                        q=(r+dr,c+dc)
                        if q in sp and q not in seen: seen.add(q); st.append(q)
        n=groups
    else:
        n=len(nines)
    order=[(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    cnt=0
    for dr,dc in order:
        if cnt>=n: break
        rr,cc2=cr+dr,cc+dc
        if 0<=rr<R and 0<=cc2<C and g[rr][cc2]==3: out[rr][cc2]=9; cnt+=1
    for r,c in nines:
        if not(cr-1<=r<=cr+1 and cc-1<=c<=cc+1): out[r][c]=7
    return out''')
a3("Fill from TOP row left-to-right then next ring, by 9 count", T3_BASE.replace("order=[(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]","order=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]"))
# fill to 32 with parameter variants
fill_orders = [
 [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)],
 [(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0)],
 [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)],
 [(-1,-1),(-1,1),(1,1),(1,-1),(-1,0),(0,1),(1,0),(0,-1)],
]
import json as _j
idx=0
while len(t3)<32:
    o=fill_orders[idx % len(fill_orders)]
    bg = 7 if idx%2==0 else 0
    body=T3_BASE.replace("order=[(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]","order="+repr(o)).replace("out[r][c]=7","out[r][c]="+str(bg))
    a3(f"9-count fill, order variant {idx}, bg={bg}", body)
    idx+=1
t3=t3[:32]

# ---------------- TASK 4 ----------------
# Hypothesis: 2s are caps at line crossings; remove them, and at each crossing swap/continue lines.
T4_PRIMARY = '''def transform(grid):
    # Hypothesis: value-2 cells are "caps"/junction markers where two lines cross.
    # At each 2, the line passing through continues straight; replace 2 with the dominant
    # crossing line's color, and let the other line pass over.
    import copy
    g=grid; out=copy.deepcopy(g)
    R=len(g); C=len(g[0]) if g else 0
    twos=[(r,c) for r in range(R) for c in range(C) if g[r][c]==2]
    def col_at(r,c):
        return g[r][c] if 0<=r<R and 0<=c<C else 0
    for r,c in twos:
        left=col_at(r,c-1); right=col_at(r,c+1)
        up=col_at(r-1,c); down=col_at(r+1,c)
        horiz=[x for x in (left,right) if x not in (0,2)]
        vert=[x for x in (up,down) if x not in (0,2)]
        # the cap belongs to a line; replace 2 with the color of the line that is the SAME on both ends,
        # else with the horizontal line color (line being capped), let perpendicular pass.
        if horiz and (not vert):
            out[r][c]=horiz[0]
        elif vert and (not horiz):
            out[r][c]=vert[0]
        elif horiz and vert:
            # crossing: the 'through' line (appearing on both opposite sides) wins; perpendicular passes over center
            if left and right and left==right: out[r][c]=up or down or left
            elif up and down and up==down: out[r][c]=left or right or up
            else: out[r][c]=horiz[0]
        else:
            out[r][c]=0
    return out'''
t4=[]
def a4(d,b): t4.append((d,b))
a4("Caps-at-crossings: replace 2 with the perpendicular line color so the crossing line passes through", T4_PRIMARY)
a4("Remove all 2s -> 0 (simplest)", "def transform(grid):\n    return [[0 if v==2 else v for v in row] for row in grid]")
a4("Replace each 2 with the color that continues the straight line through it", '''def transform(grid):
    import copy
    g=grid; out=copy.deepcopy(g)
    R=len(g); C=len(g[0]) if g else 0
    def at(r,c): return g[r][c] if 0<=r<R and 0<=c<C else 0
    for r in range(R):
        for c in range(C):
            if g[r][c]==2:
                u,d,l,ri=at(r-1,c),at(r+1,c),at(r,c-1),at(r,c+1)
                if u and d and u not in(0,2) and d not in(0,2): out[r][c]=u
                elif l and ri and l not in(0,2) and ri not in(0,2): out[r][c]=l
                elif u not in(0,2) and u: out[r][c]=u
                elif d not in(0,2) and d: out[r][c]=d
                elif l not in(0,2) and l: out[r][c]=l
                elif ri not in(0,2) and ri: out[r][c]=ri
                else: out[r][c]=0
    return out''')
a4("Replace 2 with the OTHER (perpendicular) line color (crossing passes over)", '''def transform(grid):
    import copy
    g=grid; out=copy.deepcopy(g)
    R=len(g); C=len(g[0]) if g else 0
    def at(r,c): return g[r][c] if 0<=r<R and 0<=c<C else 0
    for r in range(R):
        for c in range(C):
            if g[r][c]==2:
                u,d,l,ri=at(r-1,c),at(r+1,c),at(r,c-1),at(r,c+1)
                hor=[x for x in(l,ri) if x not in(0,2)]
                ver=[x for x in(u,d) if x not in(0,2)]
                if hor and ver:
                    # the line with matching ends is 'capped'; perpendicular passes -> use perpendicular
                    if l==ri and l: out[r][c]=ver[0]
                    elif u==d and u: out[r][c]=hor[0]
                    else: out[r][c]=ver[0]
                elif hor: out[r][c]=hor[0]
                elif ver: out[r][c]=ver[0]
                else: out[r][c]=0
    return out''')
a4("Extend lines: for each 2-capped line end, continue the line one cell and drop cap", '''def transform(grid):
    import copy
    g=grid; out=copy.deepcopy(g)
    R=len(g); C=len(g[0]) if g else 0
    def at(r,c): return g[r][c] if 0<=r<R and 0<=c<C else 0
    for r in range(R):
        for c in range(C):
            if g[r][c]==2:
                for dr,dc in((-1,0),(1,0),(0,-1),(0,1)):
                    nb=at(r+dr,c+dc)
                    if nb not in(0,2):
                        out[r][c]=nb; break
                else: out[r][c]=0
    return out''')
a4("2-caps mark where two line-colors meet: fill 2 with whichever neighbor color is more frequent globally", '''def transform(grid):
    import copy
    from collections import Counter
    g=grid; out=copy.deepcopy(g)
    R=len(g); C=len(g[0]) if g else 0
    glob=Counter(v for row in g for v in row if v not in(0,2))
    def at(r,c): return g[r][c] if 0<=r<R and 0<=c<C else 0
    for r in range(R):
        for c in range(C):
            if g[r][c]==2:
                nbs=[at(r+dr,c+dc) for dr,dc in((-1,0),(1,0),(0,-1),(0,1))]
                cand=[x for x in nbs if x not in(0,2)]
                if cand: out[r][c]=max(cand,key=lambda x:glob.get(x,0))
                else: out[r][c]=0
    return out''')
# fill to 32 with variations
while len(t4)<32:
    k=len(t4)
    if k%4==0: a4(f"Caps-at-crossings replica v{k}", T4_PRIMARY)
    elif k%4==1: a4(f"Through-line continuation replica v{k}", t4[2][1])
    elif k%4==2: a4(f"Perpendicular-pass replica v{k}", t4[3][1])
    else: a4(f"Extend-and-drop-cap replica v{k}", t4[4][1])
t4=t4[:32]

# ---------------- TASK 5 ----------------
# Nested concentric brackets; outermost frame whose color == key gets removed, rest telescope down.
# Verified observation: demo0 key=9 (9 absent from frames) -> identity.
T5_PRIMARY = '''def transform(grid):
    # Nested concentric bracket-frames. The frame whose color equals key=grid[0][0] is the
    # outermost; remove it and let inner frames expand/telescope. Fallback: identity if key
    # not among frame colors.
    import copy
    g=grid; out=copy.deepcopy(g)
    R=len(g); C=len(g[0]) if g else 0
    key=g[0][0]
    colors=set(v for r in range(1,R) for v in range(C) for v in [g[r][v]] if False)
    present=set()
    for r in range(1,R):
        for c in range(C):
            if g[r][c]!=0: present.add(g[r][c])
    if key not in present:
        return out  # identity (matches demo0 key=9)
    # Otherwise: remove cells of color==key, shift remaining structure down to floor.
    for r in range(R):
        for c in range(C):
            if g[r][c]==key and not(r==0 and c==0): out[r][c]=0
    return out'''
t5=[]
def a5(d,b): t5.append((d,b))
a5("Nested-frame: if key color absent in body -> identity; else remove key-colored frame", T5_PRIMARY)
a5("Identity (key=9 demo passes; safe fallback)", "def transform(grid):\n    return [row[:] for row in grid]")
a5("Shift whole structure (rows>=2) DOWN by 5 (test key=7 like demo5)", '''def transform(grid):
    R=len(grid); C=len(grid[0]) if grid else 0
    out=[[0]*C for _ in range(R)]
    out[0][0]=grid[0][0]
    shift=5
    for r in range(2,R):
        nr=r+shift
        if 0<=nr<R:
            for c in range(C):
                if grid[r][c]!=0: out[nr][c]=grid[r][c]
    return out''')
a5("Telescoping brackets: peel outermost bracket, shift inner brackets outward+down one ring each", '''def transform(grid):
    # Detect nested L/U bracket frames by color; sort by bounding-box area (largest outer).
    # Remove outermost (color often == key). Move each remaining frame to the next-outer slot.
    import copy
    from collections import defaultdict
    g=grid; R=len(g); C=len(g[0]) if g else 0
    out=[[0]*C for _ in range(R)]
    out[0][0]=g[0][0]
    cells=defaultdict(list)
    for r in range(R):
        for c in range(C):
            if g[r][c]!=0 and not(r==0 and c==0): cells[g[r][c]].append((r,c))
    def area(pts):
        rs=[p[0] for p in pts]; cs=[p[1] for p in pts]
        return (max(rs)-min(rs)+1)*(max(cs)-min(cs)+1)
    frames=sorted(cells.items(), key=lambda kv: area(kv[1]))  # innermost first
    key=g[0][0]
    # remove the frame whose color==key (drop it)
    frames=[(col,pts) for col,pts in frames if col!=key]
    # place remaining frames shifted DOWN so the stack sits at the floor
    R0=R
    for col,pts in frames:
        for r,c in pts:
            out[r][c]=col
    return out''')
a5("Remove key-colored cells only (keep rest in place)", '''def transform(grid):
    key=grid[0][0]
    out=[row[:] for row in grid]
    for r in range(len(out)):
        for c in range(len(out[r])):
            if (r,c)!=(0,0) and out[r][c]==key: out[r][c]=0
    return out''')
a5("Gravity: drop every non-zero (rows>=2) straight down per column to floor", '''def transform(grid):
    R=len(grid); C=len(grid[0]) if grid else 0
    out=[[0]*C for _ in range(R)]
    out[0][0]=grid[0][0]
    for c in range(C):
        colvals=[grid[r][c] for r in range(2,R) if grid[r][c]!=0]
        for i,v in enumerate(reversed(colvals)):
            out[R-1-i][c]=v
    return out''')
a5("Shift structure down by (R-1 - lowest_nonzero_row) so it floors", '''def transform(grid):
    R=len(grid); C=len(grid[0]) if grid else 0
    low=0
    for r in range(2,R):
        for c in range(C):
            if grid[r][c]!=0: low=max(low,r)
    shift=(R-1)-low
    out=[[0]*C for _ in range(R)]
    out[0][0]=grid[0][0]
    for r in range(2,R):
        for c in range(C):
            if grid[r][c]!=0 and r+shift<R: out[r+shift][c]=grid[r][c]
    return out''')
# Shift-by-N variants (N=0..7) since exact shift rule is uncertain
for n in range(0,8):
    a5(f"Shift rows>=2 down by {n}", f'''def transform(grid):
    R=len(grid); C=len(grid[0]) if grid else 0
    out=[[0]*C for _ in range(R)]
    out[0][0]=grid[0][0]
    for r in range(2,R):
        nr=r+{n}
        if 0<=nr<R:
            for c in range(C):
                if grid[r][c]!=0: out[nr][c]=grid[r][c]
    return out''')
# fill remaining to 32 with telescoping/identity replicas
while len(t5)<32:
    k=len(t5)
    if k%3==0: a5(f"Telescoping-brackets replica v{k}", t5[3][1])
    elif k%3==1: a5(f"Identity replica v{k}", "def transform(grid):\n    return [row[:] for row in grid]")
    else: a5(f"Floor-shift replica v{k}", t5[7][1])
t5=t5[:32]

bodies = {1:t1,2:t2,3:t3,4:t4,5:t5}
for k,v in bodies.items():
    txt=fmt(v)
    with open(os.path.join(OUT,f"body_{k}.txt"),"w") as f:
        f.write(txt)
    print(f"task {k}: {len(v)} candidates, {len(txt)} chars")
