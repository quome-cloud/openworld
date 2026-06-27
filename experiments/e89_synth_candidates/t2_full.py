demos=[
({"i":[[0,0,0,0,0,0,0,3,3,3],[0,0,5,5,5,5,5,3,4,3],[0,5,0,0,0,0,0,3,3,3],[0,5,4,4,4,0,0,0,0,0],[5,0,4,2,4,0,0,6,6,6],[0,5,4,4,4,0,5,6,1,6],[0,5,5,5,5,5,0,6,6,6],[0,0,1,1,1,0,0,0,0,0],[0,0,1,3,1,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0]],
  "o":[[0,0,0,0,0,0,0,3,3,3],[0,0,5,5,5,5,5,3,2,3],[0,5,0,0,0,0,0,3,3,3],[0,5,0,0,0,0,0,0,0,0],[5,0,0,0,0,0,0,6,6,6],[0,5,0,0,0,0,5,6,3,6],[0,5,5,5,5,5,0,6,6,6],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]}),
({"i":[[2,2,2,0,0,0,3,3,3,0],[2,6,2,0,0,0,3,2,3,0],[2,2,2,0,5,0,3,3,3,0],[0,0,0,5,0,5,5,5,5,0],[8,8,8,0,5,0,0,1,1,1],[8,3,8,0,0,5,0,1,4,1],[8,8,8,0,5,0,0,1,1,1],[0,5,0,5,4,4,4,0,0,0],[0,5,5,0,4,8,4,0,0,0],[0,0,0,0,4,4,4,0,0,0]],
  "o":[[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,5,0,0,0,0,0],[0,0,0,5,0,5,5,5,5,0],[8,8,8,0,5,0,0,1,1,1],[8,2,8,0,0,5,0,1,8,1],[8,8,8,0,5,0,0,1,1,1],[0,5,0,5,0,0,0,0,0,0],[0,5,5,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]}),
({"i":[[1,1,1,0,0,0,0,4,4,4],[1,2,1,0,0,0,0,4,3,4],[1,1,1,0,0,5,0,4,4,4],[0,5,0,0,5,0,5,0,0,0],[0,5,0,5,3,3,3,5,0,0],[0,0,5,0,3,6,3,0,5,0],[0,0,0,0,3,3,3,0,5,0],[2,2,2,0,0,0,0,7,7,7],[2,9,2,0,0,0,0,7,4,7],[2,2,2,0,0,0,0,7,7,7]],
  "o":[[1,1,1,0,0,0,0,0,0,0],[1,9,1,0,0,0,0,0,0,0],[1,1,1,0,0,5,0,0,0,0],[0,5,0,0,5,0,5,0,0,0],[0,5,0,5,0,0,0,5,0,0],[0,0,5,0,0,0,0,0,5,0],[0,0,0,0,0,0,0,0,5,0],[0,0,0,0,0,0,0,7,7,7],[0,0,0,0,0,0,0,7,3,7],[0,0,0,0,0,0,0,7,7,7]]}),
]
def find_shapes(g):
    shapes=[];R=len(g);C=len(g[0])
    for r in range(1,R-1):
        for c in range(1,C-1):
            b=[g[r-1][c-1],g[r-1][c],g[r-1][c+1],g[r][c-1],g[r][c+1],g[r+1][c-1],g[r+1][c],g[r+1][c+1]]
            if len(set(b))==1 and b[0]!=0 and g[r][c]!=0 and g[r][c]!=b[0]:
                shapes.append((r,c,b[0],g[r][c]))
    return shapes

def solve(g):
    import copy
    R=len(g);C=len(g[0])
    out=copy.deepcopy(g)
    shapes=find_shapes(g)
    by_border={s[2]:s for s in shapes}
    survivors=[]; removed=set()
    # A shape "survives" if its center value matches some other shape's border (points to it)
    for s in shapes:
        r,c,bord,cen=s
        if cen in by_border and cen!=bord:
            survivors.append(s); removed.add(by_border[cen][:3])
    # Determine cells to clear: 5-region detail. Simplest: remove all shapes that are NOT survivors, plus clear their cells.
    # And for survivors, set center = pointed shape's center.
    surv_ids={s[:2] for s in survivors}
    # clear non-survivor shapes (their 3x3 block) and also some 5-path cells
    def clear_block(r,c):
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                out[r+dr][c+dc]=0
    for s in shapes:
        if s[:2] not in surv_ids:
            clear_block(s[0],s[1])
    for s in survivors:
        r,c,bord,cen=s
        target=by_border[cen]
        out[r][c]=target[3]
    return out

for i,d in enumerate(demos):
    got=solve(d["i"])
    ok = got==d["o"]
    print(f"demo{i}: shapes={find_shapes(d['i'])} match={ok}")
    if not ok:
        for r in range(len(got)):
            if got[r]!=d["o"][r]:
                print("  row",r,"got",got[r])
                print("       exp",d["o"][r])
