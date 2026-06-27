# Understand which shapes survive vs removed, relative to the 5-structure.
# demo0: removed 4@(4,3),1@(8,3); survive 3@(1,8),6@(5,8)
#   The 5-region: there's a 5-bordered blob on left. shapes inside the 5-area get removed.
#   survivors are on the right (cols 7-9), away from 5s.
# demo1: removed: which? OUT shapes were the survivors. Let me recompute survivors from output.
demos_out=[
[[0,0,0,0,0,0,0,3,3,3],[0,0,5,5,5,5,5,3,2,3],[0,5,0,0,0,0,0,3,3,3],[0,5,0,0,0,0,0,0,0,0],[5,0,0,0,0,0,0,6,6,6],[0,5,0,0,0,0,5,6,3,6],[0,5,5,5,5,5,0,6,6,6],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]],
[[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,5,0,0,0,0,0],[0,0,0,5,0,5,5,5,5,0],[8,8,8,0,5,0,0,1,1,1],[8,2,8,0,0,5,0,1,8,1],[8,8,8,0,5,0,0,1,1,1],[0,5,0,5,0,0,0,0,0,0],[0,5,5,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]],
[[1,1,1,0,0,0,0,0,0,0],[1,9,1,0,0,0,0,0,0,0],[1,1,1,0,0,5,0,0,0,0],[0,5,0,0,5,0,5,0,0,0],[0,5,0,5,0,0,0,5,0,0],[0,0,5,0,0,0,0,0,5,0],[0,0,0,0,0,0,0,0,5,0],[0,0,0,0,0,0,0,7,7,7],[0,0,0,0,0,0,0,7,3,7],[0,0,0,0,0,0,0,7,7,7]],
]
def find_shapes(g):
    shapes=[];R=len(g);C=len(g[0])
    for r in range(1,R-1):
        for c in range(1,C-1):
            b=[g[r-1][c-1],g[r-1][c],g[r-1][c+1],g[r][c-1],g[r][c+1],g[r+1][c-1],g[r+1][c+1],g[r+1][c]]
            if len(set(b))==1 and b[0]!=0 and g[r][c]!=0 and g[r][c]!=b[0]:
                shapes.append((r,c,b[0],g[r][c]))
    return shapes
demos_in=[
[[0,0,0,0,0,0,0,3,3,3],[0,0,5,5,5,5,5,3,4,3],[0,5,0,0,0,0,0,3,3,3],[0,5,4,4,4,0,0,0,0,0],[5,0,4,2,4,0,0,6,6,6],[0,5,4,4,4,0,5,6,1,6],[0,5,5,5,5,5,0,6,6,6],[0,0,1,1,1,0,0,0,0,0],[0,0,1,3,1,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0]],
[[2,2,2,0,0,0,3,3,3,0],[2,6,2,0,0,0,3,2,3,0],[2,2,2,0,5,0,3,3,3,0],[0,0,0,5,0,5,5,5,5,0],[8,8,8,0,5,0,0,1,1,1],[8,3,8,0,0,5,0,1,4,1],[8,8,8,0,5,0,0,1,1,1],[0,5,0,5,4,4,4,0,0,0],[0,5,5,0,4,8,4,0,0,0],[0,0,0,0,4,4,4,0,0,0]],
[[1,1,1,0,0,0,0,4,4,4],[1,2,1,0,0,0,0,4,3,4],[1,1,1,0,0,5,0,4,4,4],[0,5,0,0,5,0,5,0,0,0],[0,5,0,5,3,3,3,5,0,0],[0,0,5,0,3,6,3,0,5,0],[0,0,0,0,3,3,3,0,5,0],[2,2,2,0,0,0,0,7,7,7],[2,9,2,0,0,0,0,7,4,7],[2,2,2,0,0,0,0,7,7,7]],
]
for i in range(3):
    si=find_shapes(demos_in[i]); so=find_shapes(demos_out[i])
    surv={s[:2] for s in so}
    print(f"demo{i}:")
    for s in si:
        status="SURVIVE" if s[:2] in surv else "REMOVED"
        print(f"   {s} -> {status}")
    print("  out shapes:",so)

print("\n=== Graph analysis ===")
for i in range(3):
    si=find_shapes(demos_in[i]); so=find_shapes(demos_out[i])
    surv={s[:2] for s in so}
    by_border={}
    for s in si: by_border.setdefault(s[2],[]).append(s)
    print(f"demo{i}:")
    for s in si:
        r,c,b,cen=s
        points_to=by_border.get(cen,[])
        st="SURV" if s[:2] in surv else "rem"
        # is this shape's border pointed to by another?
        pointed_by=[x for x in si if x[3]==b]
        print(f"   shape b={b} cen={cen} [{st}] points_to_border={[(x[2],x[3]) for x in points_to]} pointed_by={[(x[2],x[3]) for x in pointed_by]}")
