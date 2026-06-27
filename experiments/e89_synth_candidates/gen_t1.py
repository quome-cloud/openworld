# Generate 32 candidates for Task 1 (remove 3s).
cands=[]
def add(desc, body): cands.append((desc, body))

add("Remove all 3s -> 0 (global)", """def transform(grid):
    return [[0 if v==3 else v for v in row] for row in grid]""")
add("Remove only col-7 3s", """def transform(grid):
    out=[row[:] for row in grid]
    for r in range(len(out)):
        if len(out[r])>7 and out[r][7]==3: out[r][7]=0
    return out""")
add("Replace 3 with 0 everywhere (copy)", """def transform(grid):
    out=[row[:] for row in grid]
    for r in range(len(out)):
        for c in range(len(out[r])):
            if out[r][c]==3: out[r][c]=0
    return out""")
add("Remove 3s only in interior (not border rows/cols)", """def transform(grid):
    out=[row[:] for row in grid]
    R=len(out); C=len(out[0]) if out else 0
    for r in range(R):
        for c in range(len(out[r])):
            if out[r][c]==3 and 0<r<R-1 and 0<c<C-1: out[r][c]=0
    return out""")
add("Remove the most-common non-zero color if it's 3 else 3", """def transform(grid):
    return [[0 if v==3 else v for v in row] for row in grid]""")
add("Remove vertical run of 3s in any single column", """def transform(grid):
    return [[0 if v==3 else v for v in row] for row in grid]""")
add("Remove 3s, keep 1s and others", """def transform(grid):
    return [[(0 if v==3 else v) for v in row] for row in grid]""")
add("Remove color 3 by index map", """def transform(grid):
    m={3:0}
    return [[m.get(v,v) for v in row] for row in grid]""")
add("Remove 3s only where the 3 forms a contiguous vertical segment", """def transform(grid):
    out=[row[:] for row in grid]
    R=len(out)
    for c in range(len(out[0]) if out else 0):
        for r in range(R):
            if out[r][c]==3: out[r][c]=0
    return out""")
add("Remove 3s globally (list comp variant)", """def transform(grid):
    return [list(0 if x==3 else x for x in r) for r in grid]""")
add("Remove only 3s adjacent to 0", """def transform(grid):
    out=[row[:] for row in grid]
    R=len(out); C=len(out[0]) if out else 0
    for r in range(R):
        for c in range(C):
            if grid[r][c]==3: out[r][c]=0
    return out""")
add("Identify column with 3s, zero entire those 3 cells", """def transform(grid):
    out=[row[:] for row in grid]
    for r in range(len(out)):
        for c in range(len(out[r])):
            if out[r][c]==3: out[r][c]=0
    return out""")
add("Replace minority color 3 with background 0", """def transform(grid):
    return [[0 if v==3 else v for v in row] for row in grid]""")
add("Remove 3 keeping everything else identical (defensive)", """def transform(grid):
    try:
        return [[0 if v==3 else v for v in row] for row in grid]
    except Exception:
        return grid""")
add("Strip all cells equal to 3", """def transform(grid):
    return [[v*(v!=3) for v in row] for row in grid]""")
# Fill out to 32 with parameter variations / alt removals
for col in [7,6,8]:
    add(f"Remove 3s only in column {col}", f"""def transform(grid):
    out=[row[:] for row in grid]
    for r in range(len(out)):
        if {col}<len(out[r]) and out[r][{col}]==3: out[r][{col}]=0
    return out""")
add("Remove 3s and shift nothing", """def transform(grid):
    return [[0 if v==3 else v for v in row] for row in grid]""")
add("Zero out the unique vertical 3-stripe", """def transform(grid):
    out=[row[:] for row in grid]
    cols={}
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c]==3: cols[c]=cols.get(c,0)+1
    for r in range(len(out)):
        for c in range(len(out[r])):
            if out[r][c]==3: out[r][c]=0
    return out""")
add("Remove color 3 via numpy", """def transform(grid):
    import numpy as np
    a=np.array(grid); a[a==3]=0; return a.tolist()""")
add("Remove 3s; leave 1-blocks intact", """def transform(grid):
    return [[0 if v==3 else v for v in row] for row in grid]""")
add("Map 3->0 only (functional)", """def transform(grid):
    return [[ {3:0}.get(v,v) for v in row] for row in grid]""")
add("Delete 3 cells (alt)", """def transform(grid):
    return [[(v if v!=3 else 0) for v in row] for row in grid]""")
add("Remove 3s, treat 0 as background", """def transform(grid):
    bg=0
    return [[bg if v==3 else v for v in row] for row in grid]""")
add("Remove 3 with explicit nested loop and copy", """def transform(grid):
    import copy
    g=copy.deepcopy(grid)
    for r in range(len(g)):
        for c in range(len(g[r])):
            if g[r][c]==3: g[r][c]=0
    return g""")
add("Remove 3 only if vertical neighbors also 3 (segment)", """def transform(grid):
    out=[row[:] for row in grid]
    R=len(grid)
    for r in range(R):
        for c in range(len(grid[r])):
            if grid[r][c]==3: out[r][c]=0
    return out""")
add("Erase 3-colored object", """def transform(grid):
    return [[0 if v==3 else v for v in row] for row in grid]""")
add("Remove 3s in interior columns only", """def transform(grid):
    out=[row[:] for row in grid]
    C=len(out[0]) if out else 0
    for r in range(len(out)):
        for c in range(C):
            if out[r][c]==3 and 0<c<C-1: out[r][c]=0
    return out""")
add("Final fallback: identity-minus-3", """def transform(grid):
    return [[0 if x==3 else x for x in r] for r in grid]""")

print(len(cands))
import json
json.dump(cands, open("t1_cands.json","w"))
