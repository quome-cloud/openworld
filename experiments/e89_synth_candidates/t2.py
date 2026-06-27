demos=[
({"i":[[0,0,0,0,0,0,0,3,3,3],[0,0,5,5,5,5,5,3,4,3],[0,5,0,0,0,0,0,3,3,3],[0,5,4,4,4,0,0,0,0,0],[5,0,4,2,4,0,0,6,6,6],[0,5,4,4,4,0,5,6,1,6],[0,5,5,5,5,5,0,6,6,6],[0,0,1,1,1,0,0,0,0,0],[0,0,1,3,1,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0]],
  "o":[[0,0,0,0,0,0,0,3,3,3],[0,0,5,5,5,5,5,3,2,3],[0,5,0,0,0,0,0,3,3,3],[0,5,0,0,0,0,0,0,0,0],[5,0,0,0,0,0,0,6,6,6],[0,5,0,0,0,0,5,6,3,6],[0,5,5,5,5,5,0,6,6,6],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]}),
]
# Identify 3x3 shapes: a center cell surrounded by 8 cells of same border color
def find_shapes(g):
    shapes=[]
    R=len(g);C=len(g[0])
    for r in range(1,R-1):
        for c in range(1,C-1):
            border=[g[r-1][c-1],g[r-1][c],g[r-1][c+1],g[r][c-1],g[r][c+1],g[r+1][c-1],g[r+1][c],g[r+1][c+1]]
            if len(set(border))==1 and border[0]!=0 and g[r][c]!=0 and g[r][c]!=border[0]:
                shapes.append((r,c,border[0],g[r][c]))
    return shapes
d=demos[0]
print("IN shapes (cr,cc,border,center):")
for s in find_shapes(d["i"]): print("  ",s)
print("OUT shapes:")
for s in find_shapes(d["o"]): print("  ",s)

print("\n=== Analysis demo1 ===")
# IN shapes: 3-bordered(c=4) at (1,8); 4-bordered(c=2) at (4,3); 6-bordered(c=1) at (5,8); 1-bordered(c=3) at (8,3)
# OUT: 3-bordered center became 2; 6-bordered center became 3; 4-bordered and 1-bordered REMOVED
# So which got removed? 4-bordered@(4,3) and 1-bordered@(8,3) -- both in left/middle area (inside 5 region)
# 3-bordered@(1,8) and 6-bordered@(5,8) -- right column, survive but center changes
# 4-bordered center was 2 -> 3-bordered new center is 2. 1-bordered center was 3 -> 6-bordered new center is 3.
# So: removed shapes' CENTER values get transferred to surviving shapes.
# Matching: 4(c=2) -> 3-shape gets 2. 1(c=3) -> 6-shape gets 3.
# How matched? 4-shape is at row4, 3-shape at row1 (both... hmm). 
# Maybe the BORDER color of removed shape == ? 4-shape border=4, its center 2. 
# Note: 3-shape center was 4 = border color of the removed 4-shape! And 3-shape new center=2 = center of 4-shape.
# 6-shape center was 1 = border color of removed 1-shape! 6-shape new center=3 = center of 1-shape.
# RULE: each surviving shape's center value NAMES a removed shape (by border color). 
#        Replace surviving center with that removed shape's center value. Remove the named shapes.
print("CONFIRMED rule: surviving shape center C points to removed shape with border=C; new center = that shape's center")
