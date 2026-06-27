# Demo0: nested frames. 6-frame is a maze. Only 2 box cells convert despite 12 outside 9s.
# Demo2: no frame, 4 outside 9s, 4 convert.
# Demo1: 6-frame (single), box fills completely (8) with 15 outside 9s.
# Hypothesis refinement: count 9s that can "reach" the box (not blocked by frame)?
# This is too uncertain. Alternative simple hyp for demo2/demo1: fill = number of outside-9 GROUPS? 
# demo2: 9 groups: (1,9),(2,2),(2,10),(9,8) = 4 isolated -> 4 fill. MATCHES count.
# demo0: 12 9s but grouped: (1,8)(1,9)(2,8)(2,9)=1grp; (2,2)(2,3)=1; (5,9)=1; (6,1)(7,0)(7,1)=1; (8,7)(8,8)=1 => 5 groups but only 2 convert. NO.
# Demo0 has TWO nested boxes (6-frame outer + 3-box inner). Maybe only 9s INSIDE outer frame count.
inp0=[[7,7,7,7,7,7,7,7,7,7,7],[7,6,6,6,6,6,6,6,9,9,7],[7,6,9,9,7,7,7,6,9,9,7],[7,6,6,6,6,6,6,6,7,7,7],[7,7,7,6,3,3,3,6,7,7,7],[7,7,7,6,3,2,3,6,7,7,7],[7,9,7,6,3,3,3,6,7,7,7],[9,9,7,6,6,7,6,6,6,6,7],[7,7,7,6,7,7,7,9,9,6,7],[7,7,7,6,6,6,6,6,6,6,7],[7,7,7,7,7,7,7,7,7,7,7]]
# 9s inside the 6 frame region (rows4-8,cols4-7 inner area roughly): (2,2)(2,3) inside the 6-frame's left wing; (8,7)(8,8) inside.
# Actually the 6 frame is a complicated maze. 9s embedded WITHIN the maze corridors: (2,2),(2,3) and (8,7),(8,8) => 2 groups => 2 fills. The 9s at (1,8),(1,9),(2,8),(2,9),(5,9),(6,1),(7,0),(7,1) are OUTSIDE the 6 maze.
# That gives 2 fills for demo0. CONSISTENT.
# Demo1: 6 frame is a closed box rows3-9. 9s inside it: (4,8)? no that's outside right... inside cols4-6 rows4-8. 
# Inside-frame 9s in demo1: none obvious... but box filled fully(8). Hmm inconsistent unless different rule.
print("Demo0 supports: count 9-groups embedded inside the frame maze -> fill that many box cells clockwise from top-left")
print("Task3 remains uncertain; will provide diverse candidates")
