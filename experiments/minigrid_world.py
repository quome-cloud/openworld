"""MiniGrid (DoorKey) as a verified OpenWorld world -- the shared benchmark.

This expresses the MiniGrid DoorKey transition dynamics as explicit, verified code
over symbolic state, so the SAME task can be run by (a) OpenWorld's verified world
(here) and (b) learned/perceptual world models on the GCP instance (DreamerV3 /
TD-MPC2 / V-JEPA-2 / PoE-World), which consume MiniGrid's pixel or symbolic
observations. The on-instance benchmark validates that this transition reproduces
the real `minigrid` environment bit-for-bit over random rollouts before any
head-to-head, so the comparison is on a genuinely shared environment.

Level: a DoorKey-6x6-style grid -- border walls, a split wall at x=3 with a locked
door at (3,2), a key at (1,3), a goal at (4,4), agent starting at (1,1) facing
right. Actions follow MiniGrid: left/right (turn), forward, pickup, toggle, done.
State is symbolic: agent (x,y,dir), has_key, door_open, terminated.
"""

from openworld import World, CodeTransition
from openworld.state import Action

MINIGRID_INITIAL = {"x": 1, "y": 1, "dir": 0, "has_key": False,
                    "door_open": False, "terminated": False}
MINIGRID_ACTIONS = ["left", "right", "forward", "pickup", "toggle", "done"]
MINIGRID_RULES = [
    "Grid is 6x6 with border walls and a split wall at column x=3 (a locked door at (3,2)).",
    "A key sits at (1,3); the goal at (4,4); the agent starts at (1,1) facing right (dir 0).",
    "dir is 0=right,1=down,2=left,3=up. 'left'/'right' rotate; 'forward' advances one cell.",
    "'forward' is blocked by the grid edge, a wall, the key-on-ground, or a closed door.",
    "'pickup' takes the key if the agent faces the key cell; 'toggle' opens the door if "
    "the agent faces it while carrying the key.",
    "Stepping onto the goal sets terminated; a terminated state never changes.",
]

# Self-contained verified transition (layout baked in; pure (state, action) -> state).
MINIGRID_CODE = '''
def transition(state, action):
    s = dict(state)
    if s.get("terminated"):
        return s
    W = H = 6
    walls = {(x, y) for x in range(W) for y in range(H) if x in (0, W - 1) or y in (0, H - 1)}
    walls |= {(3, y) for y in (1, 3, 4)}          # split wall, gap at the door (3,2)
    key_pos, door_pos, goal_pos = (1, 3), (3, 2), (4, 4)
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    name = action["name"]
    d = s["dir"]
    if name == "left":
        s["dir"] = (d - 1) % 4
    elif name == "right":
        s["dir"] = (d + 1) % 4
    elif name == "forward":
        dx, dy = dirs[d]
        nx, ny = s["x"] + dx, s["y"] + dy
        blocked = (nx < 0 or ny < 0 or nx >= W or ny >= H or (nx, ny) in walls
                   or ((nx, ny) == door_pos and not s["door_open"])
                   or ((nx, ny) == key_pos and not s["has_key"]))
        if not blocked:
            s["x"], s["y"] = nx, ny
            if (nx, ny) == goal_pos:
                s["terminated"] = True
    elif name == "pickup":
        dx, dy = dirs[d]
        if (s["x"] + dx, s["y"] + dy) == key_pos and not s["has_key"]:
            s["has_key"] = True
    elif name == "toggle":
        dx, dy = dirs[d]
        if (s["x"] + dx, s["y"] + dy) == door_pos and s["has_key"]:
            s["door_open"] = True
    return s
'''

# A known optimal-ish solution, used to self-test the dynamics offline.
SOLUTION = ["right", "forward", "pickup", "left", "forward", "toggle",
            "forward", "forward", "right", "forward", "forward"]


def build_minigrid_world():
    return World(name="minigrid-doorkey",
                 description="MiniGrid DoorKey-6x6 as a verified symbolic world.",
                 initial_state=dict(MINIGRID_INITIAL),
                 actions=list(MINIGRID_ACTIONS),
                 rules=list(MINIGRID_RULES),
                 transition=CodeTransition(MINIGRID_CODE))


def solved(state):
    return bool(state.get("terminated"))


if __name__ == "__main__":
    w = build_minigrid_world()
    s = dict(MINIGRID_INITIAL)
    for a in SOLUTION:
        s = dict(w.transition.step(s, Action(a)))
    print("final:", s, "-> solved:", solved(s))
    assert solved(s), "the known solution should reach the goal"
    print("ok: MiniGrid DoorKey world solves in %d steps" % len(SOLUTION))
