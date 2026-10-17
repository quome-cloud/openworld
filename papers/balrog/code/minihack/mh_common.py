"""Core observation parsing, mapping, and pathfinding for the MiniHack arm.

Honest-observation contract: everything here consumes ONLY the observation dict
that BALROG's wrapper stack serves to its agents:
  obs["obs"]  -> glyphs, blstats, tty_chars, tty_cursor, inv_letters, inv_strs, tty_colors
  obs["text"] -> the rendered language observation (we use it for messages)
No env.unwrapped access, no simulator cloning, no privileged state.
"""

import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, "pylib") not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, "pylib"))
    sys.path.insert(0, _HERE)

import numpy as np
from nle import nethack as nh

ROWS, COLS = 21, 79

# ---------------------------------------------------------------- glyph decode

GLYPH_MON_OFF = nh.GLYPH_MON_OFF        # 0     hostile/peaceful monsters
GLYPH_PET_OFF = nh.GLYPH_PET_OFF        # 381   pets
GLYPH_INVIS_OFF = nh.GLYPH_INVIS_OFF    # 762
GLYPH_DETECT_OFF = nh.GLYPH_DETECT_OFF  # 763
GLYPH_BODY_OFF = nh.GLYPH_BODY_OFF      # 1144
GLYPH_RIDDEN_OFF = nh.GLYPH_RIDDEN_OFF  # 1525
GLYPH_OBJ_OFF = nh.GLYPH_OBJ_OFF        # 1906
GLYPH_CMAP_OFF = nh.GLYPH_CMAP_OFF      # 2359
GLYPH_EXPLODE_OFF = nh.GLYPH_EXPLODE_OFF
MAX_GLYPH = nh.MAX_GLYPH

# terrain codes (our semantic map)
UNKNOWN, WALL, FLOOR, CORRIDOR, DOORWAY, DOOR_OPEN, DOOR_CLOSED = 0, 1, 2, 3, 4, 5, 6
STAIRS_DOWN, STAIRS_UP, LAVA, ICE, FOUNTAIN, TRAP, WATER, IRONBARS, TREE, ALTAR = (
    7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

_CMAP_TO_TERRAIN = {}
for i in range(nh.MAXPCHARS):
    _CMAP_TO_TERRAIN[i] = None
for i in range(1, 12):
    _CMAP_TO_TERRAIN[i] = WALL
_CMAP_TO_TERRAIN[0] = None            # stone / unseen dark
_CMAP_TO_TERRAIN[12] = DOORWAY        # broken/no door
_CMAP_TO_TERRAIN[13] = DOOR_OPEN
_CMAP_TO_TERRAIN[14] = DOOR_OPEN
_CMAP_TO_TERRAIN[15] = DOOR_CLOSED
_CMAP_TO_TERRAIN[16] = DOOR_CLOSED
_CMAP_TO_TERRAIN[17] = IRONBARS
_CMAP_TO_TERRAIN[18] = TREE
_CMAP_TO_TERRAIN[19] = FLOOR
_CMAP_TO_TERRAIN[20] = FLOOR          # dark part of a room (floor, once seen)
_CMAP_TO_TERRAIN[21] = CORRIDOR
_CMAP_TO_TERRAIN[22] = CORRIDOR
_CMAP_TO_TERRAIN[23] = STAIRS_UP
_CMAP_TO_TERRAIN[24] = STAIRS_DOWN
_CMAP_TO_TERRAIN[25] = STAIRS_UP      # ladder
_CMAP_TO_TERRAIN[26] = STAIRS_DOWN
_CMAP_TO_TERRAIN[27] = ALTAR
_CMAP_TO_TERRAIN[30] = FLOOR          # sink
_CMAP_TO_TERRAIN[31] = FOUNTAIN
_CMAP_TO_TERRAIN[32] = WATER
_CMAP_TO_TERRAIN[33] = ICE
_CMAP_TO_TERRAIN[34] = LAVA
_CMAP_TO_TERRAIN[35] = FLOOR          # lowered drawbridge
_CMAP_TO_TERRAIN[36] = FLOOR
for i in range(42, 65):
    _CMAP_TO_TERRAIN[i] = TRAP

PASSABLE = {FLOOR, CORRIDOR, DOORWAY, DOOR_OPEN, STAIRS_DOWN, STAIRS_UP, ICE,
            FOUNTAIN, ALTAR, TRAP}

BOULDER_GLYPH = None
for _g in range(GLYPH_OBJ_OFF, GLYPH_CMAP_OFF):
    _idx = _g - GLYPH_OBJ_OFF
    try:
        oc = nh.objclass(_idx)
        if nh.OBJ_NAME(oc) == "boulder":
            BOULDER_GLYPH = _g
            break
    except Exception:
        pass

DIRS = {
    "north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0),
    "northeast": (1, -1), "southeast": (1, 1), "southwest": (-1, 1), "northwest": (-1, -1),
}
DIR_OF = {v: k for k, v in DIRS.items()}
CARDINALS = ["north", "east", "south", "west"]


def is_monster_glyph(g):
    return (GLYPH_MON_OFF <= g < GLYPH_INVIS_OFF) or g == GLYPH_INVIS_OFF or \
           (GLYPH_DETECT_OFF <= g < GLYPH_BODY_OFF) or (GLYPH_RIDDEN_OFF <= g < GLYPH_OBJ_OFF)


def is_pet_glyph(g):
    return GLYPH_PET_OFF <= g < GLYPH_INVIS_OFF


def monster_char(g):
    """Display char of a monster glyph (species class letter)."""
    if GLYPH_PET_OFF <= g < GLYPH_INVIS_OFF:
        idx = g - GLYPH_PET_OFF
    elif GLYPH_MON_OFF <= g < GLYPH_PET_OFF:
        idx = g - GLYPH_MON_OFF
    else:
        return "?"
    try:
        pm = nh.permonst(nh.glyph_to_mon(g))
        return chr(nh.class_sym.from_mlet(pm.mlet).sym)
    except Exception:
        return "?"


class LevelState:
    """Persistent per-episode map memory + current-frame entities.

    Dark-wall inference (engine lesson): NetHack never renders walls in
    unlit areas, even at distance 1. But night-vision radius 1 always
    reveals adjacent floor/objects/monsters. Therefore any cell that is
    still glyph-0 (stone/unseen) while the agent stands adjacent to it is
    provably impassable stone/wall. We write that negative information
    back into the map (inferred=True cells).
    """

    def __init__(self, infer_dark_walls=True):
        self.infer_dark_walls = infer_dark_walls
        self.terrain = np.zeros((ROWS, COLS), dtype=np.int8)  # UNKNOWN
        self.explored = np.zeros((ROWS, COLS), dtype=bool)    # glyph ever non-stone here
        self.inferred_wall = np.zeros((ROWS, COLS), dtype=bool)
        self.agent = (0, 0)            # (x, y)
        self.monsters = []             # list[(x, y, char, is_pet)]
        self.boulders = set()          # {(x, y)}
        self.items = set()             # object piles (non-boulder)
        self.hp = self.hpmax = 1
        self.time = 0
        self.message = ""
        self.steps = 0

    def update(self, obs):
        raw = obs["obs"]
        glyphs = raw["glyphs"]
        tty = raw.get("tty_chars")
        bl = raw["blstats"]
        self.agent = (int(bl[nh.NLE_BL_X]), int(bl[nh.NLE_BL_Y]))
        self.hp, self.hpmax = int(bl[nh.NLE_BL_HP]), int(bl[nh.NLE_BL_HPMAX])
        self.time = int(bl[nh.NLE_BL_TIME])
        self.message = extract_message(obs)
        self.monsters = []
        self.boulders = set()
        self.items = set()
        ax, ay = self.agent
        for y in range(ROWS):
            row = glyphs[y]
            for x in range(COLS):
                g = int(row[x])
                if g == GLYPH_CMAP_OFF:  # cmap 0: stone/unseen — no information
                    continue
                self.explored[y][x] = True
                self.inferred_wall[y][x] = False
                if GLYPH_CMAP_OFF <= g < GLYPH_CMAP_OFF + nh.MAXPCHARS:
                    t = _CMAP_TO_TERRAIN[g - GLYPH_CMAP_OFF]
                    if t is not None:
                        self.terrain[y][x] = t
                elif g == BOULDER_GLYPH:
                    self.boulders.add((x, y))
                    if self.terrain[y][x] == UNKNOWN:
                        self.terrain[y][x] = FLOOR  # boulder sits on something walkable
                elif GLYPH_OBJ_OFF <= g < GLYPH_CMAP_OFF:
                    self.items.add((x, y))
                    if self.terrain[y][x] == UNKNOWN:
                        self.terrain[y][x] = FLOOR
                elif GLYPH_BODY_OFF <= g < GLYPH_RIDDEN_OFF:
                    if self.terrain[y][x] == UNKNOWN:
                        self.terrain[y][x] = FLOOR
                elif is_monster_glyph(g):
                    if (x, y) != (ax, ay):
                        # species char from the tty screen (map starts row 1)
                        ch = chr(tty[y + 1][x]) if tty is not None else "?"
                        self.monsters.append((x, y, ch, is_pet_glyph(g)))
                    if self.terrain[y][x] == UNKNOWN:
                        self.terrain[y][x] = FLOOR
        # a cell we occupy is certainly passable ground
        if self.terrain[ay][ax] == UNKNOWN or self.inferred_wall[ay][ax]:
            self.terrain[ay][ax] = FLOOR
            self.inferred_wall[ay][ax] = False
        self.explored[ay][ax] = True
        if self.infer_dark_walls:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = ax + dx, ay + dy
                    if 0 <= nx < COLS and 0 <= ny < ROWS and \
                       not self.explored[ny][nx]:
                        self.terrain[ny][nx] = WALL
                        self.explored[ny][nx] = True
                        self.inferred_wall[ny][nx] = True

    # ------------------------------------------------------------- pathfinding

    def passable(self, x, y, ignore_boulders=False, doors_ok=True):
        if not (0 <= x < COLS and 0 <= y < ROWS):
            return False
        t = self.terrain[y][x]
        if t in (LAVA, WATER):
            return False
        if t == DOOR_CLOSED:
            return doors_ok           # closed door = passable via open action
        if t not in PASSABLE:
            return False
        if not ignore_boulders and (x, y) in self.boulders:
            return False
        return True

    def neighbors(self, x, y, diagonals=True, doors_ok=True, avoid=frozenset()):
        t_here = self.terrain[y][x]
        for name, (dx, dy) in DIRS.items():
            if not diagonals and (dx and dy):
                continue
            nx, ny = x + dx, y + dy
            if (nx, ny) in avoid:
                continue
            if not self.passable(nx, ny, doors_ok=doors_ok):
                continue
            if dx and dy:
                t_there = self.terrain[ny][nx]
                # NetHack: no diagonal into/out of a door cell
                if t_here in (DOORWAY, DOOR_OPEN, DOOR_CLOSED) or \
                   t_there in (DOORWAY, DOOR_OPEN, DOOR_CLOSED):
                    continue
                # No diagonal squeeze between two impassable orthogonals
                if not self.passable(x + dx, y, doors_ok=False) and \
                   not self.passable(x, y + dy, doors_ok=False):
                    continue
            yield name, (nx, ny)

    def bfs(self, start, goals, diagonals=True, doors_ok=True, avoid=frozenset()):
        """Shortest path from start to nearest goal. Returns list of action
        names, or None."""
        from collections import deque
        goals = set(goals)
        if start in goals:
            return []
        q = deque([start])
        prev = {start: None}
        while q:
            cur = q.popleft()
            for name, nxt in self.neighbors(*cur, diagonals=diagonals,
                                            doors_ok=doors_ok, avoid=avoid):
                if nxt in prev:
                    continue
                prev[nxt] = (cur, name)
                if nxt in goals:
                    path = []
                    node = nxt
                    while prev[node] is not None:
                        node, nm = prev[node]
                        path.append(nm)
                    return path[::-1]
                q.append(nxt)
        return None

    def frontier_cells(self):
        """Known-passable cells adjacent (8-dir) to unexplored cells."""
        out = []
        for y in range(ROWS):
            for x in range(COLS):
                if not self.passable(x, y):
                    continue
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < COLS and 0 <= ny < ROWS and \
                           not self.explored[ny][nx]:
                            out.append((x, y))
                            break
                    else:
                        continue
                    break
        return out

    def find_terrain(self, code):
        ys, xs = np.where(self.terrain == code)
        return [(int(x), int(y)) for x, y in zip(xs, ys)]


def extract_message(obs):
    txt = obs["text"]["long_term_context"]
    # first block: "message:\n<msg...>\n\nlanguage observation:"
    if txt.startswith("message:\n"):
        rest = txt[len("message:\n"):]
        return rest.split("\n\nlanguage observation:")[0].strip()
    return ""


def inventory(obs):
    """[(letter, description)] from raw inv arrays."""
    raw = obs["obs"]
    letters = raw["inv_letters"]
    strs = raw["inv_strs"]
    out = []
    for i, l in enumerate(letters):
        if l == 0:
            break
        desc = bytes(strs[i]).partition(b"\x00")[0].decode("latin-1")
        out.append((chr(l), desc))
    return out
