"""Core observation parsing, multi-level mapping and pathfinding for the
full-NetHack (BALROG 'nle' / NetHackChallenge-v0) arm.

Honest-observation contract: everything here consumes ONLY the observation
dict that BALROG's wrapper stack serves to its agents:
  obs["obs"]  -> glyphs, blstats, tty_chars, tty_colors, tty_cursor,
                 inv_letters, inv_strs, inv_oclasses, misc, text_message
  obs["text"] -> the rendered language observation
No env.unwrapped access, no simulator cloning, no privileged state reads.

Offline world-model provenance (disclosed in the report): terrain code
tables, monster species facts (permonst: name/level/difficulty), and object
class constants come from the NLE python API / NetHack source read OFFLINE.
At runtime they are pure lookup tables keyed by served glyph ids.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "pylib"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from nle import nethack as nh

ROWS, COLS = 21, 79

# ------------------------------------------------------------------ blstats
BL = nh  # alias for constants: nh.NLE_BL_X etc.

# hunger states (NetHack hunger_state): 0 Satiated 1 NotHungry 2 Hungry
# 3 Weak 4 Fainting 5 Fainted 6 Starved
HUNGRY, WEAK, FAINTING = 2, 3, 4

# ------------------------------------------------------------- glyph ranges
GLYPH_MON_OFF = nh.GLYPH_MON_OFF
GLYPH_PET_OFF = nh.GLYPH_PET_OFF
GLYPH_INVIS_OFF = nh.GLYPH_INVIS_OFF
GLYPH_DETECT_OFF = nh.GLYPH_DETECT_OFF
GLYPH_BODY_OFF = nh.GLYPH_BODY_OFF
GLYPH_RIDDEN_OFF = nh.GLYPH_RIDDEN_OFF
GLYPH_OBJ_OFF = nh.GLYPH_OBJ_OFF
GLYPH_CMAP_OFF = nh.GLYPH_CMAP_OFF
GLYPH_SWALLOW_OFF = getattr(nh, "GLYPH_SWALLOW_OFF", None)
GLYPH_WARNING_OFF = getattr(nh, "GLYPH_WARNING_OFF", None)
GLYPH_EXPLODE_OFF = nh.GLYPH_EXPLODE_OFF
GLYPH_ZAP_OFF = getattr(nh, "GLYPH_ZAP_OFF", None)
MAX_GLYPH = nh.MAX_GLYPH

# ------------------------------------------------------------ terrain codes
(UNKNOWN, WALL, FLOOR, CORRIDOR, DOORWAY, DOOR_OPEN, DOOR_CLOSED,
 STAIRS_DOWN, STAIRS_UP, LAVA, ICE, FOUNTAIN, TRAP, WATER, IRONBARS, TREE,
 ALTAR, HOLE_DOWN, BAD_TRAP, SLOW_TRAP, GRAVE, THRONE, SINK, AIR,
 CLOUD) = range(25)

# cmap index -> terrain (verified against nh.symdef table for balrog-nle 0.9)
_CMAP_TO_TERRAIN = {i: None for i in range(nh.MAXPCHARS)}
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
_CMAP_TO_TERRAIN[20] = FLOOR          # dark part of a room
_CMAP_TO_TERRAIN[21] = CORRIDOR
_CMAP_TO_TERRAIN[22] = CORRIDOR
_CMAP_TO_TERRAIN[23] = STAIRS_UP
_CMAP_TO_TERRAIN[24] = STAIRS_DOWN
_CMAP_TO_TERRAIN[25] = STAIRS_UP      # ladder up
_CMAP_TO_TERRAIN[26] = STAIRS_DOWN    # ladder down
_CMAP_TO_TERRAIN[27] = ALTAR
_CMAP_TO_TERRAIN[28] = GRAVE
_CMAP_TO_TERRAIN[29] = THRONE
_CMAP_TO_TERRAIN[30] = SINK
_CMAP_TO_TERRAIN[31] = FOUNTAIN
_CMAP_TO_TERRAIN[32] = WATER
_CMAP_TO_TERRAIN[33] = ICE
_CMAP_TO_TERRAIN[34] = LAVA
_CMAP_TO_TERRAIN[35] = FLOOR          # lowered drawbridge
_CMAP_TO_TERRAIN[36] = FLOOR
_CMAP_TO_TERRAIN[37] = WALL           # raised drawbridge
_CMAP_TO_TERRAIN[38] = WALL
_CMAP_TO_TERRAIN[39] = AIR
_CMAP_TO_TERRAIN[40] = CLOUD
_CMAP_TO_TERRAIN[41] = WATER
# traps 42..63 (cmap idx -> nature of trap)
_TRAP_KIND = {
    42: TRAP, 43: TRAP, 44: TRAP, 45: TRAP,           # arrow/dart/rock/board
    46: SLOW_TRAP,                                    # bear trap
    47: BAD_TRAP,                                     # land mine
    48: TRAP, 49: TRAP, 50: TRAP, 51: TRAP,           # boulder/sleep/rust/fire
    52: SLOW_TRAP, 53: SLOW_TRAP,                     # pit / spiked pit
    54: HOLE_DOWN, 55: HOLE_DOWN,                     # hole / trap door
    56: BAD_TRAP, 57: BAD_TRAP, 58: BAD_TRAP,         # tele / level-tele / portal
    59: SLOW_TRAP,                                    # web
    60: TRAP, 61: TRAP, 62: TRAP,
    63: BAD_TRAP,                                     # polymorph trap
}
for _i, _t in _TRAP_KIND.items():
    _CMAP_TO_TERRAIN[_i] = _t
_CMAP_TO_TERRAIN[64] = FLOOR          # vibrating square

PASSABLE = {FLOOR, CORRIDOR, DOORWAY, DOOR_OPEN, STAIRS_DOWN, STAIRS_UP, ICE,
            FOUNTAIN, ALTAR, TRAP, HOLE_DOWN, SLOW_TRAP, BAD_TRAP, GRAVE,
            THRONE, SINK, CLOUD}

DIRS = {
    "north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0),
    "northeast": (1, -1), "southeast": (1, 1), "southwest": (-1, 1),
    "northwest": (-1, -1),
}
DIR_OF = {v: k for k, v in DIRS.items()}
CARDINALS = ["north", "east", "south", "west"]

BOULDER_GLYPH = None
for _g in range(GLYPH_OBJ_OFF, GLYPH_CMAP_OFF):
    try:
        oc = nh.objclass(_g - GLYPH_OBJ_OFF)
        if nh.OBJ_NAME(oc) == "boulder":
            BOULDER_GLYPH = _g
            break
    except Exception:
        pass

# ---------------------------------------------------- monster species model
# Offline species table (permonst). Keyed by monster index.
_SPECIES = {}
for _i in range(nh.NUMMONS):
    _p = nh.permonst(_i)
    _sym = nh.class_sym.from_mlet(_p.mlet).sym
    _SPECIES[_i] = (_p.mname, int(_p.mlevel), int(_p.difficulty),
                    int(_p.mmove), _sym if isinstance(_sym, str) else chr(_sym))

# Species we never initiate melee against (touch/explosion/passive death):
NEVER_MELEE = {
    "floating eye", "cockatrice", "chickatrice", "gas spore", "yellow light",
    "blue jelly", "spotted jelly", "ochre jelly", "acid blob", "green slime",
    "Medusa", "brown mold", "yellow mold", "green mold", "red mold",
}
# Species that cannot move (safe to route around; never chase us):
IMMOBILE = {
    "gas spore", "acid blob", "lichen", "blue jelly", "spotted jelly",
    "ochre jelly", "brown mold", "yellow mold", "green mold", "red mold",
    "shrieker", "violet fungus",
}

# Corpses safe to eat when fresh (offline food-safety facts):
SAFE_CORPSES = {
    "grid bug", "newt", "gecko", "lichen", "lizard", "jackal", "coyote",
    "fox", "sewer rat", "giant rat", "gnome", "gnome lord", "gnomish wizard",
    "hobbit", "dwarf", "bat", "giant bat", "rock piercer", "floating eye",
    "pony", "horse", "rothe", "woodchuck", "goblin", "hobgoblin", "orc",
    "hill orc", "Mordor orc", "Uruk-hai", "orc-captain", "soldier ant",
    "fire ant", "giant beetle", "wolf", "warg", "rock mole", "gnome king",
    "dwarf lord", "dwarf king", "bugbear", "jaguar", "panther", "tiger",
    "mumak", "leocrotta", "dingo", "wild dog", "large dog", "little dog",
    "kitten", "housecat", "large cat", "raven", "monkey", "ape", "owlbear",
}


def mon_info(g):
    """(name, level, difficulty, speed, class_char, is_pet) for a monster
    glyph, or None."""
    pet = False
    if GLYPH_PET_OFF <= g < GLYPH_INVIS_OFF:
        idx, pet = g - GLYPH_PET_OFF, True
    elif GLYPH_MON_OFF <= g < GLYPH_PET_OFF:
        idx = g - GLYPH_MON_OFF
    elif GLYPH_DETECT_OFF <= g < GLYPH_BODY_OFF:
        idx = g - GLYPH_DETECT_OFF
    elif GLYPH_RIDDEN_OFF <= g < GLYPH_OBJ_OFF:
        idx = g - GLYPH_RIDDEN_OFF
    elif g == GLYPH_INVIS_OFF:
        return ("invisible monster", 5, 8, 12, "I", False)
    else:
        return None
    name, lvl, diff, speed, cls = _SPECIES.get(idx, ("unknown", 5, 8, 12, "?"))
    return (name, lvl, diff, speed, cls, pet)


def is_monster_glyph(g):
    return (GLYPH_MON_OFF <= g < GLYPH_INVIS_OFF) or g == GLYPH_INVIS_OFF or \
           (GLYPH_DETECT_OFF <= g < GLYPH_BODY_OFF) or \
           (GLYPH_RIDDEN_OFF <= g < GLYPH_OBJ_OFF)


def is_swallow_glyph(g):
    if GLYPH_SWALLOW_OFF is None:
        return False
    return GLYPH_SWALLOW_OFF <= g < (GLYPH_WARNING_OFF or MAX_GLYPH)


# ------------------------------------------------------------- observations

def message_of(obs):
    raw = obs["obs"]
    if "text_message" in raw:
        return raw["text_message"]
    txt = obs["text"]["long_term_context"]
    if txt.startswith("message:\n"):
        return txt[len("message:\n"):].split("\n\nlanguage observation:")[0].strip()
    return ""


def misc_of(obs):
    """(in_yn_function, in_getlin, xwaitingforspace) — served obs key."""
    m = obs["obs"].get("misc")
    if m is None:
        return (0, 0, 0)
    return (int(m[0]), int(m[1]), int(m[2]))


def inventory(obs):
    """[(letter, description, oclass)] from raw inv arrays."""
    raw = obs["obs"]
    letters = raw["inv_letters"]
    strs = raw["inv_strs"]
    ocls = raw.get("inv_oclasses")
    out = []
    for i, l in enumerate(letters):
        if l == 0:
            break
        desc = bytes(strs[i]).partition(b"\x00")[0].decode("latin-1")
        oc = int(ocls[i]) if ocls is not None else -1
        out.append((chr(l), desc, oc))
    return out


FOOD_CLASS = nh.FOOD_CLASS


# ------------------------------------------------------------------ mapping
class Monster:
    __slots__ = ("x", "y", "name", "level", "difficulty", "speed", "cls",
                 "pet", "glyph")

    def __init__(self, x, y, g):
        info = mon_info(g)
        self.x, self.y, self.glyph = x, y, g
        self.name, self.level, self.difficulty, self.speed, self.cls, self.pet = info

    @property
    def pos(self):
        return (self.x, self.y)


class LevelMap:
    """Persistent per-level map memory + current-frame entities.

    Dark-cell negative inference (engine lesson from the MiniHack arm):
    a cell still glyph-0 while we stand adjacent is provably stone/wall
    (night vision radius 1 always reveals adjacent floor). Suppressed
    while blind."""

    def __init__(self, key):
        self.key = key                 # (dungeon_number, level_number)
        self.terrain = np.zeros((ROWS, COLS), dtype=np.int8)
        self.explored = np.zeros((ROWS, COLS), dtype=bool)
        self.inferred_wall = np.zeros((ROWS, COLS), dtype=bool)
        self.stairs_down = set()
        self.stairs_up = set()
        self.holes = set()             # known holes/trapdoors (descend!)
        self.monsters = []
        self.boulders = set()
        self.items = set()
        self.no_attack = set()         # peaceful-monster cells (Really attack? -> n)
        self.boulder_blocked = set()   # (from_cell, dir) push attempts that failed
        self.search_counts = {}
        self.undiggable = False
        self.visits = 0

    def integrate(self, glyphs, tty, agent, blind=False):
        ax, ay = agent
        self.monsters = []
        self.boulders = set()
        self.items = set()
        for y in range(ROWS):
            row = glyphs[y]
            for x in range(COLS):
                g = int(row[x])
                if g == GLYPH_CMAP_OFF:      # stone / unseen: no info
                    continue
                self.explored[y][x] = True
                self.inferred_wall[y][x] = False
                if GLYPH_CMAP_OFF <= g < GLYPH_CMAP_OFF + nh.MAXPCHARS:
                    t = _CMAP_TO_TERRAIN[g - GLYPH_CMAP_OFF]
                    if t is not None:
                        self.terrain[y][x] = t
                        if t == STAIRS_DOWN:
                            self.stairs_down.add((x, y))
                        elif t == STAIRS_UP:
                            self.stairs_up.add((x, y))
                        elif t == HOLE_DOWN:
                            self.holes.add((x, y))
                elif g == BOULDER_GLYPH:
                    self.boulders.add((x, y))
                    if self.terrain[y][x] == UNKNOWN:
                        self.terrain[y][x] = FLOOR
                elif GLYPH_OBJ_OFF <= g < GLYPH_CMAP_OFF or \
                        GLYPH_BODY_OFF <= g < GLYPH_RIDDEN_OFF:
                    self.items.add((x, y))
                    if self.terrain[y][x] == UNKNOWN:
                        self.terrain[y][x] = FLOOR
                elif is_monster_glyph(g):
                    if (x, y) != (ax, ay):
                        self.monsters.append(Monster(x, y, g))
                    if self.terrain[y][x] == UNKNOWN:
                        self.terrain[y][x] = FLOOR
        # own cell is certainly walkable
        if self.terrain[ay][ax] == UNKNOWN or self.inferred_wall[ay][ax]:
            if self.terrain[ay][ax] == UNKNOWN:
                self.terrain[ay][ax] = FLOOR
            self.inferred_wall[ay][ax] = False
        self.explored[ay][ax] = True
        if not blind:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = ax + dx, ay + dy
                    if 0 <= nx < COLS and 0 <= ny < ROWS and \
                            not self.explored[ny][nx]:
                        self.terrain[ny][nx] = WALL
                        self.explored[ny][nx] = True
                        self.inferred_wall[ny][nx] = True
        # drop stale peaceful markers that no longer hold a monster
        if self.no_attack:
            mcells = {m.pos for m in self.monsters}
            self.no_attack &= mcells

    # -------------------------------------------------------------- queries
    def passable(self, x, y, doors_ok=True, boulders_ok=False,
                 bad_traps_ok=False):
        if not (0 <= x < COLS and 0 <= y < ROWS):
            return False
        t = self.terrain[y][x]
        if t in (LAVA, WATER, AIR, IRONBARS, TREE, WALL):
            return False
        if t == DOOR_CLOSED:
            return doors_ok
        if t == BAD_TRAP and not bad_traps_ok:
            return False
        if t not in PASSABLE:
            return False
        if not boulders_ok and (x, y) in self.boulders:
            return False
        return True

    def neighbors(self, x, y, diagonals=True, doors_ok=True, avoid=frozenset(),
                  bad_traps_ok=False):
        t_here = self.terrain[y][x]
        for name, (dx, dy) in DIRS.items():
            if not diagonals and (dx and dy):
                continue
            nx, ny = x + dx, y + dy
            if (nx, ny) in avoid:
                continue
            if not self.passable(nx, ny, doors_ok=doors_ok,
                                 bad_traps_ok=bad_traps_ok):
                continue
            if dx and dy:
                t_there = self.terrain[ny][nx]
                if t_here in (DOORWAY, DOOR_OPEN, DOOR_CLOSED) or \
                        t_there in (DOORWAY, DOOR_OPEN, DOOR_CLOSED):
                    continue
                if not self.passable(x + dx, y, doors_ok=False) and \
                        not self.passable(x, y + dy, doors_ok=False):
                    continue
            yield name, (nx, ny)

    def bfs(self, start, goals, diagonals=True, doors_ok=True,
            avoid=frozenset(), bad_traps_ok=False):
        from collections import deque
        goals = set(goals)
        if start in goals:
            return []
        q = deque([start])
        prev = {start: None}
        while q:
            cur = q.popleft()
            for name, nxt in self.neighbors(*cur, diagonals=diagonals,
                                            doors_ok=doors_ok, avoid=avoid,
                                            bad_traps_ok=bad_traps_ok):
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


class Atlas:
    """Multi-level belief state: one LevelMap per (dungeon_number, level)."""

    def __init__(self):
        self.levels = {}
        self.key = None
        self.agent = (0, 0)
        self.hp = self.hpmax = 1
        self.time = 0
        self.depth = 1
        self.dnum = 0
        self.dlevel = 1
        self.xplvl = 1
        self.hunger = 1
        self.condition = 0
        self.ac = 10
        self.message = ""
        self.swallowed = False
        self.level_changed = False

    def update(self, obs):
        raw = obs["obs"]
        bl = raw["blstats"]
        glyphs = raw["glyphs"]
        self.agent = (int(bl[nh.NLE_BL_X]), int(bl[nh.NLE_BL_Y]))
        self.hp = int(bl[nh.NLE_BL_HP])
        self.hpmax = int(bl[nh.NLE_BL_HPMAX])
        self.time = int(bl[nh.NLE_BL_TIME])
        self.depth = int(bl[nh.NLE_BL_DEPTH])
        self.dnum = int(bl[nh.NLE_BL_DNUM])
        self.dlevel = int(bl[nh.NLE_BL_DLEVEL])
        self.xplvl = int(bl[nh.NLE_BL_XP])
        self.hunger = int(bl[nh.NLE_BL_HUNGER])
        self.condition = int(bl[nh.NLE_BL_CONDITION])
        self.ac = int(bl[nh.NLE_BL_AC])
        self.message = message_of(obs)
        key = (self.dnum, self.dlevel)
        self.level_changed = (key != self.key)
        self.key = key
        if key not in self.levels:
            self.levels[key] = LevelMap(key)
            self.levels[key].visits += 1
        elif self.level_changed:
            self.levels[key].visits += 1
        # swallowed detection: adjacent swallow glyphs
        ax, ay = self.agent
        self.swallowed = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < COLS and 0 <= ny < ROWS and \
                        is_swallow_glyph(int(glyphs[ny][nx])):
                    self.swallowed = True
        blind = bool(self.condition & getattr(nh, "BL_MASK_BLIND", 0))
        if not self.swallowed:
            self.level.integrate(glyphs, raw.get("tty_chars"), self.agent,
                                 blind=blind)

    @property
    def level(self):
        return self.levels[self.key]

    @property
    def blind(self):
        return bool(self.condition & getattr(nh, "BL_MASK_BLIND", 0))
