"""Symbolic world model for BALROG BabyAI (BabyAI-MixedTrainLocal-v0).

Synthesized by Fable 5 from reading the environment source
(BartekCupial/Minigrid fork used by BALROG):
  - minigrid/minigrid_env.py           step(), gen_obs(), gen_obs_grid(), _reward()
  - minigrid/core/grid.py              slice(), rotate_left(), process_vis(), encode()
  - minigrid/core/world_object.py      Door/Key/Ball/Box semantics
  - minigrid/envs/babyai/core/verifier.py       GoTo/Pickup/Open/PutNext/Before/After
  - minigrid/envs/babyai/core/roomgrid_level.py step() plumbing (drop -> update_objs_poss -> verify)

Pure code, LLM-free at runtime. Exactness contract: lock-stepped fidelity
sweep vs the real env must show 0 disagreements (see validate_model.py).

Coordinates: (x, y), x = column (grid width axis), y = row.
Directions: 0=east(+x), 1=south(+y), 2=west(-x), 3=north(-y)  [DIR_TO_VEC]
Actions:    0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle
            (BALROG's BabyAITextCleanLangWrapper maps its 6 text actions to
             exactly these ints, in this order.)
"""

from __future__ import annotations

DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]

LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE = range(6)

# minigrid.core.constants
OBJECT_TO_IDX = {
    "unseen": 0, "empty": 1, "wall": 2, "floor": 3, "door": 4,
    "key": 5, "ball": 6, "box": 7, "goal": 8, "lava": 9, "agent": 10,
}
IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}


class ModelUnsupported(Exception):
    """Raised when the environment contains objects outside the model's scope."""


class Obj:
    """A world object. Identity (id) is preserved so instruction object-sets
    can track specific instances exactly like the env verifier does."""

    __slots__ = ("oid", "type", "color", "is_open", "is_locked")

    def __init__(self, oid, type_, color, is_open=False, is_locked=False):
        self.oid = oid
        self.type = type_
        self.color = color
        self.is_open = is_open
        self.is_locked = is_locked

    # --- semantics from world_object.py ---
    def can_overlap(self):
        return self.type == "door" and self.is_open

    def can_pickup(self):
        return self.type in ("key", "ball", "box")

    def see_behind(self):
        if self.type == "wall":
            return False
        if self.type == "door":
            return self.is_open
        return True

    def encode(self):
        if self.type == "door":
            state = 0 if self.is_open else (2 if self.is_locked else 1)
        else:
            state = 0
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def clone(self):
        return Obj(self.oid, self.type, self.color, self.is_open, self.is_locked)

    def __repr__(self):
        s = f"{self.color} {self.type}#{self.oid}"
        if self.type == "door":
            s += f"[{'open' if self.is_open else ('locked' if self.is_locked else 'closed')}]"
        return s


class SymState:
    """Full symbolic state of the env."""

    __slots__ = ("width", "height", "grid", "objs", "agent_pos", "agent_dir",
                 "carrying", "step_count", "max_steps")

    def __init__(self, width, height, max_steps):
        self.width = width
        self.height = height
        self.grid = {}            # (x, y) -> oid
        self.objs = {}            # oid -> Obj
        self.agent_pos = (0, 0)
        self.agent_dir = 0
        self.carrying = None      # oid or None
        self.step_count = 0
        self.max_steps = max_steps

    def obj_at(self, pos):
        oid = self.grid.get(pos)
        return self.objs[oid] if oid is not None else None

    def front_pos(self):
        dx, dy = DIR_TO_VEC[self.agent_dir]
        return (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

    def pos_of(self, oid):
        """Current grid position of object oid, or None if carried/off-grid."""
        for pos, o in self.grid.items():
            if o == oid:
                return pos
        return None

    def clone(self):
        s = SymState(self.width, self.height, self.max_steps)
        s.grid = dict(self.grid)
        s.objs = {k: v.clone() for k, v in self.objs.items()}
        s.agent_pos = self.agent_pos
        s.agent_dir = self.agent_dir
        s.carrying = self.carrying
        s.step_count = self.step_count
        return s

    def key(self):
        """Hashable full-state key (for search visited-sets)."""
        return (
            self.agent_pos, self.agent_dir, self.carrying,
            frozenset((pos, o, self.objs[o].is_open, self.objs[o].is_locked)
                      for pos, o in self.grid.items()
                      if self.objs[o].type != "wall"),
        )

    # ---------------- exact step semantics (minigrid_env.step) -------------
    def step(self, action):
        """Mutating step. Returns (moved_forward: bool). Reward/termination is
        the verifier's job (see EpisodeModel)."""
        self.step_count += 1
        fwd = self.front_pos()
        fwd_obj = self.obj_at(fwd)

        if action == LEFT:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == RIGHT:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == FORWARD:
            if fwd_obj is None or fwd_obj.can_overlap():
                self.agent_pos = fwd
            # goal / lava do not occur in MixedTrainLocal levels
            if fwd_obj is not None and fwd_obj.type in ("goal", "lava"):
                raise ModelUnsupported("goal/lava cell encountered")
        elif action == PICKUP:
            if fwd_obj is not None and fwd_obj.can_pickup() and self.carrying is None:
                self.carrying = fwd_obj.oid
                del self.grid[fwd]
        elif action == DROP:
            if fwd_obj is None and self.carrying is not None \
                    and 0 <= fwd[0] < self.width and 0 <= fwd[1] < self.height:
                # env: grid.set(fwd) succeeds only inside bounds; front cell of an
                # in-room agent is always inside bounds in these levels.
                self.grid[fwd] = self.carrying
                self.carrying = None
        elif action == TOGGLE:
            if fwd_obj is not None:
                if fwd_obj.type == "door":
                    if fwd_obj.is_locked:
                        car = self.objs.get(self.carrying) if self.carrying is not None else None
                        if car is not None and car.type == "key" and car.color == fwd_obj.color:
                            fwd_obj.is_locked = False
                            fwd_obj.is_open = True
                    else:
                        fwd_obj.is_open = not fwd_obj.is_open
                elif fwd_obj.type == "box":
                    # Box.toggle replaces the box by its contents; in this suite
                    # boxes are always empty -> cell becomes empty.
                    del self.grid[fwd]
        else:
            raise ValueError(f"unknown action {action}")


# ======================= verifier (exact mirror) ===========================

class SymDesc:
    """ObjDesc mirror: (type, color) with tracked object identity sets.
    loc descriptions are disabled in MixedTrainLocal (locations=False)."""

    def __init__(self, type_, color=None):
        self.type = type_
        self.color = color
        self.obj_set = []    # list of oids (fixed at reset)
        self.obj_poss = []   # list of positions (refreshed on drop actions)

    def matches(self, obj):
        if self.type is not None and obj.type != self.type:
            return False
        if self.color is not None and obj.color != self.color:
            return False
        return True

    def find_matching(self, state, use_location=True):
        """Mirror of ObjDesc.find_matching_objs. Iterates grid in (x, y) order.
        use_location=True: rebuild obj_set from scratch (reset time).
        use_location=False: keep obj_set, refresh obj_poss from grid."""
        if use_location:
            self.obj_set = []
        self.obj_poss = []
        for i in range(state.width):
            for j in range(state.height):
                oid = state.grid.get((i, j))
                if oid is None:
                    continue
                if not use_location and oid not in self.obj_set:
                    continue
                obj = state.objs[oid]
                if use_location and not self.matches(obj):
                    continue
                if use_location:
                    self.obj_set.append(oid)
                self.obj_poss.append((i, j))

    def clone(self):
        d = SymDesc(self.type, self.color)
        d.obj_set = list(self.obj_set)
        d.obj_poss = list(self.obj_poss)
        return d


def pos_next_to(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


class SymInstr:
    def reset_verifier(self, state):
        raise NotImplementedError

    def verify(self, state, action):
        raise NotImplementedError

    def update_objs_poss(self, state):
        for attr in ("desc", "desc_move", "desc_fixed"):
            if hasattr(self, attr):
                getattr(self, attr).find_matching(state, use_location=False)

    def clone(self):
        raise NotImplementedError


class SymGoTo(SymInstr):
    def __init__(self, desc):
        self.desc = desc

    def reset_verifier(self, state):
        self.desc.find_matching(state)

    def verify(self, state, action):
        fp = state.front_pos()
        for pos in self.desc.obj_poss:
            if pos == fp:
                return "success"
        return "continue"

    def clone(self):
        return SymGoTo(self.desc.clone())


class SymPickup(SymInstr):
    def __init__(self, desc):
        self.desc = desc
        self.preCarrying = None

    def reset_verifier(self, state):
        self.preCarrying = None
        self.desc.find_matching(state)

    def verify(self, state, action):
        preCarrying = self.preCarrying
        self.preCarrying = state.carrying
        if action != PICKUP:
            return "continue"
        for oid in self.desc.obj_set:
            if preCarrying is None and state.carrying == oid:
                return "success"
        self.preCarrying = state.carrying
        return "continue"

    def clone(self):
        c = SymPickup(self.desc.clone())
        c.preCarrying = self.preCarrying
        return c


class SymOpen(SymInstr):
    def __init__(self, desc):
        self.desc = desc

    def reset_verifier(self, state):
        self.desc.find_matching(state)

    def verify(self, state, action):
        if action != TOGGLE:
            return "continue"
        front = state.obj_at(state.front_pos())
        for oid in self.desc.obj_set:
            if front is not None and front.oid == oid and front.is_open:
                return "success"
        return "continue"

    def clone(self):
        return SymOpen(self.desc.clone())


class SymPutNext(SymInstr):
    def __init__(self, desc_move, desc_fixed):
        self.desc_move = desc_move
        self.desc_fixed = desc_fixed
        self.preCarrying = None

    def reset_verifier(self, state):
        self.preCarrying = None
        self.desc_move.find_matching(state)
        self.desc_fixed.find_matching(state)

    def verify(self, state, action):
        preCarrying = self.preCarrying
        self.preCarrying = state.carrying
        if action != DROP:
            return "continue"
        for oid in self.desc_move.obj_set:
            if preCarrying != oid:
                continue
            pos_a = state.pos_of(oid)
            if pos_a is None:
                continue
            for pos_b in self.desc_fixed.obj_poss:
                if pos_next_to(pos_a, pos_b):
                    return "success"
        return "continue"

    def clone(self):
        c = SymPutNext(self.desc_move.clone(), self.desc_fixed.clone())
        c.preCarrying = self.preCarrying
        return c


class SymBefore(SymInstr):
    """instr_a, then instr_b."""

    def __init__(self, instr_a, instr_b):
        self.instr_a = instr_a
        self.instr_b = instr_b
        self.a_done = False
        self.b_done = False

    def reset_verifier(self, state):
        self.instr_a.reset_verifier(state)
        self.instr_b.reset_verifier(state)
        self.a_done = False
        self.b_done = False

    def update_objs_poss(self, state):
        self.instr_a.update_objs_poss(state)
        self.instr_b.update_objs_poss(state)

    def verify(self, state, action):
        if self.a_done == "success":
            self.b_done = self.instr_b.verify(state, action)
            if self.b_done == "success":
                return "success"
        else:
            self.a_done = self.instr_a.verify(state, action)
            if self.a_done == "success":
                return self.verify(state, action)
        return "continue"

    def clone(self):
        c = SymBefore(self.instr_a.clone(), self.instr_b.clone())
        c.a_done = self.a_done
        c.b_done = self.b_done
        return c


class SymAfter(SymInstr):
    """instr_a after you instr_b (b first)."""

    def __init__(self, instr_a, instr_b):
        self.instr_a = instr_a
        self.instr_b = instr_b
        self.a_done = False
        self.b_done = False

    def reset_verifier(self, state):
        self.instr_a.reset_verifier(state)
        self.instr_b.reset_verifier(state)
        self.a_done = False
        self.b_done = False

    def update_objs_poss(self, state):
        self.instr_a.update_objs_poss(state)
        self.instr_b.update_objs_poss(state)

    def verify(self, state, action):
        if self.b_done == "success":
            self.a_done = self.instr_a.verify(state, action)
            if self.a_done == "success":
                return "success"
        else:
            self.b_done = self.instr_b.verify(state, action)
            if self.b_done == "success":
                return self.verify(state, action)
        return "continue"

    def clone(self):
        c = SymAfter(self.instr_a.clone(), self.instr_b.clone())
        c.a_done = self.a_done
        c.b_done = self.b_done
        return c


# ===================== episode model (env + verifier) ======================

class EpisodeModel:
    """SymState + verifier, mirroring RoomGridLevel.step exactly."""

    def __init__(self, state: SymState, instr: SymInstr, reset_verifier=True):
        self.state = state
        self.instr = instr
        if reset_verifier:
            self.instr.reset_verifier(state)

    def step(self, action):
        """Returns (reward, terminated, truncated). Mirrors:
        MiniGridEnv.step -> truncation check, then RoomGridLevel.step:
        drop -> update_objs_poss, verify -> success => terminated + _reward()."""
        s = self.state
        s.step(action)
        reward = 0.0
        terminated = False
        truncated = s.step_count >= s.max_steps
        if action == DROP:
            self.instr.update_objs_poss(s)
        status = self.instr.verify(s, action)
        if status == "success":
            terminated = True
            reward = 1 - 0.9 * (s.step_count / s.max_steps)
        return reward, terminated, truncated

    def clone(self):
        return EpisodeModel(self.state.clone(), self.instr.clone(), reset_verifier=False)


# ===================== privileged extraction from env ======================

def extract_state(env):
    """Read the full symbolic state from a (reset) env. Privileged channel."""
    u = env.unwrapped
    st = SymState(u.grid.width, u.grid.height, u.max_steps)
    id2oid = {}
    next_oid = [0]

    def register(cell):
        oid = next_oid[0]
        next_oid[0] += 1
        id2oid[id(cell)] = oid
        if cell.type == "door":
            st.objs[oid] = Obj(oid, "door", cell.color, cell.is_open, cell.is_locked)
        elif cell.type in ("wall", "key", "ball", "box"):
            if cell.type == "box" and cell.contains is not None:
                raise ModelUnsupported("box with contents")
            st.objs[oid] = Obj(oid, cell.type, cell.color)
        else:
            raise ModelUnsupported(f"object type {cell.type}")
        return oid

    for i in range(u.grid.width):
        for j in range(u.grid.height):
            cell = u.grid.get(i, j)
            if cell is None:
                continue
            st.grid[(i, j)] = register(cell)

    st.agent_pos = tuple(u.agent_pos)
    st.agent_dir = int(u.agent_dir)
    if u.carrying is not None:
        st.carrying = register(u.carrying)
    st.step_count = int(u.step_count)
    return st, id2oid


def extract_instr(env, id2oid):
    """Convert env.unwrapped.instrs into a Sym instruction tree, preserving
    the already-resolved obj_set identities (privileged channel)."""
    from minigrid.envs.babyai.core import verifier as V

    def conv_desc(d):
        sd = SymDesc(d.type, d.color)
        if getattr(d, "loc", None) is not None:
            raise ModelUnsupported("loc-based description")
        sd.obj_set = [id2oid[id(o)] for o in d.obj_set]
        sd.obj_poss = [tuple(p) for p in d.obj_poss]
        return sd

    def conv(instr):
        if isinstance(instr, V.GoToInstr):
            return SymGoTo(conv_desc(instr.desc))
        if isinstance(instr, V.PickupInstr):
            if instr.strict:
                raise ModelUnsupported("strict pickup")
            si = SymPickup(conv_desc(instr.desc))
            si.preCarrying = None if instr.preCarrying is None else id2oid[id(instr.preCarrying)]
            return si
        if isinstance(instr, V.OpenInstr):
            if instr.strict:
                raise ModelUnsupported("strict open")
            return SymOpen(conv_desc(instr.desc))
        if isinstance(instr, V.PutNextInstr):
            if instr.strict:
                raise ModelUnsupported("strict putnext")
            si = SymPutNext(conv_desc(instr.desc_move), conv_desc(instr.desc_fixed))
            si.preCarrying = None if instr.preCarrying is None else id2oid[id(instr.preCarrying)]
            return si
        if isinstance(instr, V.BeforeInstr):
            si = SymBefore(conv(instr.instr_a), conv(instr.instr_b))
            si.a_done, si.b_done = instr.a_done, instr.b_done
            return si
        if isinstance(instr, V.AfterInstr):
            si = SymAfter(conv(instr.instr_a), conv(instr.instr_b))
            si.a_done, si.b_done = instr.a_done, instr.b_done
            return si
        raise ModelUnsupported(f"instruction {type(instr).__name__}")

    return conv(env.unwrapped.instrs)


# ======================= observation model (for clean arm) =================

def render_view(state: SymState):
    """Replicate gen_obs()['image'] exactly: 7x7 egocentric slice, rotated,
    occlusion via process_vis, carried object at agent view cell.
    Returns a 7x7 array-of-tuples: view[vx][vy] = (type_idx, color_idx, st)."""
    n = 7
    ax, ay = state.agent_pos
    d = state.agent_dir
    if d == 0:
        topX, topY = ax, ay - n // 2
    elif d == 1:
        topX, topY = ax - n // 2, ay
    elif d == 2:
        topX, topY = ax - n + 1, ay - n // 2
    else:
        topX, topY = ax - n // 2, ay - n + 1

    # slice (out-of-bounds -> wall)
    WALL = Obj(-1, "wall", "grey")
    cells = [[None] * n for _ in range(n)]  # cells[i][j]
    for j in range(n):
        for i in range(n):
            x, y = topX + i, topY + j
            if 0 <= x < state.width and 0 <= y < state.height:
                cells[i][j] = state.obj_at((x, y))
            else:
                cells[i][j] = WALL

    # rotate_left (agent_dir + 1) times
    for _ in range(d + 1):
        newc = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                newc[j][n - 1 - i] = cells[i][j]
        cells = newc

    # process_vis with agent at (n//2, n-1)
    mask = [[False] * n for _ in range(n)]
    mask[n // 2][n - 1] = True
    for j in reversed(range(n)):
        for i in range(n - 1):
            if not mask[i][j]:
                continue
            cell = cells[i][j]
            if cell is not None and not cell.see_behind():
                continue
            mask[i + 1][j] = True
            if j > 0:
                mask[i + 1][j - 1] = True
                mask[i][j - 1] = True
        for i in reversed(range(1, n)):
            if not mask[i][j]:
                continue
            cell = cells[i][j]
            if cell is not None and not cell.see_behind():
                continue
            mask[i - 1][j] = True
            if j > 0:
                mask[i - 1][j - 1] = True
                mask[i][j - 1] = True
    for j in range(n):
        for i in range(n):
            if not mask[i][j]:
                cells[i][j] = None

    # carried object shown at agent cell; else None
    if state.carrying is not None:
        cells[n // 2][n - 1] = state.objs[state.carrying]
    else:
        cells[n // 2][n - 1] = None

    # encode
    out = [[(0, 0, 0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if mask[i][j]:
                c = cells[i][j]
                out[i][j] = (1, 0, 0) if c is None else c.encode()
    return out


def full_grid_encode(state: SymState):
    """Encode the full grid like grid.encode() (no agent)."""
    out = []
    for i in range(state.width):
        row = []
        for j in range(state.height):
            c = state.obj_at((i, j))
            row.append((1, 0, 0) if c is None else c.encode())
        out.append(row)
    return out
