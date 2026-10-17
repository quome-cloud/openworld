"""Quest-Easy / Quest-Medium agent.

Task-scoped model (from dat/quest_easy.des, dat/quest_medium.des + probes):
  - Both start the agent ON a 2-item stack: wand of cold (bottom) + frost
    horn (top), in a lit room; a 1-wide lava column separates the goal room
    (stairs down) from the rest of the level. Easy: lava is adjacent to the
    start room. Medium: a corridor + a large room with 6 giant rats precede
    the lava.
  - Menus are auto-closed by the NLE stack (xwaitforspace auto-SPACE), so
    the 2-item pickup menu is UNUSABLE. Single-item pickup bypasses menus.
    => acquisition: step aside, KICK the horn off the stack (away from the
    lava), then ',' picks up the wand alone.
  - Prompt protocol: getobj/direction prompts stay open (skill envs have
    allow_all_yn_questions=True; 'direction?' is whitelisted). Inventory
    letters are answered by actions whose raw keypress IS that letter
    (south='j', eat='e', ...). The zap prompt itself reveals the wand's
    letter: 'What do you want to zap? [j or ?*]'.
  - Zapping cold east at the lava column: 'The lava cools and solidifies.'
    -> the cell becomes walkable floor; the ray also kills rats in line.
  - Random role => random inventory size s; wand letter = letter(s) or
    letter(s+1) (we control pickup order). If neither is a reachable
    keypress the episode is mechanically unsolvable for us (rare; ~1 role).
  - Prayer (confirm prompt answered with 'northwest'='y') as HP emergency.
"""

import mh_common as C
from agent_explore import ExploreAgent

# action name -> raw key it sends (used to answer inventory-letter prompts)
KEY_OF_ACTION = {
    "north": "k", "east": "l", "south": "j", "west": "h",
    "northeast": "u", "southeast": "n", "southwest": "b", "northwest": "y",
    "far north": "K", "far east": "L", "far south": "J", "far west": "H",
    "far northeast": "U", "far southeast": "N", "far southwest": "B",
    "far northwest": "Y",
    "apply": "a", "close": "c", "eat": "e", "open": "o", "quaff": "q",
    "search": "s", "zap": "z", "puton": "P",
}
ACTION_OF_KEY = {v: k for k, v in KEY_OF_ACTION.items()}


class QuestAgent(ExploreAgent):
    def __init__(self, **kw):
        kw.setdefault("frontier_mass_w", 0.75)
        super().__init__(**kw)
        self.phase = "init"
        self.item_cell = None
        self.horn_cell = None
        self.wand_letter = None
        self.zapped = False
        self.prayed = False
        self.script = []
        self.expect = None
        self.pickups = 0
        self.acquire_gaveup = False
        # rat-pack war state (Quest-Medium)
        self.pack_mode = False
        self.kills = 0
        self.quiet = 0
        self.combat_zaps = 0
        self.osc = False
        self.baseline_letters = None      # letters present before any pickup
        self.bad_tools = set()            # letters that zapped/applied to no effect
        self.pending_tool = None          # letter being zapped/applied right now

    # --------------------------------------------------------------- helpers
    def _reachable_letters(self):
        return {KEY_OF_ACTION[a] for a in self._actions(None) if a in KEY_OF_ACTION}

    def _inv(self, obs):
        return C.inventory(obs)

    def _first_free_letters(self, obs, n=2):
        used = {l for l, _ in self._inv(obs)}
        out = []
        for o in range(26):
            ch = chr(ord("a") + o)
            if ch not in used:
                out.append(ch)
                if len(out) >= n:
                    break
        return out

    def _wand_in_inv(self, obs):
        """The quest wand of cold: a wand at a letter we did NOT start with
        (some roles start with their own -- useless -- wand), and not one
        we have already proven dead ('Nothing happens.')."""
        for l, d in self._inv(obs):
            if "wand" in d.lower() and l not in self.baseline_letters and \
               l not in self.bad_tools:
                return l
        return None

    def _horn_in_inv(self, obs):
        for l, d in self._inv(obs):
            if "horn" in d.lower() and l not in self.baseline_letters and \
               l not in self.bad_tools:
                return l
        return None

    # ----------------------------------------------------------------- main
    def _decide(self, obs):
        L = self.level
        msg = L.message
        if self.item_cell is None:
            self.item_cell = L.agent          # we start on the item stack
        if self.baseline_letters is None:
            self.baseline_letters = {l for l, _ in self._inv(obs)}

        # zap-effect verification: a dead or wrong wand answers the zap with
        # 'Nothing happens.' -- blacklist it and fall back to other tools
        if "Nothing happens" in msg and self.pending_tool is not None:
            self.bad_tools.add(self.pending_tool)
            self.log(f"  CROSS: tool '{self.pending_tool}' is a dud; blacklisting")
            self.pending_tool = None

        # scripted step pending?
        if self.script:
            return self.script.pop(0)

        low = msg.lower()
        self.kills += low.count("you kill") + low.count("you destroy")

        hostiles = [(mx, my, ch) for (mx, my, ch, pet) in L.monsters if not pet]
        if len(hostiles) >= 3:
            self.pack_mode = True

        # emergency prayer -- only when disengaged (praying while surrounded
        # just donates free attacks)
        adj_h = [h for h in hostiles
                 if max(abs(h[0] - L.agent[0]), abs(h[1] - L.agent[1])) == 1]
        if L.hp <= max(3, L.hpmax // 5) and not self.prayed and not adj_h and \
           "pray" in self._actions(obs):
            self.prayed = True
            self.script = ["northwest"]       # 'y' confirm
            return "pray"

        wand = self._wand_in_inv(obs)
        horn = self._horn_in_inv(obs)
        # a tool at an unreachable letter is unusable: never initiate a
        # zap/apply we cannot answer (else: apply -> 'Never mind.' loop)
        reach = self._reachable_letters()
        if wand is not None and wand not in reach:
            wand = None
        if horn is not None and horn not in reach:
            horn = None

        # ---------------- acquisition phase (until the WAND is in inventory)
        if wand is None and not self.acquire_gaveup and not self.zapped:
            if self.pickups >= 2:
                self.acquire_gaveup = True     # picked twice, still no wand
            else:
                return self._acquire(obs, L)

        # ---------------- rat-pack war (Quest-Medium): hold a chokepoint
        if self.pack_mode and self.kills < 6 and self.quiet < 22 and \
           not self.zapped:
            act = self._pack_war(obs, L, hostiles, adj_h, wand)
            if act is not None:
                return act

        # ---------------- lava crossing phase
        if not self.zapped:
            act = self._cross(obs, L, wand, horn)
            if act is not None:
                return act

        # ---------------- otherwise: explore/fight/descend
        return super()._decide(obs)

    # ------------------------------------------------------------ acquisition
    def _do_pickup(self, obs):
        """Issue a pickup and remember inventory size for verification."""
        self._pending_pickup = len(self._inv(obs))
        return "pickup"

    def _acquire(self, obs, L):
        ax, ay = L.agent
        ix, iy = self.item_cell

        # verify the previous pickup actually acquired something. A corpse
        # (e.g. a jackal killed on the stack) re-creates the 2-item menu
        # deadlock -> menu auto-closes -> silent no-op. Remedy: re-kick the
        # stack to knock the corpse off, then pick up again.
        # fight before manipulating items: acquisition steps with a live
        # hostile adjacent just donate free bites
        atk = self._combat(L)
        if atk is not None:
            return atk

        pend = getattr(self, "_pending_pickup", None)
        if pend is not None:
            self._pending_pickup = None
            if len(self._inv(obs)) > pend:
                self.pickups += 1
            else:
                self._failed_pickups = getattr(self, "_failed_pickups", 0) + 1
                self.log(f"  ACQUIRE: pickup was a no-op (stack menu-blocked,"
                         f" likely a corpse); attempt {self._failed_pickups}")
                if self._failed_pickups <= 3:
                    # eat the corpse off the stack (yn prompt: 'northwest'='y')
                    self.script = ["northwest"]
                    return "eat"
                self.acquire_gaveup = True

        # figure out pickup order
        free = self._first_free_letters(obs, 2)
        reach = self._reachable_letters()
        plan_two = free and free[0] not in reach and len(free) > 1 and \
            free[1] in reach
        if free and free[0] not in reach and not plan_two:
            # neither letter reachable: give up on tools (log + explore)
            self.log(f"  ACQUIRE: free letters {free} unreachable; giving up tools")
            self.acquire_gaveup = True
            return super()._decide(obs)

        if (ax, ay) == (ix, iy) and not getattr(self, "_kicked", False):
            # step east (away from nothing in particular; lava is always
            # east, so kicking west keeps the horn safe)
            for name in ("east", "southeast", "northeast", "south", "north"):
                dx, dy = C.DIRS[name]
                if L.passable(ax + dx, ay + dy):
                    self._kick_back = C.DIR_OF[(-dx, -dy)]
                    return name
        # adjacent to item cell: kick the horn off the stack
        d = (ix - ax, iy - ay)
        if d in C.DIR_OF and not getattr(self, "_kicked", False):
            self._kicked = True
            self._kick_dir = C.DIR_OF[d]
            self.script = [C.DIR_OF[d]]
            return "kick"
        # after kick: pick up the wand (walk onto stack, pickup);
        # in the two-step plan grab the (kicked-away) horn first so the
        # wand lands on the second free letter.
        if self.pickups == 0:
            # remember where the kicked horn landed -- ONCE, and only on the
            # kick ray (monsters may drop/throw other items nearby, and an
            # item under the agent is invisible: glyphs show cell tops only)
            if getattr(self, "horn_cell", None) is None and \
               getattr(self, "_kick_dir", None):
                kdx, kdy = C.DIRS[self._kick_dir]
                for c in sorted(L.items,
                                key=lambda c: abs(c[0] - ix) + abs(c[1] - iy)):
                    vx, vy = c[0] - ix, c[1] - iy
                    on_ray = ((kdx == 0 and vx == 0) or vx * kdx > 0) and \
                             ((kdy == 0 and vy == 0) or vy * kdy > 0) and \
                             (vx or vy)
                    if on_ray:
                        self.horn_cell = c
                        break
            if plan_two:
                hc = getattr(self, "horn_cell", None)
                if hc is not None:
                    if (ax, ay) == hc:
                        return self._do_pickup(obs)
                    path = L.bfs(L.agent, [hc])
                    if path:
                        return path[0]
            if (ax, ay) != (ix, iy):
                path = L.bfs(L.agent, [(ix, iy)])
                if path:
                    return path[0]
            return self._do_pickup(obs)
        if self.pickups == 1:
            # second pickup: the wand is still on the original stack cell
            if (ax, ay) == (ix, iy):
                return self._do_pickup(obs)
            path = L.bfs(L.agent, [(ix, iy)])
            if path:
                return path[0]
        self.acquire_gaveup = True
        return super()._decide(obs)

    # ------------------------------------------------------------- rat war
    def _pack_war(self, obs, L, hostiles, adj_h, wand):
        """Hold a 1-wide corridor cell and let the rats queue up."""
        msg = L.message
        ax, ay = L.agent
        # opportunistic line zap: >=2 hostiles straight east in our row
        if wand is not None and self.combat_zaps < 2 and \
           "zap" in self._actions(obs) and not adj_h:
            in_row = [h for h in hostiles if h[1] == ay and 0 < h[0] - ax <= 7]
            clear = all(L.terrain[ay][x] not in (C.WALL, C.IRONBARS)
                        for h in in_row for x in range(ax + 1, h[0]))
            if len(in_row) >= 2 and clear:
                self.combat_zaps += 1
                self._zap_dir = "east"
                return "zap"
        if "What do you want to zap" in msg or "In what direction" in msg:
            return None                        # let _cross's prompt logic run

        self.quiet = 0 if hostiles else self.quiet + 1

        # attack adjacent (explorer combat layer would too, but with its own
        # retreat rule; here: attack if <=1 adjacent, else fall back west)
        hold = self._hold_cell(L)
        if adj_h:
            if len(adj_h) >= 2 and hold is not None and (ax, ay) != hold:
                path = L.bfs(L.agent, [hold])
                if path:
                    return path[0]
            for (mx, my, ch) in adj_h:
                if ch == "e":
                    continue
                d = (mx - ax, my - ay)
                if d in C.DIR_OF:
                    return C.DIR_OF[d]
            return None
        # no contact: sit at the chokepoint, oscillating to pass world time
        if hold is None:
            return None                        # no corridor known: normal flow
        if (ax, ay) != hold:
            path = L.bfs(L.agent, [hold],
                         avoid={(m[0], m[1]) for m in L.monsters})
            if path:
                return path[0]
            return None
        self.osc = not self.osc
        if self.osc:
            for name in ("west", "north", "south"):
                dx, dy = C.DIRS[name]
                if L.passable(ax + dx, ay + dy):
                    return name
        return "east" if L.passable(ax + 1, ay) else None

    @staticmethod
    def _hold_cell(L):
        """Easternmost corridor cell whose east neighbor is corridor/floor:
        one step back from the room mouth (degree-1 exposure)."""
        import numpy as np
        ys, xs = np.where(L.terrain == C.CORRIDOR)
        cells = sorted(zip(xs.tolist(), ys.tolist()))
        if not cells:
            return None
        mouth = None
        for (x, y) in cells:
            if L.terrain[y][x + 1] in (C.FLOOR, C.DOORWAY):
                if mouth is None or x > mouth[0]:
                    mouth = (x, y)
        if mouth is None:
            return cells[-1]
        # one west of the mouth if that's corridor, else the mouth itself
        wx, wy = mouth[0] - 1, mouth[1]
        if L.terrain[wy][wx] == C.CORRIDOR:
            return (wx, wy)
        return mouth

    # ------------------------------------------------------------- crossing
    def _cross(self, obs, L, wand, horn):
        msg = L.message
        # answer an open zap/apply prompt
        if "What do you want to zap" in msg or \
           "What do you want to use or apply" in msg:
            # the prompt lists the letter(s): [j or ?*]
            import re
            m = re.search(r"\[([a-zA-Z]+)( or \?\*)?\]", msg)
            letters = list(m.group(1)) if m else []
            # ONLY ever select our own tool's letter -- selecting an
            # arbitrary applicable item (e.g. a knight's lance) drops the
            # game into cursor prompts we cannot escape.
            pref = [wand, horn] if "zap" in msg else [horn]
            for l in pref:
                if l and (not letters or l in letters) and \
                   l in ACTION_OF_KEY and ACTION_OF_KEY[l] in self._actions(obs):
                    self.pending_tool = l
                    return ACTION_OF_KEY[l]
            return "more"                      # bail out of the prompt
        if "In what direction" in msg:
            return self._zap_dir or "east"
        if "lava cools" in msg:
            self.zapped = True
            return None

        lava = L.find_terrain(C.LAVA)
        if not lava:
            # no lava seen yet (Medium: still en route) or already frozen
            if self.zapped:
                return None
            # explore toward the east until lava visible
            return None if self._saw_lava_gone(L) else None
        # find zap positions: same row as a lava cell, west of it, clear line
        cands = []
        for (lx, ly) in lava:
            for dist in range(1, 7):
                x = lx - dist
                if not (0 <= x < C.COLS):
                    break
                t = L.terrain[ly][x]
                if t in (C.WALL, C.LAVA, C.IRONBARS, C.UNKNOWN):
                    break
                cands.append((x, ly))
        if not cands:
            return None                        # let the explorer get closer
        path = L.bfs(L.agent, cands,
                     avoid={(m[0], m[1]) for m in L.monsters})
        if path is None:
            return None                        # blocked (rats): fight first
        if path:
            return self._step_path(path, L)
        # in position: zap (or apply horn as fallback)
        self._zap_dir = "east"
        if wand is not None and "zap" in self._actions(obs):
            return "zap"
        if horn is not None and "apply" in self._actions(obs):
            return "apply"
        self.acquire_gaveup = True
        return None

    @staticmethod
    def _saw_lava_gone(L):
        return False
