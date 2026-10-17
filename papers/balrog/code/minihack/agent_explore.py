"""Closed-loop explorer agent for MazeWalk / Corridor / CorridorBattle,
also the navigation+combat substrate for Quest.

Model assumptions (task-scoped, synthesized from env sources + probes):
  - Goal for all these tasks: stand on the staircase-down cell (internal
    stairs_down flag ends episode; no '>' keypress needed).
  - Unlit areas: cells revealed at radius 1 + corridor line-of-sight; map
    memory is monotone (LevelState.explored).
  - Doors: no diagonal through door cells; closed doors need open(+dir);
    locked doors need kick(+dir) repeatedly. Bumping a closed door may
    auto-open it in some configs -- we use explicit open.
  - Combat: moving into a hostile attacks it. Never attack pets. Avoid
    melee on 'e' (floating eye: paralysis), 'F' (lichen), 'a' (acid blob).
  - Every action costs one env step (100 cap), so futile bumps are pruned
    via a suspect-wall memory when a move doesn't change position/time.
"""

import mh_common as C

# Species never to melee: floating eye ('e') paralyzes on melee = death.
# 'F' (lichen/molds) is NOT here: they are immobile and often block 1-wide
# passages under the 100-step cap; passive mold damage is a price worth
# paying (discovered via Quest-Easy seed 1601: a lichen plugging the only
# corridor turned into an 84-step search loop).
AVOID_MELEE = {"e"}


class ExploreAgent:
    def __init__(self, log=print, prefer_dir=None, combat=True, maze_parity=False,
                 frontier_mass_w=0.75, frontier_mass_r=3, memory=None):
        self.level = C.LevelState()
        self.log = log
        self.prefer_dir = prefer_dir      # e.g. "east" bias for CorridorBattle
        self.combat = combat
        self.maze_parity = maze_parity    # MAZEWALK lattice: one parity class
                                          # (x%2,y%2) is always wall/stone
        self.frontier_mass_w = frontier_mass_w
        self.frontier_mass_r = frontier_mass_r
        self.memory = memory              # TaskMemory or None (condition A)
        self._mem_fired_beeline = False
        self.queue = []                   # pending multi-action sequence
        self.last_pos = None
        self.last_time = -1
        self.last_action = None
        self.suspect_walls = set()
        self.search_counts = {}
        self.kick_target = None
        # frozen-time (movement energy) tracking for the win-step protocol:
        # BALROG progression requires final reward >= 1.0; a zero-game-time
        # move yields 0.99 (penalty) even on TASK_SUCCESSFUL. Fast characters
        # (speed>12) make ~1/5-1/2 of moves in zero time. Empirically + by the
        # energy model, a frozen move is never followed by another frozen
        # move, so the move right after a frozen one is guaranteed safe.
        self.saw_frozen = False
        self.last_move_frozen = False
        self.steps_taken = 0

    # -------------------------------------------------------------- main
    def act(self, obs):
        L = self.level
        L.update(obs)
        msg = L.message

        # message-driven bookkeeping
        if "This door is locked" in msg or "WHAMM" in msg:
            # last open failed; queue kicks
            if self.kick_target:
                self.queue = ["kick", self.kick_target[0]]
        if "You succeed in unlocking" in msg or "The door opens" in msg or \
           "As you kick the door, it crashes open" in msg:
            self.queue = []
            self.kick_target = None

        # frozen-move tracking (position changed but game time did not)
        if self.last_action in C.DIRS and self.last_pos is not None:
            if L.agent != self.last_pos:
                self.last_move_frozen = (L.time == self.last_time)
                if self.last_move_frozen:
                    self.saw_frozen = True
            self.steps_taken += 1

        # stuck detection: move action that changed nothing -> suspect wall
        if self.last_action in C.DIRS and self.last_pos == L.agent and \
           L.time == self.last_time:
            dx, dy = C.DIRS[self.last_action]
            sx, sy = L.agent[0] + dx, L.agent[1] + dy
            tgt = (sx, sy)
            if not any(m[0] == sx and m[1] == sy for m in L.monsters):
                self.suspect_walls.add(tgt)

        self.last_pos, self.last_time = L.agent, L.time

        if self.maze_parity:
            self._apply_maze_parity(L)

        a = self._decide(obs)
        self.last_action = a
        return a

    def _decide(self, obs):
        L = self.level
        if self.queue:
            return self.queue.pop(0)

        # 1. combat layer
        if self.combat:
            act = self._combat(L)
            if act:
                return act

        avoid = set(self.suspect_walls)
        # cells adjacent to avoid-melee monsters are dangerous to end on
        for (mx, my, ch, pet) in L.monsters:
            if not pet and ch in AVOID_MELEE:
                avoid.add((mx, my))
        # monsters as obstacles (non-adjacent handling in combat layer)
        mcells = {(m[0], m[1]) for m in L.monsters}

        # 2. stairs known? go there
        stairs = L.find_terrain(C.STAIRS_DOWN)
        if stairs:
            path = L.bfs(L.agent, stairs, avoid=avoid | (mcells - set(stairs)))
            if path is None:
                path = L.bfs(L.agent, stairs, avoid=avoid)  # fight through
            if path:
                if len(path) == 1:
                    dance = self._win_step_guard(L)
                    if dance is not None:
                        return dance
                return self._step_path(path, L)
            if path == []:
                return "wait"  # standing on stairs: episode should have ended

        # 3. frontier exploration (with target persistence: nearest-frontier
        # alone oscillates between maze branches and wastes the step budget)
        frontier = L.frontier_cells()
        if frontier:
            fset = set(frontier)
            tgt = getattr(self, "explore_target", None)
            path = None
            if tgt in fset:
                path = L.bfs(L.agent, [tgt], avoid=avoid | mcells)
            if not path:                      # None or []
                tgt2 = None
                if self.frontier_mass_w > 0:
                    tgt2 = self._pick_frontier(L, fset, avoid | mcells)
                if tgt2 is not None:
                    path = L.bfs(L.agent, [tgt2], avoid=avoid | mcells)
                    self.explore_target = tgt2
                if not path:
                    path = L.bfs(L.agent, frontier, avoid=avoid | mcells)
                    if path:
                        x, y = L.agent
                        for stp in path:
                            dx, dy = C.DIRS[stp]
                            x, y = x + dx, y + dy
                        self.explore_target = (x, y)
            if path is None:
                path = L.bfs(L.agent, frontier, avoid=avoid)
            if path is not None and path != []:
                return self._step_path(path, L)
            if path == []:
                # we are ON a frontier cell; take any passable step toward
                # unknown: prefer preferred direction
                cand = []
                for name, (dx, dy) in C.DIRS.items():
                    nx, ny = L.agent[0] + dx, L.agent[1] + dy
                    if not (0 <= nx < C.COLS and 0 <= ny < C.ROWS):
                        continue
                    if (nx, ny) in self.suspect_walls or (nx, ny) in mcells:
                        continue
                    if not L.explored[ny][nx]:
                        cand.append(name)
                if cand:
                    if self.prefer_dir in cand:
                        return self.prefer_dir
                    # cardinal first (diagonal into unknown risks door rules)
                    for c in cand:
                        if c in C.CARDINALS:
                            return c
                    return cand[0]

        # 4. closed doors we haven't opened yet
        doors = L.find_terrain(C.DOOR_CLOSED)
        if doors:
            # walk adjacent (cardinal) to nearest closed door, then open
            targets = set()
            for (dx_, dy_) in doors:
                for name in C.CARDINALS:
                    ddx, ddy = C.DIRS[name]
                    ax, ay = dx_ - ddx, dy_ - ddy
                    if L.passable(ax, ay, doors_ok=False):
                        targets.add((ax, ay))
            path = L.bfs(L.agent, targets, avoid=self.suspect_walls | mcells,
                         doors_ok=False)
            if path == []:
                # adjacent: open toward the door
                for (door_c) in doors:
                    ddx, ddy = door_c[0] - L.agent[0], door_c[1] - L.agent[1]
                    if (ddx, ddy) in C.DIR_OF and (ddx == 0 or ddy == 0):
                        dname = C.DIR_OF[(ddx, ddy)]
                        self.kick_target = (dname, door_c)
                        self.queue = [dname]
                        return "open"
            elif path:
                return self._step_path(path, L)

        # 5. no frontier, no stairs: hidden passages (SCORR/SDOOR from
        # RANDOM_CORRIDORS). Search from cells with the most unknown map
        # within Chebyshev radius 2 (search reveals hidden features in the
        # 8 surrounding cells); rotate in rounds of 4 searches per cell.
        cands = []
        for y in range(C.ROWS):
            for x in range(C.COLS):
                if not L.passable(x, y):
                    continue
                pot = 0
                for yy in range(max(0, y - 2), min(C.ROWS, y + 3)):
                    for xx in range(max(0, x - 2), min(C.COLS, x + 3)):
                        if not L.explored[yy][xx]:
                            pot += 1
                if pot > 0:
                    rounds = self.search_counts.get((x, y), 0) // 4
                    cands.append((rounds, -pot, abs(x - L.agent[0]) +
                                  abs(y - L.agent[1]), (x, y)))
        if cands:
            cands.sort()
            tgt = cands[0][3]
            if tgt == L.agent:
                self.search_counts[tgt] = self.search_counts.get(tgt, 0) + 1
                if "search" in self._actions(obs):
                    return "search"
            else:
                path = L.bfs(L.agent, [tgt], avoid=mcells)
                if path:
                    return self._step_path(path, L)
        cnt = self.search_counts.get(L.agent, 0)
        self.search_counts[L.agent] = cnt + 1
        return "search" if "search" in self._actions(obs) else "wait"

    def _pick_frontier(self, L, fset, avoid):
        """Frontier target = argmin(dist - w * unknown mass nearby): biases
        exploration toward large unexplored regions (where goal rooms are)
        instead of small leftover pockets."""
        from collections import deque
        dist = {L.agent: 0}
        q = deque([L.agent])
        while q:
            cur = q.popleft()
            for _n, nxt in L.neighbors(*cur, avoid=avoid):
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        hints = self.memory.stairs_hint() if self.memory else []
        best = None
        for cell in fset:
            if cell not in dist:
                continue
            x, y = cell
            mass = 0
            r = self.frontier_mass_r
            for yy in range(max(0, y - r), min(C.ROWS, y + r + 1)):
                for xx in range(max(0, x - r), min(C.COLS, x + r + 1)):
                    if not L.explored[yy][xx]:
                        mass += 1
            score = dist[cell] - self.frontier_mass_w * mass
            if hints:
                # memory effect E1: bias exploration toward remembered
                # stairs locations (fixed-layout tasks only)
                hd = min(abs(x - hx) + abs(y - hy) for (hx, hy) in hints)
                score += 0.8 * hd
            if best is None or score < best[0]:
                best = (score, cell)
        if best and hints and not self._mem_fired_beeline:
            self._mem_fired_beeline = True
            self.memory.record_fired(
                f"E1 beeline: frontier biased toward remembered stairs {hints[:3]}")
        return best[1] if best else None

    def _apply_maze_parity(self, L):
        """MAZEWALK generates corridors on a lattice: nodes at one parity,
        walls at the opposite parity (both-even vs both-odd relative to the
        region origin). Once enough floor is seen, the parity class with
        zero floor observations is provably all wall/stone; write it back."""
        counts = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        floors = 0
        for y in range(C.ROWS):
            for x in range(C.COLS):
                if L.explored[y][x] and not L.inferred_wall[y][x] and \
                   L.terrain[y][x] in (C.FLOOR, C.CORRIDOR, C.STAIRS_DOWN,
                                       C.STAIRS_UP):
                    counts[(x % 2, y % 2)] += 1
                    floors += 1
        if floors < 14:
            return
        zeros = [k for k, v in counts.items() if v == 0]
        if len(zeros) != 1:
            return
        wx, wy = zeros[0]
        for y in range(C.ROWS):
            for x in range(C.COLS):
                if x % 2 == wx and y % 2 == wy and not L.explored[y][x]:
                    L.terrain[y][x] = C.WALL
                    L.explored[y][x] = True
                    L.inferred_wall[y][x] = True

    def _win_step_guard(self, L):
        """About to make the winning move onto the stairs. If the character
        is fast and the previous move consumed game time, the winning move
        may be a zero-time move (reward 0.99 -> progression 0). Burn moves
        dancing to an adjacent cell until we arrive on a frozen move; the
        next move is then guaranteed to consume time. Returns a dance action
        or None (safe to finish)."""
        if not self.saw_frozen or self.last_move_frozen:
            return None
        if self.steps_taken > 88:      # not enough margin: gamble instead
            return None
        stairs = set(L.find_terrain(C.STAIRS_DOWN))
        for name, (nx, ny) in L.neighbors(*L.agent):
            if (nx, ny) in stairs or (nx, ny) in self.suspect_walls:
                continue
            if any(m[0] == nx and m[1] == ny for m in L.monsters):
                continue
            return name
        return None                    # nowhere to dance: just go

    # ------------------------------------------------------------ helpers
    def _actions(self, obs):
        return getattr(self, "_action_cache", None) or []

    def set_actions(self, names):
        self._action_cache = list(names)

    def _deadends(self, L):
        out = []
        for y in range(C.ROWS):
            for x in range(C.COLS):
                if L.passable(x, y):
                    n = sum(1 for _ in L.neighbors(x, y))
                    if n <= 1:
                        out.append((x, y))
        return out or [L.agent]

    def _combat(self, L):
        ax, ay = L.agent
        adj = []
        for (mx, my, ch, pet) in L.monsters:
            if pet:
                continue
            if max(abs(mx - ax), abs(my - ay)) == 1:
                adj.append((mx, my, ch))
        if not adj:
            return None
        # low HP + several attackers: retreat to cell with fewest adjacent foes
        if L.hp < 0.3 * L.hpmax and len(adj) >= 2:
            best = None
            for name, (nx, ny) in L.neighbors(ax, ay):
                if any(m[0] == nx and m[1] == ny for m in L.monsters):
                    continue
                danger = sum(1 for (mx, my, _c) in adj
                             if max(abs(mx - nx), abs(my - ny)) == 1)
                if best is None or danger < best[0]:
                    best = (danger, name)
            if best and best[0] < len(adj):
                return best[1]
        # attack a safe-to-melee adjacent monster (prefer cardinal, low HP risk)
        for (mx, my, ch) in adj:
            if ch in AVOID_MELEE:
                continue
            d = (mx - ax, my - ay)
            # diagonal attack across a doorway is illegal; check
            t_here = L.terrain[ay][ax]
            t_there = L.terrain[my][mx]
            if d[0] and d[1] and (t_here in (C.DOORWAY, C.DOOR_OPEN, C.DOOR_CLOSED)
                                  or t_there in (C.DOORWAY, C.DOOR_OPEN, C.DOOR_CLOSED)):
                continue
            return C.DIR_OF[d]
        return None

    def _step_path(self, path, L):
        step = path[0]
        dx, dy = C.DIRS[step]
        nx, ny = L.agent[0] + dx, L.agent[1] + dy
        # if a closed door is the next cell, open it instead of bumping
        if L.terrain[ny][nx] == C.DOOR_CLOSED:
            if dx == 0 or dy == 0:
                self.kick_target = (step, (nx, ny))
                self.queue = [step]
                return "open"
            return "search"  # diagonal to a closed door shouldn't happen
        return step
