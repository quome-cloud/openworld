"""Boxoban (Sokoban-on-NLE) agent: full classical solver.

World model (task-scoped, synthesized from minihack/envs/boxohack.py + probes):
  - Premapped, lit, deterministic; no monsters. 4 boulders, 4 fountains.
  - Actions: N/E/S/W only. Walking into a boulder pushes it iff the cell beyond
    is passable floor/fountain (verified: 'With great effort you move the
    boulder.'); otherwise the action wastes an env step ('...but in vain.').
  - Env step = one action; episode cap 100 steps (BALROG config).
  - Success = every fountain covered by a boulder; final reward exactly 1.

Planner: A* over states (frozenset boulders, agent pos); successors are push
macros (BFS walk to push-approach cell + 1 push), g = total env steps,
h = min-cost matching of boulder->fountain push-BFS distances. Deadlock
pruning via reverse-pull reachability ("alive" squares) + 2x2 freeze checks.
Plan executed closed-loop with per-step verification against observations.
"""

import heapq
import time as _time
from collections import deque
from itertools import permutations

import numpy as np

import mh_common as C

CARD = [("north", (0, -1)), ("south", (0, 1)), ("east", (1, 0)), ("west", (-1, 0))]


class BoxobanSolver:
    def __init__(self, level, log=print):
        self.log = log
        self.walls = set()      # impassable static cells
        self.floors = set()
        self.fountains = set()
        for y in range(C.ROWS):
            for x in range(C.COLS):
                t = level.terrain[y][x]
                if t == C.UNKNOWN:
                    continue
                if t in (C.FLOOR, C.FOUNTAIN, C.DOORWAY, C.CORRIDOR, C.ICE):
                    self.floors.add((x, y))
                    if t == C.FOUNTAIN:
                        self.fountains.add((x, y))
                else:
                    self.walls.add((x, y))
        self.boulders0 = frozenset(level.boulders)
        self.agent0 = level.agent
        self.alive = self._alive_squares()
        self.push_dist = self._push_distances()

    # -- squares from which a lone boulder can still reach some fountain (pulls)
    def _alive_squares(self):
        alive = set()
        q = deque()
        for f in self.fountains:
            alive.add(f)
            q.append(f)
        while q:
            bx, by = q.popleft()
            for _, (dx, dy) in CARD:
                # pull boulder from (bx-dx, by-dy) to (bx, by): agent stands
                # at (bx+dx, by+dy) -- wait, reverse of push: boulder at p can
                # be pushed to cur if agent at p-(d) and both free. Reverse
                # BFS: predecessor p = (bx-dx, by-dy), needs agent cell
                # (bx-2dx, by-2dy) free.
                p = (bx - dx, by - dy)
                a = (bx - 2 * dx, by - 2 * dy)
                if p in self.floors and a in self.floors and p not in alive:
                    alive.add(p)
                    q.append(p)
        return alive

    # -- per-cell min pushes to each fountain (single boulder, empty board)
    def _push_distances(self):
        dists = {}
        for f in self.fountains:
            d = {f: 0}
            q = deque([f])
            while q:
                cur = q.popleft()
                bx, by = cur
                for _, (dx, dy) in CARD:
                    p = (bx - dx, by - dy)
                    a = (bx - 2 * dx, by - 2 * dy)
                    if p in self.floors and a in self.floors and p not in d:
                        d[p] = d[cur] + 1
                        q.append(p)
            dists[f] = d
        return dists

    def _h(self, boulders):
        """Min-cost perfect matching boulder->fountain over push distances."""
        fl = list(self.fountains)
        bl = list(boulders)
        best = None
        for perm in permutations(range(len(fl))):
            s = 0
            ok = True
            for bi, fi in enumerate(perm):
                d = self.push_dist[fl[fi]].get(bl[bi])
                if d is None:
                    ok = False
                    break
                s += d
            if ok and (best is None or s < best):
                best = s
        return best  # None => some boulder can't reach any fountain

    def _frozen_deadlock(self, boulders):
        """2x2 block of boulders/walls containing an off-goal boulder."""
        solid = self.walls
        for (x, y) in boulders:
            if (x, y) in self.fountains:
                continue
            for cx, cy in ((x, y), (x - 1, y), (x, y - 1), (x - 1, y - 1)):
                cells = [(cx, cy), (cx + 1, cy), (cx, cy + 1), (cx + 1, cy + 1)]
                if all(c in solid or c in boulders for c in cells):
                    if (x, y) in cells:
                        return True
        return False

    def _reach_and_paths(self, agent, boulders):
        """Cells reachable by walking; parent map for path extraction."""
        prev = {agent: None}
        q = deque([agent])
        while q:
            cur = q.popleft()
            x, y = cur
            for name, (dx, dy) in CARD:
                nxt = (x + dx, y + dy)
                if nxt in prev or nxt not in self.floors or nxt in boulders:
                    continue
                prev[nxt] = (cur, name)
                q.append(nxt)
        return prev

    @staticmethod
    def _walk(prev, cell):
        path = []
        node = cell
        while prev[node] is not None:
            node, nm = prev[node]
            path.append(nm)
        return path[::-1]

    def solve(self, max_seconds=240.0, max_nodes=400_000, weight=1.0):
        t0 = _time.time()
        start = (self.boulders0, self.agent0)
        h0 = self._h(self.boulders0)
        if h0 is None:
            return None, "unreachable_matching"
        openq = [(h0 * weight, 0, 0, start, [])]
        gbest = {(self.boulders0, self.agent0): 0}
        nodes = 0
        tie = 0
        while openq:
            if nodes % 2048 == 0 and _time.time() - t0 > max_seconds:
                return None, f"timeout({nodes} nodes)"
            f, g, _, (boulders, agent), plan = heapq.heappop(openq)
            nodes += 1
            if nodes > max_nodes:
                return None, f"node_cap({nodes})"
            if all(fn in boulders for fn in self.fountains):
                return plan, f"solved({nodes} nodes, {round(_time.time()-t0,1)}s)"
            if gbest.get((boulders, agent), 1 << 30) < g:
                continue
            prev = self._reach_and_paths(agent, boulders)
            for b in boulders:
                bx, by = b
                for name, (dx, dy) in CARD:
                    approach = (bx - dx, by - dy)
                    dest = (bx + dx, by + dy)
                    if approach not in prev:
                        continue
                    if dest not in self.floors or dest in boulders:
                        continue
                    if dest not in self.alive:
                        continue
                    nb = frozenset((boulders - {b}) | {dest})
                    if self._frozen_deadlock(nb):
                        continue
                    walk = self._walk(prev, approach)
                    ng = g + len(walk) + 1
                    if ng > 97:      # 100-step cap with margin
                        continue
                    key = (nb, b)    # agent ends on boulder's old cell
                    if gbest.get(key, 1 << 30) <= ng:
                        continue
                    nh = self._h(nb)
                    if nh is None:
                        continue
                    gbest[key] = ng
                    tie += 1
                    heapq.heappush(openq, (ng + nh * weight, ng, tie,
                                           (nb, b), plan + walk + [name]))
        return None, f"exhausted({nodes} nodes)"


class BoxobanAgent:
    """Closed-loop executor: solve once, verify every step, replan on any
    model misprediction (never observed in probes, but kept for honesty)."""

    def __init__(self, log=print):
        self.plan = None
        self.level = C.LevelState()
        self.log = log
        self.solve_note = None

    def set_actions(self, names):
        self._action_cache = list(names)

    def act(self, obs):
        self.level.update(obs)
        if self.plan is None:
            solver = BoxobanSolver(self.level, self.log)
            for w, secs, cap in ((1.0, 150.0, 250_000),
                                 (1.5, 60.0, 150_000),
                                 (2.5, 60.0, 150_000)):
                plan, note = solver.solve(max_seconds=secs, max_nodes=cap, weight=w)
                self.solve_note = f"w={w}: {note}"
                self.log(f"  solver {self.solve_note}")
                if plan is not None:
                    break
            if plan is None:
                self.plan = []
                self.expected = None
            else:
                self.plan = plan
                self._simulate_expectations()
        if not self.plan:
            return "wait_done"          # sentinel: nothing useful to do
        # verify state matches expectation of executed prefix
        idx = getattr(self, "_i", 0)
        if idx > 0:
            exp_agent, exp_boulders = self.expected[idx - 1]
            if self.level.agent != exp_agent or set(self.level.boulders) != exp_boulders:
                self.log(f"  MISPREDICTION at step {idx}: replanning "
                         f"(agent {self.level.agent} vs {exp_agent})")
                self.plan = None
                self._i = 0
                return self.act(obs)
        if idx >= len(self.plan):
            return "wait_done"
        self._i = idx + 1
        return self.plan[idx]

    def _simulate_expectations(self):
        """Forward-simulate the plan on the synthesized model to get the
        expected (agent, boulders) after every action."""
        self.expected = []
        agent = self.level.agent
        boulders = set(self.level.boulders)
        walls_or_unknown = set()
        for a in self.plan:
            dx, dy = dict((n, d) for n, d in CARD)[a]
            nxt = (agent[0] + dx, agent[1] + dy)
            if nxt in boulders:
                dest = (nxt[0] + dx, nxt[1] + dy)
                boulders.discard(nxt)
                boulders.add(dest)
                agent = nxt
            else:
                agent = nxt
            self.expected.append((agent, set(boulders)))
        self._i = 0
