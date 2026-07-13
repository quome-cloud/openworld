"""Closed-loop hierarchical planner for Crafter (shared by both protocols).

This is where the recipe evolves beyond the deterministic Baba arm:

  Baba Is AI (deterministic): synthesize exact model -> search offline for a
  full action sequence -> execute open-loop. Correct because the model is
  exact and nothing else moves.

  Crafter (stochastic): mobs, spawning and one recipe outcome are random;
  observation (honest arm) is partial. Open-loop is unsound. The planner is
  restructured into three layers, replanned EVERY step against the newest
  belief state:

    L1 REACTIVE SAFETY  — worst-case bounds from the stochastic spec
       (zombie closes 1 cell/tick and hits for 2 with cooldown 5; arrows fly
       1/tick for 2; lava = death). Overrides everything.
    L2 VITALS SCHEDULER — the deterministic vitals clocks (hunger 1/25 ticks,
       thirst 1/20, fatigue 1/31, recover ±1/25|15) give exact deadlines;
       thresholds are chosen so the worst-case travel time to a known
       resource fits inside the deadline.
    L3 ACHIEVEMENT DAG  — subgoals ordered by the tech-tree DAG
       (wood -> table -> wood tools -> stone -> stone tools + furnace ->
       coal/iron -> iron tools -> diamond) interleaved with opportunistic
       goals (sapling, plant, cow, mobs, sleep). Each subgoal compiles to
       "navigate + face + primitive" via A* over the believed map with
       mine-through costs; search happens on the synthesized model, never
       by env simulation.

  Determinization: mobs are treated as static obstacles with a soft danger
  cost for planning (replanning absorbs their motion); their *attack*
  dynamics use worst-case bounds in L1 (sound because bounds are exact).
"""

import heapq

import numpy as np

from crafter_model import (
    WALKABLE, PLAYER_WALKABLE, DIR_ACTION, MOB_HP, damage, is_night,
    WORLD_AREA, PLANT_RIPE_AGE)

MINE_TOOL = {'tree': None, 'stone': 'wood_pickaxe', 'coal': 'wood_pickaxe',
             'iron': 'stone_pickaxe', 'diamond': 'iron_pickaxe'}
MARGIN = 4          # keep away from map edge (text-obs window validity)
D4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def mdist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Brain:
    def __init__(self, belief, privileged=False, params=None):
        from memory import DEFAULT_PARAMS
        self.b = belief
        self.p = dict(DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        self.privileged = privileged
        self.goal_log = []
        self.wood_target = 5
        self.stone_target = 9
        self.sapling_give_up = 0
        self.skel_hunt_until = 0
        self.skel_cooldown_until = 0
        self.zombie_hits = 0           # damage dealt to current adjacent zombie
        self.skel_hits = 0
        self.shelter_mode = False
        self.explore_target = None
        self.last_goal = None
        self.stuck = 0
        self.last_pos_face = None
        self.mine_back_cell = None     # stone we placed for shelter, to re-mine
        self.skel_sightings = []       # (pos, t) persistent danger zones
        self.home = None               # persistent burrow (c1, c2) once dug
        self.burrow_blacklist = set()  # rejected corridors (verified leaky)
        self.topping_drink = False
        self.topping_food = False

    # ── main entry ─────────────────────────────────────────────────────────
    def act(self):
        b = self.b
        s = b.report('skeleton')
        if s:
            ref = s.exact or s.cells[0]
            if not any(mdist(ref, p) <= 2 for p, _ in self.skel_sightings):
                self.skel_sightings.append((ref, b.t))
        if b.sleeping:
            return 'noop', 'sleeping'
        if is_night(b.t):
            # night: shelter dominates; vitals only for dire emergencies
            a = (self._safety() or self._night() or self._vitals()
                 or self._goals() or self._explore() or ('noop', 'idle'))
        else:
            a = (self._safety() or self._vitals() or self._prep()
                 or self._goals() or self._explore() or ('noop', 'idle'))
        action, goal = a
        if action is None:
            action, goal = 'noop', goal + '?none'
        # stuck detector: same pos+facing and same goal for too long
        key = (b.pos, b.facing, goal)
        if key == self.last_pos_face and action.startswith('move_'):
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_pos_face = key
        if self.stuck > 6:
            self.explore_target = None
            action = DIR_ACTION[D4[b.t % 4]]   # t steps by 1: cycles all 4
            goal = 'unstick'
        self.last_goal = goal
        return action, goal

    # ── generic geometry helpers ───────────────────────────────────────────
    def _dir_to(self, cell):
        dx, dy = cell[0] - self.b.pos[0], cell[1] - self.b.pos[1]
        if abs(dx) + abs(dy) != 1:
            return None
        return (dx and (1 if dx > 0 else -1), 0) if dx else (0, 1 if dy > 0 else -1)

    def _face_or(self, cell, action):
        """If adjacent to cell: face it (turn) then run `action`."""
        d = self._dir_to(cell)
        if d is None:
            return None
        if tuple(self.b.facing) == d:
            return action
        return DIR_ACTION[d]           # blocked-or-not, this turns toward it

    def _passable(self, p, inv, allow_mine=True):
        m = self.b.mat(p)
        if m in WALKABLE or m == 'floor':
            return 1.0
        if m is None:                  # unknown (honest arm)
            return 1.6
        if allow_mine and m in MINE_TOOL:
            tool = MINE_TOOL[m]
            if tool is None or inv.get(tool, 0) > 0:
                return 3.0
        return None

    def _danger(self, p):
        cost = 0.0
        for kind, rad, w in (('zombie', 2, 6.0), ('skeleton', 3, 5.0),
                             ('arrow', 2, 4.0)):
            r = self.b.report(kind)
            if r:
                for c in r.cells:
                    d = mdist(p, c)
                    if d <= rad:
                        cost += w / max(1, d)
        return cost

    def _skel_zone(self, p, ttl=700, rad=5):
        """Structural danger: skeletons live in fixed tunnel systems, so a
        sighting poisons its neighbourhood for a long time."""
        b = self.b
        for pos, t in self.skel_sightings:
            if b.t - t < ttl and mdist(p, pos) <= rad:
                return True
        return False

    def _lava_safe(self, p):
        """Certainly-not-lava test for an adjacent target cell."""
        m = self.b.mat(p)
        if m is not None:
            return m != 'lava'
        r = self.b.report('lava')
        return not (r and r.dist == 1 and (r.exact == tuple(p) or r.exact is None))

    def path_next(self, targets, adjacent=True, allow_mine=True):
        """A* from pos to any target (or a cell adjacent to one). Returns
        (next_cell, first_target, path_len) or None."""
        b = self.b
        inv = b.inventory
        goals = set()
        tgt_of = {}
        for t in targets:
            t = tuple(t)
            if adjacent:
                for d in D4:
                    c = (t[0] + d[0], t[1] + d[1])
                    if self._passable(c, inv) is not None or c == b.pos:
                        goals.add(c)
                        tgt_of.setdefault(c, t)
            else:
                goals.add(t)
                tgt_of.setdefault(t, t)
        if not goals:
            return None
        start = tuple(b.pos)
        if start in goals:
            return (start, tgt_of[start], 0)
        h0 = min(mdist(start, g) for g in goals)
        openq = [(h0, 0.0, start, None)]
        came, seen = {}, {start: 0.0}
        found = None
        expansions = 0
        while openq and expansions < 6000:
            f, g, cur, parent = heapq.heappop(openq)
            if cur in came:
                continue
            came[cur] = parent
            expansions += 1
            if cur in goals:
                found = cur
                break
            for d in D4:
                nxt = (cur[0] + d[0], cur[1] + d[1])
                if not (MARGIN <= nxt[0] < WORLD_AREA[0] - MARGIN and
                        MARGIN <= nxt[1] < WORLD_AREA[1] - MARGIN):
                    continue
                step = self._passable(nxt, inv)
                if step is None:
                    continue
                # never path THROUGH a mob-occupied cell (except goal cells)
                if nxt not in goals and self.b.obj_at(nxt) is not None:
                    continue
                ng = g + step + self._danger(nxt) + \
                    (2.5 if self._skel_zone(nxt, ttl=400, rad=self.p['skel_zone_rad']) else 0.0)
                if nxt not in seen or ng < seen[nxt] - 1e-9:
                    seen[nxt] = ng
                    hh = min(mdist(nxt, gg) for gg in goals) if len(goals) < 12 \
                        else 0
                    heapq.heappush(openq, (ng + hh, ng, nxt, cur))
        if found is None:
            return None
        # walk back to first step
        node, prev = found, came[found]
        length = 0
        while prev is not None and prev != start:
            node, prev = prev, came[prev]
            length += 1
        return (node if prev == start else found, tgt_of.get(found, found),
                length + 1)

    def _step_toward(self, cell_next):
        """Emit the primitive that progresses onto/through cell_next."""
        b = self.b
        d = self._dir_to(cell_next)
        if d is None:
            return None
        m = b.mat(cell_next)
        obj = b.obj_at(cell_next)
        if obj is not None:
            kind = obj['kind']
            if kind in ('zombie', 'skeleton', 'cow'):
                return self._face_or(cell_next, 'do')     # clear the blocker
            if kind == 'plant' and cell_next != b.plant_pos:
                return self._face_or(cell_next, 'do')
            return None                                    # our plant: reroute
        if m in WALKABLE or m == 'floor':
            if not self._lava_safe(cell_next):
                return None
            return DIR_ACTION[d]
        if m is None:
            if not self._lava_safe(cell_next):
                return None
            return DIR_ACTION[d]       # unknown but lava-excluded: try it
        if m in MINE_TOOL:
            tool = MINE_TOOL[m]
            if tool is None or b.inventory.get(tool, 0) > 0:
                return self._face_or(cell_next, 'do')      # mine it
        return None

    def goto_and(self, targets, final_action, allow_mine=True):
        """Navigate to adjacency of a target; when adjacent, face + act."""
        b = self.b
        for t in targets:
            if mdist(b.pos, t) == 1:
                a = self._face_or(t, final_action)
                if a:
                    return a
        r = self.path_next(targets, adjacent=True, allow_mine=allow_mine)
        if r is None:
            return None
        nxt, tgt, _ = r
        if nxt == tuple(b.pos):        # already adjacent (goal was our cell)
            a = self._face_or(tgt, final_action)
            if a:
                return a
            return None
        return self._step_toward(nxt)

    # ── L1: reactive safety ────────────────────────────────────────────────
    def _safety(self):
        b = self.b
        z = b.report('zombie')
        if z and z.dist == 1 and z.exact:
            # always fight an adjacent zombie: we out-damage it (2-5/hit vs
            # its 2 per 6 ticks) and fleeing keeps it adjacent 90% of ticks
            self.zombie_hits += damage(b.inventory)
            if self.zombie_hits >= MOB_HP['zombie']:
                b.ach.add('defeat_zombie')
                self.zombie_hits = 0
            return self._face_or(z.exact, 'do'), 'fight_zombie'
        else:
            self.zombie_hits = 0
        # low health: disengage from any zombie that could reach us soon
        # (but never abandon an in-progress burrow at night: it is the exit)
        if z and 2 <= z.dist <= 3 and b.inventory.get('health', 9) <= 2 and \
                not (is_night(b.t) and self.burrow_cells):
            ref = z.exact or z.cells[0]
            best, bd = None, -1
            for d in D4:
                c = (b.pos[0] + d[0], b.pos[1] + d[1])
                if (b.mat(c) in WALKABLE or b.mat(c) == 'floor') and b.obj_at(c) is None and \
                        self._lava_safe(c):
                    dd = mdist(c, ref)
                    if dd > bd:
                        best, bd = DIR_ACTION[d], dd
            if best and bd > z.dist:
                return best, 'flee_zombie'
        arrow = b.report('arrow')
        if arrow and arrow.dist <= 2 and arrow.exact:
            ax, ay = arrow.exact
            if ax == b.pos[0] or ay == b.pos[1]:
                perp = [(1, 0), (-1, 0)] if ax == b.pos[0] else [(0, 1), (0, -1)]
                for d in perp:
                    c = (b.pos[0] + d[0], b.pos[1] + d[1])
                    if (b.mat(c) in WALKABLE or b.mat(c) == 'floor') and b.obj_at(c) is None and \
                            self._lava_safe(c):
                        return DIR_ACTION[d], 'dodge_arrow'
        s = b.report('skeleton')
        if s and s.dist <= 5 and not (is_night(b.t) and self.burrow_cells) \
                and not (b.inventory.get('food', 9) == 0 or
                         b.inventory.get('drink', 9) == 0):
            hunting = ('defeat_skeleton' not in b.ach and
                       b.inventory.get('health', 0) >= self.p['skel_hunt_min_hp'] and
                       damage(b.inventory) >= 3 and
                       b.t >= self.skel_cooldown_until)
            if hunting:
                if self.skel_hunt_until == 0:
                    self.skel_hunt_until = b.t + 50
                if b.t > self.skel_hunt_until:
                    self.skel_hunt_until = 0
                    self.skel_cooldown_until = b.t + 200
                else:
                    if s.dist == 1 and s.exact:
                        self.skel_hits += damage(b.inventory)
                        if self.skel_hits >= MOB_HP['skeleton']:
                            b.ach.add('defeat_skeleton')
                            self.skel_hits = 0
                            self.skel_hunt_until = 0
                        return self._face_or(s.exact, 'do'), 'fight_skeleton'
                    tgt = s.exact or s.cells[0]
                    a = self.goto_and([tgt], 'do', allow_mine=False)
                    if a:
                        return a, 'hunt_skeleton'
            else:
                # committed retreat: pick a refuge >=8 away once and walk it
                ref = s.exact or s.cells[0]
                if self.retreat_target is None or b.t > self.retreat_until:
                    self.retreat_target = self._pick_refuge(ref)
                    self.retreat_until = b.t + 14
                if self.retreat_target:
                    r = self.path_next([self.retreat_target],
                                       adjacent=False, allow_mine=False)
                    if r:
                        a = self._step_toward(r[0])
                        if a:
                            return a, 'retreat_skeleton'
                # cornered: fight it if adjacent
                if s.dist == 1 and s.exact:
                    return self._face_or(s.exact, 'do'), 'fight_skeleton'
        elif self.retreat_target is not None and b.mob_dist('skeleton') > 6:
            self.retreat_target = None
        return None

    retreat_target = None
    retreat_until = 0

    def _pick_refuge(self, threat):
        """Known walkable cell far from the threat, biased toward grass."""
        b = self.b
        best, bestscore = None, -1
        xs, ys = np.where(np.isin(b.map, ['grass', 'sand', 'path']))
        n = len(xs)
        stride = max(1, n // 400)
        for i in range(0, n, stride):
            c = (int(xs[i]), int(ys[i]))
            d_t = mdist(c, threat)
            d_us = mdist(c, b.pos)
            if d_us > 14 or d_t < 8:
                continue
            score = d_t - 0.5 * d_us + (2 if b.map[c] == 'grass' else 0)
            if score > bestscore:
                best, bestscore = c, score
        return best

    # ── L2: vitals ─────────────────────────────────────────────────────────
    def _vitals(self):
        b = self.b
        inv = b.inventory
        night = is_night(b.t)
        if night:
            # dire emergencies only, and only for close-by resources; the
            # sleep state halves hunger/thirst drain, so a lean night beats
            # a moonlit hike through zombie spawns
            limit = 1 if self.burrow_cells else 4
            if inv.get('drink', 9) == 0:
                w = b.nearest_material({'water'}, limit=limit)
                if w:
                    a = self.goto_and([w], 'do', allow_mine=False)
                    if a:
                        return a, 'drink'
            if inv.get('food', 9) == 0 and not self.burrow_cells:
                c = b.report('cow')
                if c and c.dist <= 4:
                    a = self._eat()
                    if a:
                        return a, 'eat'
            return None
        drink = inv.get('drink', 9)
        if drink <= self.p['vitals_floor'] or (self.topping_drink and drink < 9):
            a = self._drink()
            if a:
                self.topping_drink = True
                return a, 'drink'
        self.topping_drink = False
        food = inv.get('food', 9)
        if food <= self.p['vitals_floor'] or (self.topping_food and food < 8):
            a = self._eat()
            if a:
                self.topping_food = True
                return a, 'eat'
        self.topping_food = False
        if food <= 6:
            c = b.report('cow')
            if c and c.dist <= 5 and damage(inv) >= 2:
                a = self._eat()
                if a:
                    return a, 'eat'
        if inv.get('energy', 9) <= 1:
            # daytime emergency: sleep in the open only when provably calm
            if b.mob_dist('zombie') > 9 and b.mob_dist('skeleton') > 7:
                return 'sleep', 'emergency_sleep'
        return None

    # ── pre-night preparation (t%300 in [105,148): top up, dig burrow) ─────
    def _prep(self):
        b = self.b
        inv = b.inventory
        phase = b.t % 300
        # morning nap: recover energy (and wake_up) while zombies are
        # despawned by daylight — the cheap, seal-free sleep window.
        # Sleep is a commitment (no voluntary wake), so require the area to
        # be COMPLETELY clear and enough daylight left to finish the nap.
        if phase < 45 and inv.get('energy', 9) <= 4 and \
                b.mob_dist('zombie') > 90 and b.mob_dist('skeleton') > 7:
            return 'sleep', 'morning_nap'
        if not (90 <= phase < 148):
            return None
        # top up vitals for the night: a corked sleeper cannot eat or drink,
        # and the hp-bleed loop (sleep -> hurt -> wake -> sleep) with empty
        # food/drink was a top-3 death cause
        if inv.get('drink', 9) <= 7:
            w = b.nearest_material({'water'}, limit=20)
            if w:
                a = self.goto_and([w], 'do', allow_mine=False)
                if a:
                    return a, 'prep_drink'
        if inv.get('food', 9) <= 6 and damage(inv) >= 2:
            c = b.report('cow')
            if c and c.dist <= 12:
                a = self._eat()
                if a:
                    return a, 'prep_eat'
        # dig tonight's burrow early enough to be corked by nightfall;
        # start earlier when we still lack a cork stone (travel time).
        # With an established home, leave exactly enough time to walk back.
        if self.home is not None:
            c1, _ = self.home
            slack = 146 - phase
            if mdist(b.pos, c1) + 10 >= slack:
                a = self._burrow()
                if a:
                    return a, 'prep_burrow'
            return None
        start = self.p['prep_start'] - (8 if inv.get('stone', 0) == 0 else 0)
        if phase >= start and inv.get('wood_pickaxe', 0) > 0:
            a = self._burrow()
            if a:
                return a, 'prep_burrow'
        return None

    # ── burrow: dig 2 cells into a stone mass, back up 1, cork behind ──────
    # The only self-sealing shelter Crafter's action set allows: place_stone
    # acts on the FACED cell, and facing follows movement, so the cork is
    # placed while stepping back out of the dug corridor. Net stone: +1.
    burrow_cells = None                # (c1, c2) current burrow corridor

    def _burrow_ok(self, c1, c2, strict=True):
        """Corridor c1->c2 must have non-walkable flanks so the only way in
        is the entrance we cork. Any non-walkable material (stone, water,
        lava, tree, table, ores...) is a valid wall: nothing in the
        environment digs. strict=False (honest arm, partial map) tolerates
        UNKNOWN flanks at selection time; they are then verified in-place
        while digging via the distance-1 report lemma, aborting on a
        walkable flank."""
        b = self.b
        d = (c2[0] - c1[0], c2[1] - c1[1])
        perp = [(d[1], d[0]), (-d[1], -d[0])]
        for cell in (c1, c2):
            for p in perp:
                m = b.mat((cell[0] + p[0], cell[1] + p[1]))
                if m in WALKABLE or m == 'floor':
                    return False
                if m is None and strict:
                    return False
        far = b.mat((c2[0] + d[0], c2[1] + d[1]))
        if far in WALKABLE or far == 'floor':
            return False
        if far is None and strict:
            return False
        return True

    def _cork_action(self, at=None):
        """Primitive that seals the faced entrance cell: stone if we have
        one, else a table (2 wood). Arrows DESTROY tables (objects.py
        Arrow.update), so table corks are only sound outside skeleton
        habitat — no path cells (their spawn substrate) near the burrow."""
        b = self.b
        inv = b.inventory
        if inv.get('stone', 0) > 0:
            return 'place_stone'
        # NOTE: a table would also seal the entrance, but data.yaml has no
        # collect entry for 'table' — players cannot remove tables. A table
        # cork is therefore a self-entombment (only skeleton arrows destroy
        # tables). Stone-only.
        return None

    def _find_burrow(self):
        """Pick E0 (standing cell) + direction d with c1=E0+d, c2=E0+2d each
        walkable-or-minable-stone, flanked by non-walkable cells (stone
        masses, water coves, tree pockets all qualify). Nearest such E0 we
        can also cork (stone in hand, or wood for a table)."""
        b = self.b
        inv = b.inventory
        if self._cork_action() is None and inv.get('stone', 0) == 0:
            pass                        # still allowed: corridor may already
                                        # have a natural dead entrance
        can_mine = inv.get('wood_pickaxe', 0) > 0
        best, bd = None, 10 ** 9

        def cellok(c, allow_unknown=False):
            m = b.mat(c)
            if m in WALKABLE or m == 'floor':
                return 0
            if m == 'stone' and can_mine:
                return 1
            if m is None and allow_unknown and can_mine:
                return 1               # optimistic: stone masses are
                                       # contiguous; verified while digging
            return None

        xs, ys = np.where(np.isin(b.map, ['stone', 'grass', 'path', 'sand']))
        cells = set(zip(map(int, xs), map(int, ys)))
        for (sx, sy) in cells:
            if abs(sx - b.pos[0]) + abs(sy - b.pos[1]) > 22:
                continue
            for d in D4:
                c1 = (sx, sy)
                c2 = (sx + d[0], sy + d[1])
                e0 = (sx - d[0], sy - d[1])
                k1 = cellok(c1)
                k2 = cellok(c2, allow_unknown=not self.privileged)
                if k1 is None or k2 is None:
                    continue
                m0 = b.mat(e0)
                if m0 not in WALKABLE:
                    continue
                if (c1, c2) in self.burrow_blacklist:
                    continue
                if not self._burrow_ok(c1, c2, strict=self.privileged):
                    continue
                mines = k1 + k2
                if mines == 0 and inv.get('stone', 0) == 0:
                    continue            # nothing to cork with (stone only)
                if mines > 0 and not can_mine:
                    continue
                dd = mdist(b.pos, e0) + 2 * mines
                if dd < bd:
                    best, bd = (e0, d, c1, c2), dd
        if best is not None and bd <= 26:
            return best
        return None

    def _burrow(self):
        """Advance the burrow state machine by one primitive. Returns None
        when no burrow is available."""
        b = self.b
        inv = b.inventory
        if inv.get('wood_pickaxe', 0) == 0:
            return None
        bc = self.burrow_cells
        if bc:
            c1, c2 = bc
            pos = tuple(b.pos)
            if pos in (c1, c2) and not self._burrow_ok(c1, c2, strict=False):
                # a revealed flank is walkable: this corridor cannot be
                # sealed — blacklist it and walk back out
                self.burrow_blacklist.add((c1, c2))
                if self.home == bc:
                    self.home = None
                self.burrow_cells = None
                d = (c2[0] - c1[0], c2[1] - c1[1])
                e0 = (c1[0] - d[0], c1[1] - d[1])
                a = self._step_toward(e0 if pos == c1 else c1)
                if a:
                    return a
            if pos not in (c1, c2):
                r = self.path_next([c1], adjacent=False, allow_mine=True)
                if r is None:
                    self.burrow_cells = None
                    return None
                return self._step_toward(r[0])
            if pos == c2:
                # step back toward c1 (turns us to face the entrance side)
                a = self._step_toward(c1)
                return a
            if pos == c1:
                d = (c2[0] - c1[0], c2[1] - c1[1])
                e0 = (c1[0] - d[0], c1[1] - d[1])
                m0 = b.mat(e0)
                if tuple(b.facing) == (-d[0], -d[1]):
                    if m0 in WALKABLE or m0 == 'floor' or m0 is None:
                        if not is_night(b.t):
                            # daytime pre-dig: corridor ready, cork at dusk
                            self.home = self.burrow_cells
                            self.burrow_cells = None
                            return None
                        cork = self._cork_action()
                        if cork is not None:
                            self.mine_back_cell = e0
                            self.home = self.burrow_cells
                            return cork
                        return None    # no cork: burrow unusable
                    return 'sleep' if inv.get('energy', 9) < 9 else 'noop'
                if m0 not in WALKABLE and m0 is not None:  # corked
                    self.home = self.burrow_cells
                    return 'sleep' if inv.get('energy', 9) < 9 else 'noop'
                # not yet facing entrance: dig deeper first if c2 not dug
                m2 = b.mat(c2)
                if m2 == 'stone' or m2 is None:
                    return self._face_or(c2, 'do')   # facing reveals; do mines
                if m2 in WALKABLE or m2 == 'floor':
                    a = self._step_toward(c2)
                    return a
                if m2 in ('tree', 'coal', 'iron', 'diamond'):
                    tool = MINE_TOOL.get(m2)
                    if tool is None or inv.get(tool, 0) > 0:
                        return self._face_or(c2, 'do')
                # c2 revealed as water/lava/boundary: corridor unusable
                self.burrow_blacklist.add((c1, c2))
                if self.home == self.burrow_cells:
                    self.home = None
                self.burrow_cells = None
                return None
        if self.home is not None:
            c1, c2 = self.home
            d = (c2[0] - c1[0], c2[1] - c1[1])
            e0 = (c1[0] - d[0], c1[1] - d[1])
            if self._burrow_ok(c1, c2, strict=self.privileged) and \
                    b.mat(c1) in WALKABLE and b.mat(c2) in WALKABLE and \
                    mdist(b.pos, c1) <= 44:
                self.burrow_cells = self.home
                return self._burrow()
            self.home = None
        found = self._find_burrow()
        if found is None:
            self.burrow_cells = None
            return None
        e0, d, c1, c2 = found
        self.burrow_cells = (c1, c2)
        pos = tuple(b.pos)
        if pos != e0 and pos not in (c1, c2):
            r = self.path_next([e0], adjacent=False, allow_mine=True)
            if r is None:
                self.burrow_cells = None
                return None
            return self._step_toward(r[0])
        if pos == e0:
            if b.mat(c1) == 'stone':
                return self._face_or(c1, 'do')
            if b.mat(c1) in WALKABLE:
                return self._step_toward(c1)
        return None

    def _pick_corner(self):
        """Nearest known cell with >=2 non-walkable sides, outside skeleton
        zones, within 10 steps."""
        b = self.b
        best, bestkey = None, None
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                d0 = abs(dx) + abs(dy)
                if d0 > 10:
                    continue
                c = (b.pos[0] + dx, b.pos[1] + dy)
                m = b.mat(c)
                if not (m in WALKABLE or m == 'floor'):
                    continue
                if c != tuple(b.pos) and b.obj_at(c) is not None:
                    continue
                if self._skel_zone(c, rad=5):
                    continue
                walls = 0
                for d in D4:
                    n = b.mat((c[0] + d[0], c[1] + d[1]))
                    if n is not None and n != 'floor' and n not in WALKABLE:
                        walls += 1
                if walls < 2:
                    continue
                key = (-walls, d0)
                if bestkey is None or key < bestkey:
                    best, bestkey = c, key
        return best

    def _in_corked_burrow(self):
        b = self.b
        if not self.burrow_cells:
            return False
        c1, c2 = self.burrow_cells
        pos = tuple(b.pos)
        if pos not in (c1, c2):
            return False
        d = (c2[0] - c1[0], c2[1] - c1[1])
        e0 = (c1[0] - d[0], c1[1] - d[1])
        m0 = b.mat(e0)
        return (m0 not in WALKABLE and m0 != 'floor' and m0 is not None and
                self._burrow_ok(c1, c2, strict=False))

    def _drink(self):
        b = self.b
        xs, ys = np.where(b.map == 'water')
        best, bd = None, 10 ** 9
        for x, y in zip(map(int, xs), map(int, ys)):
            d = mdist((x, y), b.pos) + (14 if self._skel_zone((x, y)) else 0)
            if d < bd:
                best, bd = (x, y), d
        if best is None:
            r = b.report('water')
            if r:
                best = r.exact or min(r.cells, key=lambda c: mdist(c, b.pos))
        if best is None:
            return None
        return self.goto_and([best], 'do', allow_mine=True)

    def _eat(self):
        b = self.b
        if b.plant_pos and b.plant_age > PLANT_RIPE_AGE + 10:
            a = self.goto_and([b.plant_pos], 'do', allow_mine=False)
            if a:
                return a
        c = b.report('cow')
        if c:
            tgt = c.exact or min(c.cells, key=lambda x: mdist(x, b.pos))
            a = self.goto_and([tgt], 'do', allow_mine=False)
            if a:
                return a
        return None

    # ── night / sleep ──────────────────────────────────────────────────────
    def _night(self):
        b = self.b
        if not is_night(b.t):
            # dawn: uncork and reclaim the stone, then resume the day
            if self.burrow_cells and self.mine_back_cell and \
                    b.inventory.get('wood_pickaxe', 0) > 0 and \
                    mdist(b.pos, self.mine_back_cell) == 1 and \
                    b.mat(self.mine_back_cell) == 'stone':
                a = self._face_or(self.mine_back_cell, 'do')
                if a:
                    return a, 'uncork'
            self.burrow_cells = None
            self.mine_back_cell = None
            return None
        # inside a corked burrow: sleep / hold until dawn
        if self._in_corked_burrow():
            if b.inventory.get('energy', 9) < 9:
                return 'sleep', 'night_sleep'
            return 'noop', 'hold_burrow'
        # engage a zombie only if it is already on our doorstep
        if ('defeat_zombie' not in b.ach and damage(b.inventory) >= 3 and
                b.inventory.get('health', 0) >= 7):
            z = b.report('zombie')
            if z and z.dist <= 3:
                tgt = z.exact or min(z.cells, key=lambda x: mdist(x, b.pos))
                a = self.goto_and([tgt], 'do', allow_mine=False)
                if a:
                    return a, 'hunt_zombie'
        # try to reach/dig/cork a burrow
        a = self._burrow()
        if a:
            return a, 'night_burrow'
        # no burrow available: hold a defensive corner — a cell with >=2
        # walled sides limits engagements to ~one zombie at a time (they
        # attack only orthogonally); kiting into unmapped grass just finds
        # fresh spawns (observed death mode)
        corner = self._pick_corner()
        if corner is not None:
            if tuple(b.pos) == corner:
                z = b.report('zombie')
                if z and z.dist == 1 and z.exact:
                    a = self._face_or(z.exact, 'do')
                    if a:
                        return a, 'hold_corner'
                return 'noop', 'hold_corner'
            r = self.path_next([corner], adjacent=False, allow_mine=False)
            if r:
                a = self._step_toward(r[0])
                if a:
                    return a, 'goto_corner'
        z = b.report('zombie')
        if z and z.dist <= 5:
            threats = b.threat_cells('zombie') or ([z.exact] if z.exact
                                                   else [z.cells[0]])
            cur = min(mdist(b.pos, c) for c in threats)
            best, bd = None, -1
            for d in D4:
                c = (b.pos[0] + d[0], b.pos[1] + d[1])
                if (b.mat(c) in WALKABLE or b.mat(c) == 'floor') and b.obj_at(c) is None and \
                        self._lava_safe(c):
                    dd = min(mdist(c, t) for t in threats)
                    if dd > bd:
                        best, bd = DIR_ACTION[d], dd
            if best and bd > cur:
                return best, 'kite_zombie'
        # NO open-air sleeping at night: crafter's sleep is a commitment
        # (actions are locked to 'sleep' until energy is full or we get
        # hurt), so a "calm moment" nap is a death sentence once zombies
        # converge. Corked burrow or stay awake.
        return None

    # ── L3: achievement DAG ────────────────────────────────────────────────
    def _goals(self):
        b = self.b
        inv = b.inventory
        ach = b.ach

        # opportunistic quickies (cheap, adjacent-only)
        q = self._opportunistic()
        if q:
            return q

        # 1. wood — chop just enough for the NEXT unmet recipe; hoarding
        # before the table exists cost entire tech tiers (observed)
        if 'place_table' not in ach:
            need_now = 5               # table 2 + pickaxe 1 + sword 1 + spare
        elif 'make_wood_pickaxe' not in ach or 'make_wood_sword' not in ach:
            need_now = 1
        else:
            need_now = 0
        if need_now and inv.get('wood', 0) < need_now:
            t = b.nearest_material({'tree'})
            if t:
                a = self.goto_and([t], 'do')
                if a:
                    return a, 'collect_wood'

        # 2. table
        if 'place_table' not in ach and inv.get('wood', 0) >= 2:
            a = self._place_on_ground('place_table')
            if a:
                return a, 'place_table'

        # 3. wood tools
        for tool in ('wood_pickaxe', 'wood_sword'):
            if f'make_{tool}' not in ach and inv.get('wood', 0) >= 1:
                a = self._at_station(('table',), f'make_{tool}')
                if a:
                    return a, f'make_{tool}'

        # 3b. (memory-driven) grab the cork stone before anything else
        if self.p.get('home_stone_urgency') and self.home is None and \
                inv.get('wood_pickaxe', 0) > 0 and inv.get('stone', 0) == 0:
            s = b.nearest_material({'stone'})
            if s:
                a = self.goto_and([s], 'do')
                if a:
                    return a, 'collect_stone'

        # 4. sapling (early, so the plant has time to ripen)
        if ('collect_sapling' not in ach and self.sapling_give_up < self.p['sapling_budget'] and
                'make_wood_pickaxe' in ach):
            f = (b.pos[0] + b.facing[0], b.pos[1] + b.facing[1])
            if b.mat(f) == 'grass' and b.obj_at(f) is None:
                self.sapling_give_up += 1      # count actual draws only
                return 'do', 'collect_sapling'
            if self.home is not None:          # tech race first on day 1
                for d in D4:                    # face an adjacent grass cell
                    c = (b.pos[0] + d[0], b.pos[1] + d[1])
                    if b.mat(c) == 'grass' and b.obj_at(c) is None:
                        return DIR_ACTION[d], 'collect_sapling'
        if 'place_plant' not in ach and inv.get('sapling', 0) >= 1:
            a = self._place_on_ground('place_plant', where={'grass'})
            if a:
                return a, 'place_plant'

        # 5. first stones (enough for tools + a cork), then USE them —
        # topping up to a full bag before visiting the table starves the
        # tech tree (observed failure mode)
        if inv.get('wood_pickaxe', 0) > 0 and inv.get('stone', 0) < 3 and \
                ('make_stone_pickaxe' not in ach or
                 'make_stone_sword' not in ach or 'place_stone' not in ach):
            s = b.nearest_material({'stone'})
            if s:
                a = self.goto_and([s], 'do')
                if a:
                    return a, 'collect_stone'

        # 6. place_stone (cheap achievement; re-mineable)
        if 'place_stone' not in ach and inv.get('stone', 0) >= 1:
            a = self._place_on_ground('place_stone',
                                      where={'grass', 'sand', 'path'})
            if a:
                return a, 'place_stone'

        # 7. stone tools (need table nearby; build a fresh one if far)
        # keep 1 stone in reserve for the night burrow cork
        for tool in ('stone_pickaxe', 'stone_sword'):
            if f'make_{tool}' not in ach and inv.get('wood', 0) >= 1 and \
                    inv.get('stone', 0) >= 2:
                a = self._at_station(('table',), f'make_{tool}')
                if a:
                    return a, f'make_{tool}'

        # 7b. prepare tonight's home burrow EARLY — open-field nights are
        # mathematically losing (zombie queue DPS > our regen+kill rate),
        # so the corked burrow is a day-1 primary goal, not a dusk scramble
        if self.home is None and inv.get('wood_pickaxe', 0) > 0 and \
                inv.get('stone', 0) >= 1:
            a = self._burrow()
            if a:
                return a, 'prepare_home'

        # 8. furnace next to table (keep 1 cork stone in reserve)
        if 'place_furnace' not in ach:
            if inv.get('stone', 0) >= 5:
                a = self._furnace_by_table()
                if a:
                    return a, 'place_furnace'
            elif inv.get('wood_pickaxe', 0) > 0:
                s = b.nearest_material({'stone'})
                if s:
                    a = self.goto_and([s], 'do')
                    if a:
                        return a, 'collect_stone'

        # 8b. top up the stone bag for corks / plant walls
        if inv.get('wood_pickaxe', 0) > 0 and \
                inv.get('stone', 0) < self._stone_needed():
            s = b.nearest_material({'stone'}, limit=12)
            if s:
                a = self.goto_and([s], 'do')
                if a:
                    return a, 'collect_stone'

        # 9. coal & iron
        if inv.get('wood_pickaxe', 0) > 0 and inv.get('coal', 0) < \
                self._coal_needed():
            c = b.nearest_material({'coal'})
            if c:
                a = self.goto_and([c], 'do')
                if a:
                    return a, 'collect_coal'
        if inv.get('stone_pickaxe', 0) > 0 and inv.get('iron', 0) < \
                self._iron_needed():
            i = b.nearest_material({'iron'})
            if i:
                a = self.goto_and([i], 'do')
                if a:
                    return a, 'collect_iron'

        # 10. iron tools (table + furnace nearby)
        for tool in ('iron_pickaxe', 'iron_sword'):
            if f'make_{tool}' not in ach and inv.get('wood', 0) >= 1 and \
                    inv.get('coal', 0) >= 1 and inv.get('iron', 0) >= 1:
                a = self._at_station(('table', 'furnace'), f'make_{tool}')
                if a:
                    return a, f'make_{tool}'

        # 10b. wood buffer for corks / remaining recipes
        if inv.get('wood', 0) < self._wood_needed():
            t = b.nearest_material({'tree'}, limit=14)
            if t:
                a = self.goto_and([t], 'do')
                if a:
                    return a, 'collect_wood'

        # 11. diamond
        if 'collect_diamond' not in ach and inv.get('iron_pickaxe', 0) > 0:
            d = b.nearest_material({'diamond'})
            if d:
                a = self.goto_and([d], 'do')
                if a:
                    return a, 'collect_diamond'

        # 12. eat_plant when ripe
        if 'eat_plant' not in ach and b.plant_pos and \
                b.plant_age > PLANT_RIPE_AGE + 10:
            a = self.goto_and([b.plant_pos], 'do', allow_mine=False)
            if a:
                return a, 'eat_plant'

        # 13. stay near plant while it ripens if nothing else pressing
        if 'eat_plant' not in ach and b.plant_pos and \
                mdist(b.pos, b.plant_pos) > 14:
            a = self.goto_and([b.plant_pos], 'noop', allow_mine=True)
            if a and a != 'noop':
                return a, 'guard_plant'
        return None

    def _opportunistic(self):
        b = self.b
        inv = b.inventory
        # adjacent cow + eat_cow unmet (or hungry-ish): attack
        c = b.report('cow')
        if c and c.dist == 1 and c.exact and \
                ('eat_cow' not in b.ach or inv.get('food', 9) <= 6):
            a = self._face_or(c.exact, 'do')
            if a:
                return a, 'eat_cow'
        # cow hunt when unmet and close
        if 'eat_cow' not in b.ach and c and c.dist <= 6 and \
                damage(inv) >= 2:
            tgt = c.exact or min(c.cells, key=lambda x: mdist(x, b.pos))
            a = self.goto_and([tgt], 'do', allow_mine=False)
            if a:
                return a, 'hunt_cow'
        # top up drink when adjacent to water and not full
        if inv.get('drink', 9) <= 7:
            for d in D4:
                cell = (b.pos[0] + d[0], b.pos[1] + d[1])
                if b.mat(cell) == 'water':
                    a = self._face_or(cell, 'do')
                    if a:
                        return a, 'sip_water'
        return None

    # helpers for goal compilation -------------------------------------
    def _wood_needed(self):
        b = self.b
        n = 0
        if 'place_table' not in b.ach:
            n += 2
        for t in ('wood_pickaxe', 'wood_sword', 'stone_pickaxe',
                  'stone_sword', 'iron_pickaxe', 'iron_sword'):
            if f'make_{t}' not in b.ach:
                n += 1
        return min(9, max(n, 1) + 3)   # buffer: table cork (2) + spare

    def _stone_needed(self):
        b = self.b
        n = 0
        if 'make_stone_pickaxe' not in b.ach:
            n += 1
        if 'make_stone_sword' not in b.ach:
            n += 1
        if 'place_furnace' not in b.ach:
            n += 4
        if 'place_stone' not in b.ach:
            n += 1
        n += 2                          # shelter budget
        return min(9, n)

    def _coal_needed(self):
        b = self.b
        return (('make_iron_pickaxe' not in b.ach) +
                ('make_iron_sword' not in b.ach))

    def _iron_needed(self):
        b = self.b
        return (('make_iron_pickaxe' not in b.ach) +
                ('make_iron_sword' not in b.ach))

    def _place_on_ground(self, place_action, where=None):
        """Place on the FACED cell when it qualifies (facing follows
        movement in crafter, so 'turn then place' walks instead of turning
        when the target is walkable — the faced cell is the only reliably
        placeable one). Otherwise walk; the walk changes the faced cell and
        the check re-fires next step."""
        b = self.b
        where = where or {'grass', 'sand', 'path'}
        f = (b.pos[0] + b.facing[0], b.pos[1] + b.facing[1])
        if b.mat(f) in where and b.obj_at(f) is None and \
                not (place_action == 'place_plant' and f == b.plant_pos):
            return place_action
        # step so that we end up facing a qualifying cell
        for d in D4:
            c = (b.pos[0] + d[0], b.pos[1] + d[1])
            m = b.mat(c)
            if (m in where or m in WALKABLE or m == 'floor') and \
                    b.obj_at(c) is None and self._lava_safe(c):
                return DIR_ACTION[d]
        tgt = b.nearest_material(where)
        if tgt:
            return self.goto_and([tgt], place_action, allow_mine=False)
        return None

    def _at_station(self, stations, make_action):
        """Stand within 3x3 of the required station materials, then make.
        Arrows destroy tables/furnaces: if the last make attempt changed
        nothing, face the believed station so the served report corrects a
        phantom map cell (v1.1 bugfix; 42-step craft churn observed)."""
        b = self.b
        if b.last_action == make_action and b.prev_inventory == b.inventory:
            for s in stations:
                cell = b.nearest_material({s})
                if cell and mdist(b.pos, cell) == 1:
                    d = self._dir_to(cell)
                    if d and tuple(b.facing) != d:
                        return DIR_ACTION[d]
        near = {b.mat((b.pos[0] + dx, b.pos[1] + dy))
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        if all(s in near for s in stations):
            return make_action
        missing = [s for s in stations if s not in near]
        # find a known cell of the missing station
        cells = []
        for s in missing:
            c = b.nearest_material({s})
            if c:
                cells.append(c)
        if len(cells) == len(missing) and cells:
            # navigate adjacent to the first missing station
            a = self.goto_and(cells, make_action, allow_mine=True)
            if a:
                return a
        # need to build the station
        if 'table' in missing and b.inventory.get('wood', 0) >= 2:
            return self._place_on_ground('place_table')
        if 'furnace' in missing and b.inventory.get('stone', 0) >= 4:
            return self._furnace_by_table()
        return None

    furnace_tries = 0

    def _furnace_by_table(self):
        """Place furnace so table+furnace share a 3x3 with a standing cell:
        stand adjacent to the table, place furnace on another neighbor.
        v1.1: after 25 stuck attempts place it anywhere legal (orbit churn
        observed; a fresh table near it can come later, wood is renewable)."""
        b = self.b
        self.furnace_tries += 1
        if self.furnace_tries > 25:
            return self._place_on_ground('place_furnace',
                                         where={'grass', 'sand', 'path'})
        table = b.nearest_material({'table'})
        if table is None:
            if b.inventory.get('wood', 0) >= 2:
                return self._place_on_ground('place_table')
            return None
        if mdist(b.pos, table) <= 1 or (abs(b.pos[0] - table[0]) <= 1 and
                                        abs(b.pos[1] - table[1]) <= 1):
            return self._place_on_ground('place_furnace',
                                         where={'grass', 'sand', 'path'})
        return self.goto_and([table], 'noop', allow_mine=True)

    # ── exploration ────────────────────────────────────────────────────────
    def _explore(self):
        b = self.b
        inv = b.inventory
        # what are we looking for?
        wants = []
        if inv.get('wood_pickaxe', 0) > 0 and (
                inv.get('stone', 0) < self._stone_needed() or
                self._coal_needed() or self._iron_needed() or
                ('collect_diamond' not in b.ach and
                 inv.get('iron_pickaxe', 0) > 0)):
            wants = ['mountain']
        if self.privileged:
            # full map known: nothing to explore; walk toward remaining ore
            for matset in (('diamond',), ('iron',), ('coal',), ('stone',)):
                tgt = b.nearest_material(set(matset))
                if tgt and wants:
                    a = self.goto_and([tgt], 'do')
                    if a:
                        return a, f'goto_{matset[0]}'
            # otherwise roam toward cows/trees
            tgt = b.nearest_material({'tree'})
            if tgt:
                a = self.goto_and([tgt], 'noop')
                if a and a != 'noop':
                    return a, 'roam'
            return None
        # honest: frontier exploration
        if self.explore_target and b.mat(self.explore_target) is None and \
                mdist(b.pos, self.explore_target) > 1:
            a = self._explore_step(self.explore_target)
            if a:
                return a, 'explore'
        tgt = self._pick_frontier(prefer_mountain=bool(wants))
        if tgt:
            self.explore_target = tgt
            a = self._explore_step(tgt)
            if a:
                return a, 'explore'
        return None

    def _explore_step(self, tgt):
        a = self.goto_and([tgt], 'noop', allow_mine=True)
        if a and a != 'noop':
            return a
        self.explore_target = None
        # arrival/failure: never idle — take any safe step, cycling by t
        for i in range(4):
            d = D4[(self.b.t + i) % 4]
            c = (self.b.pos[0] + d[0], self.b.pos[1] + d[1])
            m = self.b.mat(c)
            if (m in WALKABLE or m == 'floor' or m is None) and \
                    self.b.obj_at(c) is None and self._lava_safe(c):
                return DIR_ACTION[d]
        return None

    def _pick_frontier(self, prefer_mountain=False):
        """Nearest unknown cell adjacent to known terrain (BFS ring); when
        hunting ores, prefer unknown cells adjacent to stone/path."""
        b = self.b
        from collections import deque
        start = tuple(b.pos)
        seen = {start}
        q = deque([start])
        fallback = None
        while q:
            cur = q.popleft()
            for d in D4:
                nxt = (cur[0] + d[0], cur[1] + d[1])
                if nxt in seen:
                    continue
                if not (MARGIN <= nxt[0] < WORLD_AREA[0] - MARGIN and
                        MARGIN <= nxt[1] < WORLD_AREA[1] - MARGIN):
                    continue
                seen.add(nxt)
                m = b.mat(nxt)
                if m is None:
                    if not prefer_mountain:
                        return nxt
                    if b.mat(cur) in ('stone', 'path', 'coal', 'iron', 'lava'):
                        return nxt
                    if fallback is None:
                        fallback = nxt
                elif m in WALKABLE or m == 'floor' or m in MINE_TOOL:
                    q.append(nxt)
            if len(seen) > 3500:
                break
        return fallback
