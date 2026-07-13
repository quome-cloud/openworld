"""Belief-state providers for the two protocols.

PrivilegedBelief — diagnostic arm: reads env internals (full 64x64 material
map, exact object list, player state). Mirrors the Baba arm's privileged
protocol.

TextBelief — honest arm: sees ONLY the BALROG text observation
(obs["text"]["long_term_context"] + ["short_term_context"]) and knows the
published observation format + the synthesized world model. It maintains:
  - dead-reckoned absolute position (spawn is always the area centre (32,32),
    a constant of the environment, cf. crafter env.py reset()),
  - an incrementally built material map from exact-position facts,
  - closest-per-type mob/terrain reports for the current step,
  - a 2-hypothesis position tracker for the rare ambiguous moves.

Key exactness lemma used throughout (from the wrapper source, env.py
describe_env): the text reports, for every item type present in the 9x7
window, the *closest* instance with its Manhattan distance and a signed
direction class. Therefore, for any adjacent cell c (distance 1):
  material(c) ∈ {t : report_t.dist == 1} — because if material(c)=t then the
  closest t is at distance 1. Distance-1 reports are always pure-cardinal,
  hence exact cells. Objects at distance 1 are likewise exactly localized.
This makes adjacent-cell walkability (and lava danger) certain whenever the
distance-1 report set is unambiguous, which is what the executor relies on.
"""

import re
import numpy as np

from crafter_model import (
    MATERIALS, WALKABLE, PLAYER_WALKABLE, DIRS, SEM_OBJECTS, ID_TO_ITEM,
    UPDATE_RADIUS, SPAWN, WORLD_AREA, VIEW_X, VIEW_Y, daylight)

MOB_KINDS = ('cow', 'zombie', 'skeleton', 'arrow', 'plant')
MATERIAL_NAMES = set(m for m in MATERIALS if m) | {'None'}
VITALS = ('health', 'food', 'drink', 'energy')


class Report:
    """One closest-per-type line: distance + candidate cells (absolute)."""
    __slots__ = ('kind', 'dist', 'cells')

    def __init__(self, kind, dist, cells):
        self.kind, self.dist, self.cells = kind, dist, cells

    @property
    def exact(self):
        return self.cells[0] if len(self.cells) == 1 else None


class BaseBelief:
    def __init__(self):
        self.t = 1                     # env step counter (reset does 1 noop)
        self.pos = SPAWN
        self.facing = (0, 1)
        self.inventory = {}
        self.sleeping = False
        self.dead = False
        self.map = np.zeros(WORLD_AREA, dtype=object)  # material str or None
        self.map[:] = None
        self.reports = {}              # kind -> Report (this step)
        self.ach = set()               # believed achievements
        self.plant_pos = None
        self.plant_age = 0
        self.prev_inventory = None
        self.notes = []                # diagnostics

    # -- helpers shared by the brain --
    def mat(self, p):
        x, y = p
        if not (0 <= x < WORLD_AREA[0] and 0 <= y < WORLD_AREA[1]):
            return 'None'
        return self.map[x, y]

    def daylight(self):
        return daylight(self.t)

    def report(self, kind):
        return self.reports.get(kind)

    def mob_dist(self, kind):
        r = self.reports.get(kind)
        return r.dist if r else 99

    def tick_plant(self):
        if self.plant_pos is not None:
            d = abs(self.pos[0] - self.plant_pos[0]) + abs(self.pos[1] - self.plant_pos[1])
            if d < UPDATE_RADIUS - 1:   # margin of 1 on the exact <18 rule
                self.plant_age += 1


LINE_RE = re.compile(
    r'^- (?P<kind>[A-Za-z_]+) (?P<dist>\d+) steps? to your (?P<dir>[a-z-]+)$')
FACE_RE = re.compile(r'^You face (?P<kind>[A-Za-z_]+) at your front\.$')
INV_RE = re.compile(r'^- (?P<item>[a-z_]+): (?P<n>\d+)(?:/9)?$')


def _candidates(pos, dist, dirword):
    """Absolute candidate cells for a (dist, direction-class) report,
    restricted to the 9x7 window."""
    parts = dirword.split('-')
    ns = next((p for p in parts if p in ('north', 'south')), None)
    ew = next((p for p in parts if p in ('west', 'east')), None)
    out = []
    for dx in range(-VIEW_X, VIEW_X + 1):
        for dy in range(-VIEW_Y, VIEW_Y + 1):
            if abs(dx) + abs(dy) != dist:
                continue
            if ns == 'north' and dy >= 0:
                continue
            if ns == 'south' and dy <= 0:
                continue
            if ns is None and dy != 0:
                continue
            if ew == 'west' and dx >= 0:
                continue
            if ew == 'east' and dx <= 0:
                continue
            if ew is None and dx != 0:
                continue
            out.append((pos[0] + dx, pos[1] + dy))
    return out


class Hypothesis:
    __slots__ = ('pos', 'score', 'writes')

    def __init__(self, pos):
        self.pos = pos
        self.score = 0
        self.writes = []               # buffered (cell, material) facts


class TextBelief(BaseBelief):
    """Parses BALROG text obs; dead-reckons position; builds map."""

    def __init__(self):
        super().__init__()
        self.hyps = [Hypothesis(SPAWN)]
        self.last_action = None
        self.faced_kind = None         # raw faced-report token
        self.ambiguous_steps = 0
        self.relocalizations = 0
        self.sapling_tries = 0
        self.slept_this_night = False
        self._pending_kill = {}        # kind -> damage dealt to adjacent mob
        self.ore_hints = {}            # material -> set of candidate cells

    # ---- parsing ----
    def observe(self, long_text, short_text, action_taken):
        """Integrate one observation. action_taken = crafter action string we
        emitted last step (None on reset)."""
        self.last_action = action_taken
        raw_reports = {}
        faced = None
        sleeping = 'You are sleeping' in long_text
        self.dead = 'You died' in long_text
        for line in long_text.splitlines():
            line = line.strip()
            m = LINE_RE.match(line)
            if m:
                raw_reports[m['kind']] = (int(m['dist']), m['dir'])
                continue
            m = FACE_RE.match(line)
            if m:
                faced = m['kind']
        inv = {}
        for line in short_text.splitlines():
            m = INV_RE.match(line.strip())
            if m:
                inv[m['item']] = int(m['n'])
        for v in VITALS:
            inv.setdefault(v, 0)
        prev_inv = self.inventory
        self.prev_inventory = prev_inv
        self.inventory = inv
        was_sleeping = self.sleeping
        self.sleeping = sleeping

        # ---- dead reckoning ----
        if action_taken is not None:
            self.t += 1
            self._advance_hypotheses(action_taken, was_sleeping)
        # score hypotheses against this obs, using type-absence + faced cell
        self._score_and_collapse(raw_reports, faced)

        self.pos = self.hyps[0].pos
        # materialize reports (absolute cells) under leading hypothesis
        self.faced_prev = self.faced_kind
        self.reports = {}
        for kind, (dist, dirword) in raw_reports.items():
            cells = _candidates(self.pos, dist, dirword)
            # drop candidates contradicted by known map (for materials)
            if kind in MATERIAL_NAMES:
                sharpened = [c for c in cells if self.mat(c) in (None, kind)]
                if sharpened:
                    cells = sharpened
            self.reports[kind] = Report(kind, dist, cells)
        self.faced_kind = faced

        # ---- map writes (only when unambiguous position) ----
        if len(self.hyps) == 1:
            self._write_map(faced)

        # ---- inventory-delta achievement inference ----
        self._infer_achievements(prev_inv, inv, action_taken, was_sleeping)
        self.tick_plant()

    def _advance_hypotheses(self, action, was_sleeping):
        if was_sleeping or not action.startswith('move_'):
            return
        d = {'move_left': (-1, 0), 'move_right': (1, 0),
             'move_up': (0, -1), 'move_down': (0, 1)}[action]
        self.facing = d
        new = []
        for h in self.hyps:
            tgt = (h.pos[0] + d[0], h.pos[1] + d[1])
            known = self.mat(tgt)
            blocked_by_obj = self._obj_at_from_reports(h.pos, tgt)
            if blocked_by_obj:
                new.append(h)          # object blocks: stayed
            elif known is not None:
                if known in PLAYER_WALKABLE:
                    h.pos = tgt
                new.append(h)
            else:
                # unknown material: candidates from last step's dist-1 set
                cands = self._dist1_types(h.pos, d)
                walk = [c in PLAYER_WALKABLE for c in cands] if cands else []
                if walk and all(walk):
                    h.pos = tgt
                    new.append(h)
                elif walk and not any(walk):
                    new.append(h)
                else:
                    self.ambiguous_steps += 1
                    h2 = Hypothesis(tgt)
                    h2.score = h.score
                    h2.writes = list(h.writes)
                    new.append(h)      # stayed
                    new.append(h2)     # moved
        # dedupe by pos
        seen, dedup = set(), []
        for h in new:
            if h.pos not in seen:
                seen.add(h.pos)
                dedup.append(h)
        self.hyps = dedup[:4]

    def _dist1_types(self, pos, d):
        """Possible materials of pos+d per the CURRENT (pre-move) reports."""
        out = set()
        for kind, r in self.reports.items():
            if kind in MATERIAL_NAMES and r.dist == 1:
                out.add(kind)
        known = self.mat((pos[0] + d[0], pos[1] + d[1]))
        if known is not None:
            return {known}
        return out

    def _obj_at_from_reports(self, pos, tgt):
        for kind in MOB_KINDS:
            r = self.reports.get(kind)
            if r and r.dist == 1 and r.exact == tgt:
                return True
        return False

    def _score_and_collapse(self, raw_reports, faced):
        if len(self.hyps) > 1:
            for h in self.hyps:
                h.score += self._consistency(h.pos, raw_reports, faced)
            self.hyps.sort(key=lambda h: -h.score)
            if (self.hyps[0].score - self.hyps[1].score >= 2 or
                    len(self.hyps) > 2):
                self.hyps = [self.hyps[0]]
            # hard cap: never carry ambiguity longer than 5 steps
            elif self.ambiguous_steps % 5 == 0:
                self.hyps = [self.hyps[0]]
        if len(self.hyps) == 1 and self.hyps[0].writes:
            for cell, matname in self.hyps[0].writes:
                self._set_map(cell, matname)
            self.hyps[0].writes = []

    def _consistency(self, pos, raw_reports, faced):
        score = 0
        # faced-cell check
        f = (pos[0] + self.facing[0], pos[1] + self.facing[1])
        known = self.mat(f)
        if faced and faced in MATERIAL_NAMES and known is not None \
                and known != 'floor':
            score += 1 if known == faced else -2
        # absence check: map shows material t in window but no report for t
        x0, x1 = pos[0] - VIEW_X, pos[0] + VIEW_X
        y0, y1 = pos[1] - VIEW_Y, pos[1] + VIEW_Y
        for t in ('water', 'stone', 'tree', 'sand'):
            in_window = False
            for x in range(max(0, x0), min(WORLD_AREA[0], x1 + 1)):
                for y in range(max(0, y0), min(WORLD_AREA[1], y1 + 1)):
                    if self.map[x, y] == t:
                        in_window = True
                        break
                if in_window:
                    break
            if in_window and t not in raw_reports:
                score -= 1
            # distance check
            r = raw_reports.get(t)
            if r and in_window:
                dmin = min(abs(x - pos[0]) + abs(y - pos[1])
                           for x in range(max(0, x0), min(WORLD_AREA[0], x1 + 1))
                           for y in range(max(0, y0), min(WORLD_AREA[1], y1 + 1))
                           if self.map[x, y] == t)
                if r[0] > dmin:
                    score -= 1         # report farther than a known instance
        return score

    def _set_map(self, cell, matname):
        x, y = cell
        if 0 <= x < WORLD_AREA[0] and 0 <= y < WORLD_AREA[1]:
            if matname in MATERIAL_NAMES and matname != 'None':
                self.map[x, y] = matname

    def _write_map(self, faced):
        pos = self.pos
        writes = []
        if faced and faced in MATERIAL_NAMES and faced != 'None':
            f = (pos[0] + self.facing[0], pos[1] + self.facing[1])
            writes.append((f, faced))
        # standing cell is walkable; if unknown mark as 'path'-like ONLY via
        # weaker info — we know it's in WALKABLE but not which; leave unknown
        # unless a report pins it. (Walkability of visited cells is tracked
        # separately via visited set.)
        for kind, r in self.reports.items():
            if kind not in MATERIAL_NAMES or kind == 'None':
                continue
            if r.exact is not None:
                writes.append((r.exact, kind))
            elif kind in ('iron', 'diamond', 'coal'):
                self.ore_hints.setdefault(kind, set()).update(r.cells)
        for cell, m in writes:
            self._set_map(cell, m)
        # confirmed ore cells clear hints
        for kind in ('iron', 'diamond', 'coal'):
            if kind in self.ore_hints:
                self.ore_hints[kind] = {
                    c for c in self.ore_hints[kind]
                    if self.mat(c) in (None, kind)}
        # standing cell is walkable but of unknown type: mark as 'floor'
        # (a pseudo-material meaning walkable-unknown; never a placement or
        # collection target, but fully usable for pathfinding)
        x, y = pos
        if self.map[x, y] is None:
            self.map[x, y] = 'floor'

    def _infer_achievements(self, prev, inv, action, was_sleeping):
        if not prev or action is None:
            return
        def delta(k):
            return inv.get(k, 0) - prev.get(k, 0)
        gains = {
            'wood': 'collect_wood', 'stone': 'collect_stone',
            'coal': 'collect_coal', 'iron': 'collect_iron',
            'diamond': 'collect_diamond', 'sapling': 'collect_sapling',
            'drink': 'collect_drink',
            'wood_pickaxe': 'make_wood_pickaxe',
            'stone_pickaxe': 'make_stone_pickaxe',
            'iron_pickaxe': 'make_iron_pickaxe',
            'wood_sword': 'make_wood_sword',
            'stone_sword': 'make_stone_sword',
            'iron_sword': 'make_iron_sword'}
        for item, achv in gains.items():
            if delta(item) > 0:
                if item == 'drink':
                    if action == 'do' and self.faced_prev == 'water':
                        self.ach.add(achv)
                else:
                    self.ach.add(achv)
        if action == 'place_table' and delta('wood') <= -2:
            self.ach.add('place_table')
        if action == 'place_stone' and delta('stone') == -1:
            self.ach.add('place_stone')
        if action == 'place_furnace' and delta('stone') <= -4:
            self.ach.add('place_furnace')
        if action == 'place_plant' and delta('sapling') == -1:
            self.ach.add('place_plant')
            f = (self.pos[0] + self.facing[0], self.pos[1] + self.facing[1])
            self.plant_pos, self.plant_age = f, 0
        if action == 'do' and delta('food') >= 5:
            self.ach.add('eat_cow')
        if action == 'do' and 3 <= delta('food') <= 4:
            self.ach.add('eat_plant')
            if self.plant_pos and self.faced_prev == 'plant':
                self.plant_age = 0     # eating resets grown (renewable farm)
        if was_sleeping and not self.sleeping and inv.get('energy', 0) >= 9 \
                and not self.dead:
            self.ach.add('wake_up')
        # combat kill inference happens in the brain (it knows hit counts)

    # brain compatibility helpers
    faced_prev = None

    def threat_cells(self, kind):
        r = self.reports.get(kind)
        return list(r.cells) if r else []

    def obj_at(self, p):
        p = tuple(p)
        for kind in MOB_KINDS:
            r = self.reports.get(kind)
            if r and r.dist == 1 and r.exact == p:
                return dict(kind=kind, health=None,
                            ripe=(self.plant_age > 310 if kind == 'plant'
                                  else False))
        return None

    def nearest_material(self, mats, limit=None):
        best, bd = None, 10 ** 9
        # current reports first (exact or nearest candidate)
        for m in mats:
            r = self.reports.get(m)
            if r:
                c = r.exact or min(
                    r.cells, key=lambda c: abs(c[0] - self.pos[0]) + abs(c[1] - self.pos[1]))
                d = r.dist
                if d < bd:
                    best, bd = c, d
        xs, ys = np.where(np.isin(self.map, list(mats)))
        for x, y in zip(xs, ys):
            d = abs(int(x) - self.pos[0]) + abs(int(y) - self.pos[1])
            if d < bd:
                best, bd = (int(x), int(y)), d
        if best is None:
            for m in mats:
                for c in self.ore_hints.get(m, ()):
                    d = abs(c[0] - self.pos[0]) + abs(c[1] - self.pos[1])
                    if d < bd:
                        best, bd = c, d
        if limit is not None and bd > limit:
            return None
        return best
