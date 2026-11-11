"""DEV-ONLY privileged belief (reads env internals). Used by offline
model validation and development diagnostics. NEVER imported by the scored
runner (run_suite.py imports TextBelief only) — see the source-leak audit
in FABLE_CRAFTER_REPORT.md."""

import numpy as np

from belief import BaseBelief, Report, MOB_KINDS


class PrivilegedBelief(BaseBelief):
    """Reads env internals every step (diagnostic protocol)."""

    def __init__(self, env):
        super().__init__()
        self.env = env                 # the raw crafter.Env

    def update(self):
        env = self.env
        p = env._player
        self.t = env._step
        self.pos = tuple(int(v) for v in p.pos)
        self.facing = tuple(int(v) for v in p.facing)
        self.inventory = dict(p.inventory)
        self.sleeping = p.sleeping
        self.dead = p.health <= 0
        # full material map
        mat_map = env._world._mat_map
        names = env._world._mat_names
        flat = self.map
        for m_id, name in names.items():
            flat[mat_map == m_id] = name
        # objects -> closest-per-kind reports (exact) + full lists
        self.obj_list = []
        best = {}
        for obj in env._world.objects:
            kind = type(obj).__name__.lower()
            if kind == 'player':
                continue
            opos = tuple(int(v) for v in obj.pos)
            d = abs(opos[0] - self.pos[0]) + abs(opos[1] - self.pos[1])
            self.obj_list.append((kind, opos, getattr(obj, 'health', 0),
                                  getattr(obj, 'ripe', False)))
            if kind in MOB_KINDS and (kind not in best or d < best[kind][0]):
                best[kind] = (d, opos)
        self.reports = {k: Report(k, d, [c]) for k, (d, c) in best.items()}
        self.ach = {k for k, v in p.achievements.items() if v > 0}
        if self.plant_pos is not None:
            alive = any(k == 'plant' and c == self.plant_pos
                        for k, c, _, _ in self.obj_list)
            if not alive:
                self.plant_pos, self.plant_age = None, 0
        self.tick_plant()

    def obj_at(self, p):
        for kind, opos, hp, ripe in self.obj_list:
            if opos == tuple(p):
                return dict(kind=kind, health=hp, ripe=ripe)
        return None

    def threat_cells(self, kind):
        return [c for k, c, _, _ in self.obj_list if k == kind]

    def nearest_material(self, mats, limit=None):
        """Nearest cell holding any material in `mats` (exact map)."""
        best, bd = None, 10 ** 9
        xs, ys = np.where(np.isin(self.map, list(mats)))
        for x, y in zip(xs, ys):
            d = abs(int(x) - self.pos[0]) + abs(int(y) - self.pos[1])
            if d < bd:
                best, bd = (int(x), int(y)), d
        if limit is not None and bd > limit:
            return None
        return best


