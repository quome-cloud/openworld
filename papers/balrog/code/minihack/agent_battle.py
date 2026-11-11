"""CorridorBattle-Dark agent: chokepoint tactics.

Task-scoped model (from minihack/envs/fightcorridor.py + probes):
  - Fixed template: small start room -- 1-wide corridor -- large room with
    6 giant rats near its east end; stairs-down 1-2 cols further east on the
    corridor row. Character is FIXED: lawful female human knight (HP~16,
    speed 12, strong melee). Level is fully DARK: radius-1 visibility.
  - Rats swarm in the open (3+ simultaneous attackers kills the knight);
    in the 1-wide corridor mouth only one rat can engage at a time.
  - Zero-time trick discovered in probes: bumping a wall consumes an env
    step but NO game time, so the world freezes -- useless for waiting.
    Waiting for rats therefore = oscillating between two corridor cells.

Policy: walk east along the corridor to its mouth; hold there oscillating,
killing whatever arrives, until 6 kills or no contact for `quiet_limit`
world-turns; then sweep east along the room's middle row to the stairs,
retreating to the chokepoint if >=2 hostiles engage at once.
"""

import mh_common as C
from agent_explore import ExploreAgent


class BattleAgent(ExploreAgent):
    def __init__(self, **kw):
        kw.setdefault("prefer_dir", "east")
        super().__init__(**kw)
        self.kills = 0
        self.quiet = 0
        self.quiet_limit = 25
        self.retreats = 0
        self.hold_cell = None
        self.osc = False

    def act(self, obs):
        # count kills before ExploreAgent consumes the message
        msg = C.extract_message(obs)
        low = msg.lower()
        self.kills += low.count("you kill")
        self.kills += low.count("you destroy")
        return super().act(obs)

    def _decide(self, obs):
        L = self.level
        ax, ay = L.agent

        # stairs seen? finish (with win-step guard) -- delegate to explorer
        if L.find_terrain(C.STAIRS_DOWN):
            return super()._decide(obs)

        hostiles = [(mx, my, ch) for (mx, my, ch, pet) in L.monsters if not pet]
        adj = [h for h in hostiles if max(abs(h[0] - ax), abs(h[1] - ay)) == 1]

        # memory effect E2: cross-episode combat statistics inform a
        # hold-vs-dash decision. If remaining expected melee damage exceeds
        # our HP budget, abandon the war of attrition and run for the
        # remembered stairs instead of finishing all 6 rats.
        if self.memory is not None and not getattr(self, "_dashing", False):
            dpk = self.memory.dmg_per_kill()
            if dpk is not None and self.kills >= 2:
                expected = dpk * (6 - self.kills)
                if L.hp < 0.85 * expected:
                    self._dashing = True
                    self.quiet = self.quiet_limit  # ends the holding phase
                    note = (f"E2 dash: hp={L.hp} < 0.85*expected remaining "
                            f"damage {expected:.1f} (dpk={dpk:.2f}, "
                            f"kills={self.kills})")
                    self.memory.record_fired(note)
                    self.log("  MEM-FIRED " + note)

        # locate chokepoint: easternmost corridor cell with room floor east
        if self.hold_cell is None:
            best = None
            for (x, y) in zip(*self._corridor_cells(L)):
                if L.terrain[y][x + 1] == C.FLOOR:
                    if best is None or x > best[0]:
                        best = (x, y)
            if best:
                self.hold_cell = best

        # 1. fight anything adjacent (at chokepoint it's 1v1)
        if adj:
            self.quiet = 0
            if len(adj) >= 2 and (ax, ay) != self.hold_cell and \
               self.hold_cell is not None and self.retreats < 3:
                # disengage toward chokepoint
                path = L.bfs(L.agent, [self.hold_cell])
                if path:
                    self.retreats += 1
                    return path[0]
            # attack: prefer cardinal targets
            adj.sort(key=lambda h: (abs(h[0] - ax) + abs(h[1] - ay)))
            for (mx, my, ch) in adj:
                d = (mx - ax, my - ay)
                if d in C.DIR_OF:
                    return C.DIR_OF[d]

        # 2. holding phase: oscillate at the chokepoint until quiet
        if self.kills < 6 and self.hold_cell is not None and \
           self.quiet < self.quiet_limit and self.retreats < 3:
            self.quiet = 0 if hostiles else self.quiet + 1
            if (ax, ay) != self.hold_cell:
                path = L.bfs(L.agent, [self.hold_cell],
                             avoid={(m[0], m[1]) for m in L.monsters})
                if path:
                    return path[0]
            # at hold cell: oscillate west<->east-of-west to pass world time
            self.osc = not self.osc
            if self.osc:
                # step west if passable, else any passable neighbor not east
                for name in ("west", "northwest", "southwest"):
                    dx, dy = C.DIRS[name]
                    if L.passable(ax + dx, ay + dy):
                        return name
            else:
                return "east" if L.passable(ax + 1, ay) else "west"

        # 3. quiet (or kills done): sweep east / explore for the stairs
        return super()._decide(obs)

    @staticmethod
    def _corridor_cells(L):
        import numpy as np
        ys, xs = np.where(L.terrain == C.CORRIDOR)
        return xs, ys
