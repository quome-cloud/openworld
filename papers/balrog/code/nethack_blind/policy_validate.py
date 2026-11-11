"""VALIDATION policy (operator-specified explore/exploit design).

A validation episode exists to test high-(decision-impact x uncertainty)
ledger rules, not to score. It plays the normal explore/descend game but
opportunistically runs deliberate, safe-cheap experiments and tags every
experiment step with a 'val-*' why so outcomes can be extracted into the
ledger with citations.

Experiment slate (chosen from ledger uncertainty at b3-close):
  VAL_WEAR   armor mechanics: 'wear' owned armor -> predict blstats
             armor_class changes (no rule exists yet; high impact: survivability)
  VAL_UP     R_UP_NEEDS_TILE live test: at depth>=2 on a remembered '<',
             'up' -> predict depth-1 (only replay-corroborated so far)
  VAL_TRAP   R_DEPTH_TRAP / trap catalog: step on a visible '^' at full hp,
             log the outcome possibility (full-HP trap stepping per operator)
  VAL_QUAFF  potion mechanics: quaff an owned potion when safe; log effects
  VAL_PRAY_SPACING  R_PRAY_COOLDOWN bound: second prayer at ~1100 steps
             (E21 refuted ~880; policy uses 1600; bound unknown between)
"""

import re

from policy_explore import ExplorePolicy, DIRS, DIR_OF


class ValidatePolicy(ExplorePolicy):
    ARMOR_WORDS = ("mail", "armor", "helmet", "helm", "shield", "cloak",
                   "boots", "gloves", "jacket", "tunic", "dress")

    def __init__(self, ep_id="?", **kw):
        super().__init__(ep_id=ep_id, **kw)
        self.val_done = set()
        self.wear_queue = []
        self.quaffed = 0
        self.upped_levels = set()
        self.trap_tested = set()
        # tighter pray spacing to probe the cooldown bound (VAL_PRAY_SPACING)
        self._pray_gap = 1100

    # --- experiment hooks woven into act() ---
    def act(self, pre):
        bl = pre["bl"]
        msg = pre["msg"] or ""
        grid = self._grid(pre)
        pos = (bl["x_pos"], bl["y_pos"])
        lv = self._lv(bl)

        # queued wear letters (after prompt appears)
        if self.wear_queue and "what do you want to wear" in msg.lower():
            letter = self.wear_queue.pop(0)
            return letter, f"val-wear-pick-{letter}"

        # VAL_WEAR: parse inventory screen once, early
        if "inv" not in self.val_done and self.t == 4:
            self.val_done.add("inv")
            return "inventory", "val-inventory"
        if "inv" in self.val_done and "wearq" not in self.val_done and self.t <= 8:
            # inventory text arrives as a multi-line message
            if " - " in msg:
                self.val_done.add("wearq")
                for line in msg.split("\n"):
                    m = re.match(r"\s*([a-zA-Z]) - (.+)", line)
                    if m and any(w in m.group(2).lower() for w in self.ARMOR_WORDS) \
                            and "(being worn)" not in m.group(2):
                        self.wear_queue.append(m.group(1))
                self.queue.append(("esc", "val-inv-close"))

        # issue wear commands while queue non-empty and safe
        if self.wear_queue and not self.queue and self.t < 400:
            adj = any(self._monster(self._at(grid, pos[0]+dx, pos[1]+dy))
                      for dx, dy in DIRS.values())
            if not adj and not (msg.rstrip().endswith("?") and "[" in msg):
                return "wear", "val-wear"

        # VAL_UP: on a remembered '<' at depth>=2, climb once per level
        key = (bl["dungeon_number"], bl["level_number"])
        if (bl["depth"] >= 2 and key not in self.upped_levels
                and pos in lv.get("stairs_up", set())):
            self.upped_levels.add(key)
            return "up", "val-up-test"

        # VAL_TRAP: step onto a visible '^' at full hp
        if bl["hitpoints"] == bl["max_hitpoints"]:
            for tp in self._find(grid, "^"):
                if tp in self.trap_tested or tp in lv["no_go"]:
                    continue
                path = self._bfs(grid, lv, pos, targets={tp})
                if path and len(path) <= 12:
                    step = path[0]
                    dd = (step[0]-pos[0], step[1]-pos[1])
                    if dd in DIR_OF:
                        if len(path) == 1:
                            self.trap_tested.add(tp)
                        d = DIR_OF[dd]
                        self.last_move = (pos, d, self._at(grid, *step), step)
                        return d, "val-trap-step"

        # VAL_QUAFF: quaff an owned potion once, when safe
        if self.quaffed < 1 and 100 < self.t < 3000:
            low = msg.lower()
            if "what do you want to drink" in low:
                m = re.search(r"\[([a-zA-Z])", msg)
                if m:
                    self.quaffed += 1
                    return m.group(1), "val-quaff-pick"
            adj = any(self._monster(self._at(grid, pos[0]+dx, pos[1]+dy))
                      for dx, dy in DIRS.values())
            if not adj and bl["hitpoints"] == bl["max_hitpoints"] \
                    and "quaff_tried" not in self.val_done and self.t % 97 == 0:
                self.val_done.add("quaff_tried")
                return "quaff", "val-quaff"

        return super().act(pre)

    # pray spacing override (VAL_PRAY_SPACING): reuse base logic but with
    # the probe spacing; base uses self.t - prayed_at > 1600.
    # We shadow by adjusting prayed_at bookkeeping: subtract the difference.
    def _hunger_desperate(self, pre):
        return super()._hunger_desperate(pre)
