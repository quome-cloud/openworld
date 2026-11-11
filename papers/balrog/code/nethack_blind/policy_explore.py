"""Explore-and-descend policy, v1. Every behavioral commitment cites a rule
in rules.json (R_*) or is flagged EXPERIMENT (hypothesis generation).

Tile passability is LEARNED: tiles_learned.json accumulates walk_ok /
walk_block counts per glyph with first-evidence citations. Nothing here is
taken from game source; glyph semantics enter only through logged outcomes.
"""

import json
import os
import re
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
TILES_PATH = os.path.join(HERE, "tiles_learned.json")

DIRS = {
    "north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0),
    "northeast": (1, -1), "southeast": (1, 1),
    "southwest": (-1, 1), "northwest": (-1, -1),
}
DIR_OF = {v: k for k, v in DIRS.items()}

BLOCK_MSGS = ("It's a wall", "It's solid stone")  # R_BLOCK_MSG


def load_tiles():
    if os.path.exists(TILES_PATH):
        with open(TILES_PATH) as f:
            return json.load(f)
    # bootstrap: '.' walked on in P0 steps 1-8 (R_MOVE evidence)
    return {".": {"walk_ok": 8, "walk_block": 0, "first_evidence": ["P0", 1]}}


def save_tiles(tiles):
    with open(TILES_PATH, "w") as f:
        json.dump(tiles, f, indent=1)


MONSTERS_PATH = os.path.join(HERE, "monsters_learned.json")


def load_monsters():
    if os.path.exists(MONSTERS_PATH):
        with open(MONSTERS_PATH) as f:
            return json.load(f)
    return {}


def save_monsters(m):
    with open(MONSTERS_PATH, "w") as f:
        json.dump(m, f, indent=1)


class ExplorePolicy:
    # glyphs we will TRY to walk into even with no walk_ok evidence yet
    # (EXPERIMENT: candidates seen on maps; outcomes recorded in tiles table)
    TRY_GLYPHS = set(".#+<>_{$[!?/=%(*)\"0^")

    def __init__(self, ep_id="?", pray_experiment=True):
        self.ep = ep_id
        self.tiles = load_tiles()
        self.monsters = load_monsters()
        pp = os.path.join(HERE, "poison_corpses.json")
        self.poison_names = (json.load(open(pp))["names"]
                             if os.path.exists(pp) else [])
        self._peaceful_target = None
        self._lv_cache = {"no_go": set()}
        self.kick_tries = {}
        self.kick_cooldown = -1
        self.queue = deque()
        self.level_mem = {}
        self.cur_level = None
        self.last_move = None      # (from_pos, dir, to_pos_expected)
        self.prayed_at = -9999
        self.pray_experiment = pray_experiment
        self.pending_eat = False
        self.search_count = 0
        self.stuck = 0
        self.t = 0
        self.pet_glyphs = set()        # R_SWAP_PET: learned from swap messages
        self.hostile_glyphs = set()    # R_COMBAT: glyphs seen attacking/being fought
        self.attack_tries = {}         # cell -> futile attack count (R_STATUE guard)

    # ---------- helpers ----------
    def _lv(self, bl):
        key = (bl["dungeon_number"], bl["level_number"])
        if key not in self.level_mem:
            self.level_mem[key] = {"edge_block": set(), "visited": set(),
                                   "searched": {}, "no_go": set(),
                                   "dark_dead": set(), "stairs_down": set()}
        self.cur_level = key
        return self.level_mem[key]

    def _grid(self, pre):
        # R_COORD: rows 1-21 are the map; map cell (x,y) = tty[y+1][x]
        return pre["tty"]

    def _at(self, grid, x, y):
        if 0 <= y + 1 < len(grid) and 0 <= x < len(grid[0]):
            return chr(grid[y + 1][x])
        return " "

    def _passable(self, ch, lv, x, y):
        if (x, y) in lv["no_go"]:
            return False
        t = self.tiles.get(ch)
        if t and t["walk_ok"] > 0 and not (t["walk_block"] > 3 * max(t["walk_ok"], 1)):
            return True
        if t and t["walk_ok"] == 0 and t["walk_block"] >= 3:
            return False
        # optimistic-until-refuted: any unseen glyph is a walk candidate
        # (E12: unknown '`' sealed the only corridor when whitelisted-only)
        return True

    def _hunger_desperate(self, pre):
        # R_HUNGER_SCALE (induced from own logs): blstats hunger_state
        # 1->2 with 'hungry' msg (168x), 2->3 with 'weak' (50x),
        # 3->4 with 'faint' (32x), ->0 after eating. >=3 is desperate.
        return pre["bl"]["hunger_state"] >= 3

    def _monster(self, ch):
        # R_MONSTERS: letters are creatures. '@' is us (R_COORD); other '@'s exist?
        return (ch.isalpha() and ch != "@") or ch in ":;'&~"

    # ---------- lifecycle ----------
    def reset(self, pre):
        self.queue.clear()
        self.t = 0
        self._hurt_recent = 0
        self._hp_prev = None

    def observe(self, pre, action, post, reward, done):
        self.t += 1
        lv = self._lv(post["bl"])
        pos = (post["bl"]["x_pos"], post["bl"]["y_pos"])
        lv["visited"].add(pos)
        ch = self._at(self._grid(post), *pos)
        msg = post["msg"] or ""
        # combat / pet learning from messages (R_SWAP_PET, R_COMBAT, R_STATUE)
        if "You swap places with" in msg and self.last_move:
            self.pet_glyphs.add(self.last_move[2])
            # E37: hp-drop override kept bumping the (provoked) pet -> death.
            # A swap identifies THIS creature as pet: exempt it from the
            # override for a while (it drifts, so track by recency not cell).
            self._pet_swap_t = self.t
        # R_FREEZE (E14:5833): some creatures freeze their attacker on contact
        if "frozen by" in msg.lower() and self.last_move and self._monster(self.last_move[2]):
            g = self.last_move[2]
            e = self.monsters.setdefault(g, {})
            e["no_attack"] = True
            e.setdefault("evidence", []).append([str(self.ep), self.t, msg[:60]])
        # fresh-kill tracking (anti-starvation: rot poisonings all came from
        # corpses of unknown age; a just-killed corpse is the freshest
        # available -> hypothesis: safe(r) to eat)
        if "You kill the" in msg and self.last_move:
            name = msg.split("You kill the", 1)[1].split("!")[0].strip()
            self._fresh_kill = (self.last_move[3], name, self.t)

        # R_KICK_DOOR / R_KICK_RISK (XKICK transcript)
        if "door is locked" in msg.lower() and self.last_move and self.t > self.kick_cooldown:
            frm, d, target_ch, tpos = self.last_move
            k = self.kick_tries.get(tpos, 0)
            if k < 6:
                self.kick_tries[tpos] = k + 1
                self.queue.append(("kick", "kick-locked-door"))
                self.queue.append((d, "kick-door-dir"))
        if "no shape for kicking" in msg.lower() or "strain a muscle" in msg.lower():
            self.kick_cooldown = self.t + 80
            self.queue.clear()
        for pat in ("bites", "hits", "misses", "kicks", "butts", "stings", "touches"):
            if f"{pat}!" in msg or f"{pat} you" in msg:
                # some creature is fighting us; mark adjacent letters hostile
                for d, (dx, dy) in DIRS.items():
                    ch = self._at(self._grid(post), pos[0] + dx, pos[1] + dy)
                    if self._monster(ch) and ch not in self.pet_glyphs:
                        self.hostile_glyphs.add(ch)

        # food-state bookkeeping (R_EAT_INV): successful eat or food pickup
        # means we have/had food again
        lowm = msg.lower()
        if ("hits the spot" in lowm or "finish eating" in lowm
                or ("ration" in lowm and " - " in msg)):
            self.no_food = False
        # E31 bug: pending_eat stuck True after "don't have anything to eat"
        # blocked corpse-eating forever while fainting on top of corpses.
        if "anything to eat" in lowm:
            self.pending_eat = False
            self._fresh_kill = None  # b3_E53: 211-step retry loop on a gone corpse
        if "tastes delicious" in lowm or "finish eating" in lowm or "rotten food" in lowm:
            self._fresh_kill = None
        if self.pending_eat and self.t - getattr(self, "_pending_eat_t", 0) > 3:
            self.pending_eat = False

        # harmful-trap memory: cell announced a trap AND cost hp -> don't
        # re-enter (darts/arrows). Depth-granting traps (trap doors/shafts,
        # R_DEPTH_TRAP) are NOT avoided - falling is free descent.
        if ("trap" in msg.lower()
                and post["bl"]["hitpoints"] < pre["bl"]["hitpoints"]
                and post["bl"]["depth"] == pre["bl"]["depth"]):
            lv["no_go"].add(pos)

        # movement outcome -> tile learning (evidence = this ep/step)
        if action in DIRS and self.last_move:
            frm, d, target_ch, tpos = self.last_move
            self._last_bump = tpos
            was_attack = self._monster(target_ch)
            prev = (pre["bl"]["x_pos"], pre["bl"]["y_pos"])
            entry = self.tiles.setdefault(
                target_ch, {"walk_ok": 0, "walk_block": 0,
                            "first_evidence": [str(self.ep), self.t]})
            if pos == tpos:
                if not was_attack:
                    entry["walk_ok"] += 1
                self.attack_tries.pop(tpos, None)
            elif pos == prev:
                if was_attack:
                    # attacking: never terrain-block (E2 pet bug); track futility
                    if ("statue" in msg.lower()
                            or (not msg and pre["bl"]["time"] == post["bl"]["time"])):
                        k = self.attack_tries.get(tpos, 0) + 1
                        self.attack_tries[tpos] = k
                        if k >= 3 or "statue" in msg.lower():
                            lv["no_go"].add(tpos)   # R_STATUE
                    elif any(w in msg for w in ("kill", "hit", "miss", "destroy")):
                        self.attack_tries.pop(tpos, None)
                elif any(m in msg for m in BLOCK_MSGS):
                    entry["walk_block"] += 1
                    lv["edge_block"].add((frm, d))
                    if target_ch == " ":
                        lv["dark_dead"].add(tpos)
                elif "closed" in msg.lower() and "door" in msg.lower():
                    # EXPERIMENT: try 'kick' + direction to break closed doors
                    self.queue.append(("open", "exp-open-door"))
                    self.queue.append((d, "exp-open-door-dir"))
                elif "in the way" in msg:
                    pass  # transient blocker (pet); retry next step
                else:
                    lv["edge_block"].add((frm, d))
                    if target_ch == " ":
                        lv["dark_dead"].add(tpos)
        self.last_move = None

    # ---------- prompt handling (R_PROMPT_ESC / R_PROMPT_LETTERS) ----------
    def _handle_prompt(self, msg):
        low = msg.lower()
        if self.pending_eat and "what do you want to eat" in low:
            m = re.search(r"\[([a-zA-Z])", msg)
            if m:
                self.pending_eat = False
                return m.group(1), "eat-pick-letter"
        if "[yn" in msg:
            if "pray" in low and self.t - self.prayed_at < 5:
                return "y", "pray-confirm"
            if "Still climb" in msg:
                return "esc", "decline-upstairs-exit"  # R_UP_NEEDS_TILE
            if "eat" in low and ("corpse" in low or "here" in low):
                return "y", "eat-floor-yes"  # EXPERIMENT: floor food offer
            if "really attack" in low:
                # R_PEACEFUL: mark the last bumped cell no_go and decline
                # (E35: stale target var left an acid blob unmarked -> bump
                # loop -> env no-progress quit)
                tgt = getattr(self, "_last_bump", None) or self._peaceful_target
                if tgt:
                    self._lv_cache["no_go"].add(tgt)
                return "esc", "decline-peaceful"
            return "esc", "decline-yn"
        if "In what direction" in msg:
            return "esc", "cancel-direction"
        if "[ynq" in msg:
            if "eat" in low or "corpse" in low:
                # R_CORPSE_ROT (E40/E41/E42 v7 + E41 v8: 'Poisoned by a rotted
                # X corpse'): floor corpses only in desperation, and never a
                # name that has poisoned us before.
                desperate = (getattr(self, "_last_desperate", False)
                             or getattr(self, "_fresh_floor_ok", False))
                poisoned = any(n in low for n in self.poison_names)
                if desperate and not poisoned:
                    return "y", "eat-floor-yes"
                return "n", "decline-floor-food"
            return "esc", "decline-ynq"
        if "what do you want" in low or "or ?*]" in msg:
            self.pending_eat = False
            return "esc", "clear-prompt"
        stripped = msg.rstrip()
        if (stripped.endswith("?") and "[" in msg) or msg.count("\n") > 4:
            return "esc", "clear-prompt"
        return None

    # ---------- main ----------
    def act(self, pre):
        if self.queue:
            a, why = self.queue.popleft()
            if why == "pray-go":
                self.prayed_at = self.t
            return a, why

        msg = pre["msg"] or ""
        pr = self._handle_prompt(msg)
        if pr:
            return pr

        bl = pre["bl"]
        lv = self._lv(bl)
        self._lv_cache = lv
        grid = self._grid(pre)
        pos = (bl["x_pos"], bl["y_pos"])
        lv["visited"].add(pos)

        # ANTI-STALL (E15/E19/E35: env aborts the run as 'quit' after long
        # zero-game-time loops - its no-progress guard). If game time froze
        # for 60+ policy steps, burn a real turn ('wait' advances time, R_WAIT)
        # and randomize next target.
        if bl["time"] == getattr(self, "_stall_time", -1):
            self._stall_n = getattr(self, "_stall_n", 0) + 1
        else:
            self._stall_time, self._stall_n = bl["time"], 0
        if self._stall_n >= 60:
            self._stall_n = 0
            import random
            if random.random() < 0.5:
                return "wait", "antistall-wait"
            return "search", "antistall-search"

        # R_STAIR_HIDDEN: our '@' hides the tile we stand on (E9: 154-step
        # oscillation on the stair). Remember every '>' seen; descend from memory.
        for sp in self._find(grid, ">"):
            lv["stairs_down"].add(sp)
        for sp in self._find(grid, "<"):
            lv.setdefault("stairs_up", set()).add(sp)

        adj_hostile = None
        self._last_desperate = bl["hunger_state"] >= 3  # R_HUNGER_SCALE
        hp_dropped = (self._hp_prev is not None) and self._hp_prev > bl["hitpoints"]
        self._hp_prev = bl["hitpoints"]
        pet_recent = self.t - getattr(self, "_pet_swap_t", -999) < 25
        for d, (dx, dy) in DIRS.items():
            ch = self._at(grid, pos[0] + dx, pos[1] + dy)
            if self._monster(ch) and (pos[0] + dx, pos[1] + dy) not in lv["no_go"]:
                if ch not in self.pet_glyphs:
                    adj_hostile = (d, ch)
                    break
                # pet-glyph creature: only treat as hostile under fire AND
                # not freshly swap-confirmed as our pet (E37)
                if (hp_dropped or self._hurt_recent > 0) and not pet_recent:
                    adj_hostile = (d, ch)
                    break
        if hp_dropped:
            self._hurt_recent = 3
        else:
            self._hurt_recent = max(0, getattr(self, "_hurt_recent", 0) - 1)

        # EXPERIMENT pray-on-low-hp — revised after E7 (killed while praying
        # with fox adjacent): only pray with no adjacent threat.
        if (self.pray_experiment and bl["hitpoints"] <= max(3, bl["max_hitpoints"] // 6)
                and self.t - self.prayed_at > 900 and adj_hostile is None):
            self.prayed_at = self.t
            return "pray", "exp-pray-lowhp"

        # hunger: R_PROMPT_LETTERS gives edible letters after 'eat' (R_EAT_INV)
        low = msg.lower()
        if "don't have anything to eat" in low:
            self.no_food = True
        if (("hungry" in low or "weak" in low or "fainting" in low
             or bl["hunger_state"] >= 2) and not self.pending_eat
                and not getattr(self, "no_food", False)):
            self.pending_eat = True
            self._pending_eat_t = self.t
            return "eat", "eat-when-hungry"
        # EXPERIMENT pray-on-starvation. R_PRAY_COOLDOWN (E21): a second prayer
        # too soon -> 'displeased', no rescue; keep a long cooldown.
        if (getattr(self, "no_food", False) and ("faint" in low or "weak" in low)
                and self.t - self.prayed_at > getattr(self, "_pray_gap", 1600)):
            self.prayed_at = self.t
            return "pray", "exp-pray-starving"
        # FRESH-KILL EATING (v9): hungry + no food + a kill within 60 steps
        # whose name never poisoned us -> go eat it where it dropped.
        fk = getattr(self, "_fresh_kill", None)
        if (fk and bl["hunger_state"] >= 2 and getattr(self, "no_food", False)
                and self.t - fk[2] < 60
                and not any(n in fk[1].lower() for n in self.poison_names)):
            self._fresh_floor_ok = True
            if pos == fk[0]:
                if not self.pending_eat:
                    self.pending_eat = True
                    self._pending_eat_t = self.t
                    return "eat", "eat-fresh-kill"
            else:
                path = self._bfs(grid, lv, pos, targets={fk[0]})
                if path:
                    step = path[0]
                    dd = (step[0] - pos[0], step[1] - pos[1])
                    if dd in DIR_OF:
                        d = DIR_OF[dd]
                        self.last_move = (pos, d, self._at(grid, *step), step)
                        return d, "goto-fresh-kill"
        else:
            self._fresh_floor_ok = False

        # R_EAT_FLOOR, revised after E40 (poisoned by ROTTED corpse): floor
        # corpses can be spoiled -> eat them only in true desperation
        # (weak/fainting with no inventory food), fresh-looking or not.
        if ("corpse" in low and "see here" in low and not self.pending_eat
                and getattr(self, "no_food", False)
                and ("weak" in low or "faint" in low
                     or self._hunger_desperate(pre))):
            self.pending_eat = True
            self._pending_eat_t = self.t
            return "eat", "eat-floor-corpse"

        # FOOD PICKUP (lexicon induced from own eat evidence: 'food ration'
        # E2:1014 'really hits the spot'; corpses eaten from floor): grab
        # anything matching learned food words for later hunger.
        if "you see here" in low and any(w in low for w in ("food", "ration", "corpse")) \
                and getattr(self, "_picked_at", None) != (pos, self.t // 50):
            self._picked_at = (pos, self.t // 50)
            if "corpse" not in low or getattr(self, "no_food", False):
                return "pickup", "pickup-food"

        # REST-TO-HEAL (R_HP_BOUND regen evidence P0:116): when wounded, safe
        # (no adjacent hostile, not hurt recently) and not starving, wait in
        # place to regenerate before engaging the world again.
        if (bl["hitpoints"] < 0.6 * bl["max_hitpoints"] and adj_hostile is None
                and self._hurt_recent == 0 and not getattr(self, "no_food", False)
                and "hungry" not in low and "weak" not in low and "faint" not in low):
            if getattr(self, "_rest_budget", 0) > 0:
                self._rest_budget -= 1
                return "search", "rest-heal"
            if getattr(self, "_rest_cool", 0) < self.t:
                self._rest_budget = 60
                self._rest_cool = self.t + 400
                return "search", "rest-heal"
        elif adj_hostile is not None or hp_dropped:
            self._rest_budget = 0

        # adjacent hostile -> bump attack (R_COMBAT; statues per R_STATUE;
        # pet-glyph collision handled via hp-drop trigger: E7 fox=='d'==pet)
        if adj_hostile:
            d, ch = adj_hostile
            nx, ny = pos[0] + DIRS[d][0], pos[1] + DIRS[d][1]
            no_atk = self.monsters.get(ch, {}).get("no_attack", False)  # R_FREEZE
            if not no_atk and ((ch not in self.pet_glyphs and
                    (ch in self.hostile_glyphs or self.attack_tries.get((nx, ny), 0) < 3))
                    or hp_dropped or self._hurt_recent > 0):
                self.last_move = (pos, d, ch, (nx, ny))
                self._peaceful_target = (nx, ny)
                self.attack_tries[(nx, ny)] = self.attack_tries.get((nx, ny), 0) + 1
                return d, f"attack-{ch}"

        # standing on a remembered '>' (R_STAIR_HIDDEN): descend
        if pos in lv["stairs_down"]:
            return "down", "descend"

        # path to remembered '>' (R_STAIR_HIDDEN)
        tgt = lv["stairs_down"]
        path = self._bfs(grid, lv, pos, targets=set(tgt)) if tgt else None
        why = "goto->"
        if not path:
            # explore frontier
            fr = self._frontier(grid, lv)
            path = self._bfs(grid, lv, pos, targets=fr) if fr else None
            why = "explore"
        if path and len(path) >= 1:
            step = path[0]
            d = DIR_OF[(step[0] - pos[0], step[1] - pos[1])]
            self.last_move = (pos, d, self._at(grid, *step), step)
            self._last_bump = step
            self.stuck = 0
            return d, why

        # nothing reachable: search for hidden passages (EXPERIMENT).
        # Prioritize cells bordering large unknown regions (E10/E16/E21 lost
        # thousands of steps searching low-value spots).
        self.stuck += 1

        def unknown_score(c):
            s = 0
            for ddx in range(-3, 4):
                for ddy in range(-3, 4):
                    if self._at(grid, c[0] + ddx, c[1] + ddy) == " ":
                        s += 1
            return s

        here_n = lv["searched"].get(pos, 0)
        if here_n < 8 and unknown_score(pos) > 6:
            lv["searched"][pos] = here_n + 1
            return "search", "search-hidden"
        cand = [c for c in lv["visited"] if lv["searched"].get(c, 0) < 8]
        if cand:
            scored = sorted(cand, key=lambda c: -unknown_score(c))[:12]
            best = [c for c in scored if unknown_score(c) > 6] or scored[:4]
            path = self._bfs(grid, lv, pos, targets=set(best) - {pos})
            if path:
                step = path[0]
                dd = (step[0] - pos[0], step[1] - pos[1])
                if dd in DIR_OF:
                    d = DIR_OF[dd]
                    self.last_move = (pos, d, self._at(grid, *step), step)
                    return d, "relocate-search"
            if pos in best and here_n < 8:
                lv["searched"][pos] = here_n + 1
                return "search", "search-hidden"
        else:
            lv["searched"] = {}  # all spots exhausted: start another round
        # last resort: probe a dark neighbor (EXPERIMENT dark-walk)
        for d, (dx, dy) in DIRS.items():
            nx, ny = pos[0] + dx, pos[1] + dy
            if self._at(grid, nx, ny) == " " and (pos, d) not in lv["edge_block"] \
                    and 0 < ny < 21 and 0 <= nx < 79:
                self.last_move = (pos, d, " ", (nx, ny))
                return d, "probe-dark"
        return "search", "stuck-search"

    # ---------- geometry ----------
    def _find(self, grid, glyph):
        out = []
        for y in range(21):
            for x in range(80):
                if self._at(grid, x, y) == glyph:
                    out.append((x, y))
        return out

    def _frontier(self, grid, lv):
        """Unknown ' ' cells adjacent to a known-passable/visited cell.
        Dark corridors only reveal when stepped on (E3-E5 evidence: 3500-step
        episodes revealed almost no map while only targeting lit cells), so
        the probe target IS the dark cell."""
        fr = set()
        for y in range(1, 20):
            for x in range(1, 79):
                if self._at(grid, x, y) != " " or (x, y) in lv["no_go"]:
                    continue
                if (x, y) in lv["dark_dead"]:
                    continue
                for dx, dy in DIRS.values():
                    nx, ny = x + dx, y + dy
                    ch = self._at(grid, nx, ny)
                    if (nx, ny) in lv["visited"] or (ch != " " and self._passable(ch, lv, nx, ny)):
                        fr.add((x, y))
                        break
        return fr

    def _bfs(self, grid, lv, start, targets):
        if not targets:
            return None
        targets = set(targets) - {start}
        if not targets:
            return None
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur in targets:
                path = []
                while cur != start:
                    path.append(cur)
                    cur = prev[cur]
                return list(reversed(path))
            for d, (dx, dy) in DIRS.items():
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in prev or (cur, d) in lv["edge_block"]:
                    continue
                x, y = nxt
                if not (0 <= x < 80 and 0 <= y < 21):
                    continue
                ch = self._at(grid, x, y)
                if ch == " " and nxt not in lv["visited"] and nxt not in targets:
                    continue  # unknown cells enterable only as final target
                if nxt in targets or self._passable(ch, lv, x, y) or nxt in lv["visited"]:
                    if self._monster(ch) and ch not in self.pet_glyphs and nxt not in targets:
                        continue
                    prev[nxt] = cur
                    q.append(nxt)
        return None

    def finish(self):
        save_tiles(self.tiles)
        save_monsters(self.monsters)
