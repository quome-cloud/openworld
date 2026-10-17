"""DiveAgent: layered belief-state policy for BALROG NetHackChallenge.

Objective model (from balrog/environments/nle/progress.py, offline read):
progression = max over achieved milestones Dlvl:n / Xp:n; depth dominates
(Dlvl:10 = 0.126, Dlvl:13 = 0.257). Survival per se scores nothing — the
optimal policy is depth-before-death maximization with cheap survival
maintenance (rest, pray, food) that buys more descent.

Layers (checked in order every step):
  P0 prompt/menu state machine (misc flags + message grammar)
  P1 scripted-sequence queue (dig / engrave / eat / pray / kick / pickup)
  P2 swallowed -> attack engulfer
  P3 emergency survival (pray, Elbereth, retreat)
  P4 food clock (eat inventory food / fresh safe corpse)
  P5 combat tactician (threat-budgeted melee, never-melee species)
  P6 rest-to-heal gate before descending
  P7 descent: dig down if digger held; else stairs/holes
  P8 pick-axe acquisition detour (object-glyph spotting)
  P9 frontier exploration (mass-biased, target persistence)
  P10 closed/locked doors (open, kick)
  P11 hidden-passage search rotation
  P12 anti-no-progress fallback (search/wait)

All decisions replan per step from the Atlas belief state.
"""

import re

import nh_common as C
from nh_common import DIRS, DIR_OF, CARDINALS

from nle import nethack as nh

# object glyphs for diggers (tool appearances are not shuffled)
DIGGER_GLYPHS = set()
for _g in range(C.GLYPH_OBJ_OFF, C.GLYPH_CMAP_OFF):
    try:
        _name = nh.OBJ_NAME(nh.objclass(_g - C.GLYPH_OBJ_OFF))
    except Exception:
        continue
    if _name in ("pick-axe", "dwarvish mattock"):
        DIGGER_GLYPHS.add(_g)

# NOTE: fingertip-Elbereth is UNREACHABLE in BALROG's NLE action space:
# the "What do you want to write with?" getobj prompt needs the '-' key
# (fingers), but BALROG's NLE action list has no "minus"/'-' action
# (TextCharacters are stripped from USEFUL_ACTIONS; only letters/digits/
# space are added back). Engraving is therefore disabled.
OUR_SPEED = 12          # all starting roles move at speed 12

# V1.1 L2 — shallow opportunistic hunting. DROPPED from v1.1 after dev
# validation (coordinator rule: drop levers that don't clearly pay):
# isolated effect +0.63 [-0.42,+1.82] on 24 paired dev seeds, and the
# 12-seed extension block regressed to -0.34 — the initial gains were
# dev-noise seed luck (the Crafter v2 lesson). Kept as an off-by-default
# toggle for future work.
import os as _os
HUNT_SHALLOW = _os.environ.get("NH_HUNT", "0") == "1"

RE_KILLED = re.compile(r"You (?:kill|destroy) the ([a-zA-Z' -]+?)!")
RE_SEE_HERE = re.compile(r"You see here (?:an? |the )?([^.]*)\.")

TRAP_HOLD_MSGS = ("pit", "bear trap", "web", "You are stuck")


class DiveAgent:
    def __init__(self, log=print, memory=None):
        self.atlas = C.Atlas()
        self.log = log
        self.memory = memory                    # NetHackMemory or None
        self.queue = []                         # scripted action sequence
        self.queue_tag = None
        self.last_action = None
        self.last_pos = None
        self.last_time = -1
        self.last_hp = None
        self.suspect_walls = {}                 # key -> set of cells
        self.explore_target = None
        self.no_time_steps = 0                  # env steps w/o game time
        self.last_prompt_sig = None
        self.prompt_repeats = 0
        self.prayed_at = None
        self.pray_count = 0
        self.engraved_at = {}                   # cell -> time
        self.kick_dir = None
        self.kick_count = 0
        self.door_giveup = set()
        self.door_target = None                 # cell of last open/kick attempt
        self.hunt_turns = {}                    # level key -> game turns spent hunting
        self.fresh_kills = []                   # (cell, species, time)
        self.role = None
        self.race = None
        self.rest_budget = {}                   # level key -> turns rested
        self.dig_attempts = {}                  # level key -> attempts
        self.no_dig_cells = set()               # (key, cell)
        self.pickup_wanted = None               # letter to verify after pickup
        self.digger_letter = None
        self.digging = False
        self.need_look = True
        self.retreat_ups = 0
        self.grind_start = {}                   # level key -> game time
        self.grind_note = set()
        self.descended_from = None              # (key, cell) of last '>' taken
        self.mines_entrances = {}               # level key -> {cells}
        self.mines_avoid_since = {}             # level key -> game time
        self.commit_mines = False               # ban expired: stop retreating
        self.throws_at = {}                     # (key, cell) -> throw count
        self.steps = 0
        self.recent_max_hit = 0                 # worst single-step hp loss, decayed
        self.notes = []                         # sparse decision log
        self.mem_fired = []                     # memory-driven decisions
        self._mem_avoid = set()
        self._mem_danger_depth = None
        if memory is not None:
            self._mem_avoid = set(memory.avoid_species())
            self._mem_danger_depth = memory.danger_depth()
            if self._mem_avoid:
                self._fire(f"M1 loaded avoid-species {sorted(self._mem_avoid)}")
            if self._mem_danger_depth is not None:
                self._fire(f"M2 loaded danger depth {self._mem_danger_depth}")

    # ------------------------------------------------------------------ api
    def set_actions(self, names):
        self.actions = set(names)

    def note(self, s):
        self.notes.append((self.steps, s))

    def _fire(self, s):
        self.mem_fired.append((self.steps, s))
        if self.memory is not None:
            self.memory.record_fired(s)

    def act(self, obs):
        A = self.atlas
        A.update(obs)
        msg = A.message
        self._bookkeeping(obs, msg)
        a = self._decide(obs, msg)
        self.last_action = a
        self.last_pos = A.agent
        self.last_time = A.time
        self.last_hp = A.hp
        self.steps += 1
        return a

    # ---------------------------------------------------------- bookkeeping
    def _bookkeeping(self, obs, msg):
        A = self.atlas
        if self.role is None and "welcome to NetHack" in msg:
            m = re.search(r"You are an? ([a-z ]+) (\w+) (\w+)\.", msg)
            if m:
                self.role = m.group(3)
                self.race = m.group(2)
                self.note(f"role={self.role} race={self.race}")

        if A.level_changed:
            # mines-entrance learning: if taking that '>' put us in the
            # Gnomish Mines (dnum 2), remember the fork-level cell
            if self.descended_from and A.dnum == 2 and \
                    self.descended_from[0][0] == 0:
                fkey, fcell = self.descended_from
                self.mines_entrances.setdefault(fkey, set()).add(fcell)
                self.note(f"learned Mines entrance at {fkey}:{fcell}")
            self.descended_from = None
            self.explore_target = None
            self.queue = []
            self.queue_tag = None
            self.kick_dir = None
            self.digging = False
            self.need_look = True      # terrain under agent is invisible
            self.retreat_ups = 0
            self.note(f"level -> {A.key} depth={A.depth}")

        # message-driven terrain-under-agent knowledge (from 'look')
        low = msg.lower()
        if "staircase down here" in low or "ladder down here" in low:
            A.level.stairs_down.add(A.agent)
            A.level.terrain[A.agent[1]][A.agent[0]] = C.STAIRS_DOWN
        elif "staircase up here" in low or "ladder up here" in low:
            A.level.stairs_up.add(A.agent)
            A.level.terrain[A.agent[1]][A.agent[0]] = C.STAIRS_UP

        # no-game-time counter (env aborts at 150)
        if A.time == self.last_time:
            self.no_time_steps += 1
        else:
            self.no_time_steps = 0

        # suspect-wall aging: stale entries (monster-blocked cells, opened
        # doors) otherwise wall off corridors forever; re-verifying costs
        # one bump each
        if self.steps % 250 == 249:
            self.suspect_walls.pop(A.key, None)

        # damage tracking (crisis detection + memory ledger)
        if self.last_hp is not None:
            dmg = self.last_hp - A.hp
            if dmg > 0:
                self.recent_max_hit = max(self.recent_max_hit, dmg)
                if self.memory is not None:
                    adj = [m for m in A.level.monsters
                           if not m.pet and max(abs(m.x - A.agent[0]),
                                                abs(m.y - A.agent[1])) <= 1]
                    for m in adj:
                        self.memory.record_exchange(m.name, dmg / len(adj))
            elif self.steps % 12 == 0 and self.recent_max_hit > 0:
                self.recent_max_hit -= 1        # decay when not being hit

        # kill log (fresh corpses for the food layer); the corpse drops on
        # the victim's cell = the cell we attacked into, not our own
        for mname in RE_KILLED.findall(msg):
            cell = A.agent
            if self.last_action in DIRS:
                dx, dy = DIRS[self.last_action]
                cell = (A.agent[0] + dx, A.agent[1] + dy)
            self.fresh_kills.append((cell, mname.strip(), A.time))
            if self.memory is not None:
                self.memory.record_kill(mname.strip())
        self.fresh_kills = [k for k in self.fresh_kills
                            if A.time - k[2] < 40][-20:]

        # stuck detection -> suspect walls (not while held by a trap)
        if self.last_action in DIRS and self.last_pos == A.agent and \
                A.time == self.last_time and \
                not any(t in msg for t in TRAP_HOLD_MSGS):
            dx, dy = DIRS[self.last_action]
            tgt = (A.agent[0] + dx, A.agent[1] + dy)
            if not any(m.pos == tgt for m in A.level.monsters):
                self.suspect_walls.setdefault(A.key, set()).add(tgt)

        # boulder push failure
        if "but in vain" in msg and self.last_action in DIRS:
            dx, dy = DIRS[self.last_action]
            tgt = (A.agent[0] + dx, A.agent[1] + dy)
            A.level.boulder_blocked.add((A.agent, self.last_action))
            self.suspect_walls.setdefault(A.key, set()).add(tgt)

        # dig outcome tracking (self.digging set when a dig queue is issued)
        if getattr(self, "digging", False):
            low2 = msg.lower()
            if "stairs are too hard to dig" in low2:
                # we are standing on invisible stairs: identify them
                self.digging = False
                self.queue = []
                self.queue_tag = None
                self.need_look = True
                self.no_dig_cells.add((A.key, A.agent))
            elif "too hard to dig" in low2 or "can't dig" in low2 or \
                    "cannot dig" in low2 or "you don't have anything" in low2:
                self.no_dig_cells.add((A.key, A.agent))
                self.digging = False
                self.queue = []
                self.queue_tag = None

        # a queue answer that missed its prompt: abort the script
        if self.queue and ("You don't have that object" in msg or
                           "Never mind" in msg or "never mind" in msg):
            self.queue = []
            self.queue_tag = None
            self.digging = False

        # phantom corpse: kill was logged but no corpse dropped (drop is
        # probabilistic) -> "eat" refuses in zero time, which would freeze
        # the fresh-kill clock forever (dev seed 103 abort)
        if "don't have anything to eat" in msg:
            self.fresh_kills = [k for k in self.fresh_kills
                                if k[0] != A.agent]

        # locked door outcomes
        if "This door is locked" in msg and self.kick_dir:
            pass  # kick sequence continues in _decide
        if "crashes open" in msg or "The door opens" in msg or \
                "You succeed" in msg:
            self.kick_dir = None
            self.kick_count = 0

        # V1.1 L1 — stale-door terrain correction (v1 failure catalog #8:
        # an item glyph covering an opened door left remembered terrain
        # "closed"; the open/direction loop then churned 100k-step episodes
        # at ~zero game time). The message channel is authoritative:
        if self.door_target is not None:
            tx, ty = self.door_target
            if "This door is already open" in msg or "The door opens" in msg:
                A.level.terrain[ty][tx] = C.DOOR_OPEN
                self.door_target = None
            elif "You see no door there" in msg or "no door" in msg.lower():
                A.level.terrain[ty][tx] = C.DOORWAY
                self.door_target = None
            elif "This door is broken" in msg:
                A.level.terrain[ty][tx] = C.DOORWAY
                self.door_target = None

    # -------------------------------------------------------------- helpers
    def _suspects(self):
        return self.suspect_walls.get(self.atlas.key, set())

    def _mcells(self, hostile_only=True):
        """Monster cells to treat as path obstacles. Pets are NOT obstacles
        (moving into a pet swaps places) — treating them as walls let a
        following kitten box the agent into corridor dead-ends forever
        (dev seeds 106/103: thousands of stationary searches)."""
        L = self.atlas.level
        return {m.pos for m in L.monsters if not (hostile_only and m.pet)}

    def _inv(self, obs):
        return C.inventory(obs)

    def _find_digger(self, obs):
        for letter, desc, oc in self._inv(obs):
            d = desc.lower()
            if "pick-axe" in d or "mattock" in d:
                return letter
        return None

    def _food_letter(self, obs):
        best = None
        for letter, desc, oc in self._inv(obs):
            if oc != C.FOOD_CLASS:
                continue
            d = desc.lower()
            if "corpse" in d:
                # exact species match: substring matching accepted
                # "dwarf zombie corpse" via "dwarf" (wrath epidemic root #2)
                m = re.search(r"([a-z' -]+?) corpses?", d)
                species = m.group(1).strip() if m else ""
                if species in C.SAFE_CORPSES and not self._cannibal(species):
                    best = best or letter
                continue
            if "tin " in d or d.endswith("tin") or "tins" in d:
                continue
            return letter                       # prepared food: take first
        return best

    _RACE_ROOT = {"dwarven": "dwarf", "gnomish": "gnome", "elven": "elf",
                  "orcish": None,  # orcs may eat orcs
                  "human": "human"}

    def _cannibal(self, desc):
        d = desc.lower()
        if not self.race:
            # race unknown (welcome message missed): refuse any corpse that
            # could be own-race — cannibalism angers the god (wrath deaths)
            return any(r in d for r in ("dwarf", "gnome", "elf"))
        root = self._RACE_ROOT.get(self.race.lower(), self.race.lower())
        return bool(root) and root in d

    def _pray_ok(self, last_resort=False):
        # prayer discipline (dev seed 104: Healer smote by Hermes for
        # praying early+often): first prayer after turn 300, >=1500-turn
        # gaps, at most 3 per game. last_resort (imminent death) waives the
        # turn gates: an angry god is no worse than the ant eating you.
        A = self.atlas
        if self.pray_count >= 3:
            return False
        if last_resort:
            # one gamble per crisis, not a wrath-farming loop (dev seed 105)
            return self.prayed_at is None or A.time - self.prayed_at > 500
        if self.prayed_at is None:
            # NetHack's initial prayer timeout is rnz(350) (long-tailed):
            # T>700 clears most of the distribution
            return A.time > 700
        return A.time - self.prayed_at > 1500

    def _adjacent_hostiles(self):
        A = self.atlas
        ax, ay = A.agent
        out = []
        for m in A.level.monsters:
            if m.pet or m.pos in A.level.no_attack:
                continue
            if max(abs(m.x - ax), abs(m.y - ay)) == 1:
                out.append(m)
        return out

    def _mobile_hostiles(self):
        return [m for m in self.atlas.level.monsters
                if not m.pet and m.name not in C.IMMOBILE and
                m.pos not in self.atlas.level.no_attack]

    def _never_melee(self, m):
        if m.name in C.NEVER_MELEE:
            return True
        if m.name in self._mem_avoid and self.atlas.xplvl <= 6:
            return True
        return False

    def _dir_to(self, tgt):
        ax, ay = self.atlas.agent
        d = (tgt[0] - ax, tgt[1] - ay)
        return DIR_OF.get(d)

    # ------------------------------------------------------------- decision
    def _decide(self, obs, msg):
        A = self.atlas
        L = A.level

        # ---- P0: prompts ------------------------------------------------
        in_yn, in_getlin, waitspace = C.misc_of(obs)
        prompt_open = in_yn or in_getlin or waitspace
        if prompt_open:
            if self.queue:
                self.last_prompt_sig = None
                return self._pop_queue()
            sig = (in_yn, in_getlin, waitspace, msg[-60:])
            if sig == self.last_prompt_sig:
                self.prompt_repeats += 1
            else:
                self.prompt_repeats = 0
            self.last_prompt_sig = sig
            if self.prompt_repeats >= 4:
                # cycle escapes for prompts that refuse our default
                return ["esc", "space", "more", "n"][self.prompt_repeats % 4]
            return self._answer_prompt(obs, msg, in_yn, in_getlin, waitspace)
        else:
            self.last_prompt_sig = None
            self.prompt_repeats = 0
            if not self.queue:
                self.queue_tag = None   # tags only live into their prompt

        # zero-time livelock breaker: the env aborts after 150 env steps
        # without game-time advance; search always consumes a turn
        if self.no_time_steps > 60:
            self.queue = []
            self.queue_tag = None
            return "search"

        # ---- P1: scripted queue ------------------------------------------
        if self.queue:
            # prompt-answer entries are only valid inside a prompt
            if self.queue_tag in ("pray", "eat", "eat_corpse"):
                self.queue = []
                self.queue_tag = None
            else:
                return self._pop_queue()

        # ---- P2: swallowed ------------------------------------------------
        if A.swallowed:
            self.note("swallowed: attacking engulfer")
            return "west"

        # ---- P2.5: identify the (invisible) terrain under our feet -------
        if self.need_look and not self._adjacent_hostiles():
            self.need_look = False
            return "look"

        # ---- P3: emergency survival --------------------------------------
        adj = self._adjacent_hostiles()
        # prayer heals only in "major trouble" (hp < hpmax/7): fire it there
        if A.hp <= max(A.hpmax // 7, 5) and \
                self._pray_ok(last_resort=bool(adj) and A.hp <= 4):
            self.prayed_at = A.time
            self.pray_count += 1
            self.note(f"pray (hp {A.hp}/{A.hpmax})")
            self.queue = ["y"]
            self.queue_tag = "pray"
            return "pray"
        # crisis zone: below ~28% max HP, or worst recent hit could kill us
        # within two more exchanges -> disengage from slower monsters
        crisis = A.hp <= max(A.hpmax * 0.28, 6) or \
            (self.recent_max_hit * 2 >= A.hp and self.recent_max_hit > 0)
        if crisis and adj:
            act = self._flee(adj)
            if act:
                return act

        # hunger crisis handled with priority right below emergencies
        if A.hunger >= C.WEAK:
            fl = self._food_letter(obs)
            if fl:
                self.note(f"eat inventory food {fl} (hunger {A.hunger})")
                self.queue = [fl]
                self.queue_tag = "eat"
                return "eat"
            corpse = self._fresh_corpse_here()
            if corpse:
                self.note(f"eat fresh corpse here ({corpse})")
                self.queue = ["y"]
                self.queue_tag = "eat_corpse"
                return "eat"
            # walk to a fresh safe corpse nearby
            cells = [k[0] for k in self.fresh_kills
                     if k[1] in C.SAFE_CORPSES and not self._cannibal(k[1])
                     and k[0] != A.agent]
            if cells:
                path = A.level.bfs(A.agent, cells,
                                   avoid=self._suspects() | self._mcells())
                if path and len(path) <= 12:
                    return self._step_path(path)
            fainting_ok = (A.hunger >= C.FAINTING and self.pray_count < 5 and
                           (self.prayed_at is None or
                            A.time - self.prayed_at > 400))
            if self._pray_ok() or fainting_ok:
                self.prayed_at = A.time
                self.pray_count += 1
                self.note(f"pray (hunger {A.hunger})")
                self.queue = ["y"]
                self.queue_tag = "pray"
                return "pray"
        elif A.hunger == C.HUNGRY:
            fl = self._food_letter(obs)
            if fl:
                self.queue = [fl]
                self.queue_tag = "eat"
                return "eat"

        # held by a sticky monster: kill it, fleeing is impossible
        if "cannot escape from" in msg:
            m2 = re.search(r"cannot escape from (?:the |an? )?([a-z' -]+?)!", msg)
            if m2:
                for m in self._adjacent_hostiles():
                    if m.name == m2.group(1).strip() and \
                            m.name not in C.NEVER_MELEE:
                        d = (m.x - A.agent[0], m.y - A.agent[1])
                        if d in DIR_OF:
                            return DIR_OF[d]

        # ---- P5: combat ---------------------------------------------------
        if adj:
            act = self._combat(adj)
            if act:
                return act

        # ---- P5.5: ranged removal of never-melee blockers ------------------
        # (dev seed 116: a sleeping floating eye parked in a 1-wide corridor
        # walled off the only route to the '>' for 4,000 turns)
        act = self._throw_at_blocker(obs)
        if act:
            return act

        # blind: sit tight until it clears (map can't be trusted)
        if A.blind:
            return "search"

        # global rest gate: badly hurt, nothing visible hunting us -> heal
        if A.hp < 0.35 * A.hpmax and self._rest_here_ok() and \
                self.rest_budget.get(A.key, 0) < 900:
            self.rest_budget[A.key] = self.rest_budget.get(A.key, 0) + 1
            return "search"

        # ---- P5.7: shallow opportunistic hunting (V1.1 L2) -----------------
        if HUNT_SHALLOW and A.depth <= 4 and A.hp >= 0.6 * A.hpmax and \
                not self.digger_letter and \
                self.hunt_turns.get(A.key, 0) < 250:
            prey = [m for m in self._mobile_hostiles()
                    if m.difficulty <= A.xplvl + 1 and m.speed <= OUR_SPEED
                    and not self._never_melee(m)
                    and max(abs(m.x - A.agent[0]),
                            abs(m.y - A.agent[1])) <= 6]
            if prey:
                tgt = min(prey, key=lambda m: max(abs(m.x - A.agent[0]),
                                                  abs(m.y - A.agent[1])))
                path = A.level.bfs(A.agent, [tgt.pos],
                                   avoid=self._suspects() |
                                   (self._mcells() - {tgt.pos}))
                if path:
                    t0 = self.hunt_turns.setdefault(A.key, 0)
                    self.hunt_turns[A.key] = t0 + 1
                    return self._step_path(path)

        # ---- P6/P7: descent (dig > stairs), with rest gate -----------------
        act = self._descend(obs)
        if act:
            return act

        # ---- P8: digger acquisition detour ---------------------------------
        act = self._acquire_digger(obs)
        if act:
            return act

        # opportunistic floor-food pickup (long games starve otherwise)
        m3 = RE_SEE_HERE.search(msg)
        if m3 and self.queue_tag != "pickup":
            it = m3.group(1)
            if any(k in it for k in ("food ration", "cram ration", "lembas",
                                     "K-ration", "C-ration", "pancake",
                                     "candy bar", "fortune cookie", "apple",
                                     "orange", "pear", "banana", "melon",
                                     "carrot", "meatball", "meat stick")):
                self.note(f"picking up food: {it}")
                self.queue_tag = "pickup"
                return "pickup"

        # ---- P9: explore ----------------------------------------------------
        act = self._explore(obs)
        if act:
            return act

        # ---- P10: doors ------------------------------------------------------
        act = self._doors(obs)
        if act:
            return act

        # ---- P11: hidden passages -------------------------------------------
        act = self._hidden_search(obs)
        if act:
            return act

        # ---- P12: fallback ----------------------------------------------------
        return "search"

    # --------------------------------------------------------------- queue
    def _pop_queue(self):
        a = self.queue.pop(0)
        if not self.queue:
            self.queue_tag = None
        return a

    # ------------------------------------------------------------- prompts
    def _answer_prompt(self, obs, msg, in_yn, in_getlin, waitspace):
        A = self.atlas
        if in_getlin:
            return "esc"
        if in_yn:
            if "Really attack" in msg:
                # peaceful: mark the intended cell and decline
                if self.last_action in DIRS:
                    dx, dy = DIRS[self.last_action]
                    cell = (A.agent[0] + dx, A.agent[1] + dy)
                    A.level.no_attack.add(cell)
                    self.note(f"peaceful at {cell}: declining attack")
                return "n"
            if "eat it?" in msg or "eat one?" in msg:
                m = re.search(r"There (?:is|are) (?:an? )?([a-z' -]+?) corpse", msg)
                if m and m.group(1) in C.SAFE_CORPSES and \
                        not self._cannibal(m.group(1)):
                    return "y"
                return "n"
            if "Are you sure you want to pray" in msg:
                return "y" if self.queue_tag == "pray" else "n"
            if "In what direction" in msg:
                return "esc"
            if "Do you want to add to the current engraving" in msg:
                return "n"
            if "lock it?" in msg or "Force its lock" in msg:
                return "esc"
            if "Shall I remove" in msg or "loot it?" in msg:
                return "n"
            if "Continue?" in msg:
                return "y" if self.queue_tag == "dig" else "n"
            if "What do you want" in msg:
                return "esc"
            return "esc"
        # xwaitingforspace: menus / overview screens
        if "Pick up what" in msg or self._tty_has(obs, "Pick up what"):
            letter = self._menu_letter_for(obs, ("pick-axe", "mattock"))
            if letter and self.queue_tag == "pickup":
                self.queue = ["more"]
                return letter
            return "esc"
        return "esc"

    def _tty_has(self, obs, text):
        tty = obs["obs"]["tty_chars"]
        for row in tty:
            if text in "".join(chr(c) for c in row):
                return True
        return False

    def _menu_letter_for(self, obs, keywords):
        tty = obs["obs"]["tty_chars"]
        for row in tty:
            line = "".join(chr(c) for c in row)
            m = re.search(r"([a-zA-Z]) - (.*)", line)
            if m and any(k in m.group(2) for k in keywords):
                return m.group(1)
        return None

    def _fresh_corpse_here(self):
        A = self.atlas
        for (cell, species, t) in reversed(self.fresh_kills):
            if cell == A.agent and species in C.SAFE_CORPSES and \
                    not self._cannibal(species):
                return species
        return None

    # -------------------------------------------------------------- combat
    def _combat(self, adj):
        A = self.atlas
        L = A.level
        ax, ay = A.agent
        threats = [m for m in adj if not self._never_melee(m)]
        nm_threats = [m for m in adj if self._never_melee(m) and
                      m.name not in C.IMMOBILE]

        # threat budget: flee/hold when outmatched
        danger = sum(m.difficulty for m in adj if m.name not in C.IMMOBILE)
        if (A.hp < 0.3 * A.hpmax and len([m for m in adj
                                          if m.name not in C.IMMOBILE]) >= 2):
            # retreat to the neighbor cell with fewest adjacent threats
            best = None
            for name, (nx, ny) in L.neighbors(ax, ay, avoid=self._suspects()):
                if any(m.x == nx and m.y == ny for m in L.monsters):
                    continue
                dgr = sum(1 for m in adj
                          if max(abs(m.x - nx), abs(m.y - ny)) <= 1)
                if best is None or dgr < best[0]:
                    best = (dgr, name)
            if best and best[0] < len(adj):
                self.note(f"retreat (hp {A.hp}, danger {danger})")
                return best[1]

        # attack the weakest attackable adjacent hostile (cardinal-legal)
        threats.sort(key=lambda m: m.difficulty)
        for m in threats:
            if m.name in C.IMMOBILE and not self._blocks_path(m):
                continue
            d = (m.x - ax, m.y - ay)
            t_here = L.terrain[ay][ax]
            t_there = L.terrain[m.y][m.x]
            if d[0] and d[1] and (
                    t_here in (C.DOORWAY, C.DOOR_OPEN, C.DOOR_CLOSED) or
                    t_there in (C.DOORWAY, C.DOOR_OPEN, C.DOOR_CLOSED)):
                continue
            return DIR_OF[d]

        # only never-melee mobile threats adjacent: step away if possible
        if nm_threats:
            for name, (nx, ny) in L.neighbors(ax, ay, avoid=self._suspects()):
                if all(max(abs(m.x - nx), abs(m.y - ny)) > 1
                       for m in nm_threats) and \
                        not any(m.x == nx and m.y == ny for m in L.monsters):
                    self.note(f"stepping away from {nm_threats[0].name}")
                    return name
            return "search"     # nowhere better: pass time, don't touch it
        return None

    def _flee(self, adj):
        """Crisis disengage: NetHack speed system makes running away work
        against slower species (dwarf 6, zombie 6, mold 0) and suicide
        against faster ones -- flee only when every adjacent mobile threat
        is slower than us; otherwise keep fighting (E2 lesson from the
        MiniHack arm: dashing while surrounded by fast monsters is worse
        than trading blows)."""
        A = self.atlas
        L = A.level
        mobile = [m for m in adj if m.name not in C.IMMOBILE]
        if not mobile:
            return None
        # outrunning needs a real speed margin: a speed-9 monster still
        # attacks on ~3 of 4 turns while "outrun" at speed 12
        if any(m.speed > (OUR_SPEED * 2) // 3 for m in mobile):
            return None
        if "cannot escape" in A.message or "still in a pit" in A.message or \
                "You fall into a pit" in A.message:
            return None                # held: fleeing burns turns for nothing
        ax, ay = A.agent
        # on known up stairs: escape the level entirely
        if A.agent in L.stairs_up and self.retreat_ups < 3:
            self.retreat_ups += 1
            self.note(f"crisis: escaping upstairs (hp {A.hp})")
            return "up"
        # move to the neighbor that maximizes distance from threats;
        # sticky direction (dev seed 109: direction flip-flop vs a speed-9
        # crocodile gained zero distance and donated bites)
        best = None
        prev = self.last_action if self.last_action in DIRS else None
        for name, (nx, ny) in L.neighbors(ax, ay, avoid=self._suspects()):
            if any(m.x == nx and m.y == ny for m in L.monsters):
                continue
            score = min(max(abs(m.x - nx), abs(m.y - ny)) for m in mobile)
            adjcnt = sum(1 for m in mobile
                         if max(abs(m.x - nx), abs(m.y - ny)) <= 1)
            sticky = 0 if name == prev else 1
            if best is None or (adjcnt, -score, sticky) < best[0]:
                best = ((adjcnt, -score, sticky), name)
        # flee only when the move fully disengages (0 adjacent threats after
        # it); partial retreats just donate free attacks (dev seed 103)
        if best and best[0][0] == 0:
            self.note(f"crisis: fleeing {mobile[0].name} (hp {A.hp})")
            return best[1]
        return None

    def _throwable_letter(self, obs):
        for letter, desc, oc in self._inv(obs):
            d = desc.lower()
            # "not wielded" contains "wielded" — only skip actual wields
            if "weapon in hand" in d or "weapons in hands" in d or \
                    ("wielded" in d and "not wielded" not in d):
                continue
            if any(k in d for k in ("dagger", "dart", "arrow", "spear",
                                    "shuriken", "rock", "aklys")):
                return letter
        return None

    def _throw_at_blocker(self, obs):
        A = self.atlas
        L = A.level
        ax, ay = A.agent
        letter = self._throwable_letter(obs)
        if not letter:
            return None
        for m in L.monsters:
            if m.pet or not self._never_melee(m):
                continue
            dx, dy = m.x - ax, m.y - ay
            dist = max(abs(dx), abs(dy))
            if dist < 1 or dist > 8:
                continue
            if not (dx == 0 or dy == 0 or abs(dx) == abs(dy)):
                continue
            key = (A.key, m.pos)
            if self.throws_at.get(key, 0) >= 8:
                continue
            # ray must be clear (passable, no other monster) up to the target
            sx = (dx > 0) - (dx < 0)
            sy = (dy > 0) - (dy < 0)
            cx, cy = ax + sx, ay + sy
            clear = True
            while (cx, cy) != m.pos:
                if not L.passable(cx, cy, bad_traps_ok=True) or \
                        any(mm.pos == (cx, cy) for mm in L.monsters):
                    clear = False
                    break
                cx, cy = cx + sx, cy + sy
            if not clear:
                continue
            # only spend ammo when it actually blocks us (goal or frontier
            # unreachable without its cell) or it is adjacent
            if dist > 1 and not self._blocks_route(m):
                continue
            self.throws_at[key] = self.throws_at.get(key, 0) + 1
            self.note(f"throwing {letter} at {m.name} at {m.pos} "
                      f"(dist {dist})")
            self.queue = [letter, DIR_OF[(sx, sy)]]
            self.queue_tag = "throw"
            return "throw"
        return None

    def _blocks_route(self, m):
        A = self.atlas
        L = A.level
        goals = set(L.stairs_down) | set(L.holes)
        if not goals:
            cells = L.frontier_cells()
            if not cells:
                return True     # fully explored + never-melee around: clear it
            goals = set(cells)
        path_avoiding = L.bfs(A.agent, goals,
                              avoid=self._suspects() | {m.pos})
        return path_avoiding is None    # no route without its cell = blocker

    def _blocks_path(self, m):
        # immobile monster attacked only if it sits on our next planned cell
        tgt = self._current_goal_cell()
        if tgt is None:
            return False
        L = self.atlas.level
        path = L.bfs(self.atlas.agent, [tgt],
                     avoid=self._suspects() | {mm.pos for mm in L.monsters
                                               if mm.pos != m.pos})
        if not path:
            return True
        dx, dy = DIRS[path[0]]
        nxt = (self.atlas.agent[0] + dx, self.atlas.agent[1] + dy)
        return nxt == m.pos

    def _current_goal_cell(self):
        L = self.atlas.level
        goals = list(L.stairs_down | L.holes)
        if goals:
            return goals[0]
        if self.explore_target:
            return self.explore_target
        return None

    # ------------------------------------------------------------- descent
    def _rest_threshold(self):
        A = self.atlas
        lo, hi = 0.6, 0.85
        # V1.1 L4 — shallow rest discipline: the 2000-block's early deaths
        # (13/25 episodes at depth 2-6, xp 1-2) went in at part health;
        # shallow floors are the cheapest place to buy HP
        if A.depth <= 3:
            lo, hi = 0.75, 0.92
        if self._mem_danger_depth is not None and \
                A.depth >= self._mem_danger_depth - 1:
            if (lo, hi) != (0.9, 0.95) and A.hp < 0.9 * A.hpmax:
                self._fire(f"M2 rest gate tightened at depth {A.depth} "
                           f"(remembered danger depth {self._mem_danger_depth})")
            lo, hi = 0.9, 0.95
        return lo, hi

    def _should_rest(self):
        A = self.atlas
        lo, hi = self._rest_threshold()
        key = A.key
        used = self.rest_budget.get(key, 0)
        cap = 800 if (self._mem_danger_depth is not None and
                      A.depth >= self._mem_danger_depth - 1) else 400
        if used >= cap:
            return False
        if A.hp >= hi * A.hpmax:
            return False
        if A.hp > lo * A.hpmax and used > 0:
            return True    # continue an ongoing rest up to hi
        return A.hp <= lo * A.hpmax

    def _rest_here_ok(self):
        # rest only when no mobile hostile is visible
        return not self._mobile_hostiles() and self.atlas.hunger < C.WEAK

    def _pace_gate(self):
        """XP pacing experiment — DISABLED after dev A/B: seed 102's
        Archeologist died grinding at depth 7 (0.048) where the ungated
        dig-dive reached depth 10 (0.126); observed non-digger deaths all
        happened at depth < xplvl+6, so the gate never fires for them.
        Depth-before-death favors continuous descent; kept for the record
        and for the memory condition to re-enable selectively (M2)."""
        return None
        A = self.atlas
        if A.depth < A.xplvl + 6:
            return None
        if A.hunger >= C.HUNGRY:
            return None
        start = self.grind_start.setdefault(A.key, A.time)
        if A.time - start > 700:
            return None
        if A.key not in self.grind_note:
            self.grind_note.add(A.key)
            self.note(f"pace gate: grinding at depth {A.depth} "
                      f"(xplvl {A.xplvl})")
        L = A.level
        # hunt the nearest beatable mobile hostile
        prey = [m for m in self._mobile_hostiles()
                if m.difficulty <= A.xplvl + 1 and not self._never_melee(m)]
        if prey and A.hp > 0.55 * A.hpmax:
            tgt = min(prey, key=lambda m: max(abs(m.x - A.agent[0]),
                                              abs(m.y - A.agent[1])))
            path = L.bfs(A.agent, [tgt.pos],
                         avoid=self._suspects() | (self._mcells() - {tgt.pos}))
            if path:
                return self._step_path(path)
        # otherwise rest/wait for spawns (search also reveals hidden ways)
        return "search"

    def _descend(self, obs):
        A = self.atlas
        L = A.level

        act = self._pace_gate()
        if act:
            return act

        # dig straight down if we hold a digger and the spot allows it
        digger = self._find_digger(obs)
        self.digger_letter = digger
        # digging with a mobile hostile nearby donates free attacks: clear
        # the area first (hunt it down), then dig. Only CATCHABLE threats
        # are worth hunting — chasing a speed-22 bat around the level got
        # the condition-A run-1 digger killed; fast flitters are left to
        # the adjacency combat layer while digging continues.
        near_threats = [
            m for m in self._mobile_hostiles()
            if max(abs(m.x - A.agent[0]), abs(m.y - A.agent[1])) <= 3
            and 6 <= m.speed <= OUR_SPEED and not self._never_melee(m)]
        threat_near = bool(near_threats)
        if digger and not threat_near and not L.undiggable and \
                (A.key, A.agent) not in self.no_dig_cells and \
                L.terrain[A.agent[1]][A.agent[0]] not in (
                    C.STAIRS_DOWN, C.STAIRS_UP, C.ALTAR, C.FOUNTAIN,
                    C.THRONE, C.SINK):
            if self._should_rest() and self._rest_here_ok():
                self.rest_budget[A.key] = self.rest_budget.get(A.key, 0) + 1
                return "search"
            att = self.dig_attempts.get(A.key, 0)
            if att < 40:
                self.dig_attempts[A.key] = att + 1
                self.queue = [digger, "down"]
                self.queue_tag = "dig"
                self.digging = True
                if att == 0:
                    self.note(f"digging down at {A.agent} depth {A.depth}")
                return "apply"
        elif digger and threat_near and not L.undiggable:
            # close and kill the interloper so the dig can proceed
            tgt = min(near_threats,
                      key=lambda m: max(abs(m.x - A.agent[0]),
                                        abs(m.y - A.agent[1])))
            path = L.bfs(A.agent, [tgt.pos],
                         avoid=self._suspects() |
                         (self._mcells() - {tgt.pos}))
            if path:
                return self._step_path(path)
            return "search"            # unreachable: let it come, pass time
        elif digger and (A.key, A.agent) in self.no_dig_cells:
            # try a nearby diggable floor cell
            tries = sum(1 for (k, c) in self.no_dig_cells if k == A.key)
            if tries < 4:
                for name, (nx, ny) in L.neighbors(*A.agent,
                                                  avoid=self._suspects()):
                    if L.terrain[ny][nx] in (C.FLOOR, C.CORRIDOR) and \
                            (A.key, (nx, ny)) not in self.no_dig_cells:
                        return name
            else:
                L.undiggable = True
                self.note(f"level {A.key} marked undiggable")

        # Mines policy (condition A run 1: 3/5 episodes erased by early-
        # Mines dwarves at xplvl 1-2): while weak, back out of the Mines'
        # top levels and use the main-branch '>' on the fork level instead;
        # give up avoiding after 800 fruitless turns (starving is worse).
        if A.dnum == 2 and A.xplvl <= 3 and A.dlevel <= 2 and not digger \
                and not self.commit_mines:
            up = set(L.stairs_up)
            if A.agent in up:
                self.note(f"Mines retreat: going up (xplvl {A.xplvl})")
                return "up"
            if up:
                p = L.bfs(A.agent, up,
                          avoid=self._suspects() | (self._mcells() - up))
                if p is None:
                    p = L.bfs(A.agent, up, avoid=self._suspects())
                if p:
                    return self._step_path(p)

        goals = set(L.stairs_down) | set(L.holes)
        banned = self.mines_entrances.get(A.key, set())
        if banned and A.xplvl <= 3:
            since = self.mines_avoid_since.setdefault(A.key, A.time)
            if A.time - since < 800:
                if goals - banned:
                    goals = goals - banned
                elif goals:
                    # only the Mines '>' is known: keep exploring for the
                    # main-branch one instead of re-entering
                    goals = set()
            else:
                if not self.commit_mines:
                    self.commit_mines = True
                    self.note("Mines avoidance timed out: committing to Mines")
        if not goals:
            return None
        mcells = self._mcells()
        avoid = self._suspects() | (mcells - goals)
        path = L.bfs(A.agent, goals, avoid=avoid)
        if path is None:
            path = L.bfs(A.agent, goals, avoid=self._suspects())
        if path is None:
            path = L.bfs(A.agent, goals, avoid=self._suspects(),
                         bad_traps_ok=True)
        if path == []:
            # standing on the goal
            if self._should_rest() and self._rest_here_ok():
                self.rest_budget[A.key] = self.rest_budget.get(A.key, 0) + 1
                return "search"
            self.descended_from = (A.key, A.agent)
            return "down"
        if path:
            return self._step_path(path)
        return None

    # ------------------------------------------------- digger acquisition
    def _acquire_digger(self, obs):
        A = self.atlas
        if self.digger_letter:
            return None
        msg = A.message
        m = RE_SEE_HERE.search(msg)
        if m and ("pick-axe" in m.group(1) or "mattock" in m.group(1)):
            self.note(f"picking up digger: {m.group(1)}")
            self.queue_tag = "pickup"
            return "pickup"
        # spot digger object glyphs on the map and detour if close
        glyphs = obs["obs"]["glyphs"]
        best = None
        ax, ay = A.agent
        for y in range(C.ROWS):
            for x in range(C.COLS):
                if int(glyphs[y][x]) in DIGGER_GLYPHS:
                    d = max(abs(x - ax), abs(y - ay))
                    if d <= 20 and (best is None or d < best[0]):
                        best = (d, (x, y))
        if best:
            if best[1] == A.agent:
                self.queue_tag = "pickup"
                return "pickup"
            path = A.level.bfs(A.agent, [best[1]],
                               avoid=self._suspects() | self._mcells())
            if path:
                if len(path) == 1:
                    self.note(f"digger spotted at {best[1]}: stepping on")
                return self._step_path(path)
        return None

    # ------------------------------------------------------------- explore
    def _explore(self, obs):
        A = self.atlas
        L = A.level
        frontier = L.frontier_cells()
        if not frontier:
            return None
        fset = set(frontier)
        mcells = self._mcells()
        avoid = self._suspects() | mcells
        path = None
        if self.explore_target in fset:
            path = L.bfs(A.agent, [self.explore_target], avoid=avoid)
        if not path:
            tgt = self._pick_frontier(L, fset, avoid)
            if tgt is not None:
                path = L.bfs(A.agent, [tgt], avoid=avoid)
                self.explore_target = tgt
            if not path:
                path = L.bfs(A.agent, frontier, avoid=avoid)
        if path is None:
            path = L.bfs(A.agent, frontier, avoid=self._suspects())
        if path:
            return self._step_path(path)
        if path == []:
            # on a frontier cell: any passable step into the unknown
            for name, (dx, dy) in DIRS.items():
                nx, ny = A.agent[0] + dx, A.agent[1] + dy
                if not (0 <= nx < C.COLS and 0 <= ny < C.ROWS):
                    continue
                if (nx, ny) in self._suspects() or (nx, ny) in mcells:
                    continue
                if not L.explored[ny][nx]:
                    if name in CARDINALS:
                        return name
            for name, (dx, dy) in DIRS.items():
                nx, ny = A.agent[0] + dx, A.agent[1] + dy
                if 0 <= nx < C.COLS and 0 <= ny < C.ROWS and \
                        not L.explored[ny][nx] and \
                        (nx, ny) not in self._suspects():
                    return name
        return None

    def _pick_frontier(self, L, fset, avoid):
        from collections import deque
        dist = {self.atlas.agent: 0}
        q = deque([self.atlas.agent])
        while q:
            cur = q.popleft()
            for _n, nxt in L.neighbors(*cur, avoid=avoid):
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        best = None
        for cell in fset:
            if cell not in dist:
                continue
            x, y = cell
            mass = 0
            for yy in range(max(0, y - 3), min(C.ROWS, y + 4)):
                for xx in range(max(0, x - 3), min(C.COLS, x + 4)):
                    if not L.explored[yy][xx]:
                        mass += 1
            score = dist[cell] - 0.55 * mass
            if best is None or score < best[0]:
                best = (score, cell)
        return best[1] if best else None

    # --------------------------------------------------------------- doors
    def _doors(self, obs):
        A = self.atlas
        L = A.level
        doors = [d for d in L.find_terrain(C.DOOR_CLOSED)
                 if d not in self.door_giveup]
        if not doors:
            return None
        mcells = self._mcells()
        # adjacent (cardinal) closed door? open / kick it
        for door in doors:
            ddx, ddy = door[0] - A.agent[0], door[1] - A.agent[1]
            if (ddx, ddy) in DIR_OF and (ddx == 0 or ddy == 0):
                dname = DIR_OF[(ddx, ddy)]
                self.door_target = door
                if "This door is locked" in A.message or self.kick_dir == dname:
                    self.kick_dir = dname
                    self.kick_count += 1
                    if self.kick_count > 12:
                        self.door_giveup.add(door)
                        self.kick_dir = None
                        self.kick_count = 0
                        self.note(f"giving up on locked door {door}")
                        return None
                    self.queue = [dname]
                    self.queue_tag = "kick"
                    return "kick"
                self.queue = [dname]
                self.queue_tag = "open"
                return "open"
        # walk cardinal-adjacent to the nearest closed door
        targets = set()
        for (dx_, dy_) in doors:
            for name in CARDINALS:
                ddx, ddy = DIRS[name]
                ax2, ay2 = dx_ - ddx, dy_ - ddy
                if L.passable(ax2, ay2, doors_ok=False):
                    targets.add((ax2, ay2))
        if targets:
            path = L.bfs(A.agent, targets, avoid=self._suspects() | mcells,
                         doors_ok=False)
            if path:
                return self._step_path(path)
        return None

    # ------------------------------------------------------ hidden search
    def _hidden_search(self, obs):
        A = self.atlas
        L = A.level
        counts = L.search_counts
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
                # dead-end bonus: hidden corridors continue from dead ends
                deg = sum(1 for _ in L.neighbors(x, y))
                # secret doors live in ROOM WALLS: any cell cardinal-adjacent
                # to a real (non-inferred) wall is a candidate host (dev seed
                # 103: 19.5k searches at dead ends only, secret door in a
                # room wall never probed)
                wall_adj = 0
                for name in CARDINALS:
                    ddx, ddy = DIRS[name]
                    nx2, ny2 = x + ddx, y + ddy
                    if 0 <= nx2 < C.COLS and 0 <= ny2 < C.ROWS and \
                            L.terrain[ny2][nx2] == C.WALL and \
                            not L.inferred_wall[ny2][nx2]:
                        wall_adj += 1
                if pot > 0 or deg <= 1 or wall_adj > 0:
                    rounds = counts.get((x, y), 0) // 6
                    cands.append((rounds,
                                  -(pot + (4 if deg <= 1 else 0) + wall_adj),
                                  abs(x - A.agent[0]) + abs(y - A.agent[1]),
                                  (x, y)))
        if cands:
            cands.sort()
            # walk the ranking until a REACHABLE candidate (a never-melee
            # blocker can make the top pick unreachable forever — dev 116)
            for _r, _p, _d, tgt in cands[:60]:
                if tgt == A.agent:
                    counts[tgt] = counts.get(tgt, 0) + 1
                    return "search"
                path = L.bfs(A.agent, [tgt], avoid=self._mcells())
                if path:
                    return self._step_path(path)
        counts[A.agent] = counts.get(A.agent, 0) + 1
        return "search"

    # ---------------------------------------------------------- path steps
    def _step_path(self, path):
        A = self.atlas
        L = A.level
        step = path[0]
        dx, dy = DIRS[step]
        nx, ny = A.agent[0] + dx, A.agent[1] + dy
        # closed door ahead: open instead of bumping
        if L.terrain[ny][nx] == C.DOOR_CLOSED:
            if dx == 0 or dy == 0:
                self.door_target = (nx, ny)
                if "This door is locked" in A.message or self.kick_dir == step:
                    self.kick_dir = step
                    self.kick_count += 1
                    if self.kick_count > 12:
                        self.door_giveup.add((nx, ny))
                        self.suspect_walls.setdefault(A.key, set()).add((nx, ny))
                        self.kick_dir = None
                        self.kick_count = 0
                        return "search"
                    self.queue = [step]
                    self.queue_tag = "kick"
                    return "kick"
                self.queue = [step]
                self.queue_tag = "open"
                return "open"
            return "search"
        # monster on the next cell
        for m in L.monsters:
            if m.pos == (nx, ny):
                if m.pet:
                    return step            # swap places with pet
                if m.pos in L.no_attack:
                    return self._detour(path, (nx, ny))
                if self._never_melee(m):
                    return self._detour(path, (nx, ny))
                return step                # attack by moving in
        return step

    def _detour(self, path, cell):
        A = self.atlas
        L = A.level
        # rebuild path avoiding this cell; fall back to waiting
        tgt = self._path_end(path)
        p = L.bfs(A.agent, [tgt], avoid=self._suspects() | {cell})
        if p:
            return p[0] if p[0] != path[0] else p[0]
        return "search"

    def _path_end(self, path):
        x, y = self.atlas.agent
        for stp in path:
            dx, dy = DIRS[stp]
            x, y = x + dx, y + dy
        return (x, y)
