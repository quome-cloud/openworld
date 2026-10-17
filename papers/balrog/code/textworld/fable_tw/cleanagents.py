"""CLEAN-protocol agents: text-only, closed-loop (act -> parse -> update belief -> replan).

Inputs per step: the BALROG-served observation text and the done flag. Nothing else.
Pure code at runtime; grammar knowledge lives in textparse.py, world dynamics knowledge
(verbs, preconditions) mirrors the synthesized model in worldmodel.py.
"""
from collections import deque

from .textparse import parse, CONTAINER_NOUNS, KEY_NOUNS, APPLIANCES, base_name

DIRS = ["north", "east", "south", "west"]
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
COOK_VERB_TO_APP = {"fry": "stove", "grill": "BBQ", "roast": "oven"}
CUT_VERBS = ("slice", "chop", "dice")


class Belief:
    def __init__(self):
        self.rooms = {}          # name -> {"exits": {dir: {...}}, "visited": bool}
        self.cur = None
        self.start = None
        self.last_dir = None
        self.items = {}          # portable name -> {"kind","holder","room"}
        self.containers = {}     # name -> {"room","state","open_attempted","lock_key","revealed"}
        self.inventory = set()
        self.appliances = {}     # BBQ/oven/stove -> room
        self.recipe = None
        self.cooked = set()
        self.cut = set()
        self.prepared = False
        self.locked_doors = {}   # door -> {"key": name|None}
        self.dead_takes = set()  # items that errored on take
        self.ever_locked = set() # doors/containers ever observed locked
        self.failed_unlocks = set()  # (thing, key_item_name) attempts that failed

    def room(self, name):
        if name not in self.rooms:
            self.rooms[name] = {"exits": {}, "visited": False}
        return self.rooms[name]

    # ------------------------------------------------------------------
    def update(self, obs_text, last_cmd):
        p = parse(obs_text)
        ev_names = [e[0] for e in p["events"]]

        # feedback events
        for e in p["events"]:
            if e[0] == "took":
                self.inventory.add(e[1])
                self.items.pop(e[1], None)
            elif e[0] == "revealed":
                c, items = e[1], e[2]
                cont = self.containers.setdefault(c, {"room": self.cur, "state": None, "open_attempted": True,
                                                      "lock_key": None, "revealed": False})
                cont["state"] = "open"
                cont["revealed"] = True
                for it in items:
                    self.items[it] = {"kind": "in", "holder": c, "room": cont["room"] or self.cur}
            elif e[0] == "opened":
                name = e[1]
                if name in self.containers:
                    self.containers[name]["state"] = "open"
                    self.containers[name]["revealed"] = True  # opened with no reveal = empty
                self._set_door_state(name, "open")
            elif e[0] == "unlocked":
                name = e[1]
                if name in self.containers:
                    self.containers[name]["state"] = "closed"
                    self.containers[name]["open_attempted"] = False  # reopen after unlock
                self._set_door_state(name, "closed")
                self.locked_doors.pop(name, None)
            elif e[0] == "locked":
                thing, key = e[1], e[2]
                self.ever_locked.add(thing)
                if thing in self.containers or (last_cmd or "").startswith("open ") and thing not in self._door_names():
                    cont = self.containers.setdefault(thing, {"room": self.cur, "state": None,
                                                              "open_attempted": True, "lock_key": None,
                                                              "revealed": False})
                    cont["state"] = "locked"
                    cont["lock_key"] = key
                if thing in self._door_names():
                    self._set_door_state(thing, "locked")
                    self.locked_doors[thing] = {"key": key}
            elif e[0] == "must_open":
                self._set_door_state(e[1], "closed")
            elif e[0] == "cooked":
                self.cooked.add(e[2])
            elif e[0] == "cut":
                self.cut.add(e[2])
            elif e[0] == "prepared":
                self.prepared = True
                self.inventory.add("meal")
            elif e[0] in ("wrong_key", "which_mean") and last_cmd and last_cmd.startswith("unlock "):
                thing, key = last_cmd[7:].split(" with ")
                self.failed_unlocks.add((thing, key))
            elif e[0] == "carrying":
                for it in e[1]:
                    self.inventory.add(it)
                    self.items.pop(it, None)
            elif e[0] == "cant_see" and last_cmd:
                # display names carry flavor adjectives the parser can't split from the
                # entity name; degrade gracefully by dropping the leading word and retrying
                if last_cmd.startswith("open "):
                    name = last_cmd[5:]
                    if name in self.containers:
                        info = self.containers.pop(name)
                        if len(name.split()) > 1:
                            short = " ".join(name.split()[1:])
                            info["open_attempted"] = False
                            self.containers.setdefault(short, info)
                elif last_cmd.startswith("take "):
                    nm = last_cmd[5:].split(" from ")[0]
                    if nm in self.items and len(nm.split()) > 1:
                        self.items[" ".join(nm.split()[1:])] = self.items.pop(nm)
                    else:
                        self.items.pop(nm, None)
                        self.dead_takes.add(nm)
            elif e[0] == "cant_go" and last_cmd and last_cmd.startswith("go ") and self.cur:
                d = last_cmd[3:]
                self.room(self.cur)["exits"].setdefault(d, {}).update({"dead": True})

        if p["recipe"]:
            self.recipe = p["recipe"]

        # room snapshot
        if p["room"]:
            prev = self.cur
            newroom = p["room"]
            moved = (last_cmd or "").startswith("go ") and newroom != prev
            self.cur = newroom
            R = self.room(newroom)
            R["visited"] = True
            if moved and prev:
                d = last_cmd[3:]
                self.room(prev)["exits"].setdefault(d, {"door": None, "door_state": None})["dest"] = newroom
                R["exits"].setdefault(OPP[d], {"door": None, "door_state": None})["dest"] = prev
                self.last_dir = d
            for d, ex in p["exits"].items():
                slot = R["exits"].setdefault(d, {"door": None, "door_state": None})
                slot["door"] = ex["door"] or slot.get("door")
                if ex["door_state"]:
                    # don't downgrade a known-locked door to closed on re-描述 (desc says closed for locked)
                    if not (slot.get("door_state") == "locked" and ex["door_state"] == "closed"):
                        slot["door_state"] = ex["door_state"]
            for it in p["floor"]:
                if it not in self.inventory:
                    self.items[it] = {"kind": "floor", "holder": newroom, "room": newroom}
            for sup, its in p["on"].items():
                for it in its:
                    if it not in self.inventory:
                        self.items[it] = {"kind": "on", "holder": sup, "room": newroom}
            for c, its in p["contains"].items():
                cont = self.containers.setdefault(c, {"room": newroom, "state": "open", "open_attempted": True,
                                                      "lock_key": None, "revealed": True})
                cont.update({"room": newroom, "state": "open", "revealed": True})
                for it in its:
                    if it not in self.inventory:
                        self.items[it] = {"kind": "in", "holder": c, "room": newroom}
            for c, st in p["containers"].items():
                if self._is_container_noun(c):
                    cont = self.containers.setdefault(c, {"room": newroom, "state": None, "open_attempted": False,
                                                          "lock_key": None, "revealed": False})
                    cont["room"] = newroom
                    if cont["state"] != "open":
                        cont["state"] = st
            for s in p["sightings"]:
                if self._is_container_noun(s) and s not in self.containers:
                    self.containers[s] = {"room": newroom, "state": None, "open_attempted": False,
                                          "lock_key": None, "revealed": False}
            for app in p["appliances"]:
                self.appliances.setdefault(app, newroom)
            if self.start is None:
                self.start = newroom
        return p

    def _is_container_noun(self, name):
        last = name.split()[-1] if name else ""
        return last in [c.split()[-1] for c in CONTAINER_NOUNS] or name in CONTAINER_NOUNS

    def _door_names(self):
        out = set()
        for r in self.rooms.values():
            for ex in r["exits"].values():
                if ex.get("door"):
                    out.add(ex["door"])
        return out

    def _set_door_state(self, door, state):
        for r in self.rooms.values():
            for ex in r["exits"].values():
                if ex.get("door") == door:
                    ex["door_state"] = state

    # ------------------------------------------------------------------
    def bfs(self, src, goal_pred, allow_locked=False):
        """Shortest command-path to nearest room satisfying goal_pred. Returns (room, cmds)."""
        q = deque([(src, [])])
        seen = {src}
        while q:
            r, cmds = q.popleft()
            if goal_pred(r) and r != src:
                return r, cmds
            R = self.rooms.get(r)
            if not R:
                continue
            for d, ex in sorted(R["exits"].items()):
                dest = ex.get("dest")
                if dest is None or dest in seen or ex.get("dead"):
                    continue
                st = ex.get("door_state")
                if st == "locked" and not allow_locked:
                    continue
                step = ([f"open {ex['door']}"] if st == "closed" else []) + [f"go {d}"]
                seen.add(dest)
                q.append((dest, cmds + step))
        return None, None

    def path_to(self, room_name):
        if room_name == self.cur:
            return []
        _, cmds = self.bfs(self.cur, lambda r: r == room_name)
        return cmds

    def unexplored_exit(self, room_name):
        R = self.rooms.get(room_name)
        if not R:
            return None
        prefs = [self.last_dir] + DIRS if self.last_dir else DIRS
        for d in prefs:
            ex = R["exits"].get(d)
            if ex and ex.get("dest") is None and not ex.get("dead") and ex.get("door_state") != "locked":
                return d, ex
        return None


class BaseAgent:
    def __init__(self, memory=None):
        # memory: optional, fully generic, built ONLY from this game's own prior clean
        # episodes (see run_memory.py): {"avoid": set of object names whose take ended an
        # episode in a loss, "target": object name whose take ended an episode in a win}
        self.memory = memory or {"avoid": set(), "target": None}
        self.b = Belief()
        self.last_cmd = None
        self.queue = deque()
        self.steps = 0

    def act(self, obs_text):
        self.b.update(obs_text, self.last_cmd)
        cmd = None
        if self.queue:
            cmd = self.queue.popleft()
        else:
            cmd = self.policy()
        if cmd is None:
            cmd = "look"
        # loop guard: the same non-movement command 3x in a row means a blind spot;
        # hard-invalidate the underlying belief entry
        self.recent = getattr(self, "recent", [])
        self.recent.append(cmd)
        if len(self.recent) >= 3 and self.recent[-1] == self.recent[-2] == self.recent[-3]:
            if cmd.startswith("unlock "):
                thing, key = cmd[7:].split(" with ")
                self.b.failed_unlocks.add((thing, key))
            elif cmd.startswith("open "):
                name = cmd[5:]
                if name in self.b.containers:
                    self.b.containers[name]["state"] = "open"  # stop retrying
            elif cmd.startswith("take "):
                self.b.dead_takes.add(cmd[5:].split(" from ")[0])
        self.last_cmd = cmd
        self.steps += 1
        return cmd

    # exploration primitive shared by all agents
    def explore_cmd(self):
        b = self.b
        ue = b.unexplored_exit(b.cur)
        if ue:
            d, ex = ue
            if ex.get("door_state") == "closed":
                return f"open {ex['door']}"
            return f"go {d}"
        room, cmds = b.bfs(b.cur, lambda r: b.unexplored_exit(r) is not None)
        if cmds:
            return cmds[0]
        return None


class CoinAgent(BaseAgent):
    def policy(self):
        b = self.b
        it = b.items.get("coin")
        if it and it["room"] == b.cur:
            return "take coin"
        if it:
            p = b.path_to(it["room"])
            if p:
                return p[0]
        return self.explore_cmd()


class TreasureAgent(BaseAgent):
    def __init__(self, memory=None):
        super().__init__(memory)
        self.inv_checked = False
        self.harvest_goal = None

    def policy(self):
        b = self.b
        if not self.inv_checked:
            self.inv_checked = True
            return "inventory"
        # 1. probe containers here once (locked ones included: the refusal names the key)
        for c, info in b.containers.items():
            if info["room"] == b.cur and not info["open_attempted"] and info["state"] != "open":
                info["open_attempted"] = True
                return f"open {c}"
        # 2. explore
        e = self.explore_cmd()
        if e:
            return e
        # 3. unlock chains: locked doors/containers with named keys we can fetch
        for thing, meta in list(b.locked_doors.items()) + [
            (c, {"key": i["lock_key"]}) for c, i in b.containers.items() if i["state"] == "locked"
        ]:
            key = meta.get("key")
            if not key:
                continue
            # display names may prefix flavor adjectives; the refusal names the canonical
            # noun. Candidates: exact-name key first, then suffix matches; commands always
            # use the candidate's own full name (avoids parser disambiguation questions).
            def rank(n):
                return 0 if n == key else 1
            held = sorted([n for n in b.inventory
                           if (n == key or n.endswith(" " + key)) and (thing, n) not in b.failed_unlocks],
                          key=rank)
            if held:
                room = self._room_of_thing(thing)
                if room is None:
                    continue
                if b.cur != room:
                    p = b.path_to(room)
                    if p:
                        return p[0]
                    continue
                # on success the 'unlocked' event flips the state; the container/exit
                # branches then open it (no blind pre-queue)
                return f"unlock {thing} with {held[0]}"
            key_item = next((n for n in sorted(b.items, key=rank)
                             if (n == key or n.endswith(" " + key)) and n not in b.dead_takes
                             and (thing, n) not in b.failed_unlocks), None)
            if key_item:
                ki = b.items[key_item]
                if ki["room"] != b.cur:
                    p = b.path_to(ki["room"])
                    if p:
                        return p[0]
                    continue
                return self._take(key_item)
        # 4. harvest: candidates by acquisition depth from start, descending (locked doors
        # crossed weigh extra: the target sits at the end of the quest's unlock chain)
        if self.harvest_goal and (self.harvest_goal in b.inventory or self.harvest_goal in b.dead_takes
                                  or self.harvest_goal not in b.items):
            self.harvest_goal = None
        if self.harvest_goal is None and self.memory.get("target") in b.items:
            self.harvest_goal = self.memory["target"]  # remembered winning take
        if self.harvest_goal is None:
            cands = []
            for it, info in b.items.items():
                if it in b.inventory or it in b.dead_takes:
                    continue
                if it in self.memory.get("avoid", ()):  # remembered fatal take
                    continue
                depth = self._depth(b.start, info["room"])
                if depth is None:
                    depth = 0
                if info["kind"] == "in":
                    depth += 2 if info["holder"] in b.ever_locked else 1
                cands.append((-depth, it))
            cands.sort()
            if cands:
                self.harvest_goal = cands[0][1]
        if self.harvest_goal:
            info = b.items[self.harvest_goal]
            if info["room"] == b.cur:
                return self._take(self.harvest_goal)
            p = b.path_to(info["room"])
            if p:
                return p[0]
            self.b.dead_takes.add(self.harvest_goal)  # unreachable
            self.harvest_goal = None
        return None

    def _take(self, item):
        info = self.b.items.get(item, {})
        if info.get("kind") == "on":
            return f"take {item} from {info['holder']}"
        if info.get("kind") == "in":
            c = self.b.containers.get(info["holder"], {})
            if c.get("state") != "open":
                return f"open {info['holder']}"
            return f"take {item} from {info['holder']}"
        return f"take {item}"

    def _depth(self, a, room):
        """Dijkstra from start; edge cost 1, +4 per (ever-)locked door crossed."""
        if a is None or room is None:
            return None
        import heapq
        pq = [(0, a)]
        best = {a: 0}
        while pq:
            c, r = heapq.heappop(pq)
            if r == room:
                return c
            R = self.b.rooms.get(r)
            if not R:
                continue
            for d, ex in R["exits"].items():
                dest = ex.get("dest")
                if dest is None or ex.get("dead"):
                    continue
                w = 1
                if ex.get("door"):
                    w += 1  # doors start closed: opening cost on first traversal
                    if ex.get("door_state") == "locked" or ex.get("door") in self.b.ever_locked:
                        w += 1  # unlock cost
                nc = c + w
                if nc < best.get(dest, 1e9):
                    best[dest] = nc
                    heapq.heappush(pq, (nc, dest))
        return None

    def _room_of_thing(self, thing):
        if thing in self.b.containers:
            return self.b.containers[thing]["room"]
        for rname, R in self.b.rooms.items():
            for d, ex in R["exits"].items():
                if ex.get("door") == thing:
                    return rname
        return None

    def _door_dir(self, room, door):
        R = self.b.rooms.get(room, {"exits": {}})
        for d, ex in R["exits"].items():
            if ex.get("door") == door:
                return d
        return None


class CookingAgent(BaseAgent):
    def __init__(self, memory=None):
        super().__init__(memory)
        self.examined = False
        self.cook_plan = None

    def policy(self):
        b = self.b
        # open containers opportunistically (fridge etc.)
        for c, info in b.containers.items():
            if info["room"] == b.cur and not info["open_attempted"] and info["state"] in (None, "closed"):
                info["open_attempted"] = True
                return f"open {c}"
        # 1. get the recipe
        if b.recipe is None:
            if b.cur == "kitchen" and not self.examined:
                self.examined = True
                return "examine cookbook"
            k = b.rooms.get("kitchen")
            if k and b.cur != "kitchen":
                p = b.path_to("kitchen")
                if p:
                    return p[0]
            return self.explore_cmd()
        # 2. collect needed items
        needed = list(b.recipe["ingredients"])
        if b.recipe["cuts"]:
            needed.append("knife")
        missing = [n for n in needed if n not in b.inventory]
        located = [(n, b.items[n]) for n in missing if n in b.items and n not in b.dead_takes]
        if located:
            # nearest first
            scored = []
            for n, info in located:
                d = 0 if info["room"] == b.cur else len(b.path_to(info["room"]) or [99])
                scored.append((d, n, info))
            scored.sort(key=lambda x: x[0])
            d, n, info = scored[0]
            if info["room"] != b.cur:
                p = b.path_to(info["room"])
                if p:
                    return p[0]
            return self._take(n)
        if missing:
            e = self.explore_cmd()
            if e:
                return e
            return None
        # 3. cook (cook-first order, proven by walkthrough corpus)
        for ing, verb in b.recipe["cooks"].items():
            if ing in b.cooked:
                continue
            app = COOK_VERB_TO_APP[verb]
            room = b.appliances.get(app)
            if room is None:
                e = self.explore_cmd()
                return e
            if b.cur != room:
                p = b.path_to(room)
                if p:
                    return p[0]
                continue
            return f"cook {ing} with {app}"
        # 4. cut
        for ing, verb in b.recipe["cuts"].items():
            if ing not in b.cut:
                return f"{verb} {ing} with knife"
        # 5. prepare + eat at kitchen
        if not b.prepared:
            if b.cur != "kitchen":
                p = b.path_to("kitchen")
                if p:
                    return p[0]
            return "prepare meal"
        return "eat meal"

    def _take(self, item):
        info = self.b.items.get(item, {})
        if info.get("kind") == "on":
            return f"take {item} from {info['holder']}"
        if info.get("kind") == "in":
            c = self.b.containers.get(info["holder"], {})
            if c.get("state") != "open":
                return f"open {info['holder']}"
            return f"take {item} from {info['holder']}"
        return f"take {item}"


def make_agent(task, memory=None):
    cls = {"coin_collector": CoinAgent, "treasure_hunter": TreasureAgent, "the_cooking_game": CookingAgent}[task]
    return cls(memory)
