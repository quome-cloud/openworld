"""Synthesized symbolic world model of the three BALROG TextWorld domains + planners.

The model is a STRIPS-like reimplementation of the TextWorld dynamics these games use
(verbs: go/open/unlock/take/slice/chop/dice/cook/prepare meal/eat meal), synthesized from
the game files' embedded KB logic rules (see FABLE_TEXTWORLD_REPORT.md section on provenance).

PRIVILEGED protocol: initial state is read from the game's .json spec (world facts + quests);
plans are computed by forward search over this model and executed through the BALROG wrapper.
Nothing is read from metadata.walkthrough / policy_commands at any time.
"""
import heapq
import itertools
import json
from collections import defaultdict

DIRS = ["north", "south", "east", "west"]
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
COOK_STATE_TO_APPLIANCE = {"fried": "stove", "roasted": "oven", "grilled": "toaster"}
CUT_STATES = ("chopped", "sliced", "diced")
COOK_STATES = ("fried", "roasted", "grilled")


class World:
    """Symbolic state loaded from a game .json spec."""

    def __init__(self, path):
        j = json.load(open(path))
        self.infos = dict(j["infos"])
        self.name = {k: v["name"] for k, v in self.infos.items()}
        self.type = {k: v["type"] for k, v in self.infos.items()}
        facts = [(f["name"], [a["name"] for a in f["arguments"]]) for f in j["world"]]
        self.facts = facts

        # rooms + exits
        self.exits = defaultdict(dict)  # room -> dir -> other room
        self.doors = {}  # door -> {state, rooms}
        self.door_on_edge = {}  # (r1,r2) -> door
        self.at = {}  # entity -> room (objects on floor, fixtures)
        self.on = {}  # obj -> supporter
        self.in_ = {}  # obj -> container
        self.state = {}  # door/container -> 'open'|'closed'|'locked'
        self.match = {}  # lock -> key
        self.start_room = None
        self.initial_inv = set()

        for name, args in facts:
            if name in ("north_of", "south_of", "east_of", "west_of"):
                a, b = args  # a is <dir>_of b : from b going <dir> leads to a
                d = name[:-3]
                self.exits[b][d] = a
                self.exits[a][OPP[d]] = b
            elif name == "link":
                r1, d, r2 = args
                self.door_on_edge[(r1, r2)] = d
                self.door_on_edge[(r2, r1)] = d
            elif name == "at":
                ent, r = args
                if ent == "P":
                    self.start_room = r
                else:
                    self.at[ent] = r
            elif name == "on":
                self.on[args[0]] = args[1]
            elif name == "in":
                if args[1] == "I":
                    self.initial_inv.add(args[0])
                elif args[1] != "RECIPE":
                    self.in_[args[0]] = args[1]
            elif name in ("open", "closed", "locked"):
                self.state[args[0]] = name
            elif name == "match":
                self.match[args[1]] = args[0]

        # quests
        self.quests = j["quests"]
        self.metadata = j["metadata"]

        # cooking recipe (if present)
        self.recipe = self._parse_recipe(facts)

    # -- recipe -------------------------------------------------------------
    def _parse_recipe(self, facts):
        base = {}
        slot_states = defaultdict(set)
        for name, args in facts:
            if name == "base":
                base[args[1]] = args[0]  # ingredient_k -> food id
            elif name in CUT_STATES + COOK_STATES and args and args[0].startswith("ingredient"):
                slot_states[args[0]].add(name)
        if not base:
            return None
        recipe = []
        for slot, food in sorted(base.items()):
            cut = next((s for s in CUT_STATES if s in slot_states[slot]), None)
            cook = next((s for s in COOK_STATES if s in slot_states[slot]), None)
            recipe.append({"food": food, "name": self.name[food], "cut": cut, "cook": cook})
        return recipe

    # -- helpers ------------------------------------------------------------
    def room_of(self, ent):
        """Room an object is (transitively) located in."""
        seen = set()
        while ent not in self.at:
            if ent in self.on:
                ent = self.on[ent]
            elif ent in self.in_:
                ent = self.in_[ent]
            else:
                return None
            if ent in seen:
                return None
            seen.add(ent)
        return self.at[ent]

    def holder_of(self, ent):
        """('floor', room) | ('on', supporter) | ('in', container)."""
        if ent in self.on:
            return ("on", self.on[ent])
        if ent in self.in_:
            return ("in", self.in_[ent])
        return ("floor", self.at.get(ent))

    def door_between(self, r1, r2):
        return self.door_on_edge.get((r1, r2))

    def fixtures_in(self, room, typ):
        return [e for e, r in self.at.items() if r == room and self.type.get(e) == typ]

    # -- quest targets ------------------------------------------------------
    def quest_target_and_fail(self):
        """treasure_hunter / coin_collector: (target_obj, fail_obj|None)."""
        target = fail = None
        for q in self.quests:
            for e in q.get("win_events", []):
                for p in (e.get("condition") or {}).get("preconditions", []):
                    if p["name"] == "in" and p["arguments"][1]["name"] == "I":
                        target = p["arguments"][0]["name"]
            for e in q.get("fail_events", []):
                for p in (e.get("condition") or {}).get("preconditions", []):
                    if p["name"] == "in" and p["arguments"][1]["name"] == "I":
                        fail = p["arguments"][0]["name"]
        return target, fail


# ---------------------------------------------------------------------------
# Simulator over the symbolic model: executes a command list, tracking state.
# Used to (a) verify plans before touching the env, (b) count steps exactly.
# ---------------------------------------------------------------------------
class SimError(Exception):
    pass


class Sim:
    def __init__(self, world: World):
        self.w = world
        self.room = world.start_room
        self.state = dict(world.state)
        self.at = dict(world.at)
        self.on = dict(world.on)
        self.in_ = dict(world.in_)
        self.inv = set(world.initial_inv)
        self.cut = defaultdict(lambda: None)
        self.cooked = defaultdict(lambda: None)
        self.prepared = False
        self.eaten = False

    def visible_portables(self):
        vis = []
        for ent in list(self.at) + list(self.on) + list(self.in_):
            if self.w.type.get(ent) not in ("o", "k", "f"):
                continue
            kind, holder = self._holder(ent)
            if kind == "floor" and holder == self.room:
                vis.append(ent)
            elif kind == "on" and self.at.get(holder) == self.room:
                vis.append(ent)
            elif kind == "in" and self.at.get(holder) == self.room and self.state.get(holder) == "open":
                vis.append(ent)
        return vis

    def _holder(self, ent):
        if ent in self.inv:
            return ("inv", None)
        if ent in self.on:
            return ("on", self.on[ent])
        if ent in self.in_:
            return ("in", self.in_[ent])
        return ("floor", self.at.get(ent))

    def step(self, cmd):
        w = self.w
        if cmd.startswith("go "):
            d = cmd[3:]
            if d not in w.exits[self.room]:
                raise SimError(f"no exit {d} from {w.name.get(self.room, self.room)}")
            dest = w.exits[self.room][d]
            door = w.door_between(self.room, dest)
            if door and self.state.get(door) != "open":
                raise SimError(f"door {w.name[door]} not open")
            self.room = dest
        elif cmd.startswith("open "):
            ent = self._resolve(cmd[5:])
            st = self.state.get(ent)
            if st == "locked":
                raise SimError(f"{w.name[ent]} locked")
            if st != "closed":
                raise SimError(f"{w.name[ent]} not closed")
            self.state[ent] = "open"
        elif cmd.startswith("unlock "):
            body = cmd[7:]
            lock_name, key_name = body.split(" with ")
            lock = self._resolve(lock_name)
            key = self._resolve(key_name)
            if self.state.get(lock) != "locked":
                raise SimError(f"{lock} not locked")
            if key not in self.inv:
                raise SimError("key not held")
            if w.match.get(lock) != key:
                raise SimError("key mismatch")
            self.state[lock] = "closed"
        elif cmd.startswith("take "):
            name = cmd[5:]
            if " from " in name:
                name = name.split(" from ")[0]
            ent = self._resolve(name)
            if ent not in self.visible_portables():
                raise SimError(f"{name} not visible")
            self.on.pop(ent, None)
            self.in_.pop(ent, None)
            self.at.pop(ent, None)
            self.inv.add(ent)
        elif any(cmd.startswith(v + " ") for v in ("slice", "chop", "dice")):
            verb = cmd.split()[0]
            name = cmd[len(verb) + 1:].split(" with ")[0]
            ent = self._resolve(name)
            if ent not in self.inv:
                raise SimError("food not held")
            knife = next((k for k in self.inv if self.w.name.get(k) == "knife"), None)
            if not knife:
                raise SimError("no knife")
            if self.cut[ent] is not None:
                raise SimError("already cut -> would fail quest")
            self.cut[ent] = {"slice": "sliced", "chop": "chopped", "dice": "diced"}[verb]
        elif cmd.startswith("cook "):
            name = cmd[5:].split(" with ")[0]
            app_name = cmd.split(" with ")[1]
            ent = self._resolve(name)
            if ent not in self.inv:
                raise SimError("food not held")
            apps = [e for e, r in self.at.items() if r == self.room and self.w.name.get(e) == app_name]
            if not apps:
                raise SimError(f"no {app_name} here")
            if self.cooked[ent] is not None:
                raise SimError("already cooked -> burned")
            typ = self.w.type[apps[0]]
            self.cooked[ent] = {"stove": "fried", "oven": "roasted", "toaster": "grilled"}[typ]
        elif cmd == "prepare meal":
            if not self.w.recipe:
                raise SimError("no recipe")
            for slot in self.w.recipe:
                f = slot["food"]
                if f not in self.inv:
                    raise SimError(f"missing {slot['name']}")
                if self.cut[f] != slot["cut"] or self.cooked[f] != slot["cook"]:
                    raise SimError(f"{slot['name']} state {self.cut[f]},{self.cooked[f]} != {slot['cut']},{slot['cook']}")
            # cooking_location is the kitchen: room named 'kitchen'
            if self.w.name.get(self.room) != "kitchen":
                raise SimError("not at cooking location")
            for slot in self.w.recipe:  # make/meal consumes the ingredients (non-$ in(f,I) in the KB rule)
                self.inv.discard(slot["food"])
            self.prepared = True
        elif cmd == "eat meal":
            if not self.prepared:
                raise SimError("no meal")
            self.eaten = True
        else:
            raise SimError(f"unmodeled command {cmd}")

    def _resolve(self, name):
        cands = [e for e, n in self.w.name.items() if n == name]
        if not cands:
            raise SimError(f"unknown entity {name}")
        return cands[0]


# ---------------------------------------------------------------------------
# Navigation on the modeled map (doors open along the way, opens persist).
# ---------------------------------------------------------------------------
def nav(world, room_from, room_to, door_states, keys_held=frozenset()):
    """Dijkstra over rooms; cost = steps incl. door open/unlock. Returns (cost, commands, doors_opened, doors_unlocked)."""
    if room_from == room_to:
        return 0, [], [], []
    pq = [(0, room_from, ())]
    best = {room_from: 0}
    back = {}
    while pq:
        c, r, _ = heapq.heappop(pq)
        if r == room_to:
            break
        if c > best.get(r, 1e9):
            continue
        for d, dest in world.exits[r].items():
            door = world.door_between(r, dest)
            step_cost = 1
            if door:
                st = door_states.get(door, "open")
                if st == "closed":
                    step_cost += 1
                elif st == "locked":
                    key = world.match.get(door)
                    if key not in keys_held:
                        continue
                    step_cost += 2
            nc = c + step_cost
            if nc < best.get(dest, 1e9):
                best[dest] = nc
                back[dest] = (r, d, door)
                heapq.heappush(pq, (nc, dest, ()))
    if room_to not in back:
        return None
    cmds, opened, unlocked = [], [], []
    seq = []
    r = room_to
    while r != room_from:
        pr, d, door = back[r]
        seq.append((pr, d, door, r))
        r = pr
    seq.reverse()
    ds = dict(door_states)
    for pr, d, door, r in seq:
        if door:
            st = ds.get(door, "open")
            if st == "locked":
                cmds.append(f"unlock {world.name[door]} with {world.name[world.match[door]]}")
                unlocked.append(door)
                st = "closed"
            if st == "closed":
                cmds.append(f"open {world.name[door]}")
                opened.append(door)
                ds[door] = "open"
        cmds.append(f"go {d}")
    return len(cmds), cmds, opened, unlocked


# ---------------------------------------------------------------------------
# Task planners (forward search over the model)
# ---------------------------------------------------------------------------
def plan_coin(world):
    target, _ = world.quest_target_and_fail()
    room = world.room_of(target)
    res = nav(world, world.start_room, room, dict(world.state))
    assert res is not None
    _, cmds, _, _ = res
    return cmds + [f"take {world.name[target]}"]


def take_command(world, ent, holder_kind, holder):
    n = world.name[ent]
    if holder_kind in ("on", "in"):
        return f"take {n} from {world.name[holder]}"
    return f"take {n}"


def plan_treasure(world):
    """Uniform-cost forward search over (room, inventory, door/container states)."""
    target, fail = world.quest_target_and_fail()
    keys = [e for e, t in world.type.items() if t == "k" and e != fail]
    interesting = set(keys) | {target}

    def holder(ent):
        return world.holder_of(ent)

    init_states = tuple(sorted(world.state.items()))
    start = (world.start_room, frozenset(world.initial_inv & (set(keys) | {target})), init_states)
    pq = [(0, 0, start, [])]
    seen = {start: 0}
    tiebreak = 0
    while pq:
        cost, _, (room, inv, states), cmds = heapq.heappop(pq)
        if target in inv:
            return cmds
        if cost > seen.get((room, inv, states), 1e9):
            continue
        sd = dict(states)

        succs = []
        # movement (one edge at a time keeps search exact)
        for d, dest in world.exits[room].items():
            door = world.door_between(room, dest)
            pre = []
            if door:
                st = sd.get(door, "open")
                if st == "locked":
                    key = world.match.get(door)
                    if key not in inv:
                        continue
                    pre = [f"unlock {world.name[door]} with {world.name[key]}",
                           f"open {world.name[door]}"]
                    ns = dict(sd)
                    ns[door] = "open"
                elif st == "closed":
                    pre = [f"open {world.name[door]}"]
                    ns = dict(sd)
                    ns[door] = "open"
                else:
                    ns = sd
            else:
                ns = sd
            succs.append((len(pre) + 1, pre + [f"go {d}"], dest, inv, ns))
        # take interesting visible objects here
        for ent in interesting:
            if ent in inv:
                continue
            kind, h = holder(ent)
            if kind == "floor" and h == room:
                succs.append((1, [take_command(world, ent, kind, h)], room, inv | {ent}, sd))
            elif kind == "on" and world.at.get(h) == room:
                succs.append((1, [take_command(world, ent, kind, h)], room, inv | {ent}, sd))
            elif kind == "in" and world.at.get(h) == room:
                st = sd.get(h, "open")
                pre = []
                ns = sd
                if st == "locked":
                    key = world.match.get(h)
                    if key not in inv:
                        continue
                    pre = [f"unlock {world.name[h]} with {world.name[key]}", f"open {world.name[h]}"]
                    ns = dict(sd)
                    ns[h] = "open"
                elif st == "closed":
                    pre = [f"open {world.name[h]}"]
                    ns = dict(sd)
                    ns[h] = "open"
                succs.append((len(pre) + 1, pre + [take_command(world, ent, kind, h)], room, inv | {ent}, ns))

        for dc, dcmds, nroom, ninv, nsd in succs:
            nstate = (nroom, ninv, tuple(sorted(nsd.items())))
            ncost = cost + dc
            if ncost < seen.get(nstate, 1e9):
                seen[nstate] = ncost
                tiebreak += 1
                heapq.heappush(pq, (ncost, tiebreak, nstate, cmds + dcmds))
    return None


def plan_cooking(world):
    """Collect ingredients (+knife), cook (cook-first, walkthrough-proven order), cut, prepare, eat."""
    recipe = world.recipe
    needs_cut = any(s["cut"] for s in recipe)
    needed = [s["food"] for s in recipe]
    knife = next((e for e, n in world.name.items() if n == "knife" and world.type.get(e) == "o"), None)
    if needs_cut:
        assert knife is not None
        needed = needed + [knife]

    # pickup stops grouped by room
    stops = defaultdict(list)  # room -> [(ent, kind, holder)]
    for ent in needed:
        if ent in world.initial_inv:
            continue
        kind, h = world.holder_of(ent)
        room = world.room_of(ent)
        stops[room].append((ent, kind, h))

    kitchen = next(r for r, n in world.name.items() if n == "kitchen" and world.type.get(r) == "r")
    # appliance rooms needed
    app_rooms = []
    for s in recipe:
        if s["cook"]:
            app_t = COOK_STATE_TO_APPLIANCE[s["cook"]]
            app = next(e for e, t in world.type.items() if t == app_t)
            app_rooms.append((world.at[app], app, s))

    best_plan = None
    stop_rooms = list(stops.keys())
    for perm in itertools.permutations(stop_rooms):
        cmds = []
        ds = dict(world.state)
        cur = world.start_room
        opened_containers = set()
        ok = True
        for room in perm:
            res = nav(world, cur, room, ds)
            if res is None:
                ok = False
                break
            _, ncmds, opened, unlocked = res
            cmds += ncmds
            for d in opened:
                ds[d] = "open"
            cur = room
            for ent, kind, h in stops[room]:
                if kind == "in" and ds.get(h) in ("closed", "locked") and h not in opened_containers:
                    cmds.append(f"open {world.name[h]}")
                    opened_containers.add(h)
                    ds[h] = "open"
                cmds.append(take_command(world, ent, kind, h))
        if not ok:
            continue
        # cooking trips: unique appliance rooms, try both orders
        rooms_needed = []
        for r, app, s in app_rooms:
            if r not in rooms_needed:
                rooms_needed.append(r)
        room_orders = list(itertools.permutations(rooms_needed)) or [()]
        for order in room_orders:
            c2 = list(cmds)
            ds2 = dict(ds)
            cur2 = cur
            for room in order:
                res = nav(world, cur2, room, ds2)
                if res is None:
                    break
                _, ncmds, opened, _ = res
                c2 += ncmds
                for d in opened:
                    ds2[d] = "open"
                cur2 = room
                for r, app, s in app_rooms:
                    if r == room:
                        c2.append(f"cook {s['name']} with {world.name[app]}")
            else:
                # cuts (anywhere), then kitchen: prepare + eat
                for s in recipe:
                    if s["cut"]:
                        verb = {"sliced": "slice", "chopped": "chop", "diced": "dice"}[s["cut"]]
                        c2.append(f"{verb} {s['name']} with knife")
                res = nav(world, cur2, kitchen, ds2)
                if res is None:
                    continue
                _, ncmds, _, _ = res
                c2 += ncmds + ["prepare meal", "eat meal"]
                if best_plan is None or len(c2) < len(best_plan):
                    best_plan = c2
    return best_plan


def plan(world, task):
    if task == "coin_collector":
        return plan_coin(world)
    if task == "treasure_hunter":
        return plan_treasure(world)
    if task == "the_cooking_game":
        return plan_cooking(world)
    raise ValueError(task)


def verify_plan(world, task, cmds):
    """Replay plan on the symbolic simulator; returns (ok, err, steps)."""
    sim = Sim(world)
    try:
        for c in cmds:
            sim.step(c)
    except SimError as e:
        return False, str(e), None
    if task == "coin_collector" or task == "treasure_hunter":
        target, fail = world.quest_target_and_fail()
        ok = target in sim.inv and (fail is None or fail not in sim.inv)
        return ok, None if ok else "target not held", len(cmds)
    else:
        return sim.eaten, None if sim.eaten else "meal not eaten", len(cmds)
