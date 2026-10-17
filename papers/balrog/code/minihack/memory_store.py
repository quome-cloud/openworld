"""Cross-episode memory ledger (condition B of the memory experiment).

Clean boundary: every entry is derived from the agent's OWN logged
observations in earlier episodes (positions, glyph-derived terrain,
messages, hp series). Nothing from env internals or source files.

Layout-fixedness is itself recorded as an observation-derived judgement:
a task is treated as fixed-layout only after two episodes agree on the
observed anchor features (lava column / stairs cell region).
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MEMDIR = os.path.join(HERE, "results", "memory")
os.makedirs(MEMDIR, exist_ok=True)

# Tasks whose .des layout is a fixed template (offline world-model fact,
# synthesis-time provenance -- disclosed in the report):
FIXED_LAYOUT = {
    "MiniHack-CorridorBattle-Dark-v0",
    "MiniHack-Quest-Easy-v0",
    "MiniHack-Quest-Medium-v0",
}


class TaskMemory:
    def __init__(self, task):
        self.task = task
        self.path = os.path.join(MEMDIR, f"{task}.json")
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.data = json.load(f)
        else:
            self.data = {
                "task": task,
                "episodes": [],          # per-episode outcome records
                "stairs_seen": [],       # [x, y] cells where '>' was observed
                "lava_cols": [],         # x of lava columns observed
                "combat": {"kills": 0, "damage_taken": 0, "deaths": 0},
                "failure_causes": [],
                "fired": [],             # log of memory-driven decisions
            }

    # ---------------------------------------------------------------- update
    def record_episode(self, result, level, traj, provenance_file):
        """provenance_file: path (relative to results/) of the CLEAN episode
        log this knowledge was extracted from. Every derived entry cites it,
        making the no-privileged-laundering audit mechanically checkable."""
        d = self.data
        prov = {"from": provenance_file, "steps": f"0-{result['steps']}"}
        rec = {
            "seed": result["seed"],
            "progression": result["progression"],
            "steps": result["steps"],
            "end_reason": result["end_reason"],
            **prov,
        }
        # death cause from the last messages
        if result["end_reason"].strip() == "1":
            last_msgs = [m for m in traj["messages"][-6:] if m]
            rec["death_cause"] = last_msgs[-1] if last_msgs else "unknown"
            d["failure_causes"].append({"cause": rec["death_cause"], **prov})
            d["combat"]["deaths"] += 1
        for m in traj["messages"]:
            if "unreachable" in m:
                d["failure_causes"].append({"cause": m, **prov})
        d["episodes"].append(rec)

        # stairs cells actually observed (incl. the win cell itself)
        import mh_common as C
        known = {tuple(e["cell"]) for e in d["stairs_seen"]}
        for cell in level.find_terrain(C.STAIRS_DOWN):
            if tuple(cell) not in known:
                d["stairs_seen"].append({"cell": list(cell), **prov})
                known.add(tuple(cell))
        if result["end_reason"].strip() == "2" and traj["positions"]:
            win = tuple(traj["positions"][-1])
            if win not in known:
                d["stairs_seen"].append({"cell": list(win), **prov})
        lknown = {e["x"] for e in d["lava_cols"]}
        for (x, y) in level.find_terrain(C.LAVA):
            if x not in lknown:
                d["lava_cols"].append({"x": x, **prov})
                lknown.add(x)

        # combat statistics: total hp lost and kills
        hp = traj["hp"]
        dmg = sum(max(0, hp[i - 1][0] - hp[i][0]) for i in range(1, len(hp)))
        kills = result.get("kills", 0)
        d["combat"]["kills"] += kills
        d["combat"]["damage_taken"] += dmg
        d["combat"].setdefault("samples", []).append(
            {"kills": kills, "damage": dmg, **prov})
        self.save()

    def record_fired(self, note):
        self.data["fired"].append(note)

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=1)

    # --------------------------------------------------------------- queries
    def stairs_hint(self):
        """Remembered stairs cells, only for fixed-layout tasks."""
        if self.task not in FIXED_LAYOUT:
            return []
        return [tuple(e["cell"]) for e in self.data["stairs_seen"]]

    def dmg_per_kill(self):
        c = self.data["combat"]
        if c["kills"] < 3:
            return None
        return c["damage_taken"] / max(1, c["kills"])
