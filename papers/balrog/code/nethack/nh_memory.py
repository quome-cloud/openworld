"""Cross-episode memory ledger for the NetHack arm (condition B).

Clean boundary: every entry derives from the agent's OWN logged clean
observations (messages, blstats, end_reason strings from the wrapper's
get_stats — the same method BALROG's evaluator calls). Every entry cites
the transition file + step range it came from, making the audit mechanical.

NetHack-specific design (vs the MiniHack ledger): layouts do NOT transfer
across NetHackChallenge episodes (each seed is a fresh dungeon + role), and
item appearances are shuffled per game seed, so only game-invariant
knowledge is stored:
  - species combat statistics (damage per exchange, kills, deaths caused)
  - death records (depth, xplvl, hp, cause)
Derived retrieval effects:
  M1 avoid-species: species that killed us (or with high observed damage)
     are treated as never-melee while the character is weak (xplvl <= 6).
  M2 danger-depth rest gate: the shallowest remembered death depth tightens
     the pre-descent rest threshold (0.5/0.75 -> 0.9/0.95 of hpmax) from one
     level above it, and doubles the per-level rest budget.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MEMDIR = os.path.join(HERE, "results", "memory")
os.makedirs(MEMDIR, exist_ok=True)


class NetHackMemory:
    def __init__(self, name="nethack_ledger"):
        self.path = os.path.join(MEMDIR, f"{name}.json")
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.data = json.load(f)
        else:
            self.data = {
                "episodes": [],
                "species": {},       # name -> {exchanges, damage, kills, deaths_caused}
                "deaths": [],        # {depth, xplvl, hp, cause, from, steps}
                "fired": [],
            }
        self._pending_prov = None

    # ------------------------------------------------------------- runtime
    def begin_episode(self, provenance_file):
        self._pending_prov = provenance_file
        self._pending_species = {}

    def record_exchange(self, name, dmg):
        s = self._pending_species.setdefault(
            name, {"exchanges": 0, "damage": 0.0, "kills": 0})
        s["exchanges"] += 1
        s["damage"] += float(dmg)

    def record_kill(self, name):
        s = self._pending_species.setdefault(
            name, {"exchanges": 0, "damage": 0.0, "kills": 0})
        s["kills"] += 1

    def record_fired(self, what):
        self.data["fired"].append({"what": what,
                                   "episode": self._pending_prov})

    def end_episode(self, result, steps):
        prov = {"from": self._pending_prov, "steps": f"0-{steps}"}
        rec = {
            "seed": result.get("seed"),
            "role": result.get("role"),
            "progression": result.get("progression"),
            "depth_max": result.get("depth_max"),
            "xplvl_max": result.get("xplvl_max"),
            "end_reason": result.get("end_reason"),
            **prov,
        }
        self.data["episodes"].append(rec)
        for name, s in self._pending_species.items():
            g = self.data["species"].setdefault(
                name, {"exchanges": 0, "damage": 0.0, "kills": 0,
                       "deaths_caused": 0, "sources": []})
            g["exchanges"] += s["exchanges"]
            g["damage"] += s["damage"]
            g["kills"] += s["kills"]
            g["sources"].append(prov["from"])
        er = (result.get("end_reason") or "")
        if er.startswith("DEATH"):
            cause = self._death_species(er)
            self.data["deaths"].append({
                "depth": result.get("depth_max"),
                "xplvl": result.get("xplvl_max"),
                "cause": cause,
                "combat": self._is_combat_death(result), **prov})
            if cause:
                g = self.data["species"].setdefault(
                    cause, {"exchanges": 0, "damage": 0.0, "kills": 0,
                            "deaths_caused": 0, "sources": []})
                g["deaths_caused"] += 1
                g["sources"].append(prov["from"])
        self.save()

    @staticmethod
    def _death_species(end_reason):
        import re
        m = re.search(r"(?:killed|poisoned) by (?:an? |the |a rotted )?"
                      r"([a-z' -]+?)(?:[.,]|$)", end_reason, re.IGNORECASE)
        return m.group(1).strip() if m else None

    @staticmethod
    def _is_combat_death(rec):
        er = (rec.get("end_reason") or "").lower()
        return "killed by" in er and "wrath" not in er and \
            "starved" not in er and "lack of food" not in er

    # ----------------------------------------------------------- retrieval
    def avoid_species(self):
        out = set()
        for name, s in self.data["species"].items():
            if s.get("deaths_caused", 0) >= 1:
                out.add(name)
            elif s["exchanges"] >= 4 and s["damage"] / s["exchanges"] >= 5.0:
                out.add(name)
        return out

    def danger_depth(self):
        # combat deaths only: a starvation death at Dlvl 1 says nothing
        # about where fights get lethal (v1 ledger bug: it set the rest
        # gate to depth 1 for every episode)
        depths = [d["depth"] for d in self.data["deaths"]
                  if d.get("depth") and d.get("combat")]
        return min(depths) if depths else None

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=1)
