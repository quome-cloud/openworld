"""Induced world model — source-blind arm.

Rules live in rules.json (the auditable ledger). Each rule:
  {id, statement, status: hypothesized|corroborated|refuted|revised,
   confidence, evidence: [[ep,step],...], corroborations, refutations,
   scope, revisions: [...]}

This module implements the PREDICTIVE content of rules whose status is
hypothesized or corroborated. predict() emits a possibility set BEFORE the
step; verify() checks the served observation against it AFTER. Observations
outside the set are anomalies -> the model is wrong, not the observation.

Every predictor cites the rule id(s) it implements. No predictor may encode
knowledge that lacks an evidence-cited rule in rules.json.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RULES_PATH = os.path.join(HERE, "rules.json")

DIRS = {
    "north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0),
    "northeast": (1, -1), "southeast": (1, 1),
    "southwest": (-1, 1), "northwest": (-1, -1),
}


class WorldModel:
    def __init__(self):
        self.rules = {}
        self.load()

    def load(self):
        if os.path.exists(RULES_PATH):
            with open(RULES_PATH) as f:
                data = json.load(f)
            self.rules = {r["id"]: r for r in data["rules"]}
        else:
            self.rules = {}

    def save(self):
        with open(RULES_PATH, "w") as f:
            json.dump({"rules": sorted(self.rules.values(), key=lambda r: r["id"])},
                      f, indent=1)

    def active(self, rid):
        r = self.rules.get(rid)
        return r is not None and r["status"] in ("hypothesized", "corroborated", "revised")

    # ---------------- prediction ----------------
    def predict(self, pre, action):
        """pre: dict with blstats dict 'bl', and optionally map info.
        Returns dict dimension -> possibility spec. Each spec dict:
          {rule: rid, kind: 'set'|'range'|'ge', ...}
        Dimensions not predicted are unconstrained (not checked).
        """
        p = {}
        bl = pre["bl"]

        # R_TIME: game time is non-decreasing across a step.
        if self.active("R_TIME"):
            p["time"] = {"rule": "R_TIME", "kind": "ge", "value": bl["time"]}

        # R_MOVE: single-step move in dir d -> pos becomes pos+d OR unchanged
        # (blocked / interrupted); never any other cell. Scope: only while depth
        # is unchanged (trap falls relocate arbitrarily: E25:665, E25:1175).
        if action in DIRS and self.active("R_MOVE"):
            dx, dy = DIRS[action]
            p["pos"] = {"rule": "R_MOVE", "kind": "set",
                        "value": [[bl["x_pos"] + dx, bl["y_pos"] + dy],
                                  [bl["x_pos"], bl["y_pos"]]],
                        "if_depth": bl["depth"]}

        # R_NONMOVE_POS: non-movement, non-travel actions leave position
        # unchanged (teleport-class exceptions get scope conditions when seen).
        if (action not in DIRS and not action.startswith("far") and
                action not in ("down", "up", "jump", "move", "movefar", "rush", "rush2", "travel")
                and self.active("R_NONMOVE_POS")):
            p["pos"] = {"rule": "R_NONMOVE_POS", "kind": "set",
                        "value": [[bl["x_pos"], bl["y_pos"]]],
                        "if_depth": bl["depth"]}

        # R_DEPTH_STABLE: depth only changes via down/up (or trap-scope).
        if self.active("R_DEPTH_STABLE"):
            d0 = bl["depth"]
            if action == "down":
                # R_DOWN (+1) plus shaft scope (+2, R_DEPTH_TRAP replay evidence)
                allowed = [d0, d0 + 1, d0 + 2]
                rid = "R_DOWN_NEEDS_TILE"
            elif action == "up":
                allowed = [d0, max(1, d0 - 1)]
                rid = "R_UP_NEEDS_TILE"
            else:
                # R_DEPTH_TRAP: trap door +1, shaft +2
                allowed = [d0, d0 + 1, d0 + 2]
                rid = "R_DEPTH_STABLE"
            p["depth"] = {"rule": rid, "kind": "set", "value": allowed}

        # R_HP_BOUND: hp never exceeds max_hitpoints; hp>=0.
        if self.active("R_HP_BOUND"):
            p["hp"] = {"rule": "R_HP_BOUND", "kind": "range",
                       "value": [0, None]}  # upper checked vs post max_hp

        # R_XP_MONO: experience_level non-decreasing (within an episode).
        if self.active("R_XP_MONO"):
            p["xplvl"] = {"rule": "R_XP_MONO", "kind": "ge",
                          "value": bl["experience_level"]}

        return p

    # ---------------- verification ----------------
    def verify(self, pred, post):
        """Returns list of (dimension, rule, ok, detail)."""
        bl = post["bl"]
        # R_TERMINAL_ZERO (evidence E1:1030, E2:1857): death frame zeroes
        # blstats; core rules' scopes exclude it.
        if post.get("done") and bl["time"] == 0 and bl["max_hitpoints"] == 0:
            if self.active("R_TERMINAL_ZERO"):
                return [("terminal", "R_TERMINAL_ZERO", True, "zeroed-frame")]
            return []
        out = []
        for dim, spec in pred.items():
            ok, detail = True, None
            if dim == "time":
                ok = bl["time"] >= spec["value"]
                detail = bl["time"]
            elif dim == "pos":
                if spec.get("if_depth") is not None and bl["depth"] != spec["if_depth"]:
                    continue  # scope: depth changed (trap fall) -> pos unconstrained
                pos = [bl["x_pos"], bl["y_pos"]]
                ok = pos in spec["value"]
                detail = pos
            elif dim == "depth":
                ok = bl["depth"] in spec["value"]
                detail = bl["depth"]
            elif dim == "hp":
                ok = 0 <= bl["hitpoints"] <= max(bl["max_hitpoints"], 1)
                detail = [bl["hitpoints"], bl["max_hitpoints"]]
            elif dim == "xplvl":
                ok = bl["experience_level"] >= spec["value"]
                detail = bl["experience_level"]
            out.append((dim, spec["rule"], ok, detail))
        return out

    def record(self, results, ep, step, anomaly_log):
        """Count corroborations; log anomalies."""
        for dim, rid, ok, detail in results:
            r = self.rules.get(rid)
            if r is None:
                continue
            if ok:
                r["corroborations"] = r.get("corroborations", 0) + 1
            else:
                r["refutations"] = r.get("refutations", 0) + 1
                anomaly_log.append({"ep": ep, "step": step, "rule": rid,
                                    "dim": dim, "observed": detail})
