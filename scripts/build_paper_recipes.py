"""Build the two paper-derived recipes: reusable OpenWorld worlds that distill the
transferable knowledge of the ARC-AGI-3 paper (goal-as-procedure) and the world-time-compute
paper (a verified world is an exact compute substrate). Same authoring pattern as build/intake.py:
construct a verified-code world, self-check its dynamics, then to_spec -> validate -> write.

    python scripts/build_paper_recipes.py     # writes recipes/{interactive-reasoning,world-time-compute}/*.json
"""
import os, json
from pathlib import Path
from openworld import (World, CodeTransition, Action, to_spec, from_spec,
                       validate_spec, spec_to_json)

ROOT = Path(__file__).resolve().parent.parent


def emit(world, relpath):
    """Round-trip check + validate, then write the recipe JSON."""
    spec = to_spec(world)
    problems = validate_spec(spec)
    assert not problems, problems
    # lossless: reconstruct and confirm the rollout matches on a probe sequence
    w2 = from_spec(spec, allow_code=True)
    s1 = dict(world.initial_state); s2 = dict(world.initial_state)
    for a in (world.actions * 4):
        s1 = dict(world.transition.step(s1, Action(a)))
        s2 = dict(w2.transition.step(s2, Action(a)))
        assert s1 == s2, (a, s1, s2)
    out = ROOT / relpath
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spec_to_json(spec))
    print(f"  wrote {relpath}")


# =====================================================================================
# Recipe 1 — INTERACTIVE REASONING: goal-as-procedure (the ARC-AGI-3 paper's core lesson)
# =====================================================================================
# A four-button panel with a hidden correct press ORDER. The win is an ordered *procedure*
# (press the buttons in sequence), not a state to maximize: a wrong press resets progress,
# and button 1 always lights a shiny `lit` counter --- a decoy. A method that scores the
# visible state (maximize `lit`) presses button 1 forever and never unlocks; only reasoning
# the ordered sequence wins. This is "goal-as-procedure": a reward over a single state cannot
# rank an ordered protocol. Dynamics are deterministic, verified code (discoverable by acting).
PROCEDURE_TRANSITION = """
def transition(state, action):
    ORDER = [2, 0, 3, 1]                       # the hidden correct press sequence
    s = dict(state)
    if s.get("unlocked"):
        return s                               # solved: absorbing state
    name = action["name"]
    if not name.startswith("press_"):
        return s
    btn = int(name.split("_", 1)[1])
    s["presses"] = s.get("presses", 0) + 1
    if btn == 1:
        s["lit"] = s.get("lit", 0) + 1         # decoy: button 1 always lights the lamp
    p = s.get("progress", 0)
    if btn == ORDER[p]:                         # correct next step -> advance the procedure
        p += 1
        s["progress"] = p
        if p >= len(ORDER):
            s["unlocked"] = True
    else:                                       # wrong step -> the procedure resets
        s["progress"] = 0
        s["broke"] = s.get("broke", 0) + 1
    return s
"""

proc = World(
    name="goal_as_procedure",
    description=("A hidden-order button panel: the win is an ordered PROCEDURE (press the four "
                 "buttons in the right sequence), not a state to maximize. Distills the ARC-AGI-3 "
                 "paper's central lesson --- a reward scored over a single state cannot rank an "
                 "ordered protocol, so state-scoring and blind search fail; only reasoning the "
                 "sequence wins."),
    initial_state={"progress": 0, "unlocked": False, "presses": 0, "lit": 0, "broke": 0},
    actions=["press_0", "press_1", "press_2", "press_3"],
    rules=[
        "The win is an ordered PROCEDURE: press the buttons in a fixed hidden sequence.",
        "A correct next press advances progress; any wrong press resets progress to 0.",
        "Button 1 always lights the `lit` lamp --- a decoy that looks like progress.",
        "GOAL-AS-PROCEDURE (the ARC-AGI-3 lesson): maximizing the visible state `lit` presses "
        "button 1 forever and never unlocks; only the ordered sequence [2,0,3,1] wins. A reward "
        "over a single frame cannot rank a procedure --- reason the sequence, do not score a state.",
        "Dynamics are deterministic, verified code --- discoverable by acting (source-free).",
    ],
    transition=CodeTransition(PROCEDURE_TRANSITION),
)
proc.objectives = [
    {"name": "unlock (ordered procedure)", "goal": "reach unlocked == True by pressing the ordered sequence"},
    {"name": "do not chase the decoy", "goal": "note: maximizing lit never unlocks --- lit is not the goal"},
    {"name": "efficiency", "goal": "minimize presses"},
]

# --- self-check: the ordered sequence unlocks; the decoy never does -------------------
def _step(w, s, name):
    return dict(w.transition.step(s, Action(name)))

s = dict(proc.initial_state)
for a in ["press_2", "press_0", "press_3", "press_1"]:        # the correct procedure
    s = _step(proc, s, a)
assert s["unlocked"] is True and s["progress"] == 4, s
assert s["presses"] == 4, s

s = dict(proc.initial_state)
for _ in range(50):                                            # chase the decoy: press 1 forever
    s = _step(proc, s, "press_1")
assert s["unlocked"] is False and s["lit"] == 50, s            # lit soars, never unlocks
assert s["progress"] == 0, s

s = dict(proc.initial_state)                                   # a wrong step resets progress
s = _step(proc, s, "press_2"); assert s["progress"] == 1
s = _step(proc, s, "press_3"); assert s["progress"] == 0 and s["broke"] == 1, s


# =====================================================================================
# Recipe 2 — WORLD-TIME COMPUTE: a verified world is an exact compute substrate
# =====================================================================================
# An elementary cellular automaton (Rule 110). Each step applies the rule exactly; rolling the
# world forward N generations IS "world-time compute" --- you spend inference steps traversing
# your own verified world to get an EXACT multi-step answer, with zero compounding error (unlike
# a learned next-state proxy, which drifts as depth grows). Distills the world-time-compute paper.
CA_TRANSITION = """
def transition(state, action):
    s = dict(state)
    if action["name"] != "step":
        return s
    rule = int(s.get("rule", 110))
    row = list(s.get("row", []))
    n = len(row)
    nxt = []
    for i in range(n):                          # wrap-around neighborhood
        left = row[(i - 1) % n]; center = row[i]; right = row[(i + 1) % n]
        idx = (left << 2) | (center << 1) | right    # 0..7
        nxt.append((rule >> idx) & 1)                # the rule's output bit
    s["row"] = nxt
    s["gen"] = int(s.get("gen", 0)) + 1
    s["population"] = sum(nxt)
    return s
"""

N = 21
_row0 = [0] * N; _row0[N // 2] = 1                              # classic single-cell seed
ca = World(
    name="world_time_compute",
    description=("An elementary cellular automaton (Rule 110) as a verified compute substrate. "
                 "Each step applies the rule exactly; rolling the world forward is WORLD-TIME "
                 "COMPUTE --- spend inference steps traversing your own verified world for an exact "
                 "multi-step answer, with zero compounding error. Distills the world-time-compute "
                 "paper: verified code stays exact at every rollout depth, where a learned proxy drifts."),
    initial_state={"row": _row0, "gen": 0, "rule": 110, "population": 1},
    actions=["step"],
    rules=[
        "Each `step` applies elementary cellular-automaton Rule 110 to the whole row at once.",
        "The world is a deterministic, verified compute substrate: rolling it forward N steps "
        "computes generation N exactly.",
        "WORLD-TIME COMPUTE (the lesson): traversing your own verified world is exact at EVERY "
        "depth --- no compounding error --- so inference-time rollout buys reliable multi-step "
        "prediction, unlike an LLM next-state proxy whose exact-match rate collapses with depth.",
        "Verified code, zero data, bit-exact rollouts --- reproducible on any machine.",
    ],
    transition=CodeTransition(CA_TRANSITION),
)
ca.objectives = [
    {"name": "exact rollout (world-time compute)", "goal": "step the world forward --- each generation is exact"},
    {"name": "grow structure", "goal": "evolve the row; population = number of live cells"},
]

# --- self-check: deterministic, exact, and advances one generation per step -----------
def _roll(w, s, k):
    s = dict(s)
    for _ in range(k):
        s = dict(w.transition.step(s, Action("step")))
    return s

a = _roll(ca, ca.initial_state, 20)
b = _roll(ca, ca.initial_state, 20)
assert a == b, "rollout must be deterministic (exact at depth)"
assert a["gen"] == 20 and a["population"] == sum(a["row"]), a
assert len(a["row"]) == N and set(a["row"]) <= {0, 1}, a
assert _roll(ca, ca.initial_state, 1)["gen"] == 1


if __name__ == "__main__":
    print("building paper recipes:")
    emit(proc, "recipes/interactive-reasoning/goal_as_procedure.json")
    emit(ca,   "recipes/world-time-compute/cellular_automaton.json")
    print("done.")
