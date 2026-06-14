"""E43 - Active world-model induction: acting beats observing.

E38 showed that scaling the generator does not close the rules-vs-traces gap:
passive random-policy traces are information-limited (the subtle
`bugs += debt // k` interaction only manifests at high debt, which a random
policy rarely sets up). E43 turns that negative into a positive: an agent that
ACTS to disambiguate identifies the exact rule in far fewer transitions than
passive observation - acting, not scaling, is what closes the gap.

The sprint world's dynamics are drawn from a known candidate family
parameterizing the unknowns (ship's debt delta, the divisor `k` in
`bugs += debt // k`, and the fix/refactor magnitudes). A candidate is
eliminated the moment its predicted next-state disagrees with an observed
transition. Conditions:

  passive_random      - random policy (the E38 setting); rarely reaches the
                        high-debt states that reveal k -> plateaus.
  active_versionspace - picks the action that most splits the remaining
                        candidates at the current state, tie-breaking toward
                        ship (build debt) to reach the disambiguating states.
                        Knows NOTHING about the true rule.
  clairvoyant         - a knowing-greedy reference: it already knows the true
                        rule, so it sees each true response in advance and
                        removes the most candidates per step. A strong baseline
                        (not a strict lower bound - greedy is myopic, so the
                        non-myopic build-debt move can occasionally beat it).

Metric: transitions until a UNIQUE rule remains. Deterministic, offline.
"""

import itertools
import random

from common import SPRINT_INITIAL, save_results

ACTIONS = ["ship", "fix", "refactor", "noop"]
BUDGET = 60

# Candidate family: the unknowns the agent must identify. The true rule is the
# sprint ground truth: ship_debt=1, k=4, fix=2, refactor=2.
GRID = {
    "ship_debt": [1, 2],            # debt += ship_debt on ship
    "k": [2, 3, 4, 5, 6, 8],        # bugs += debt // k on ship  (the hard one)
    "fix": [1, 2, 3],               # bugs -= fix on fix (clamped at 0)
    "refactor": [1, 2, 3],          # debt -= refactor on refactor (clamped at 0)
}
PARAMS = list(GRID)


def candidate_rules():
    return [dict(zip(PARAMS, combo)) for combo in itertools.product(*GRID.values())]


def step_with(rule, state, action):
    """The sprint transition under a candidate rule (action is a name str)."""
    s = dict(state)
    if action == "ship" and s["backlog"] > 0:
        s["backlog"] -= 1
        s["shipped"] += 1
        s["debt"] += rule["ship_debt"]
        s["bugs"] += s["debt"] // rule["k"]
    elif action == "fix":
        s["bugs"] = max(0, s["bugs"] - rule["fix"])
    elif action == "refactor":
        s["debt"] = max(0, s["debt"] - rule["refactor"])
    return s


TRUE_DEFAULT = {"ship_debt": 1, "k": 4, "fix": 2, "refactor": 2}


def consistent(cands, state, action, nxt):
    """Keep only candidates whose prediction matches the observed transition."""
    return [c for c in cands if step_with(c, state, action) == nxt]


def split_score(cands, state, action):
    """How many distinct next-states the remaining candidates predict for
    (state, action) - higher means this probe partitions the set more."""
    outs = {tuple(sorted(step_with(c, state, action).items())) for c in cands}
    return len(outs)


# --- probe-accuracy on held-out states (did we identify the true behavior?) -
PROBE_STATES = [
    {"backlog": 5, "shipped": 7, "bugs": 2, "debt": 7},
    {"backlog": 4, "shipped": 0, "bugs": 5, "debt": 0},
    {"backlog": 4, "shipped": 0, "bugs": 0, "debt": 6},
    {"backlog": 1, "shipped": 11, "bugs": 9, "debt": 11},
]


def probe_accuracy(cands, true_rule):
    """Fraction of (probe state, action) where the surviving candidates agree
    with the truth. 1.0 iff the behavior is pinned on the probe suite."""
    hits = total = 0
    for st in PROBE_STATES:
        for a in ("ship", "fix", "refactor"):
            truth = step_with(true_rule, st, a)
            if all(step_with(c, st, a) == truth for c in cands):
                hits += 1
            total += 1
    return hits / total


def run_episode(strategy, true_rule, rng):
    """Return (steps_to_unique, accuracy_curve) for one hidden rule."""
    cands = candidate_rules()
    state = dict(SPRINT_INITIAL)
    curve = []
    steps_to_unique = None
    for t in range(BUDGET):
        if strategy == "passive":
            action = rng.choice(ACTIONS)
        else:  # active version-space
            best, best_key = None, (-1, -1)
            for a in ACTIONS:
                # primary: candidate-split here; tiebreak: ship to build debt
                key = (split_score(cands, state, a), 1 if a == "ship" else 0)
                if key > best_key:
                    best_key, best = key, a
            action = best
        nxt = step_with(true_rule, state, action)        # the true world responds
        cands = consistent(cands, state, action, nxt)
        state = nxt
        curve.append(len(cands))
        if len(cands) == 1 and steps_to_unique is None:
            steps_to_unique = t + 1
        # keep backlog from emptying (so ship stays available) - refill episode
        if state["backlog"] == 0:
            state = dict(SPRINT_INITIAL)
    return steps_to_unique, curve, probe_accuracy(cands, true_rule)


def clairvoyant(true_rule):
    """Knowing-greedy reference: a prober that already knows the true rule and
    therefore, at each step, knows the true response in advance. It picks the
    action that eliminates the most inconsistent candidates given that true
    response (the active agent, by contrast, only has the split heuristic and
    cannot see which partition is the true one). Greedy, so myopic - it is a
    strong reference, not a strict lower bound."""
    cands = candidate_rules()
    state = dict(SPRINT_INITIAL)
    for step in range(BUDGET):
        best, best_after = None, None
        for a in ACTIONS:
            truth = step_with(true_rule, state, a)
            survivors = consistent(cands, state, a, truth)
            if best_after is None or len(survivors) < len(best_after):
                best_after, best = survivors, a
        cands = best_after
        state = step_with(true_rule, state, best)
        if state["backlog"] == 0:
            state = dict(SPRINT_INITIAL)
        if len(cands) == 1:
            return step + 1
    return None


def main():
    true_rules = [
        TRUE_DEFAULT,
        {"ship_debt": 2, "k": 3, "fix": 1, "refactor": 3},
        {"ship_debt": 1, "k": 8, "fix": 3, "refactor": 1},
        {"ship_debt": 2, "k": 6, "fix": 2, "refactor": 2},
        {"ship_debt": 1, "k": 2, "fix": 1, "refactor": 1},
    ]
    n_cands = len(candidate_rules())
    rows = []
    for i, rule in enumerate(true_rules):
        passive = run_episode("passive", rule, random.Random(43 + i))
        active = run_episode("active", rule, random.Random(43 + i))
        clair = clairvoyant(rule)
        rows.append({
            "true_rule": rule,
            "passive_steps": passive[0], "passive_acc": passive[2],
            "active_steps": active[0], "active_acc": active[2],
            "clairvoyant_steps": clair,
            "active_curve": active[1], "passive_curve": passive[1],
        })

    def agg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    summary = {
        "n_candidates": n_cands, "budget": BUDGET, "n_rules": len(true_rules),
        "active_mean_steps": agg("active_steps"),
        "passive_mean_steps": agg("passive_steps"),
        "clairvoyant_mean_steps": agg("clairvoyant_steps"),
        "passive_unresolved": sum(1 for r in rows if r["passive_steps"] is None),
        "active_unresolved": sum(1 for r in rows if r["active_steps"] is None),
    }
    save_results("e43_active_induction", {
        "grid": GRID, "summary": summary, "rows": rows,
    })

    print(f"Active world-model induction on the sprint world "
          f"({n_cands} candidate rules, budget {BUDGET}).\n")
    print(f"  {'true rule':<42}{'active':>8}{'passive':>9}{'clairv':>8}")
    for r in rows:
        p = "none" if r["passive_steps"] is None else r["passive_steps"]
        print(f"  {str(r['true_rule']):<42}{r['active_steps']!s:>8}{p!s:>9}{r['clairvoyant_steps']!s:>8}")
    print(f"\n  mean steps-to-identify: active {summary['active_mean_steps']}, "
          f"passive {summary['passive_mean_steps']} "
          f"({summary['passive_unresolved']}/{len(rows)} never resolved), "
          f"clairvoyant {summary['clairvoyant_mean_steps']}")
    assert summary["active_unresolved"] == 0, "active should identify every rule"
    assert summary["active_mean_steps"] < (summary["passive_mean_steps"] or 1e9), \
        "active should beat passive"


if __name__ == "__main__":
    main()
