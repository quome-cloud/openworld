"""OpenWorld for scikit-learn users - a runnable, side-by-side intro.

You already know the loop: fit a model to data, predict, score on a test set.
This script models the SAME tiny environment two ways and scores both against
the ground-truth ORACLE, in-distribution and at 10x scale:

  LEFT  (the scikit-learn way) : fit a regressor to sampled transitions.
  RIGHT (the OpenWorld way)    : declare the rule, get code dynamics, step them.

The contrast is the whole lesson. A learned model memorizes the magnitudes it
saw and cannot extrapolate; declared-then-verified code is exact at any scale,
because it encodes the rule, not the statistics.

    python tutorials/from_scikit_learn.py      # offline, deterministic

Concept map (see from_scikit_learn.md for the full Rosetta table):
  World         ~ the environment / data-generating process (NOT a dataset)
  transition    ~ the function you would normally fit
  oracle        ~ held-out ground-truth labels you score against
  world.compile ~ fit (but an LLM writes *code*, not weights; no epochs)
  world.step    ~ predict
  verification  ~ property tests / assertions (NOT a held-out accuracy number)
"""

import random

from openworld import Action, World
from openworld.transition import FunctionTransition

# ---------------------------------------------------------------------------
# The environment we want a model of: a water tank.
#
# The RULE below is the thing an ML practitioner would have to LEARN from data,
# and the thing an OpenWorld user simply DECLARES: 'fill' adds 2, 'drain'
# removes 3 (the tank can't go below empty), 'wait' does nothing.
# ---------------------------------------------------------------------------
ACTIONS = ["fill", "drain", "wait"]


def oracle(level, action):
    """Ground truth = the 'labels' you would train against and score on."""
    if action == "fill":
        return level + 2
    if action == "drain":
        return max(0, level - 3)
    return level


# In-distribution probes (levels the sklearn model will have seen the scale of)
# and 10x out-of-distribution probes (same rule, larger magnitudes).
IN_DIST = [(lvl, a) for lvl in (0, 1, 5, 12, 28) for a in ACTIONS]
OOD_10X = [(lvl, a) for lvl in (300, 305, 412) for a in ACTIONS]


# ===========================================================================
# LEFT: the scikit-learn way -- fit a regressor to sampled transitions.
# ===========================================================================
def sklearn_arm():
    try:
        from sklearn.tree import DecisionTreeRegressor
    except ImportError:
        return None  # framework stays zero-dependency; arm is optional

    rng = random.Random(0)
    # "Collect a dataset": random-policy transitions over the levels we see.
    X, y = [], []
    for _ in range(2000):
        level = rng.randint(0, 30)            # the distribution we train on
        action = rng.choice(ACTIONS)
        X.append([level] + [1 if action == a else 0 for a in ACTIONS])
        y.append(oracle(level, action))
    model = DecisionTreeRegressor(random_state=0).fit(X, y)  # <-- fit

    def predict(level, action):               # <-- predict
        feat = [[level] + [1 if action == a else 0 for a in ACTIONS]]
        return int(round(model.predict(feat)[0]))

    return predict


# ===========================================================================
# RIGHT: the OpenWorld way -- declare the rule, get code dynamics, step them.
# ===========================================================================
def tank_dynamics(state, action):
    """The dynamics as code. Offline we hand-write it; with a live model,
    world.compile() SYNTHESIZES exactly this from the plain-language rules and
    VERIFIES it before accepting (see software_engineering_sprint.py)."""
    s = dict(state)
    s["level"] = oracle(s["level"], action["name"])
    return s


def openworld_arm():
    world = World(
        name="tank",
        description="A water tank: fill adds 2, drain removes 3 (never below empty).",
        initial_state={"level": 0},
        actions=ACTIONS,
        rules=[
            "'fill' increases level by 2.",
            "'drain' decreases level by 3 but never below 0.",
            "'wait' leaves level unchanged.",
        ],
        transition=FunctionTransition(tank_dynamics),
    )

    def predict(level, action):
        world.state = world.initial_state.copy()
        world.state["level"] = level
        return world.step(Action(action))["level"]

    return world, predict


# ---------------------------------------------------------------------------
def exact_rate(predict, probes):
    hits = sum(1 for lvl, a in probes if predict(lvl, a) == oracle(lvl, a))
    return hits / len(probes)


def main():
    sk_predict = sklearn_arm()
    world, ow_predict = openworld_arm()

    # VERIFICATION (not a test score): the OpenWorld dynamics must match the
    # oracle EXACTLY -- this is a correctness assertion on the program, the way
    # you'd write property tests, not a number you hope is high enough.
    assert all(ow_predict(lvl, a) == oracle(lvl, a) for lvl, a in IN_DIST + OOD_10X), \
        "verification failed: declared dynamics disagree with the oracle"

    print("Modeling a water tank's dynamics. Exact-match accuracy vs the oracle:\n")
    print(f"  {'approach':<34}{'in-distribution':>16}{'10x out-of-dist':>18}")
    print("  " + "-" * 66)
    if sk_predict is not None:
        print(f"  {'scikit-learn (fit a regressor)':<34}"
              f"{exact_rate(sk_predict, IN_DIST):>15.0%}{exact_rate(sk_predict, OOD_10X):>18.0%}")
    else:
        print(f"  {'scikit-learn (not installed)':<34}{'--':>16}{'--':>18}")
    print(f"  {'OpenWorld (declare + verify code)':<34}"
          f"{exact_rate(ow_predict, IN_DIST):>15.0%}{exact_rate(ow_predict, OOD_10X):>18.0%}")

    print("\nWhy the difference:")
    print("  - The regressor learned the MAGNITUDES it saw (levels ~0-30); at 10x")
    print("    scale it predicts a memorized value and is wrong. It never saw the rule.")
    print("  - The OpenWorld dynamics ARE the rule, as code. They are exact at any")
    print("    scale, and verification proved it before we trusted a single step.")
    print("\nRule of thumb: declarable dynamics -> OpenWorld; a pattern with no")
    print("declarable rule that you can only learn from data -> stay in scikit-learn.")


if __name__ == "__main__":
    main()
