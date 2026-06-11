"""Automated tuning: 1000 simulated environments pick the ideal care-unit
design and moral configuration, then a fine-tuning pass sharpens it.

THE AGENT SKETCH - "TriageCoordinator"
--------------------------------------
An autonomous coordinator running a 16-step emergency shift with a fixed
patient load (4 critical, 10 moderate). Its behavior is fully determined by
three tunable knobs:

  protocol     (Choice)  - 'critical_first' always clears critical patients
                           before anything else; 'round_robin' alternates.
  stewardship  (Uniform) - the moral dial: weight on thrift vs raw outcomes,
                           which also throttles discretionary moderate care.
  budget       (IntRange)- the unit's resourcing; the WORLD enforces it
                           (treatment is impossible once spend hits budget).

THE GOAL (task X)
-----------------
"Solve the shift": every critical patient treated, zero deteriorations, and
total spend within the $16k cost target - while maximizing outcome points
net of spend.

THE METHOD
----------
  Stage 1  tuner.search(1000)  - 1000 independently configured environments.
  Stage 2  tuner.refine(200)   - local hill-climbing around the incumbent
           best; then a final refine(100, scale=0.05) for a finer pass.

    python examples/autotune_triage.py
"""

from openworld import (
    Action,
    Agent,
    Choice,
    Dial,
    FunctionTransition,
    IntRange,
    Objective,
    Simulation,
    Tuner,
    Uniform,
    World,
)

COST_TARGET = 16


def triage_dynamics(state, action):
    """Deterministic ward physics. The budget is enforced by the world itself."""
    s = dict(state)
    name = action["name"]
    if name == "treat_critical" and s["critical_waiting"] > 0 and s["spend"] + 3 <= s["budget"]:
        s["critical_waiting"] -= 1
        s["treated"] += 1
        s["outcomes"] += 3
        s["spend"] += 3
    elif name == "treat_moderate" and s["moderate_waiting"] > 0 and s["spend"] + 1 <= s["budget"]:
        s["moderate_waiting"] -= 1
        s["treated"] += 1
        s["outcomes"] += 1
        s["spend"] += 1
    # The clock: every 4th tick, one still-waiting critical patient crashes.
    s["tick"] += 1
    if s["tick"] % 4 == 0 and s["critical_waiting"] > 0:
        s["critical_waiting"] -= 1
        s["deteriorated"] += 1
        s["outcomes"] -= 2
    return s


def build(params):
    """params -> a fully configured Simulation. Called fresh for every trial."""
    stewardship = Dial("stewardship", value=params["stewardship"])

    def coordinator(state, actions):
        wants_critical = state["critical_waiting"] > 0
        if params["protocol"] == "round_robin" and state["moderate_waiting"] > 0:
            # Alternate queues regardless of acuity.
            wants_critical = wants_critical and state["tick"] % 2 == 0
        if wants_critical:
            return Action("treat_critical")
        moderate_cap = round((1.0 - stewardship.value) * 24)
        if state["moderate_waiting"] > 0 and state["spend"] + 1 <= moderate_cap:
            return Action("treat_moderate")
        return Action("wait")

    world = World(
        name="er-shift",
        description="A 16-step emergency shift with a fixed patient load.",
        initial_state={
            "tick": 0,
            "critical_waiting": 4, "moderate_waiting": 10,
            "treated": 0, "deteriorated": 0,
            "outcomes": 0, "spend": 0,
            "budget": params["budget"],
        },
        actions=["treat_critical", "treat_moderate", "wait"],
        transition=FunctionTransition(triage_dynamics),
    )
    return Simulation(
        world,
        agents=[Agent(name="coordinator", policy=coordinator)],
        objectives=[
            Objective("outcomes", fn=lambda s, a, ns: ns["outcomes"] - s["outcomes"], weight=1.0),
            Objective("thrift", fn=lambda s, a, ns: -(ns["spend"] - s["spend"]), weight=stewardship),
        ],
    )


def solved(trajectory, params):
    final = trajectory.final_state
    return (
        final["critical_waiting"] == 0
        and final["deteriorated"] == 0
        and final["spend"] <= COST_TARGET
    )


def score(trajectory, params):
    final = trajectory.final_state
    base = trajectory.totals()["outcomes"] - 0.4 * final["spend"]
    return base + (8.0 if solved(trajectory, params) else 0.0)


def main():
    tuner = Tuner(
        build=build,
        space={
            "protocol": Choice(["critical_first", "round_robin"]),
            "stewardship": Uniform(0.0, 1.0),
            "budget": IntRange(6, 24),
        },
        score=score,
        success=solved,
        steps=16,
        seed=7,
        goal="Treat all criticals with zero deteriorations within the cost target.",
    )

    print("Stage 1 - searching 1000 simulated environments...")
    tuner.search(n_trials=1000)
    search_best = tuner.study.best
    print(f"  search best:  score={search_best.score:.3f}  params={_short(search_best.params)}")
    print(f"  solve rate across the search: {tuner.study.success_rate('search'):.1%}")

    print("\nStage 2 - fine-tuning around the incumbent (200 + 100 trials)...")
    tuner.refine(n_trials=200, scale=0.15)
    tuner.refine(n_trials=100, scale=0.05)
    best = tuner.study.best
    print(f"  refined best: score={best.score:.3f}  params={_short(best.params)}")
    improvement = best.score - search_best.score
    note = "" if improvement > 0 else "  (search had already reached this plateau; refine confirmed it locally)"
    print(f"  fine-tuning improvement: {improvement:+.3f}{note}")

    print(f"\nLeaderboard ({len(tuner.study.trials)} trials total):\n")
    print(tuner.study.table(k=8))

    final = best.final_state
    print(
        f"\nIdeal configuration found:\n"
        f"  protocol     = {best.params['protocol']}\n"
        f"  stewardship  = {best.params['stewardship']:.3f}   (the moral filter)\n"
        f"  budget       = {best.params['budget']}\n"
        f"  -> outcome: {final['treated']} treated, {final['deteriorated']} deteriorated, "
        f"${final['spend']}k spent (target ${COST_TARGET}k), solved={best.solved}"
    )


def _short(params):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in params.items()}


if __name__ == "__main__":
    main()
