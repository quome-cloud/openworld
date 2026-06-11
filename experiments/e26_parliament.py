"""E26 - A moral parliament under theory uncertainty (round-4 item P4).

Four agents run the same 12-step triage shift:

  utilitarian   - one-step lookahead maximizing outcomes - 0.2*spend
  rawlsian      - one-step lookahead maximizing the worst-off patient
                  group's welfare (maximin over critical/moderate groups)
  deontologist  - duty ordering: never abandon a waiting critical patient;
                  then treat; only then wait
  parliament    - Borda vote among the three delegates (equal credence)

Each agent's trajectory is then scored under EVERY theory's own lens. The
hedging hypothesis: the parliament is never the worst agent under any lens.
Deterministic, no LLM.
"""

from openworld import Action, Agent, Delegate, MoralParliament, Simulation

from common import make_oracle_world, save_results, triage_ground_truth

STEPS = 12
ACTIONS = ["treat_critical", "treat_moderate", "wait"]


def simulate(state, action_name):
    return triage_ground_truth(dict(state), {"name": action_name, "params": {}, "agent": None})


def applicable(state, action_name):
    if action_name == "treat_critical":
        return state["critical_waiting"] > 0
    if action_name == "treat_moderate":
        return state["moderate_waiting"] > 0
    return True


def utilitarian_value(state):
    return state["outcomes"] - 0.2 * state["spend"]


def group_welfare(state):
    critical = -(state["critical_waiting"] + 3 * state["deteriorated"])
    moderate = -state["moderate_waiting"]
    return min(critical, moderate)


def rank_by(value_fn):
    def rank(state, actions, sim):
        usable = [a for a in actions if applicable(state, a)]
        return sorted(usable, key=lambda a: value_fn(sim(state, a)), reverse=True)
    return rank


def deont_rank(state, actions, sim):
    usable = [a for a in actions if applicable(state, a)]
    if state["critical_waiting"] > 0 and "treat_critical" in usable:
        order = ["treat_critical", "treat_moderate", "wait"]
    else:
        order = ["treat_moderate", "treat_critical", "wait"]
    return [a for a in order if a in usable]


DELEGATES = [
    Delegate("utilitarian", rank_by(utilitarian_value)),
    Delegate("rawlsian", rank_by(group_welfare)),
    Delegate("deontologist", deont_rank),
]


def delegate_policy(delegate):
    def policy(state, actions):
        ranking = delegate.rank(state, [a for a in ACTIONS], simulate)
        return Action(ranking[0] if ranking else "wait")
    return policy


def parliament_policy(parliament):
    def policy(state, actions):
        usable = [a for a in ACTIONS if applicable(state, a)]
        return Action(parliament.choose(state, usable, simulate) if usable else "wait")
    return policy


def main():
    agents = {d.name: delegate_policy(d) for d in DELEGATES}
    agents["parliament"] = parliament_policy(MoralParliament(delegates=DELEGATES))

    results = {}
    for name, policy in agents.items():
        world = make_oracle_world("triage")
        violation_count = {"n": 0}
        inner = policy

        def counting(state, actions, _inner=inner):
            choice = _inner(state, actions)
            if state["critical_waiting"] > 0 and choice.name != "treat_critical":
                violation_count["n"] += 1
            return choice

        sim = Simulation(world, agents=[Agent(name=name, policy=counting)])
        final = sim.run(steps=STEPS).final_state
        results[name] = {
            "final_state": dict(final),
            "lens_utilitarian": utilitarian_value(final),
            "lens_rawlsian": group_welfare(final),
            "lens_deontological_violations": violation_count["n"],
        }
        print(f"  {name}: util={results[name]['lens_utilitarian']:.1f} "
              f"rawls={results[name]['lens_rawlsian']:.1f} "
              f"violations={violation_count['n']}")

    # Hedging: is the parliament ever the strictly worst under any lens?
    lenses = {
        "lens_utilitarian": max,      # higher is better
        "lens_rawlsian": max,
        "lens_deontological_violations": min,  # lower is better
    }
    hedging = {}
    for lens, better in lenses.items():
        values = {name: r[lens] for name, r in results.items()}
        worst = (min if better is max else max)(values.values())
        hedging[lens] = {
            "parliament_value": values["parliament"],
            "worst_value": worst,
            "parliament_is_strictly_worst": (
                values["parliament"] == worst
                and sum(1 for v in values.values() if v == worst) == 1),
        }

    save_results("e26_parliament", {
        "steps": STEPS,
        "agents": results,
        "hedging": hedging,
        "parliament_never_strictly_worst": not any(
            h["parliament_is_strictly_worst"] for h in hedging.values()),
    })
    print(f"parliament never strictly worst: "
          f"{not any(h['parliament_is_strictly_worst'] for h in hedging.values())}")


if __name__ == "__main__":
    main()
