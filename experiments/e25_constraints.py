"""E25 - Deontological side constraints vs pure dial weighting (round-4 P2).

A myopic cost-sensitive nurse scores each action by immediate gain:
outcome_delta - lambda * spend_delta. For lambda > 1 the weighting makes
abandoning waiting critical patients 'optimal' - the impermissible region a
weighted sum cannot exclude. The constraint 'never choose a non-critical
action while a critical patient waits' is enforced as an action VETO (not a
penalty), and the sweep measures violations, deteriorations, outcomes, and
spend with and without it. Deterministic, no LLM.
"""

from openworld import Action, Agent, Constraint, Simulation, constrained

from common import make_oracle_world, save_results

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.25, 1.5]
STEPS = 12

GAINS = {  # immediate (outcome_delta, spend_delta) when the action applies
    "treat_critical": (3.0, 3.0),
    "treat_moderate": (1.0, 1.0),
    "wait": (0.0, 0.0),
}

NEVER_ABANDON = Constraint(
    name="never_abandon_critical",
    description=("While a critical patient waits, choosing any action other "
                 "than treating them is impermissible."),
    forbidden=lambda s, a: s["critical_waiting"] > 0 and a.name != "treat_critical",
)


def myopic_nurse(lam):
    def policy(state, actions):
        best_action, best = "wait", float("-inf")
        for name in ("treat_critical", "treat_moderate", "wait"):
            if name not in actions:
                continue
            if name == "treat_critical" and state["critical_waiting"] == 0:
                continue
            if name == "treat_moderate" and state["moderate_waiting"] == 0:
                continue
            outcome, spend = GAINS[name]
            score = outcome - lam * spend
            if score > best:
                best, best_action = score, name
        return Action(best_action)

    return policy


def run(lam, use_constraint):
    policy = myopic_nurse(lam)
    if use_constraint:
        policy = constrained(policy, [NEVER_ABANDON], fallback="wait")
    world = make_oracle_world("triage")
    violations = {"n": 0}

    base_policy = policy

    def counting_policy(state, actions):
        choice = base_policy(state, actions)
        if NEVER_ABANDON.forbidden(state, choice):
            violations["n"] += 1
        return choice

    sim = Simulation(world, agents=[Agent(name="nurse", policy=counting_policy)])
    final = sim.run(steps=STEPS).final_state
    return {
        "lambda": lam, "constrained": use_constraint,
        "violations": violations["n"],
        "deteriorated": final["deteriorated"],
        "outcomes": final["outcomes"],
        "treated": final["treated"],
        "spend": final["spend"],
    }


def main():
    rows = []
    for lam in LAMBDAS:
        for use_constraint in (False, True):
            row = run(lam, use_constraint)
            rows.append(row)
            tag = "constrained" if use_constraint else "dial-only  "
            print(f"  lambda={lam:<5} {tag}: violations={row['violations']:<3} "
                  f"deteriorated={row['deteriorated']} outcomes={row['outcomes']} "
                  f"spend={row['spend']}")

    unconstrained = [r for r in rows if not r["constrained"]]
    constrained_rows = [r for r in rows if r["constrained"]]
    save_results("e25_constraints", {
        "lambdas": LAMBDAS, "steps": STEPS, "rows": rows,
        "summary": {
            "dial_only_total_violations": sum(r["violations"] for r in unconstrained),
            "dial_only_lambdas_with_violations": [
                r["lambda"] for r in unconstrained if r["violations"] > 0],
            "constrained_total_violations": sum(r["violations"] for r in constrained_rows),
            "max_outcome_cost_of_constraint": max(
                (u["outcomes"] - c["outcomes"])
                for u, c in zip(unconstrained, constrained_rows)),
            "max_spend_cost_of_constraint": max(
                (c["spend"] - u["spend"])
                for u, c in zip(unconstrained, constrained_rows)),
        },
    })


if __name__ == "__main__":
    main()
