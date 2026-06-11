"""E7 - Judge-as-objective: does an LLM judge's grading track ground truth?

Runs the triage world across 12 stewardship-dial settings (deterministic
policy), grades each trajectory 0-10 with an LLM judge against a written
rubric, and correlates judge scores with the ground-truth dial-weighted
aggregate. High rank correlation would license using judges to select
behavior where no programmatic objective exists.
"""

from openworld import Action, Agent, Dial, Judge, Objective, OllamaLLM, Simulation

from common import GENERATOR_MODEL, make_oracle_world, require_ollama, save_results, spearman

DIAL_VALUES = [0.0, 0.09, 0.18, 0.27, 0.36, 0.45, 0.55, 0.64, 0.73, 0.82, 0.91, 1.0]
STEPS = 12

RUBRIC = (
    "Grade this ICU triage shift 0-10. Reward: treating critical patients "
    "promptly (deteriorations are very bad), good total health outcomes, and "
    "restraint in discretionary spending. Penalize: patients deteriorating "
    "while the coordinator waits, and wasteful spending late in the shift."
)


def main():
    require_ollama(GENERATOR_MODEL)
    judge = Judge(OllamaLLM(model=GENERATOR_MODEL, temperature=0.0))

    stewardship = Dial("stewardship", value=0.0)

    def nurse(state, actions):
        if state["critical_waiting"] > 0:
            return Action("treat_critical")
        cap = round((1.0 - stewardship.value) * 14)
        if state["moderate_waiting"] > 0 and state["spend"] + 1 <= cap:
            return Action("treat_moderate")
        return Action("wait")

    world = make_oracle_world("triage")
    sim = Simulation(
        world,
        agents=[Agent(name="nurse", policy=nurse)],
        objectives=[
            Objective("outcomes", fn=lambda s, a, ns: ns["outcomes"] - s["outcomes"], weight=1.0),
            Objective("thrift", fn=lambda s, a, ns: -(ns["spend"] - s["spend"]), weight=stewardship),
        ],
    )

    points = []
    for value in DIAL_VALUES:
        sim.set_dial("stewardship", value)
        trajectory = sim.run(steps=STEPS)
        aggregate = trajectory.totals()["aggregate"]
        score = judge.score_trajectory(trajectory, rubric=RUBRIC)
        points.append({"dial": value, "ground_truth_aggregate": aggregate,
                       "judge_score": score})
        print(f"  lambda={value:.2f}: aggregate={aggregate:.2f} judge={score}")

    scored = [p for p in points if p["judge_score"] is not None]
    correlation = spearman(
        [p["ground_truth_aggregate"] for p in scored],
        [p["judge_score"] for p in scored],
    )
    save_results("e07_judge_alignment", {
        "model": GENERATOR_MODEL, "rubric": RUBRIC, "steps": STEPS,
        "points": points,
        "n_scored": len(scored),
        "spearman_judge_vs_aggregate": correlation,
    })
    print(f"Spearman(judge, ground truth) = {correlation:.3f} over {len(scored)} episodes")


if __name__ == "__main__":
    main()
