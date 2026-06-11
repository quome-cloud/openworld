"""E15 - Judge-alignment robustness (review item W6).

Extends E7: two rubric phrasings over the same 12 dial-swept triage episodes,
Spearman correlation per rubric against the ground-truth aggregate, and a
permutation p-value (10,000 shuffles) for each.
"""

from openworld import Action, Agent, Dial, Judge, Objective, OllamaLLM, Simulation

from common import (
    GENERATOR_MODEL, make_oracle_world, permutation_p_spearman, require_ollama,
    save_results, spearman,
)

DIAL_VALUES = [0.0, 0.09, 0.18, 0.27, 0.36, 0.45, 0.55, 0.64, 0.73, 0.82, 0.91, 1.0]
STEPS = 12

RUBRICS = {
    "original": (
        "Grade this ICU triage shift 0-10. Reward: treating critical patients "
        "promptly (deteriorations are very bad), good total health outcomes, and "
        "restraint in discretionary spending. Penalize: patients deteriorating "
        "while the coordinator waits, and wasteful spending late in the shift."
    ),
    "paraphrase": (
        "Score the following hospital triage episode from 0 to 10. A high score "
        "means critical cases were handled without delay (any deterioration is a "
        "serious failure), overall patient outcomes were strong, and money was "
        "spent carefully. Mark the episode down for avoidable deteriorations and "
        "for unnecessary expenditure."
    ),
}


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

    # Deterministic episodes, generated once and graded under both rubrics.
    episodes = []
    for value in DIAL_VALUES:
        sim.set_dial("stewardship", value)
        trajectory = sim.run(steps=STEPS)
        episodes.append((value, trajectory, trajectory.totals()["aggregate"]))

    results = {}
    for rubric_name, rubric in RUBRICS.items():
        points = []
        for value, trajectory, aggregate in episodes:
            score = judge.score_trajectory(trajectory, rubric=rubric)
            points.append({"dial": value, "ground_truth_aggregate": aggregate,
                           "judge_score": score})
        scored = [p for p in points if p["judge_score"] is not None]
        xs = [p["ground_truth_aggregate"] for p in scored]
        ys = [p["judge_score"] for p in scored]
        rho = spearman(xs, ys)
        p_value = permutation_p_spearman(xs, ys, seed=7)
        results[rubric_name] = {
            "points": points, "n_scored": len(scored),
            "spearman": rho, "permutation_p": p_value,
        }
        print(f"  rubric={rubric_name}: rho={rho:.3f} p={p_value:.4f} "
              f"({len(scored)} episodes)")

    save_results("e15_judge_robustness", {
        "model": GENERATOR_MODEL, "steps": STEPS, "rubrics": RUBRICS,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "points"}
                    for k, v in results.items()},
        "points": {k: v["points"] for k, v in results.items()},
    })


if __name__ == "__main__":
    main()
