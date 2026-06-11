"""E2 - Synthesis reliability across worlds and model sizes.

For each of three worlds and two local models, attempt compile() five times
(different sampling seeds). Measures verified-acceptance rate, iterations to
acceptance, and semantic correctness of accepted code on ground-truth probes.
"""

from openworld import OllamaLLM, World
from openworld.state import Action
from openworld.verify import SynthesisError

from common import (
    GENERATOR_MODEL, SMALL_MODEL, SPRINT_PROBES, WORLD_SPECS, probe_accuracy,
    require_ollama, save_results, wilson_ci,
)

ATTEMPTS = 5
MAX_ITERS = 4


def world_probes(spec_name):
    if spec_name == "sprint":
        return SPRINT_PROBES
    spec = WORLD_SPECS[spec_name]
    # Generic probes: every action from the initial state plus noop.
    return [(dict(spec["initial"]), Action(a, agent="probe_agent"))
            for a in spec["actions"] + ["noop"]]


def main():
    require_ollama(GENERATOR_MODEL)  # fail fast if the server is down
    runs = []
    for model in (GENERATOR_MODEL, SMALL_MODEL):
        for spec_name, spec in WORLD_SPECS.items():
            for attempt in range(ATTEMPTS):
                llm = OllamaLLM(model=model, temperature=0.7,
                                options={"seed": 1000 + attempt})
                world = World(
                    name=spec_name, description=spec["description"],
                    initial_state=dict(spec["initial"]),
                    actions=list(spec["actions"]), rules=list(spec["rules"]),
                    llm=llm,
                )
                record = {"model": model, "world": spec_name, "attempt": attempt}
                try:
                    transition = world.compile(max_iters=MAX_ITERS)
                    record["accepted"] = True
                    record["probe_accuracy"] = probe_accuracy(
                        transition, world_probes(spec_name), spec["oracle"]
                    )
                    record["code"] = transition.code
                except SynthesisError as exc:
                    record["accepted"] = False
                    record["iterations_used"] = MAX_ITERS
                    record["probe_accuracy"] = 0.0
                    record["last_feedback"] = str(exc)[:300]
                runs.append(record)
                print(f"  {model} {spec_name} #{attempt}: "
                      f"accepted={record['accepted']} "
                      f"probe_acc={record['probe_accuracy']:.2f}")

    summary = []
    for model in (GENERATOR_MODEL, SMALL_MODEL):
        rows = [r for r in runs if r["model"] == model]
        accepted = sum(r["accepted"] for r in rows)
        low, high = wilson_ci(accepted, len(rows))
        summary.append({
            "model": model,
            "n": len(rows),
            "acceptance_rate": accepted / len(rows),
            "acceptance_ci": [low, high],
            "mean_probe_accuracy_accepted": (
                sum(r["probe_accuracy"] for r in rows if r["accepted"]) / accepted
                if accepted else 0.0
            ),
        })
    save_results("e02_synthesis", {
        "attempts_per_cell": ATTEMPTS, "max_iters": MAX_ITERS,
        "summary": summary, "runs": runs,
    })
    for s in summary:
        print(f"{s['model']}: acceptance {s['acceptance_rate']:.0%} "
              f"(CI {s['acceptance_ci'][0]:.2f}-{s['acceptance_ci'][1]:.2f}), "
              f"probe accuracy {s['mean_probe_accuracy_accepted']:.2f}")


if __name__ == "__main__":
    main()
