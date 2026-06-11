"""E18 - Repair-loop ablation (round-2 review item R3).

The 'full gate' verification both FILTERS bad candidates and REPAIRS via
feedback regeneration. This ablation separates the mechanisms: the same full
gate with max_iters=1 (filter only; rejecting every candidate is possible)
versus max_iters=4 (filter + repair loop), paired generation seeds, 3B
generator at high temperature.
"""

from openworld import OllamaLLM, WorldState
from openworld.state import Action
from openworld.verify import SynthesisError, Verifier, synthesize_transition

from common import (
    SMALL_MODEL, SPRINT_ACTIONS, SPRINT_DESCRIPTION, SPRINT_INITIAL,
    SPRINT_PROBES, SPRINT_RULES, probe_accuracy, require_ollama, save_results,
    sprint_ground_truth,
)

ATTEMPTS = 8
INVARIANTS = [(
    "counters never negative",
    lambda s: all(s[k] >= 0 for k in ("backlog", "shipped", "bugs", "debt")),
)]


def run(max_iters, attempt):
    llm = OllamaLLM(model=SMALL_MODEL, temperature=0.9,
                    options={"seed": 3000 + attempt})  # paired with E3 seeds
    verifier = Verifier(
        initial_state=WorldState(SPRINT_INITIAL),
        sample_actions=[Action(a, agent="smoke_test_agent") for a in SPRINT_ACTIONS],
        invariants=INVARIANTS,
    )
    try:
        transition = synthesize_transition(
            llm, SPRINT_DESCRIPTION, WorldState(SPRINT_INITIAL),
            SPRINT_ACTIONS, SPRINT_RULES, verifier=verifier, max_iters=max_iters,
        )
    except SynthesisError:
        return {"max_iters": max_iters, "attempt": attempt,
                "accepted": False, "probe_accuracy": 0.0}
    return {"max_iters": max_iters, "attempt": attempt, "accepted": True,
            "probe_accuracy": probe_accuracy(transition, SPRINT_PROBES, sprint_ground_truth)}


def main():
    require_ollama(SMALL_MODEL)
    runs = []
    for max_iters in (1, 4):
        for attempt in range(ATTEMPTS):
            record = run(max_iters, attempt)
            runs.append(record)
            print(f"  iters={max_iters} #{attempt}: accepted={record['accepted']} "
                  f"acc={record['probe_accuracy']:.2f}")
    summary = []
    for max_iters in (1, 4):
        rows = [r for r in runs if r["max_iters"] == max_iters]
        accepted = [r for r in rows if r["accepted"]]
        summary.append({
            "max_iters": max_iters,
            "n": len(rows),
            "acceptance_rate": len(accepted) / len(rows),
            "mean_probe_accuracy_all": sum(r["probe_accuracy"] for r in rows) / len(rows),
            "mean_probe_accuracy_accepted": (
                sum(r["probe_accuracy"] for r in accepted) / len(accepted)
                if accepted else 0.0),
        })
    save_results("e18_repair_loop", {
        "generator": SMALL_MODEL, "attempts": ATTEMPTS, "summary": summary,
        "runs": runs,
    })
    for s in summary:
        print(f"max_iters={s['max_iters']}: accept {s['acceptance_rate']:.0%}, "
              f"probe(all) {s['mean_probe_accuracy_all']:.2f}, "
              f"probe(accepted) {s['mean_probe_accuracy_accepted']:.2f}")


if __name__ == "__main__":
    main()
