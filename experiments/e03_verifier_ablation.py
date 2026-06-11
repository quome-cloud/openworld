"""E3 - Verifier ablation: what each verification gate buys.

Generates sprint-world dynamics under four acceptance regimes and measures
the semantic correctness (ground-truth probe accuracy) of what each regime
accepts:

  none      - accept the model's first code block blindly
  syntax    - accept anything that parses and defines transition()
  full      - the framework default: syntax + sandboxed smoke-runs + invariants,
              with the repair feedback loop
  critic    - full + a second-model semantic critic in the loop

The generator is deliberately the SMALL model (qwen2.5:3b) at high temperature
so that defective generations occur and the gates have something to catch; the
semantic critic is the larger 7B model. Seeds are paired across conditions.
"""

from openworld import OllamaLLM, WorldState
from openworld.parsing import extract_code
from openworld.state import Action
from openworld.transition import CodeTransition
from openworld.verify import (
    GENERATOR_SYSTEM, SynthesisError, Verifier, _world_context, synthesize_transition,
)

from common import (
    GENERATOR_MODEL, SMALL_MODEL, SPRINT_ACTIONS, SPRINT_DESCRIPTION,
    SPRINT_INITIAL, SPRINT_PROBES, SPRINT_RULES, probe_accuracy,
    require_ollama, save_results, sprint_ground_truth,
)

ATTEMPTS = 8
MAX_ITERS = 4
INVARIANTS = [(
    "counters never negative",
    lambda s: all(s[k] >= 0 for k in ("backlog", "shipped", "bugs", "debt")),
)]


def generate_raw(llm):
    prompt = (
        f"{_world_context(SPRINT_DESCRIPTION, WorldState(SPRINT_INITIAL), SPRINT_ACTIONS, SPRINT_RULES)}\n\n"
        "Write the transition function now."
    )
    return extract_code(llm.ask(prompt, system=GENERATOR_SYSTEM))


def syntax_only_verifier():
    verifier = Verifier(initial_state=WorldState(SPRINT_INITIAL), sample_actions=[])
    verifier.check_behavior = lambda code: (True, "")  # disable smoke-runs
    return verifier


def run_condition(condition, attempt):
    llm = OllamaLLM(model=SMALL_MODEL, temperature=0.9,
                    options={"seed": 3000 + attempt})
    accepted, code = True, None
    try:
        if condition == "none":
            code = generate_raw(llm)
            transition = CodeTransition(code)
        elif condition == "syntax":
            transition = synthesize_transition(
                llm, SPRINT_DESCRIPTION, WorldState(SPRINT_INITIAL),
                SPRINT_ACTIONS, SPRINT_RULES,
                verifier=syntax_only_verifier(), max_iters=MAX_ITERS,
            )
        elif condition == "full":
            verifier = Verifier(
                initial_state=WorldState(SPRINT_INITIAL),
                sample_actions=[Action(a, agent="smoke_test_agent") for a in SPRINT_ACTIONS],
                invariants=INVARIANTS,
            )
            transition = synthesize_transition(
                llm, SPRINT_DESCRIPTION, WorldState(SPRINT_INITIAL),
                SPRINT_ACTIONS, SPRINT_RULES, verifier=verifier, max_iters=MAX_ITERS,
            )
        elif condition == "critic":
            verifier = Verifier(
                initial_state=WorldState(SPRINT_INITIAL),
                sample_actions=[Action(a, agent="smoke_test_agent") for a in SPRINT_ACTIONS],
                invariants=INVARIANTS,
                critic=OllamaLLM(model=GENERATOR_MODEL, temperature=0.0),
            )
            transition = synthesize_transition(
                llm, SPRINT_DESCRIPTION, WorldState(SPRINT_INITIAL),
                SPRINT_ACTIONS, SPRINT_RULES, verifier=verifier, max_iters=MAX_ITERS,
            )
        else:
            raise ValueError(condition)
    except SynthesisError:
        return {"condition": condition, "attempt": attempt,
                "accepted": False, "probe_accuracy": 0.0}
    accuracy = probe_accuracy(transition, SPRINT_PROBES, sprint_ground_truth)
    return {"condition": condition, "attempt": attempt,
            "accepted": accepted, "probe_accuracy": accuracy}


def main():
    require_ollama(GENERATOR_MODEL)
    runs = []
    for condition in ("none", "syntax", "full", "critic"):
        for attempt in range(ATTEMPTS):
            record = run_condition(condition, attempt)
            runs.append(record)
            print(f"  {condition} #{attempt}: accepted={record['accepted']} "
                  f"probe_acc={record['probe_accuracy']:.2f}")

    summary = []
    for condition in ("none", "syntax", "full", "critic"):
        rows = [r for r in runs if r["condition"] == condition]
        summary.append({
            "condition": condition,
            "n": len(rows),
            "acceptance_rate": sum(r["accepted"] for r in rows) / len(rows),
            "mean_probe_accuracy": sum(r["probe_accuracy"] for r in rows) / len(rows),
            "perfect_rate": sum(r["probe_accuracy"] == 1.0 for r in rows) / len(rows),
        })
    save_results("e03_verifier_ablation", {
        "attempts": ATTEMPTS, "generator": SMALL_MODEL,
        "critic": GENERATOR_MODEL, "summary": summary, "runs": runs,
    })
    for s in summary:
        print(f"{s['condition']:>7}: probe accuracy {s['mean_probe_accuracy']:.2f}, "
              f"perfect {s['perfect_rate']:.0%}")


if __name__ == "__main__":
    main()
