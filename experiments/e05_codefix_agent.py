"""E5 - Program repair in the coding world: baseline LLM agent.

The flagship use case: a local LLM agent repairs the 10-task CodeFix
benchmark inside the world model. The world executes the tests bit-exactly,
so success is verifiable, and failing-test feedback flows back into the next
attempt. Metrics: pass@1, pass@budget (4 attempts), attempts-to-solve.

The repair agent is deliberately small (qwen2.5:1.5b): in pilots, both the
7B and 3B agents saturated the benchmark (pass@1 = 100% - failing-test
feedback makes these archetypes easy for capable models), leaving no headroom
to measure judge-based selection in E6. The capability ladder is itself a
finding; the 1.5B agent leaves failures to study.
"""

from openworld import OllamaLLM
from openworld.coding import BENCHMARK, build_codefix_world
from openworld.parsing import extract_code
from openworld.state import Action

from common import require_ollama, save_results, wilson_ci

AGENT_MODEL = "qwen2.5:1.5b"

MAX_ATTEMPTS = 4

REPAIR_SYSTEM = (
    "You are an expert Python debugger. You receive a buggy function, its "
    "intended behavior, and failing test feedback. Reply with ONLY a python "
    "code block containing the complete corrected source. Keep the same "
    "function name and signature. Use only pure python and math."
)


def repair_prompt(task, state):
    errors = "\n".join(f"- {e}" for e in state["last_errors"]) or "- (none reported yet)"
    return (
        f"Intended behavior: {task.description}\n\n"
        f"Current source:\n```python\n{state['source']}\n```\n\n"
        f"Tests passing: {state['tests_passed']}, failing: {state['tests_failed']}\n"
        f"Failing test feedback:\n{errors}\n\n"
        "Provide the corrected source."
    )


def solve_task(task, llm, max_attempts=MAX_ATTEMPTS, propose=None):
    """Run one episode. `propose` lets E6 swap in judge-guided proposals."""
    world = build_codefix_world(task)
    attempts_used = 0
    first_attempt_solved = False
    for attempt in range(max_attempts):
        if propose is None:
            patch = extract_code(llm.ask(repair_prompt(task, world.state), system=REPAIR_SYSTEM))
        else:
            patch = propose(task, world.state, attempt)
        world.step(Action("submit_patch", params={"source": patch}))
        attempts_used = attempt + 1
        if world.state["solved"]:
            if attempt == 0:
                first_attempt_solved = True
            break
    return {
        "task": task.name,
        "solved": bool(world.state["solved"]),
        "solved_first_attempt": first_attempt_solved,
        "attempts": attempts_used,
    }


def summarize(records, condition, model):
    n = len(records)
    solved = sum(r["solved"] for r in records)
    first = sum(r["solved_first_attempt"] for r in records)
    return {
        "condition": condition,
        "model": model,
        "n_tasks": n,
        "pass_at_1": first / n,
        "pass_at_1_ci": list(wilson_ci(first, n)),
        "pass_at_budget": solved / n,
        "pass_at_budget_ci": list(wilson_ci(solved, n)),
        "mean_attempts_when_solved": (
            sum(r["attempts"] for r in records if r["solved"]) / solved if solved else None
        ),
    }


def main():
    import sys
    agent_model = sys.argv[1] if len(sys.argv) > 1 else AGENT_MODEL
    require_ollama(agent_model)
    llm = OllamaLLM(model=agent_model, temperature=0.2, options={"seed": 41})
    records = []
    for task in BENCHMARK:
        record = solve_task(task, llm)
        records.append(record)
        print(f"  {task.name}: solved={record['solved']} attempts={record['attempts']}")
    summary = summarize(records, "baseline", agent_model)
    save_results("e05_codefix_agent", {
        "max_attempts": MAX_ATTEMPTS, "summary": summary, "records": records,
    })
    print(f"pass@1 {summary['pass_at_1']:.0%}, pass@{MAX_ATTEMPTS} "
          f"{summary['pass_at_budget']:.0%}")


if __name__ == "__main__":
    main()
