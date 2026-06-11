"""E6 - Agents-as-a-judge for behavior selection in the coding world.

At every repair attempt, sample THREE candidate patches from the small
proposer (qwen2.5:1.5b, high temperature - the same proposer as the E5
baseline), then have the 7B judge select the one to submit (it sees the
task, the test feedback, and the candidates). Compared against the E5
baseline (single low-temperature 1.5B proposal). Also audits the judge:
when at least one sampled candidate passes the tests, how often does the
judge pick a passing one?
"""

from openworld import Judge, OllamaLLM
from openworld.coding import BENCHMARK, run_tests
from openworld.parsing import extract_code

from common import GENERATOR_MODEL, require_ollama, save_results, wilson_ci
from e05_codefix_agent import AGENT_MODEL, MAX_ATTEMPTS, REPAIR_SYSTEM, repair_prompt, solve_task, summarize

N_CANDIDATES = 3


def main():
    require_ollama(GENERATOR_MODEL)
    judge = Judge(
        OllamaLLM(model=GENERATOR_MODEL, temperature=0.0),
        criteria=(
            "Pick the patch most likely to make ALL the failing tests pass while "
            "implementing the intended behavior. Prefer minimal, correct fixes."
        ),
    )
    judge_audit = {"rounds_with_passing_candidate": 0, "judge_picked_passing": 0}

    def propose(task, state, attempt):
        candidates = []
        for k in range(N_CANDIDATES):
            sampler = OllamaLLM(model=AGENT_MODEL, temperature=0.9,
                                options={"seed": 5000 + attempt * 10 + k})
            candidates.append(
                extract_code(sampler.ask(repair_prompt(task, state), system=REPAIR_SYSTEM))
            )
        passing = [run_tests(c, task.tests)["failed"] == 0 for c in candidates]
        choice = judge.choose(
            [f"```python\n{c}\n```" for c in candidates],
            context=repair_prompt(task, state),
        )
        if any(passing):
            judge_audit["rounds_with_passing_candidate"] += 1
            if passing[choice]:
                judge_audit["judge_picked_passing"] += 1
        return candidates[choice]

    records = []
    for task in BENCHMARK:
        record = solve_task(task, llm=None, propose=propose)
        records.append(record)
        print(f"  {task.name}: solved={record['solved']} attempts={record['attempts']}")

    summary = summarize(records, "judge_selected",
                        f"proposer={AGENT_MODEL}, judge={GENERATOR_MODEL}")
    n_rounds = judge_audit["rounds_with_passing_candidate"]
    audit = {
        **judge_audit,
        "judge_accuracy_when_solvable": (
            judge_audit["judge_picked_passing"] / n_rounds if n_rounds else None
        ),
        "judge_accuracy_ci": (
            list(wilson_ci(judge_audit["judge_picked_passing"], n_rounds)) if n_rounds else None
        ),
    }
    save_results("e06_judge_selection", {
        "max_attempts": MAX_ATTEMPTS, "n_candidates": N_CANDIDATES,
        "summary": summary, "judge_audit": audit, "records": records,
    })
    print(f"pass@1 {summary['pass_at_1']:.0%}, pass@{MAX_ATTEMPTS} "
          f"{summary['pass_at_budget']:.0%}; judge accuracy when solvable: "
          f"{audit['judge_accuracy_when_solvable']}")


if __name__ == "__main__":
    main()
