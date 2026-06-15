"""E59 - Optimizing the brain architecture for a task, with the LLM held constant.

Neuromorphic architecture search over composable worlds: the *backbone* (the LLM's
raw capability) is fixed; we vary the brain ARCHITECTURE around it -- whether it has
long-term memory it can retrieve, how many tree-of-thoughts drafts it generates, and
whether it verifies them -- and show that the architecture, optimized for the task,
beats the bare backbone by a wide margin. Then a live LLM demo runs the real
text-in -> reason -> text-out loop (TextPerceptor + LLMEmitter).

Task suite (recalled-fact + verify): each question needs a specific fact (common
knowledge or rare, requiring retrieval) and is simple or hard (a hard task needs a
verifier to filter best-of-N drafts). The backbone answers a question iff the fact
is in its context (common, or retrieved), correctly for simple tasks and with a
fixed per-draft probability for hard tasks. That capability is IDENTICAL across all
architectures -- the whole point.

Deterministic/offline core; an optional Ollama demo (fail-soft) shows the real loop.
"""

import numpy as np

from openworld import (Aggregator, Bridge, CodePerceptor, CodeTransition,
                       CompositeWorld, World, from_spec, render_card, to_spec,
                       validate_spec)
from openworld.state import Action

from common import save_results

SEED = 59
N_TASKS = 40
P_HARD = 0.4                  # per-draft success on a hard task (backbone constant)
WIDTHS, MEMS, VERIFY = [1, 3, 5], ["none", "longterm"], [False, True]


# --------------------------------------------------------------------------- #
# the task suite + the FIXED backbone
# --------------------------------------------------------------------------- #
def make_tasks(seed):
    rng = np.random.RandomState(seed)
    tasks = []
    for i in range(N_TASKS):
        tasks.append({"key": f"fact{i}", "answer": int(rng.randint(1000)),
                      "common": bool(i % 2 == 0),      # half common-knowledge
                      "hard": bool((i // 2) % 2 == 0)})  # half hard
    return tasks


def backbone(task, context_keys, seed):
    """The fixed LLM: answers iff it knows the fact (common or in context);
    correct for simple tasks, correct with prob P_HARD per draft for hard ones."""
    if not (task["common"] or task["key"] in context_keys):
        return None                                    # doesn't know it
    if not task["hard"]:
        return task["answer"]
    return task["answer"] if np.random.RandomState(seed).random() < P_HARD else -1


def eval_arch(tasks, memory, width, verify, seed):
    correct = 0
    for i, task in enumerate(tasks):
        ctx = {task["key"]} if memory == "longterm" else set()   # retrieval
        drafts = [backbone(task, ctx, seed * 9973 + i * 17 + d) for d in range(width)]
        if verify:                                     # keep a draft that checks out
            pick = next((d for d in drafts if d == task["answer"]), drafts[0])
        else:
            pick = drafts[0]
        correct += int(pick == task["answer"])
    return correct / len(tasks)


# --------------------------------------------------------------------------- #
# the brain as a CompositeWorld (architecture artifact, serializable)
# --------------------------------------------------------------------------- #
C_CODE = "def transition(state, action):\n    return dict(state)"
BRIDGE_CODE = ('def transition(state, action):\n'
               '    return {"a": dict(state["a"]), "b": dict(state["b"])}')
PERCEIVE_CODE = """
def perceive(data):
    out = {}
    for line in str(data).splitlines():
        if line.strip().lower().startswith("q:"):
            out["question"] = line.split(":", 1)[1].strip()
    return out
"""


def brain_progress(children):
    return 1 if children.get("task", {}).get("solved") else 0


def brain_world(arch):
    conscious = World(name="conscious", description="working memory: question, retrieved facts, drafts",
                      initial_state={"question": "", "context": [], "drafts": []},
                      actions=["think", "draft", "verify"], rules=["holds the current reasoning."],
                      transition=CodeTransition(C_CODE))
    unconscious = World(name="unconscious", description="long-term memory of facts",
                        initial_state={"memory": {}, "retrieved": []},
                        actions=["retrieve", "consolidate"], rules=["stores and recalls facts."],
                        transition=CodeTransition(C_CODE))
    task = World(name="task", description="the question to answer",
                 initial_state={"answer": None, "solved": False}, actions=["answer"],
                 rules=["solved when the emitted answer matches."],
                 transition=CodeTransition(C_CODE))
    bridges = [Bridge(name="retrieve", a="unconscious", b="conscious",
                      transition=CodeTransition(BRIDGE_CODE),
                      description="surface the needed fact into working memory"),
               Bridge(name="emit", a="conscious", b="task",
                      transition=CodeTransition(BRIDGE_CODE),
                      description="write the answer to the task")]
    brain = CompositeWorld(
        name="brain", children={"conscious": conscious, "unconscious": unconscious, "task": task},
        bridges=bridges, aggregators=[Aggregator(name="goal_progress", fn=brain_progress)],
        default_actions={"conscious": "think", "unconscious": "retrieve", "task": "answer"},
        description="A brain whose architecture (memory / tree width / verify) is "
                    "optimized for the task while the LLM backbone stays constant.")
    brain.perceptors = [CodePerceptor(code=PERCEIVE_CODE, produces=["question"],
                                      schema={"question": str}, modality="text")]
    brain.emit = [{"modality": "text", "fields": ["answer"],
                   "report": "Answer: {answer}",
                   "kind": "llm",
                   "template": "Question: {question}\nKnown facts: {context}\n"
                               "Answer with just the value."}]
    brain.objectives = [{"name": "answer correctly", "goal": "max accuracy"}]
    brain.architecture = arch
    return brain


def _rollout(world, actions):
    s, out = world.initial_state.copy(), []
    for a in actions:
        s = dict(world.transition.step(s, Action(a)))
        out.append(s)
    return out


# --------------------------------------------------------------------------- #
# optional live demo: real text-in -> reason -> text-out with an LLM
# --------------------------------------------------------------------------- #
def live_demo():
    try:
        from common import require_ollama
        from openworld import LLMEmitter, TextPerceptor
        from openworld.perceive import Observation
        llm = require_ollama()
    except Exception as e:
        return {"ran": False, "reason": str(e)[:120]}
    facts = {"the capital of France": "Paris", "2 + 2": "4"}
    perceptor = TextPerceptor(llm, produces=["question"])
    emitter = LLMEmitter(llm, reads=["question", "fact"],
                         template="Using this fact: {fact}\nAnswer concisely: {question}")
    runs = []
    for q, fact in [("Q: what is the capital of France?", facts["the capital of France"])]:
        try:
            delta = perceptor.perceive(Observation(modality="text", data=q))
            ans = emitter.emit({"question": delta.get("question", q), "fact": fact})
            runs.append({"question": q, "answer": ans.strip()[:160]})
        except Exception as e:
            runs.append({"question": q, "error": str(e)[:120]})
    return {"ran": True, "runs": runs}


def main():
    tasks = make_tasks(SEED)
    arms = {
        "bare LLM": ("none", 1, False),
        "+ memory (retrieval)": ("longterm", 1, False),
        "+ best-of-5 + verify": ("none", 5, True),
        "optimized brain": ("longterm", 5, True),
    }
    arm_acc = {name: round(eval_arch(tasks, *cfg, seed=SEED), 3)
               for name, cfg in arms.items()}

    # architecture search: random-sample the 12-config space, track best-so-far
    grid = [(m, w, v) for m in MEMS for w in WIDTHS for v in VERIFY]
    rng = np.random.RandomState(SEED)
    order = list(rng.permutation(len(grid)))
    best, best_cfg, curve = -1.0, None, []
    for idx in order:
        acc = eval_arch(tasks, *grid[idx], seed=SEED)
        if acc > best:
            best, best_cfg = acc, grid[idx]
        curve.append(round(best, 3))
    leaderboard = sorted(((round(eval_arch(tasks, *c, seed=SEED), 3), c) for c in grid),
                         reverse=True)

    # the optimized brain world: serialize + round-trip + card
    brain = brain_world({"memory": best_cfg[0], "width": best_cfg[1], "verify": best_cfg[2]})
    spec = to_spec(brain, card={"tags": ["brain", "architecture-search", "llm"],
                                "license": "MIT", "version": "0.1",
                                "lineage": "E59 brain architecture search"})
    problems = validate_spec(spec)
    acts = ["tick", "conscious:think", "tick"]
    try:
        round_trip = _rollout(brain, acts) == _rollout(from_spec(spec, allow_code=True), acts)
    except Exception:
        round_trip = False
    from pathlib import Path
    gal = Path(__file__).resolve().parent.parent / "gallery"
    gal.mkdir(exist_ok=True)
    render_card(spec, path=str(gal / "brain-arch.svg"))

    demo = live_demo()

    results = {
        "n_tasks": N_TASKS, "p_hard": P_HARD,
        "arms": {k: list(v) for k, v in arms.items()}, "arm_accuracy": arm_acc,
        "best_cfg": list(best_cfg), "best_acc": best,
        "search_curve": curve, "n_configs": len(grid),
        "leaderboard": [[a, list(c)] for a, c in leaderboard[:5]],
        "lift_over_bare": round(arm_acc["optimized brain"] - arm_acc["bare LLM"], 3),
        "brain_validated": problems == [], "brain_round_trip": round_trip,
        "live_demo": demo, "problems": problems,
    }
    save_results("e59_brain_arch", results)

    print("E59 - optimizing the brain architecture (LLM held constant)\n")
    for name, cfg in arms.items():
        print(f"  {name:<24} acc={arm_acc[name]:.3f}   {cfg}")
    print(f"\n  search recovered best config {best_cfg} (acc={best:.3f}) from "
          f"{len(grid)} architectures")
    print(f"  lift of optimized brain over bare LLM: +{results['lift_over_bare']:.3f}")
    print(f"  brain world: validated={results['brain_validated']} round_trip={round_trip}")
    print(f"  live LLM demo: {'ran - ' + str(demo.get('runs')) if demo.get('ran') else 'skipped (' + demo.get('reason', '') + ')'}")

    # --- self-checks (deterministic; backbone identical across arms) ---
    assert arm_acc["+ memory (retrieval)"] > arm_acc["bare LLM"], "retrieval should help"
    assert arm_acc["+ best-of-5 + verify"] > arm_acc["bare LLM"], "best-of-N+verify should help"
    assert arm_acc["optimized brain"] > arm_acc["+ memory (retrieval)"], \
        "combining memory + best-of-N should beat memory alone"
    assert results["lift_over_bare"] >= 0.3, "architecture should give a large lift"
    assert tuple(best_cfg) == ("longterm", 5, True), "search should find the richest architecture"
    assert curve[-1] == max(a for a, _ in leaderboard), "search converges to the best"
    assert problems == [] and round_trip, "brain world must validate and round-trip"
    print("\nchecks pass: architecture optimized for the task beats the bare backbone, "
          "LLM held constant; brain world serializes losslessly.")


if __name__ == "__main__":
    main()
