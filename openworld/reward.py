"""Verified reward / objective induction -- the missing pillar, symmetric to CodeTransition.

OpenWorld verifies *dynamics* as code (CodeTransition) but treats goals as *declared* (Objective with
a hand-written fn). For interactive problems (e.g. ARC-AGI-3) the hard part is *goal discovery*: the
win condition is unknown. This module makes the OBJECTIVE a first-class *verified, induced* artifact:

  * CodeObjective -- a reward/goal as sandboxed code `reward(state, action, next_state) -> float`,
    duck-compatible with Objective (name/score/effective_weight), runnable in the same pure-Python
    sandbox as CodeTransition (zero new dependencies).
  * induce_reward(examples, synth) -- synthesize + EXACT-MATCH VERIFY a CodeObjective from observed
    reward transitions, exactly as synthesize_transition does for dynamics. Discovers the goal.

Together with CodeTransition (verified dynamics) and a typed Perceptor (verified perception), this
completes the loop: verified perception + verified dynamics + verified reward + planning -- a world
*computer* that can solve, not just simulate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .sandbox import load_transition_code


class CodeObjective:
    """A verified reward/goal as sandboxed code (symmetric to CodeTransition).

    `code` defines `def reward(state, action, next_state) -> float` (a scalar reward; a goal predicate
    is just a 0/1 reward). Runs in the pure-Python sandbox. Duck-compatible with Objective, so it can
    be scored, weighted, and planned against anywhere an Objective is used.
    """

    def __init__(self, code: str, name: str = "induced_reward", func_name: str = "reward",
                 weight: Any = 1.0, description: str = "") -> None:
        self.code = code
        self.name = name
        self.func_name = func_name
        self.weight = weight
        self.description = description or "induced verified reward (code)"

    @property
    def effective_weight(self) -> float:
        from .objectives import Dial
        return self.weight.value if isinstance(self.weight, Dial) else float(self.weight)

    def score(self, state: Dict[str, Any], action: Any, next_state: Dict[str, Any]) -> float:
        fn = load_transition_code(self.code, self.func_name)
        a = action.to_dict() if hasattr(action, "to_dict") else (action if action is not None else {})
        return float(fn(dict(state), a, dict(next_state)))

    def is_goal(self, state: Dict[str, Any]) -> bool:
        """Treat a positive reward into `state` (from an empty no-op) as goal-reached."""
        return self.score(state, {}, state) > 0

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.write_text(self.code, encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Union[str, Path], func_name: str = "reward") -> "CodeObjective":
        return cls(Path(path).read_text(encoding="utf-8"), func_name=func_name)


def verify_reward(code: str, examples: List[Dict[str, Any]], func_name: str = "reward") -> float:
    """Fraction of observed reward transitions the code reproduces exactly (the induction gate)."""
    if not examples:
        return 0.0
    try:
        fn = load_transition_code(code, func_name)
    except Exception:  # noqa: BLE001
        return 0.0
    ok = 0
    for ex in examples:
        try:
            got = float(fn(dict(ex["state"]), dict(ex.get("action", {})), dict(ex["next_state"])))
            if abs(got - float(ex["reward"])) < 1e-6:
                ok += 1
        except Exception:  # noqa: BLE001
            pass
    return ok / len(examples)


def induce_reward(
    examples: List[Dict[str, Any]],
    synth: Callable[[str], str],
    prompt_fn: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    rounds: int = 4,
    threshold: float = 0.99,
    func_name: str = "reward",
) -> Tuple[Optional[CodeObjective], float]:
    """Induce a VERIFIED CodeObjective from observed reward transitions (symmetric to
    synthesize_transition). `examples`: list of {state, action, next_state, reward}. `synth`: maps a
    prompt to candidate code. Returns (CodeObjective | None, held-out exact-match accuracy)."""
    if len(examples) < 2:
        return None, 0.0
    cut = max(1, len(examples) * 3 // 4)
    train, held = examples[:cut], examples[cut:] or examples[:cut]
    build = prompt_fn or _default_prompt
    best_code, best_acc = None, 0.0
    feedback = ""
    for _ in range(rounds):
        code = _extract_code(synth(build(train) + feedback))
        acc = verify_reward(code, held, func_name)
        if acc > best_acc:
            best_acc, best_code = acc, code
        if acc >= threshold:
            break
        feedback = ("\n\nThe reward() was wrong on held-out transitions. It must EXACTLY reproduce the "
                    "observed reward for every transition; fix the rule.")
    if best_code is None:
        return None, 0.0
    return CodeObjective(best_code, func_name=func_name), best_acc


def _default_prompt(examples: List[Dict[str, Any]]) -> str:
    import json
    ex = "\n".join(
        f"- reward={e['reward']} for action={e.get('action')} (state->next_state changed)" for e in examples[:16]
    )
    return (
        "Induce the REWARD/win condition of a world as pure Python (no imports). Write exactly:\n\n"
        "    def reward(state, action, next_state):  # dicts; return a float reward\n\n"
        "It must reproduce the observed reward for every transition (positive only when the goal is "
        "achieved). Return ONLY a ```python code block.\n\nObserved reward transitions:\n" + ex
    )


def _extract_code(text: str) -> str:
    import re
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()
