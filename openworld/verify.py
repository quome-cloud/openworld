"""Transition synthesis with a generate -> verify -> repair loop.

A generator LLM writes executable Python dynamics for the world; a verifier
checks each candidate before it is accepted:

  1. syntactic  — the code parses and defines transition(state, action)
  2. behavioral — a sandboxed smoke-run on the initial state with each sample
                  action must return a dict and respect declared invariants
  3. semantic   — (optional) a critic LLM reviews the code against the world
                  rules and returns PASS or actionable feedback

Failed checks feed back into the next generation attempt, forming the
closed-loop correct/redo relay described in the research.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .llm import BaseLLM
from .parsing import extract_code
from .sandbox import SandboxError, run_transition_code
from .state import Action, WorldState
from .transition import CodeTransition

Invariant = Callable[[WorldState], bool]


class SynthesisError(RuntimeError):
    def __init__(self, message: str, attempts: List[str]):
        super().__init__(message)
        self.attempts = attempts


GENERATOR_SYSTEM = (
    "You write deterministic Python world-dynamics code. Reply with a single "
    "python code block defining exactly:\n"
    "    def transition(state: dict, action: dict) -> dict\n"
    "It must return the COMPLETE next state dict (copy the input, never mutate "
    "shared structures), handle every declared action including 'noop', use only "
    "pure python and the math module, no imports, no I/O, no randomness."
)

CRITIC_SYSTEM = (
    "You are a strict code reviewer for world-dynamics code. Check the code "
    "against the world description and rules. If it faithfully implements every "
    "rule and action, reply exactly PASS. Otherwise reply FAIL: followed by a "
    "short, concrete description of each problem."
)


@dataclass
class Verifier:
    """Checks candidate transition code. All checks return (ok, feedback)."""

    initial_state: WorldState
    sample_actions: List[Action] = field(default_factory=list)
    invariants: List[Tuple[str, Invariant]] = field(default_factory=list)
    critic: Optional[BaseLLM] = None
    world_context: str = ""

    def check_syntax(self, code: str) -> Tuple[bool, str]:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc}"
        defines = any(
            isinstance(node, ast.FunctionDef) and node.name == "transition"
            for node in tree.body
        )
        if not defines:
            return False, "Code must define a top-level function transition(state, action)."
        return True, ""

    def check_behavior(self, code: str) -> Tuple[bool, str]:
        actions = self.sample_actions or [Action.noop()]
        for action in actions:
            try:
                result = run_transition_code(code, dict(self.initial_state), action.to_dict())
            except SandboxError as exc:
                return False, f"On action {action.name!r}: {exc}"
            next_state = WorldState(result)
            for description, invariant in self.invariants:
                if not invariant(next_state):
                    return False, (
                        f"After action {action.name!r}, invariant violated: {description}. "
                        f"Resulting state: {next_state.to_json()}"
                    )
        return True, ""

    def check_semantics(self, code: str) -> Tuple[bool, str]:
        if self.critic is None:
            return True, ""
        verdict = self.critic.ask(
            f"{self.world_context}\n\nCandidate dynamics code:\n```python\n{code}\n```",
            system=CRITIC_SYSTEM,
        ).strip()
        if verdict.upper().startswith("PASS"):
            return True, ""
        return False, verdict

    def check(self, code: str) -> Tuple[bool, str]:
        for checker in (self.check_syntax, self.check_behavior, self.check_semantics):
            ok, feedback = checker(code)
            if not ok:
                return False, feedback
        return True, ""


def synthesize_transition(
    llm: BaseLLM,
    description: str,
    initial_state: WorldState,
    actions: List[str],
    rules: Optional[List[str]] = None,
    verifier: Optional[Verifier] = None,
    max_iters: int = 4,
) -> CodeTransition:
    """Generate verified transition code for a world. Raises SynthesisError."""
    rules = rules or []
    # Smoke-run actions carry a named agent so agent-dependent code paths
    # (e.g. per-agent tallies) are exercised realistically.
    sample_actions = [Action(name, agent="smoke_test_agent") for name in actions]
    if verifier is None:
        verifier = Verifier(initial_state=initial_state, sample_actions=sample_actions)
    verifier.world_context = verifier.world_context or _world_context(
        description, initial_state, actions, rules
    )

    prompt = (
        f"{_world_context(description, initial_state, actions, rules)}\n\n"
        "Write the transition function now."
    )
    attempts: List[str] = []
    feedback = ""
    for _ in range(max_iters):
        full_prompt = prompt if not feedback else (
            f"{prompt}\n\nYour previous attempt failed verification:\n{feedback}\n"
            "Fix the problem and output the corrected full code block."
        )
        code = extract_code(llm.ask(full_prompt, system=GENERATOR_SYSTEM))
        attempts.append(code)
        ok, feedback = verifier.check(code)
        if ok:
            return CodeTransition(code)
    raise SynthesisError(
        f"Failed to synthesize verified transition code after {max_iters} attempts. "
        f"Last feedback: {feedback}",
        attempts,
    )


def _world_context(
    description: str, initial_state: WorldState, actions: List[str], rules: List[str]
) -> str:
    rule_lines = "\n".join(f"- {rule}" for rule in rules) or "- (none beyond the description)"
    return (
        f"World description: {description}\n"
        f"State schema (example initial state):\n{initial_state.to_json(indent=2)}\n"
        f"Actions (action['name'] values, plus 'noop'): {actions}\n"
        f"Action dicts look like: {{'name': ..., 'params': {{...}}, 'agent': ...}}\n"
        f"Rules:\n{rule_lines}"
    )
