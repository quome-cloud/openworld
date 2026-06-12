"""Transition engines: pluggable world dynamics.

Three interchangeable engines, all implementing (state, action) -> next_state:

- FunctionTransition: hand-written Python, fully deterministic.
- CodeTransition:     LLM-synthesized executable Python (the Code World Model
                      paradigm). The source is a plain, editable artifact that
                      can be inspected, unit-tested, saved, and reloaded.
- LLMTransition:      the LLM predicts the next state directly as JSON each
                      step — quick to prototype, stochastic and unverified,
                      approximating the behavior of learned neural dynamics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

from .llm import BaseLLM
from .parsing import extract_json
from .sandbox import run_transition_code
from .state import Action, WorldState


class Transition:
    """Base interface for world dynamics."""

    def step(self, state: WorldState, action: Action) -> WorldState:
        raise NotImplementedError


class FunctionTransition(Transition):
    """Wraps a hand-written python function (state_dict, action_dict) -> dict."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def step(self, state: WorldState, action: Action) -> WorldState:
        result = self.fn(dict(state.copy()), action.to_dict())
        return WorldState(result)


class CodeTransition(Transition):
    """Executable Python source as the dynamics engine.

    The code must define `def transition(state: dict, action: dict) -> dict`.
    It runs in a restricted sandbox; bit-exact and training-free.
    """

    def __init__(self, code: str, func_name: str = "transition"):
        self.code = code
        self.func_name = func_name

    def step(self, state: WorldState, action: Action) -> WorldState:
        result = run_transition_code(
            self.code, dict(state), action.to_dict(), self.func_name
        )
        return WorldState(result)

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.write_text(self.code, encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Union[str, Path], func_name: str = "transition") -> "CodeTransition":
        return cls(Path(path).read_text(encoding="utf-8"), func_name=func_name)


class LLMTransition(Transition):
    """The LLM predicts the next state directly. Fast to prototype, unverified.

    Useful as a 'learned-style' baseline to compare against synthesized code.
    """

    SYSTEM = (
        "You are the dynamics engine of a simulated world. Given the current "
        "world state as JSON and an action, output ONLY the complete next world "
        "state as a single JSON object with the same keys. Apply the world rules "
        "exactly. No commentary."
    )

    def __init__(self, llm: BaseLLM, description: str = "", rules: Optional[list] = None):
        self.llm = llm
        self.description = description
        self.rules = list(rules or [])

    def step(self, state: WorldState, action: Action) -> WorldState:
        rules = "\n".join(f"- {rule}" for rule in self.rules)
        prompt = (
            f"World: {self.description}\n"
            + (f"Rules:\n{rules}\n" if rules else "")
            + f"Current state:\n{state.to_json()}\n"
            f"Action: {action.to_dict()}\n"
            "Next state JSON:"
        )
        reply = self.llm.ask(prompt, system=self.SYSTEM)
        parsed = extract_json(reply)
        if parsed is None:
            # An unparseable prediction leaves the world unchanged rather than
            # crashing a long rollout.
            return state.copy()
        return WorldState(parsed)


class PhasedTransition(Transition):
    """Dynamics whose rules change over time: ordered (trigger, transition)
    phases with sequential, irreversible advance.

    A trigger is an int N (the phase becomes eligible once the phased step
    counter reaches N) or a callable state -> bool. Phase 0's trigger is
    ignored - it is the starting regime. Before delegating each step, the
    NEXT phase's trigger is checked once; if it fires, the regime advances,
    permanently (regimes do not revert, and at most one advance per step).
    The active phase index is recorded in state[record_key] and a step
    counter in record_key + "_steps", so trajectories stay replayable and
    regime switches are visible in the record.

    Every phase is constructed (and, when synthesized, verified) BEFORE the
    run, preserving ahead-of-time verification. For parameter drift that
    fits in state, prefer encoding the regime variable in state and
    branching in the rules; PhasedTransition is for structural change.
    """

    def __init__(self, phases, record_key: str = "_phase"):
        if not phases:
            raise ValueError("PhasedTransition needs at least one phase")
        self.phases = list(phases)
        self.record_key = record_key

    def step(self, state: WorldState, action: Action) -> WorldState:
        s = state.copy()
        index = int(s.get(self.record_key, 0))
        steps = int(s.get(self.record_key + "_steps", 0))
        if index + 1 < len(self.phases):
            trigger = self.phases[index + 1][0]
            fired = steps >= trigger if isinstance(trigger, int) else bool(trigger(s))
            if fired:
                index += 1
        s = self.phases[index][1].step(s, action)
        s[self.record_key] = index
        s[self.record_key + "_steps"] = steps + 1
        return s
