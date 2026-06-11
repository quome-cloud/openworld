"""The World: a declarative container for state, actions, rules, and dynamics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .llm import BaseLLM
from .state import Action, WorldState
from .transition import CodeTransition, LLMTransition, Transition
from .verify import Verifier, synthesize_transition


class World:
    """A world model: symbolic state + declared actions/rules + a dynamics engine.

    Provide a `transition` directly (FunctionTransition / CodeTransition /
    LLMTransition), or call `compile()` to have the LLM synthesize verified
    transition code from the description and rules.
    """

    def __init__(
        self,
        name: str,
        description: str,
        initial_state: Union[WorldState, Dict[str, Any]],
        actions: List[str],
        rules: Optional[List[str]] = None,
        transition: Optional[Transition] = None,
        llm: Optional[BaseLLM] = None,
    ):
        self.name = name
        self.description = description
        self.initial_state = WorldState(initial_state)
        self.actions = list(actions)
        self.rules = list(rules or [])
        self.transition = transition
        self.llm = llm
        self.state = self.initial_state.copy()

    def compile(
        self,
        llm: Optional[BaseLLM] = None,
        critic: Optional[BaseLLM] = None,
        invariants: Optional[list] = None,
        max_iters: int = 4,
        save_to: Optional[Union[str, Path]] = None,
    ) -> CodeTransition:
        """Synthesize verified transition code and install it as the dynamics.

        invariants: list of (description, fn(state) -> bool) checked during
        verification. save_to: optionally write the accepted code to a file so
        it remains an editable artifact.
        """
        generator = llm or self.llm
        if generator is None:
            raise ValueError("compile() needs an LLM (pass llm= or set World(llm=...))")
        verifier = Verifier(
            initial_state=self.initial_state,
            sample_actions=[Action(name, agent="smoke_test_agent") for name in self.actions],
            invariants=list(invariants or []),
            critic=critic,
        )
        transition = synthesize_transition(
            generator,
            description=self.description,
            initial_state=self.initial_state,
            actions=self.actions,
            rules=self.rules,
            verifier=verifier,
            max_iters=max_iters,
        )
        if save_to is not None:
            transition.save(save_to)
        self.transition = transition
        return transition

    def use_llm_dynamics(self, llm: Optional[BaseLLM] = None) -> LLMTransition:
        """Install direct LLM next-state prediction as the dynamics engine."""
        engine = LLMTransition(llm or self.llm, self.description, self.rules)
        if engine.llm is None:
            raise ValueError("use_llm_dynamics() needs an LLM")
        self.transition = engine
        return engine

    def step(self, action: Action) -> WorldState:
        if self.transition is None:
            raise RuntimeError(
                f"World {self.name!r} has no dynamics. Pass transition=, or call "
                "compile() / use_llm_dynamics() first."
            )
        self.state = self.transition.step(self.state, action)
        return self.state

    def reset(self) -> WorldState:
        self.state = self.initial_state.copy()
        return self.state
