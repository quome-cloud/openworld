"""Agents: LLM-backed planners that act inside a world.

An Agent proposes one action per step from the world's declared action set.
A hand-written `policy` function can replace the LLM for deterministic
baselines and oracle policies.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .llm import BaseLLM
from .parsing import extract_json
from .state import Action, WorldState

Policy = Callable[[WorldState, List[str]], Action]

PLANNER_SYSTEM = (
    "You are an agent acting in a simulated world. Each turn you receive the "
    "world state, your goal, and the legal actions. Reply with ONLY a JSON "
    'object: {"action": "<name>", "params": {...}, "reason": "<short>"}.'
)


class Agent:
    def __init__(
        self,
        name: str,
        goal: str = "",
        llm: Optional[BaseLLM] = None,
        policy: Optional[Policy] = None,
        persona: str = "",
    ):
        if llm is None and policy is None:
            raise ValueError(f"Agent {name!r} needs an llm or a policy")
        self.name = name
        self.goal = goal
        self.llm = llm
        self.policy = policy
        self.persona = persona

    def act(
        self,
        state: WorldState,
        actions: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> Action:
        """Choose an action. Unparseable or illegal LLM output becomes a noop."""
        if self.policy is not None:
            action = self.policy(state, actions)
            action.agent = self.name
            return action

        context_lines = "\n".join(f"{k}: {v}" for k, v in (context or {}).items())
        prompt = (
            (f"Persona: {self.persona}\n" if self.persona else "")
            + f"Your name: {self.name}\n"
            f"Your goal: {self.goal}\n"
            + (f"{context_lines}\n" if context_lines else "")
            + f"World state:\n{state.to_json(indent=2)}\n"
            f"Legal actions: {actions}\n"
            "Choose your action."
        )
        reply = self.llm.ask(prompt, system=PLANNER_SYSTEM)
        parsed = extract_json(reply)
        if not parsed or parsed.get("action") not in actions:
            return Action.noop(agent=self.name)
        return Action(
            name=parsed["action"],
            params=dict(parsed.get("params") or {}),
            agent=self.name,
        )
