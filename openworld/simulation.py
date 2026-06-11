"""Simulation: run agents in a world and record scored trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agent import Agent
from .objectives import Dial, Objective, ObjectiveSuite
from .state import Action, WorldState
from .world import World


@dataclass
class StepRecord:
    step: int
    agent: Optional[str]
    action: Action
    state: WorldState
    scores: Dict[str, float]


@dataclass
class Trajectory:
    """The full record of one episode."""

    initial_state: WorldState
    steps: List[StepRecord] = field(default_factory=list)
    dial_settings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def final_state(self) -> WorldState:
        return self.steps[-1].state if self.steps else self.initial_state

    def totals(self) -> Dict[str, float]:
        """Sum of each objective's raw score (and the aggregate) over the episode."""
        totals: Dict[str, float] = {}
        for record in self.steps:
            for name, value in record.scores.items():
                totals[name] = totals.get(name, 0.0) + value
        return totals

    def history(self, objective: str) -> List[float]:
        return [record.scores.get(objective, 0.0) for record in self.steps]


class Simulation:
    """Orchestrates agents acting in a world under a set of objectives.

    Agents act round-robin: one world step per agent per simulation step.
    """

    def __init__(
        self,
        world: World,
        agents: List[Agent],
        objectives: Optional[List[Objective]] = None,
        on_step: Optional[Any] = None,
    ):
        self.world = world
        self.agents = list(agents)
        self.suite = ObjectiveSuite(list(objectives or []))
        self.on_step = on_step  # callback(StepRecord) for live monitoring

    @property
    def dials(self) -> Dict[str, Dial]:
        return self.suite.dials()

    def set_dial(self, name: str, value: float) -> None:
        dials = self.dials
        if name not in dials:
            raise KeyError(f"No dial named {name!r}; available: {sorted(dials)}")
        dials[name].set(value)

    def run(self, steps: int = 10, reset: bool = True) -> Trajectory:
        if reset:
            self.world.reset()
        trajectory = Trajectory(
            initial_state=self.world.state.copy(),
            dial_settings={name: dial.value for name, dial in self.dials.items()},
        )
        for step_index in range(steps):
            for agent in self.agents:
                state_before = self.world.state.copy()
                action = agent.act(
                    state_before,
                    self.world.actions + ["noop"],
                    context={"step": step_index, "world": self.world.name},
                )
                if action.name == "noop" and agent.llm is not None and agent.policy is None:
                    trajectory.warnings.append(
                        f"step {step_index}: agent {agent.name!r} fell back to noop"
                    )
                state_after = self.world.step(action)
                record = StepRecord(
                    step=step_index,
                    agent=agent.name,
                    action=action,
                    state=state_after.copy(),
                    scores=self.suite.score(state_before, action, state_after),
                )
                trajectory.steps.append(record)
                if self.on_step:
                    self.on_step(record)
        return trajectory
