"""Objectives and dials: open, editable value specifications.

Instead of baking values into trained weights, objectives are declared scoring
functions over (state, action, next_state), each weighted by a Dial that can be
adjusted at any time — including mid-simulation — to steer behavior along the
Pareto frontier between competing objectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .state import Action, WorldState

ScoreFn = Callable[[WorldState, Action, WorldState], float]


@dataclass
class Dial:
    """A tunable scalar weight (e.g. a morality dial lambda in [0, 1])."""

    name: str
    value: float = 1.0
    minimum: float = 0.0
    maximum: float = 1.0

    def set(self, value: float) -> None:
        if not (self.minimum <= value <= self.maximum):
            raise ValueError(
                f"Dial {self.name!r} value {value} outside [{self.minimum}, {self.maximum}]"
            )
        self.value = value


@dataclass
class Objective:
    """A named scoring function over a single transition.

    weight may be a float (fixed) or a Dial (tunable at inference time).
    """

    name: str
    fn: ScoreFn
    weight: Any = 1.0  # float or Dial
    description: str = ""

    @property
    def effective_weight(self) -> float:
        return self.weight.value if isinstance(self.weight, Dial) else float(self.weight)

    def score(self, state: WorldState, action: Action, next_state: WorldState) -> float:
        return float(self.fn(state, action, next_state))


@dataclass
class ObjectiveSuite:
    """A set of objectives scored together; aggregate is the dial-weighted sum."""

    objectives: List[Objective] = field(default_factory=list)

    def dials(self) -> Dict[str, Dial]:
        found: Dict[str, Dial] = {}
        for objective in self.objectives:
            if isinstance(objective.weight, Dial):
                found[objective.weight.name] = objective.weight
        return found

    def score(
        self, state: WorldState, action: Action, next_state: WorldState
    ) -> Dict[str, float]:
        """Raw score per objective plus the weighted 'aggregate'."""
        raw = {o.name: o.score(state, action, next_state) for o in self.objectives}
        raw["aggregate"] = sum(
            o.effective_weight * raw[o.name] for o in self.objectives
        )
        return raw

    def get(self, name: str) -> Optional[Objective]:
        for objective in self.objectives:
            if objective.name == name:
                return objective
        return None
