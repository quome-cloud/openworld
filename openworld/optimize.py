"""Optimization: dial sweeps, Pareto frontiers, and best-setting selection.

The core workflow from the research: sweep a tunable weight (e.g. a morality
dial lambda) across episodes, record each objective's raw totals, and trace the
Pareto frontier between competing objectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .simulation import Simulation, Trajectory


@dataclass
class SweepPoint:
    dial_value: float
    mean_totals: Dict[str, float]
    trajectories: List[Trajectory] = field(default_factory=list)


@dataclass
class SweepResult:
    dial: str
    points: List[SweepPoint] = field(default_factory=list)

    def best(self, objective: str = "aggregate", maximize: bool = True) -> SweepPoint:
        key = lambda p: p.mean_totals.get(objective, float("-inf"))
        return max(self.points, key=key) if maximize else min(self.points, key=key)

    def pareto(
        self, objectives: Sequence[str], maximize: Optional[Sequence[bool]] = None
    ) -> List[SweepPoint]:
        """Non-dominated sweep points over the named raw objectives."""
        maximize = list(maximize or [True] * len(objectives))

        def signed(point: SweepPoint) -> Tuple[float, ...]:
            return tuple(
                point.mean_totals.get(name, 0.0) * (1 if up else -1)
                for name, up in zip(objectives, maximize)
            )

        signed_points = [signed(point) for point in self.points]
        frontier = []
        for point, p in zip(self.points, signed_points):
            dominated = any(
                all(q[i] >= p[i] for i in range(len(p))) and q != p
                for q in signed_points
            )
            if not dominated:
                frontier.append(point)
        return sorted(frontier, key=lambda point: point.dial_value)

    def table(self) -> str:
        """Plain-text summary of the sweep, one row per dial value."""
        if not self.points:
            return "(empty sweep)"
        names = sorted(self.points[0].mean_totals)
        header = f"{self.dial:>10} | " + " | ".join(f"{n:>14}" for n in names)
        rows = [header, "-" * len(header)]
        for point in self.points:
            cells = " | ".join(f"{point.mean_totals.get(n, 0.0):>14.4f}" for n in names)
            rows.append(f"{point.dial_value:>10.3f} | {cells}")
        return "\n".join(rows)


def sweep(
    simulation: Simulation,
    dial: str,
    values: Sequence[float],
    steps: int = 10,
    episodes: int = 1,
) -> SweepResult:
    """Run the simulation at each dial value and aggregate objective totals."""
    result = SweepResult(dial=dial)
    for value in values:
        simulation.set_dial(dial, value)
        trajectories = [simulation.run(steps=steps) for _ in range(episodes)]
        totals_per_episode = [t.totals() for t in trajectories]
        names = {name for totals in totals_per_episode for name in totals}
        mean_totals = {
            name: sum(t.get(name, 0.0) for t in totals_per_episode) / len(totals_per_episode)
            for name in names
        }
        result.points.append(
            SweepPoint(dial_value=value, mean_totals=mean_totals, trajectories=trajectories)
        )
    return result
