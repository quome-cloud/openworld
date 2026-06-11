"""Automated tuning: search world, policy, and dial parameters for a goal.

An AutoML-style workflow for world models. Declare a parameter space over
anything that shapes an experiment — world design, agent policy thresholds,
moral dials — plus a goal (a score to maximize and optionally a success
predicate). The Tuner then:

1. search(n_trials)  — samples the space broadly, building and simulating a
                       fresh environment per trial (thousands of environments).
2. refine(n_trials)  — fine-tunes: local perturbation search around the best
                       configuration found so far, re-centering on every
                       improvement (hill climbing).

Every trial is a full, replayable simulation; the study keeps each trial's
parameters, score, success, and objective totals so the result is an auditable
table, not just a single winner.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .simulation import Simulation, Trajectory

BuildFn = Callable[[Dict[str, Any]], Simulation]
ScoreFn = Callable[[Trajectory, Dict[str, Any]], float]
SuccessFn = Callable[[Trajectory, Dict[str, Any]], bool]


class Param:
    """A tunable parameter: knows how to sample itself and perturb a value."""

    def sample(self, rng: random.Random) -> Any:
        raise NotImplementedError

    def perturb(self, value: Any, rng: random.Random, scale: float) -> Any:
        raise NotImplementedError


@dataclass
class Uniform(Param):
    low: float
    high: float

    def sample(self, rng):
        return rng.uniform(self.low, self.high)

    def perturb(self, value, rng, scale):
        span = (self.high - self.low) * scale
        return min(self.high, max(self.low, value + rng.uniform(-span, span)))


@dataclass
class IntRange(Param):
    """Integer parameter, bounds inclusive."""

    low: int
    high: int

    def sample(self, rng):
        return rng.randint(self.low, self.high)

    def perturb(self, value, rng, scale):
        span = max(1, round((self.high - self.low) * scale))
        return min(self.high, max(self.low, value + rng.randint(-span, span)))


@dataclass
class Choice(Param):
    options: Sequence[Any]

    def sample(self, rng):
        return rng.choice(list(self.options))

    def perturb(self, value, rng, scale):
        # Mostly keep the incumbent choice; occasionally jump.
        if rng.random() < max(scale, 0.05):
            return rng.choice(list(self.options))
        return value


@dataclass
class Trial:
    number: int
    stage: str  # "search" or "refine"
    params: Dict[str, Any]
    score: float
    success: Optional[float]  # mean success over episodes, None if no predicate
    totals: Dict[str, float]
    final_state: Dict[str, Any]

    @property
    def solved(self) -> bool:
        return self.success is not None and self.success >= 1.0


@dataclass
class Study:
    goal: str = ""
    trials: List[Trial] = field(default_factory=list)

    @property
    def best(self) -> Trial:
        if not self.trials:
            raise ValueError("Study has no trials yet")
        return max(self.trials, key=lambda t: t.score)

    def top(self, k: int = 5) -> List[Trial]:
        return sorted(self.trials, key=lambda t: t.score, reverse=True)[:k]

    def success_rate(self, stage: Optional[str] = None) -> Optional[float]:
        trials = [
            t for t in self.trials
            if t.success is not None and (stage is None or t.stage == stage)
        ]
        if not trials:
            return None
        return sum(t.success for t in trials) / len(trials)

    def table(self, k: int = 10) -> str:
        """Plain-text leaderboard of the top-k trials."""
        rows = self.top(k)
        if not rows:
            return "(empty study)"
        names = sorted(rows[0].params)
        header = (
            f"{'#':>5} | {'stage':>6} | {'score':>9} | {'success':>7} | "
            + " | ".join(f"{n:>12}" for n in names)
        )
        lines = [header, "-" * len(header)]
        for t in rows:
            success = "-" if t.success is None else f"{t.success:.2f}"
            cells = " | ".join(f"{_fmt(t.params[n]):>12}" for n in names)
            lines.append(
                f"{t.number:>5} | {t.stage:>6} | {t.score:>9.3f} | {success:>7} | {cells}"
            )
        return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


class Tuner:
    """Searches a parameter space for the configuration that best meets a goal.

    build:   params -> Simulation. Construct the whole experiment from the
             sampled parameters: world (and its design), agents (and their
             policy thresholds), objectives, and dial settings.
    score:   (trajectory, params) -> float to MAXIMIZE. Defaults to the
             trajectory's dial-weighted 'aggregate' total.
    success: optional (trajectory, params) -> bool predicate defining what
             'solving the task' means; tracked per trial as a solve rate.
    episodes: simulations per trial (use >1 for stochastic worlds).
    """

    def __init__(
        self,
        build: BuildFn,
        space: Dict[str, Param],
        score: Optional[ScoreFn] = None,
        success: Optional[SuccessFn] = None,
        steps: int = 10,
        episodes: int = 1,
        seed: Optional[int] = None,
        goal: str = "",
    ):
        self.build = build
        self.space = dict(space)
        self.score = score or (lambda traj, params: traj.totals().get("aggregate", 0.0))
        self.success = success
        self.steps = steps
        self.episodes = episodes
        self.rng = random.Random(seed)
        self.study = Study(goal=goal)

    def _evaluate(self, params: Dict[str, Any], stage: str) -> Trial:
        scores: List[float] = []
        successes: List[bool] = []
        last: Optional[Trajectory] = None
        for _ in range(self.episodes):
            simulation = self.build(dict(params))
            trajectory = simulation.run(steps=self.steps)
            scores.append(float(self.score(trajectory, params)))
            if self.success is not None:
                successes.append(bool(self.success(trajectory, params)))
            last = trajectory
        trial = Trial(
            number=len(self.study.trials),
            stage=stage,
            params=dict(params),
            score=sum(scores) / len(scores),
            success=(sum(successes) / len(successes)) if successes else None,
            totals=last.totals(),
            final_state=dict(last.final_state),
        )
        self.study.trials.append(trial)
        return trial

    def search(self, n_trials: int = 100) -> Study:
        """Stage 1: broad random sampling across the whole space."""
        for _ in range(n_trials):
            params = {name: p.sample(self.rng) for name, p in self.space.items()}
            self._evaluate(params, "search")
        return self.study

    def refine(
        self,
        n_trials: int = 50,
        around: Optional[Dict[str, Any]] = None,
        scale: float = 0.15,
    ) -> Study:
        """Stage 2: fine-tune locally around the best configuration.

        Perturbs each parameter within `scale` of its range and re-centers the
        search on every new best (hill climbing with random restarts of size
        `scale`). Call repeatedly with shrinking scale for finer passes.
        """
        center = dict(around or self.study.best.params)
        best_score = self.study.best.score if self.study.trials else float("-inf")
        for _ in range(n_trials):
            params = {
                name: p.perturb(center[name], self.rng, scale)
                for name, p in self.space.items()
            }
            trial = self._evaluate(params, "refine")
            if trial.score > best_score:
                best_score = trial.score
                center = dict(trial.params)
        return self.study
