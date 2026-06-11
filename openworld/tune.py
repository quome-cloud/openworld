"""Automated tuning: search world, policy, and dial parameters for a goal.

An AutoML-style workflow for world models. Declare a parameter space over
anything that shapes an experiment — world design, agent policy thresholds,
moral dials — plus a goal (a score to maximize and optionally a success
predicate). The Tuner then:

1. search(n_trials)  — samples the space broadly, building and simulating a
                       fresh environment per trial (thousands of environments).
                       strategy="random" (default) or "tpe" (Optuna-backed
                       Bayesian optimization, for expensive worlds).
2. refine(n_trials)  — fine-tunes: local perturbation search around the best
                       configuration found so far, re-centering on every
                       improvement (hill climbing).

Every trial is a full, replayable simulation; the study keeps each trial's
parameters, score, success, and objective totals so the result is an auditable
table (or CSV via study.to_csv), not just a single winner.

Moral configurations are first-class: pass Dial objects directly in the space
and they are tuned across their declared [minimum, maximum] bounds, so the
search jointly optimizes world design, agent policy, AND the value weights
that govern behavior.

Set workers > 1 to evaluate trials concurrently (threads). This pays off for
LLM-backed worlds, where trials are network-bound; pure-Python symbolic
rollouts gain little. build() must construct fully independent simulations.
"""

from __future__ import annotations

import csv
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .objectives import Dial
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
    stage: str  # "search", "tpe", or "refine"
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

    def to_csv(self, path: Union[str, Path]) -> Path:
        """Write every trial (params, score, success, totals) to a CSV file."""
        path = Path(path)
        param_names = sorted({k for t in self.trials for k in t.params})
        total_names = sorted({k for t in self.trials for k in t.totals})
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["number", "stage", "score", "success"]
                + param_names
                + [f"total_{n}" for n in total_names]
            )
            for t in self.trials:
                writer.writerow(
                    [t.number, t.stage, t.score, "" if t.success is None else t.success]
                    + [t.params.get(n, "") for n in param_names]
                    + [t.totals.get(n, "") for n in total_names]
                )
        return path


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


class Tuner:
    """Searches a parameter space for the configuration that best meets a goal.

    build:   params -> Simulation. Construct the whole experiment from the
             sampled parameters: world (and its design), agents (and their
             policy thresholds), objectives, and dial settings.
    space:   {name: Param or Dial}. A Dial is tuned as a Uniform over its
             [minimum, maximum] bounds — moral configurations are searchable
             like any other parameter.
    score:   (trajectory, params) -> float to MAXIMIZE. Defaults to the
             trajectory's dial-weighted 'aggregate' total.
    success: optional (trajectory, params) -> bool predicate defining what
             'solving the task' means; tracked per trial as a solve rate.
    episodes: simulations per trial (use >1 for stochastic worlds).
    workers: trials evaluated concurrently (threads). >1 helps LLM-backed
             worlds; build() must produce independent simulations.
    """

    def __init__(
        self,
        build: BuildFn,
        space: Dict[str, Union[Param, Dial]],
        score: Optional[ScoreFn] = None,
        success: Optional[SuccessFn] = None,
        steps: int = 10,
        episodes: int = 1,
        seed: Optional[int] = None,
        goal: str = "",
        workers: int = 1,
    ):
        self.build = build
        self.space = {
            name: Uniform(p.minimum, p.maximum) if isinstance(p, Dial) else p
            for name, p in space.items()
        }
        self.score = score or (lambda traj, params: traj.totals().get("aggregate", 0.0))
        self.success = success
        self.steps = steps
        self.episodes = episodes
        self.workers = max(1, workers)
        self.rng = random.Random(seed)
        self.study = Study(goal=goal)

    # -- evaluation ---------------------------------------------------------

    def _run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build and simulate one configuration. No shared-state mutation, so
        it is safe to call from worker threads."""
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
        return {
            "score": sum(scores) / len(scores),
            "success": (sum(successes) / len(successes)) if successes else None,
            "totals": last.totals(),
            "final_state": dict(last.final_state),
        }

    def _record(self, params: Dict[str, Any], stage: str, result: Dict[str, Any]) -> Trial:
        trial = Trial(
            number=len(self.study.trials),
            stage=stage,
            params=dict(params),
            **result,
        )
        self.study.trials.append(trial)
        return trial

    def _evaluate(self, params: Dict[str, Any], stage: str) -> Trial:
        return self._record(params, stage, self._run(params))

    def _evaluate_batch(self, batch: List[Dict[str, Any]], stage: str) -> List[Trial]:
        """Evaluate a list of parameter dicts, in parallel when workers > 1.
        Trials are recorded in batch order, keeping studies reproducible."""
        if self.workers > 1 and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                results = list(pool.map(self._run, batch))
        else:
            results = [self._run(params) for params in batch]
        return [self._record(p, stage, r) for p, r in zip(batch, results)]

    # -- stage 1: broad search ----------------------------------------------

    def search(self, n_trials: int = 100, strategy: str = "random") -> Study:
        """Broad sampling across the whole space.

        strategy="random": independent uniform samples (parallelizes freely).
        strategy="tpe":    Optuna's Tree-structured Parzen Estimator — sample-
                           efficient Bayesian optimization for expensive
                           (e.g. live-LLM) worlds. Requires `pip install optuna`.
        """
        if strategy == "tpe":
            return self._search_optuna(n_trials)
        if strategy != "random":
            raise ValueError(f"Unknown strategy {strategy!r}; use 'random' or 'tpe'")
        batch = [
            {name: p.sample(self.rng) for name, p in self.space.items()}
            for _ in range(n_trials)
        ]
        self._evaluate_batch(batch, "search")
        return self.study

    def _search_optuna(self, n_trials: int) -> Study:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "strategy='tpe' requires Optuna: pip install optuna "
                "(or pip install 'openworld[optuna]')"
            ) from exc
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.rng.randint(0, 2**31 - 1))
        optuna_study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(otrial):
            params: Dict[str, Any] = {}
            for name, p in self.space.items():
                if isinstance(p, Uniform):
                    params[name] = otrial.suggest_float(name, p.low, p.high)
                elif isinstance(p, IntRange):
                    params[name] = otrial.suggest_int(name, p.low, p.high)
                elif isinstance(p, Choice):
                    params[name] = otrial.suggest_categorical(name, list(p.options))
                else:
                    raise TypeError(
                        f"Cannot map param {name!r} ({type(p).__name__}) to an "
                        "Optuna distribution; use Uniform, IntRange, or Choice."
                    )
            return self._evaluate(params, "tpe").score

        optuna_study.optimize(objective, n_trials=n_trials)
        return self.study

    # -- stage 2: local fine-tuning -----------------------------------------

    def refine(
        self,
        n_trials: int = 50,
        around: Optional[Dict[str, Any]] = None,
        scale: float = 0.15,
    ) -> Study:
        """Fine-tune locally around the best configuration.

        Perturbs each parameter within `scale` of its range and re-centers on
        every new best (hill climbing). With workers > 1, perturbations are
        evaluated in generations of `workers` and the center moves to the best
        of each generation. Call repeatedly with shrinking scale for finer
        passes.
        """
        center = dict(around or self.study.best.params)
        best_score = self.study.best.score if self.study.trials else float("-inf")
        remaining = n_trials
        while remaining > 0:
            batch_size = min(self.workers, remaining) if self.workers > 1 else 1
            batch = [
                {name: p.perturb(center[name], self.rng, scale) for name, p in self.space.items()}
                for _ in range(batch_size)
            ]
            for trial in self._evaluate_batch(batch, "refine"):
                if trial.score > best_score:
                    best_score = trial.score
                    center = dict(trial.params)
            remaining -= batch_size
        return self.study
