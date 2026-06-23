"""Path integrals over composite-world learning trajectories.

To solve a new (possibly out-of-distribution) problem, an agent must traverse a
space of worlds/skills: master some primitives, compose them, and reach a target
capability. There are combinatorially many trajectories (the orderings of what to
learn), or literally infinitely many if revisiting is allowed. The PATH INTEGRAL
sums over all of them, weighting each trajectory by ``exp(-beta * action)`` where
the action is the total cost of the path; it is dominated by the least-action
path -- the most direct curriculum.

The sum is computed WITHOUT enumerating trajectories, as a semiring dynamic
program over capability-states (the sum-over-paths = semiring-closure identity
behind shortest paths, Viterbi, and partition functions). This reuses
``openworld.Semiring`` (the E46 many-worlds store): choosing the value semiring
chooses the question.

  TROPICAL (min, +)      -> least-action path: the optimal learning curriculum.
  LOG (logsumexp, +)     -> the full path integral: log-partition / free energy.
  COUNTING (+, *)        -> how many distinct trajectories reach the goal.
  forward * backward / Z -> each world's path-integral MARGINAL: how much of the
                            trajectory mass flows through learning it (what to
                            prioritize).

Additive module: new classes/functions only; nothing existing changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Tuple

from .manyworlds import COUNTING, Semiring


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


# least action (min cost path) and the full integral (log-sum-exp over paths)
TROPICAL = Semiring(math.inf, 0.0, min, lambda a, b: a + b, "tropical")
LOG = Semiring(-math.inf, 0.0, _logaddexp, lambda a, b: a + b, "log")


@dataclass
class Skill:
    """A world/skill the agent can master. Composing it is cheap once its
    prerequisites are mastered; learning it from scratch is expensive - this gap
    is what makes a curriculum (and compositional transfer) worth planning."""

    name: str
    prereqs: Tuple[str, ...] = ()
    compose_cost: float = 1.0
    scratch_cost: float = 10.0


@dataclass
class TrajectorySpace:
    """The space of learning trajectories from an initial capability set to a
    goal skill, over a library of skills."""

    skills: Dict[str, Skill]
    initial: FrozenSet[str]
    goal: str
    universe: FrozenSet[str] = field(default_factory=frozenset)

    def __init__(self, skills: List[Skill], initial=(), goal: str = ""):
        self.skills = {s.name: s for s in skills}
        self.initial = frozenset(initial)
        self.goal = goal
        self.universe = self._relevant()

    def _relevant(self) -> FrozenSet[str]:
        """Skills that can matter: the goal, its transitive prerequisites, and
        the initial set (others never reduce the action to the goal)."""
        seen, stack = set(), [self.goal]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(self.skills[n].prereqs)
        return frozenset(seen | set(self.initial))

    def cost(self, name: str, state: FrozenSet[str]) -> float:
        s = self.skills[name]
        return s.compose_cost if set(s.prereqs) <= state else s.scratch_cost

    def addable(self, state: FrozenSet[str]) -> List[str]:
        return [n for n in self.universe if n not in state]

    def is_goal(self, state: FrozenSet[str]) -> bool:
        return self.goal in state

    def reachable(self) -> List[FrozenSet[str]]:
        """All capability-states reachable from `initial`; goal-states are
        absorbing (a trajectory ends when the goal is first mastered)."""
        out, frontier = {self.initial}, [self.initial]
        while frontier:
            st = frontier.pop()
            if self.is_goal(st):
                continue
            for n in self.addable(st):
                nxt = st | {n}
                if nxt not in out:
                    out.add(nxt)
                    frontier.append(nxt)
        return sorted(out, key=len)

    # -- the path integral, as a semiring DP -------------------------------
    def _edge(self, sr: Semiring, c: float, beta: float):
        if sr.name == "tropical":
            return c
        if sr.name == "log":
            return -beta * c
        if sr.name == "counting":
            return 1
        raise ValueError(f"unsupported semiring {sr.name}")

    def forward(self, sr: Semiring, beta: float = 1.0):
        states = self.reachable()
        fwd = {st: (sr.one if st == self.initial else sr.zero) for st in states}
        # also track goal-terminal states' weights
        for st in states:                              # increasing size order
            if self.is_goal(st) or fwd[st] == sr.zero:
                continue
            for n in self.addable(st):
                nxt = st | {n}
                w = sr.times(fwd[st], self._edge(sr, self.cost(n, st), beta))
                if nxt in fwd:
                    fwd[nxt] = sr.plus(fwd[nxt], w)
        return fwd

    def backward(self, sr: Semiring, beta: float = 1.0):
        states = self.reachable()
        bwd = {st: (sr.one if self.is_goal(st) else sr.zero) for st in states}
        for st in sorted(states, key=len, reverse=True):
            if self.is_goal(st):
                continue
            for n in self.addable(st):
                nxt = st | {n}
                if nxt in bwd:
                    bwd[st] = sr.plus(
                        bwd[st], sr.times(self._edge(sr, self.cost(n, st), beta), bwd[nxt]))
        return bwd

    def partition(self, sr: Semiring, beta: float = 1.0):
        """Combine all goal-terminal states: min cost (tropical), log Z (log),
        or trajectory count (counting)."""
        fwd = self.forward(sr, beta)
        z = sr.zero
        for st, w in fwd.items():
            if self.is_goal(st):
                z = sr.plus(z, w)
        return z

    def least_action_path(self):
        """The optimal curriculum: the min-total-cost sequence of skills to learn
        from `initial` to the goal. Returns (steps, total_cost)."""
        fwd = self.forward(TROPICAL)
        goal_states = [st for st in fwd if self.is_goal(st) and fwd[st] < math.inf]
        if not goal_states:
            return [], math.inf
        end = min(goal_states, key=lambda st: fwd[st])
        total = fwd[end]
        # backtrack
        steps, st = [], end
        while st != self.initial:
            added = None
            for n in st:
                if n in self.initial:
                    continue
                prev = st - {n}
                if prev in fwd and abs(fwd[prev] + self.cost(n, prev) - fwd[st]) < 1e-9:
                    added, st = n, prev
                    break
            if added is None:
                break
            steps.append(added)
        steps.reverse()
        return steps, total

    def goal_cost_from_scratch(self) -> float:
        """Cost to learn the goal with NO prerequisites mastered (the no-transfer
        baseline)."""
        return self.skills[self.goal].scratch_cost

    def node_marginals(self, beta: float = 1.0) -> Dict[str, float]:
        """For each skill, the path-integral probability that a trajectory learns
        it on the way to the goal (forward * edge * backward / Z), in [0,1]."""
        fwd = self.forward(LOG, beta)
        bwd = self.backward(LOG, beta)
        logZ = self.partition(LOG, beta)
        marg: Dict[str, float] = {n: -math.inf for n in self.universe}
        for st, fw in fwd.items():
            if fw == -math.inf or self.is_goal(st):
                continue
            for n in self.addable(st):
                nxt = st | {n}
                if nxt in bwd:
                    contrib = fw + (-beta * self.cost(n, st)) + bwd[nxt]
                    marg[n] = _logaddexp(marg[n], contrib)
        return {n: (math.exp(v - logZ) if v != -math.inf else 0.0)
                for n, v in marg.items()}

    def count_trajectories(self) -> int:
        return int(self.partition(COUNTING))
