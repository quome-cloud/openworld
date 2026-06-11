"""Ethical structure beyond the weighted sum.

The Dial/Objective system aggregates values as a weighted sum - structurally,
act consequentialism with a linear social welfare function. This module adds
the structures that a weighted sum cannot express:

- Aggregators: `maximin` (Rawls's difference principle - judge a transition
  by its worst-off objective) and `lexicographic` (Berlin-compatible priority
  orderings - a higher-priority value cannot be traded for any amount of a
  lower one).
- `Constraint`: a deontological side constraint (Nozick) - a predicate that
  marks an action IMPERMISSIBLE in a state, enforced as a veto on the action
  set rather than a penalty in the score (a penalty is just more
  utilitarianism).
- `constrained`: wraps any policy so vetoed actions are never taken.
- `MoralParliament`: decision-making under moral uncertainty (MacAskill) -
  delegates for different ethical theories rank the available actions and
  vote; the parliament picks the consensus action, hedging across theories
  instead of maximizing within one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .state import Action, WorldState

ScoreFn = Callable[[WorldState, Action, WorldState], float]
ConstraintFn = Callable[[WorldState, Action], bool]


# ---------------------------------------------------------------------------
# Aggregators over a set of named scores
# ---------------------------------------------------------------------------

def weighted_sum(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """The classical utilitarian aggregate (what Objective/Dial already do)."""
    return sum(weights.get(name, 1.0) * value for name, value in scores.items())


def maximin(scores: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """Rawls's difference principle: a transition is only as good as its
    worst-off (optionally weighted) component."""
    weights = weights or {}
    return min(weights.get(name, 1.0) * value for name, value in scores.items())


def lexicographic(scores: Dict[str, float], priority: Sequence[str]) -> tuple:
    """Lexical priority ordering: returns a tuple comparable element-wise, so
    a gain on a lower-priority value can never outweigh ANY loss on a higher
    one. Compare outcomes with normal tuple comparison."""
    return tuple(scores.get(name, 0.0) for name in priority)


# ---------------------------------------------------------------------------
# Deontological side constraints
# ---------------------------------------------------------------------------

@dataclass
class Constraint:
    """An impermissibility predicate: forbidden(state, action) -> True means
    the action may not be taken in that state, whatever the consequences."""

    name: str
    forbidden: ConstraintFn
    description: str = ""

    def permits(self, state: WorldState, action: Action) -> bool:
        return not self.forbidden(state, action)


def permitted_actions(
    state: WorldState,
    actions: Sequence[str],
    constraints: Sequence[Constraint],
    agent: Optional[str] = None,
) -> List[str]:
    """Filter an action set down to those no constraint forbids."""
    out = []
    for name in actions:
        action = Action(name, agent=agent)
        if all(c.permits(state, action) for c in constraints):
            out.append(name)
    return out


def constrained(policy: Callable, constraints: Sequence[Constraint],
                fallback: str = "noop") -> Callable:
    """Wrap a policy so it can never select a forbidden action.

    The wrapped policy sees only the permitted action set; if its choice is
    nonetheless forbidden (or nothing is permitted), it degrades to the
    fallback action.
    """

    def wrapped(state, actions):
        allowed = permitted_actions(state, actions, constraints)
        if not allowed:
            return Action(fallback)
        choice = policy(state, allowed)
        if choice.name not in allowed and choice.name != fallback:
            return Action(allowed[0])
        return choice

    return wrapped


# ---------------------------------------------------------------------------
# Moral parliament: decision-making under moral uncertainty
# ---------------------------------------------------------------------------

@dataclass
class Delegate:
    """One ethical theory's representative.

    rank(state, candidate_actions, simulate) returns the candidate actions
    ordered best-first under this theory. `simulate(state, action_name)`
    gives the theory a one-step lookahead through the world model.
    """

    name: str
    rank: Callable[[WorldState, List[str], Callable], List[str]]
    credence: float = 1.0  # the operator's degree of belief in this theory


@dataclass
class MoralParliament:
    """Borda-count voting over actions among theory delegates.

    Each delegate ranks the candidate actions; ranks convert to Borda scores
    weighted by credence; the parliament returns the action with the highest
    total. Ties break toward the first delegate's preference (declared
    order = precedence), keeping the procedure deterministic.
    """

    delegates: List[Delegate] = field(default_factory=list)

    def choose(self, state: WorldState, actions: List[str],
               simulate: Callable[[Any, str], Dict[str, Any]]) -> str:
        totals = {a: 0.0 for a in actions}
        first_pref: Dict[str, int] = {}
        for d_index, delegate in enumerate(self.delegates):
            ranking = delegate.rank(state, list(actions), simulate)
            n = len(ranking)
            for position, action_name in enumerate(ranking):
                totals[action_name] = totals.get(action_name, 0.0) + \
                    delegate.credence * (n - 1 - position)
                if d_index == 0 and action_name not in first_pref:
                    first_pref[action_name] = position
        best = max(totals.values())
        winners = [a for a in actions if totals[a] == best]
        if len(winners) == 1:
            return winners[0]
        return min(winners, key=lambda a: first_pref.get(a, len(actions)))
