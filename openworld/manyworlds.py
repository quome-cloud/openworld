"""A database for many worlds: a factored, semiring-annotated world store.

A *world* is a full assignment to a set of parameters; the world space is the
Cartesian product of their domains, which grows astronomically as parameters and
ranges grow. Holding an explicit list of worlds (the version-space style of E43)
does not scale. This module stores the world set in FACTORED form and answers
queries over it without ever enumerating it.

The structure is a set of factors, one per *mechanism* of a world model's
transition. A mechanism computes one observable of the next state from the
current (observed) state and a small SCOPE of parameters. Because a transition
is observed in full (state, action, next_state), the likelihood factorizes over
observables, and each observable constrains only its mechanism's scope - so an
observation touches only small factors, never the global product.

Values live in a pluggable SEMIRING; swapping it changes the question with no
change to the update logic (the provenance-semiring idea):

    Boolean      -> version space: is an assignment still possible?
    Counting     -> exact number of globally consistent worlds.
    Probability  -> a posterior over parameters; marginals; expectations.

This module is additive: it introduces new classes only and changes nothing
existing.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Semirings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Semiring:
    """A commutative semiring (zero, one, plus, times) used to weight worlds."""

    zero: Any
    one: Any
    plus: Callable[[Any, Any], Any]
    times: Callable[[Any, Any], Any]
    name: str = ""


BOOLEAN = Semiring(False, True, lambda a, b: a or b, lambda a, b: a and b, "boolean")
COUNTING = Semiring(0, 1, lambda a, b: a + b, lambda a, b: a * b, "counting")
PROBABILITY = Semiring(0.0, 1.0, lambda a, b: a + b, lambda a, b: a * b, "probability")


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------


@dataclass
class Mechanism:
    """One observable of the next state, as a function of the current state and a
    small scope of parameters.

    `scope` is the tuple of parameter names this observable depends on.
    `observable` is the next-state key it predicts.
    `fn(state, action, params)` returns the predicted value of that observable
    (params is a dict over exactly the scope), or None if the mechanism does not
    fire for this action (then it imposes no constraint).
    """

    name: str
    observable: str
    scope: Tuple[str, ...]
    fn: Callable[[dict, dict, dict], Any]


# ---------------------------------------------------------------------------
# The factored store
# ---------------------------------------------------------------------------


class WorldStore:
    """A factored version space / posterior over a parameter world space.

    Parameters: {name: domain (sequence of values)}.
    Mechanisms decompose the transition into small-scope observables.

    Internally we keep one factor per DISTINCT scope: a dict mapping a
    scope-assignment (a tuple of values in scope order) to a semiring weight.
    Updating on an observation reweights only the factors of the mechanisms that
    fired. The global weight of a world is the product of its factors' weights;
    it is never materialized.
    """

    def __init__(self, params: Dict[str, Sequence[Any]],
                 mechanisms: Sequence[Mechanism], semiring: Semiring = BOOLEAN):
        self.params = {k: list(v) for k, v in params.items()}
        self.mechanisms = list(mechanisms)
        self.sr = semiring
        # group mechanisms by scope; one factor per scope
        self.scopes: List[Tuple[str, ...]] = []
        self._by_scope: Dict[Tuple[str, ...], List[Mechanism]] = {}
        for m in self.mechanisms:
            self._by_scope.setdefault(m.scope, []).append(m)
        self.scopes = list(self._by_scope)
        # every parameter must live in exactly one scope for the product to be a
        # clean partition (the factored-exact regime)
        covered = [p for sc in self.scopes for p in sc]
        assert len(covered) == len(set(covered)), \
            "parameters must not be shared across scopes (would couple factors)"
        self.free = [p for p in self.params if p not in set(covered)]
        self.factors: Dict[Tuple[str, ...], Dict[Tuple, Any]] = {}
        for sc in self.scopes:
            self.factors[sc] = {
                combo: self.sr.one
                for combo in product(*[self.params[p] for p in sc])
            }

    # -- updates ------------------------------------------------------------
    def observe(self, state: dict, action: dict, next_state: dict) -> None:
        """Reweight factors by consistency with one observed transition."""
        for sc, mechs in self._by_scope.items():
            factor = self.factors[sc]
            for combo in list(factor):
                if factor[combo] == self.sr.zero:
                    continue
                params = dict(zip(sc, combo))
                ok = True
                for m in mechs:
                    pred = m.fn(state, action, params)
                    if pred is not None and pred != next_state.get(m.observable):
                        ok = False
                        break
                if not ok:
                    factor[combo] = self.sr.zero

    # -- queries (no world enumeration) -------------------------------------
    def count(self) -> int:
        """Exact number of globally consistent worlds."""
        total = 1
        for sc in self.scopes:
            total *= sum(1 for v in self.factors[sc].values() if v != self.sr.zero)
        for p in self.free:                       # unconstrained parameters
            total *= len(self.params[p])
        return total

    def total_worlds(self) -> int:
        t = 1
        for d in self.params.values():
            t *= len(d)
        return t

    def is_possible(self, world: Dict[str, Any]) -> bool:
        """Is a fully-specified world still consistent?"""
        for sc in self.scopes:
            combo = tuple(world[p] for p in sc)
            if self.factors[sc].get(combo, self.sr.zero) == self.sr.zero:
                return False
        return True

    def marginal(self, param: str) -> Dict[Any, float]:
        """Normalized posterior marginal over one parameter (probability view).

        Uniform prior over surviving assignments; for scoped params we marginalize
        the other params in the same scope by counting survivors."""
        if param in self.free:
            d = self.params[param]
            return {v: 1.0 / len(d) for v in d}
        sc = next(s for s in self.scopes if param in s)
        i = sc.index(param)
        weight: Dict[Any, float] = {v: 0.0 for v in self.params[param]}
        for combo, w in self.factors[sc].items():
            if w != self.sr.zero:
                weight[combo[i]] += 1.0
        z = sum(weight.values()) or 1.0
        return {v: w / z for v, w in weight.items()}

    def predict(self, state: dict, action: dict) -> Dict[str, Dict[Any, float]]:
        """Distribution over each next-state observable, marginalizing only the
        relevant scope factor - cost independent of the global world count."""
        out: Dict[str, Dict[Any, float]] = {}
        for sc, mechs in self._by_scope.items():
            survivors = [dict(zip(sc, combo))
                         for combo, w in self.factors[sc].items()
                         if w != self.sr.zero]
            if not survivors:
                continue
            for m in mechs:
                dist: Dict[Any, float] = {}
                for params in survivors:
                    val = m.fn(state, action, params)
                    if val is None:
                        continue
                    dist[val] = dist.get(val, 0.0) + 1.0
                z = sum(dist.values())
                if z:
                    out[m.observable] = {k: v / z for k, v in dist.items()}
        return out

    def expected_next(self, state: dict, action: dict) -> Dict[str, float]:
        """Expected value of each numeric observable across consistent worlds."""
        out = {}
        for obs, dist in self.predict(state, action).items():
            out[obs] = sum(k * p for k, p in dist.items())
        return out
