"""Abstract interpretation: sound bounds on world outcomes (verified envelopes).

The framework's verified transitions give exact answers on exact inputs. When the
inputs are uncertain, abstract interpretation runs the SAME transition over an
abstract domain that SOUNDLY over-approximates the set of possible states, so the
output is a provable envelope: "the outcome is guaranteed to lie in [lo, hi]" -
not a point, not a sampled guess.

Two abstract domains:
  Interval - [lo, hi]; simple and sound but loses correlations (the 'wrapping'
             effect: it forgets that the same uncertain parameter recurs, so
             bounds blow up over iterations).
  Affine   - x = c + sum_i x_i * eps_i (eps_i in [-1,1]); tracks linear
             correlations between quantities, so it stays tight where intervals
             explode, while remaining sound.

Soundness (the point): the true reachable set is always contained in the abstract
result - unlike Monte Carlo, which only samples and silently under-covers the
worst case. Additive, numpy-free.
"""

from __future__ import annotations

from typing import Dict


class Interval:
    def __init__(self, lo: float, hi: float):
        self.lo, self.hi = (lo, hi) if lo <= hi else (hi, lo)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def contains(self, x: float, tol: float = 1e-9) -> bool:
        return self.lo - tol <= x <= self.hi + tol

    def _iv(self, o):
        return o if isinstance(o, Interval) else Interval(o, o)

    def __add__(self, o):
        o = self._iv(o)
        return Interval(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __mul__(self, o):
        o = self._iv(o)
        ps = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        return Interval(min(ps), max(ps))

    __rmul__ = __mul__

    def __repr__(self):
        return f"[{self.lo:.4g}, {self.hi:.4g}]"


class Affine:
    """Affine form c + sum coef_i * eps_i, eps_i in [-1, 1]."""
    _counter = [0]

    def __init__(self, center: float, dev: Dict[int, float] = None):
        self.c = float(center)
        self.dev = dict(dev or {})

    @classmethod
    def from_interval(cls, lo: float, hi: float) -> "Affine":
        cls._counter[0] += 1
        return cls((lo + hi) / 2.0, {cls._counter[0]: (hi - lo) / 2.0})

    @property
    def radius(self) -> float:
        return sum(abs(v) for v in self.dev.values())

    def to_interval(self) -> Interval:
        r = self.radius
        return Interval(self.c - r, self.c + r)

    def __add__(self, o):
        if not isinstance(o, Affine):
            return Affine(self.c + o, self.dev)
        dev = dict(self.dev)
        for k, v in o.dev.items():
            dev[k] = dev.get(k, 0.0) + v
        return Affine(self.c + o.c, dev)

    __radd__ = __add__

    def __mul__(self, o):
        if not isinstance(o, Affine):
            return Affine(self.c * o, {k: v * o for k, v in self.dev.items()})
        # linear part keeps correlations; nonlinear cross-term -> a fresh symbol
        dev = {}
        for k in set(self.dev) | set(o.dev):
            dev[k] = self.c * o.dev.get(k, 0.0) + o.c * self.dev.get(k, 0.0)
        Affine._counter[0] += 1
        dev[Affine._counter[0]] = self.radius * o.radius        # sound bound on cross term
        return Affine(self.c * o.c, dev)

    __rmul__ = __mul__

    def __repr__(self):
        return f"Affine(c={self.c:.4g}, r={self.radius:.4g})"
