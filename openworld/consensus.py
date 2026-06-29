"""ConsensusTransition -- a committee of world models, the "multiple worlds" primitive.

Several independently synthesized world models can model the same dynamics differently. This combines
them, verification-aware:

  * mode="select" (default): use the member with the highest verified fidelity that runs. This is the
    right default for discrete *code* world models -- per-element averaging of weak, correlated
    members empirically *hurts* (ARC-AGI-3 E95), whereas selecting the best-verified member never
    does worse than that member.
  * mode="vote": per-key majority over members' predicted next-states (useful when members make
    *independent* per-key errors).

Zero new dependencies; members are ordinary Transitions paired with a held-out fidelity. Makes
"average or choose from multiple worlds" a first-class, verification-aware part of the framework.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from .state import Action, WorldState
from .transition import Transition


class ConsensusTransition(Transition):
    def __init__(self, members: List[Tuple[Transition, float]], mode: str = "select") -> None:
        if not members:
            raise ValueError("ConsensusTransition needs at least one member")
        if mode not in ("select", "vote"):
            raise ValueError("mode must be 'select' or 'vote'")
        # highest verified fidelity first
        self.members = sorted(members, key=lambda m: -float(m[1]))
        self.mode = mode

    def step(self, state: WorldState, action: Action) -> WorldState:
        if self.mode == "select":
            for t, _ in self.members:
                try:
                    return t.step(state, action)
                except Exception:  # noqa: BLE001 -- fall through to the next-best member
                    continue
            return WorldState(dict(state))
        # vote: per-key majority over members' predicted next-states
        outs = []
        for t, _ in self.members:
            try:
                outs.append(dict(t.step(state, action)))
            except Exception:  # noqa: BLE001
                pass
        if not outs:
            return WorldState(dict(state))
        keys = set().union(*(o.keys() for o in outs))
        result = {}
        for k in keys:
            present = [o[k] for o in outs if k in o]
            counts = Counter(repr(v) for v in present)
            winner = counts.most_common(1)[0][0]
            result[k] = next(v for v in present if repr(v) == winner)  # the actual value, not its repr
        return WorldState(result)
