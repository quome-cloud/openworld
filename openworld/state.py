"""Symbolic world state and actions.

State in OpenWorld is symbolic and JSON-serializable (per the Code World Model
paradigm): a plain mapping of entity/attribute data rather than a neural latent.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class WorldState(dict):
    """A symbolic world state. Behaves like a dict, adds copy/diff/json helpers."""

    def copy(self) -> "WorldState":
        return WorldState(copy.deepcopy(dict(self)))

    def diff(self, other: "WorldState") -> Dict[str, Any]:
        """Return {key: (old, new)} for keys that differ between self and other."""
        changes: Dict[str, Any] = {}
        for key in set(self) | set(other):
            old, new = self.get(key), other.get(key)
            if old != new:
                changes[key] = (old, new)
        return changes

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self, indent=indent, sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> "WorldState":
        return cls(json.loads(text))


@dataclass
class Action:
    """An action taken in the world.

    name:   action identifier, must be one of the world's declared actions
    params: free-form parameters interpreted by the transition engine
    agent:  name of the acting agent (optional for single-actor worlds)
    """

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "params": dict(self.params), "agent": self.agent}

    @classmethod
    def noop(cls, agent: Optional[str] = None) -> "Action":
        return cls(name="noop", params={}, agent=agent)
