"""The perception boundary: map raw observations to symbolic state.

Multimodal inputs (text, audio, image, video) enter OpenWorld ONLY here, and
ONLY as a resolution to symbolic state that happens BEFORE any transition
runs. The verified symbolic-state + code-dynamics core is untouched: a
Perceptor is an untrusted sensor whose output is contract-checked by a
PerceptionGate before it is allowed to update the world.

This module is purely additive. A world that never calls `observe()` behaves
exactly as before; perception adds no required dependencies (the text and mock
perceptors need only the existing LLM client and the stdlib).

Perception is the framework's one acknowledged fallible, non-deterministic
layer: an LLM-backed perceptor may vary run to run. The symbolic state it
commits, and every transition over it, remain deterministic and replayable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .llm import BaseLLM
from .parsing import extract_json

MODALITIES = ("text", "audio", "image", "video_frame", "video_segment")


class PerceptionError(ValueError):
    """A perceptor's output violated the boundary contract."""


@dataclass
class Observation:
    """One raw input at the perception boundary.

    `data` is a `str` for text and `bytes`/path-like for media. `t` is an
    optional timestamp. Content-hashed (`.sha256`) for trajectory provenance;
    the raw payload is never stored in symbolic state.
    """

    modality: str
    data: Any
    t: Optional[float] = None

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise PerceptionError(
                f"unknown modality {self.modality!r}; expected one of {MODALITIES}")

    @property
    def sha256(self) -> str:
        raw = self.data if isinstance(self.data, (bytes, bytearray)) else str(self.data).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


class Perceptor:
    """An untrusted sensor: `perceive(observation) -> partial symbolic delta`.

    Subclasses declare `modality`, `produces` (the state fields this perceptor
    is allowed to write), and `schema` mapping each produced field to either a
    type (`int`) or a `(type, (lo, hi))` pair for a closed numeric range.
    Override `perceive`.
    """

    modality: str = "text"
    produces: List[str] = []
    schema: Dict[str, Any] = {}

    def perceive(self, observation: Observation) -> Dict[str, Any]:
        raise NotImplementedError


class PerceptionGate:
    """Contract-checks a perceptor's output before it touches the world.

    Rejects fields the perceptor does not own, wrong types, and out-of-range
    values. This is where bad perception is caught - never silently fed to the
    dynamics.
    """

    def check(self, perceptor: Perceptor, delta: Dict[str, Any]) -> Dict[str, Any]:
        owned = set(perceptor.produces)
        for key, value in delta.items():
            if key not in owned:
                raise PerceptionError(
                    f"{type(perceptor).__name__} wrote field {key!r} it does not "
                    f"own (produces={perceptor.produces})")
            spec = perceptor.schema.get(key)
            if spec is None:
                continue
            typ, bounds = (spec if isinstance(spec, tuple) else (spec, None))
            if isinstance(typ, type) and not isinstance(value, typ):
                raise PerceptionError(
                    f"field {key!r} expected {typ.__name__}, got {type(value).__name__}")
            if bounds is not None:
                lo, hi = bounds
                if not (lo <= value <= hi):
                    raise PerceptionError(
                        f"field {key!r}={value} out of range [{lo}, {hi}]")
        return dict(delta)


class MockPerceptor(Perceptor):
    """Scripted perceptor for tests and offline tutorials (mirrors MockLLM).

    Returns the given `deltas` in order; the last repeats once exhausted.
    """

    def __init__(self, produces: List[str], deltas: List[Dict[str, Any]],
                 schema: Optional[Dict[str, Any]] = None, modality: str = "text"):
        self.produces = list(produces)
        self.schema = dict(schema or {})
        self.modality = modality
        self._deltas = list(deltas)
        self._i = 0

    def perceive(self, observation: Observation) -> Dict[str, Any]:
        if not self._deltas:
            return {}
        delta = self._deltas[min(self._i, len(self._deltas) - 1)]
        self._i += 1
        return dict(delta)


class TextPerceptor(Perceptor):
    """Map free text to symbolic fields by asking an LLM to extract them as JSON.

    Deterministic with `MockLLM`. Only fields in `produces` are kept; anything
    else the model emits is dropped here and would be caught by the gate anyway.
    """

    modality = "text"
    _SYSTEM = (
        "You extract structured fields from text. Reply with ONLY a JSON object "
        "containing exactly the requested fields and nothing else.")

    def __init__(self, llm: BaseLLM, produces: List[str],
                 schema: Optional[Dict[str, Any]] = None, system: Optional[str] = None):
        self.llm = llm
        self.produces = list(produces)
        self.schema = dict(schema or {})
        self.system = system or self._SYSTEM

    def perceive(self, observation: Observation) -> Dict[str, Any]:
        prompt = (
            f"Extract these fields as JSON: {self.produces}.\n\n"
            f"Text:\n{observation.data}\n\n"
            f"Return only the JSON object with those keys.")
        data = extract_json(self.llm.ask(prompt, system=self.system)) or {}
        return {k: data[k] for k in self.produces if k in data}
