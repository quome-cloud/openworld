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

import base64
import hashlib
import json as _json
import re as _re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .llm import BaseLLM
from .parsing import extract_json

MODALITIES = ("text", "audio", "image", "video_frame", "video_segment")


class EmissionError(ValueError):
    """Raised when a world's emitted output violates its declared output contract."""


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


class CodePerceptor(Perceptor):
    """A perceptor whose extraction is verified Python code, not an LLM.

    The code defines `def perceive(data) -> dict`, run in the same restricted
    sandbox as transition code (stdlib only, no imports/IO). Because it is plain
    serializable code, a CodePerceptor round-trips through a spec and runs on a
    server with no LLM at inference time -- the deterministic, deployable way to
    turn structured input (e.g. `key: value` text) into a symbolic state delta.
    Its output is still contract-checked by the PerceptionGate.
    """

    def __init__(self, code: str, produces: List[str],
                 schema: Optional[Dict[str, Any]] = None, modality: str = "text",
                 func_name: str = "perceive"):
        self.code = code
        self.produces = list(produces)
        self.schema = dict(schema or {})
        self.modality = modality
        self.func_name = func_name

    def perceive(self, observation: Observation) -> Dict[str, Any]:
        from .sandbox import load_transition_code
        data = observation.data if isinstance(observation, Observation) else observation
        func = load_transition_code(self.code, self.func_name)
        result = func(data)
        return dict(result) if isinstance(result, dict) else {}


class JSONPerceptor(Perceptor):
    """Map a JSON / dict payload (API response, webhook, form) to state fields by
    dotted key paths. Declarative and deterministic -- no code execution, so it is
    safe to reconstruct from a spec without opting into code. Output is gated."""

    def __init__(self, paths: Dict[str, str], schema: Optional[Dict[str, Any]] = None,
                 modality: str = "text"):
        self.paths = dict(paths)               # produced field -> "a.b.c" path
        self.produces = list(paths)
        self.schema = dict(schema or {})
        self.modality = modality

    def perceive(self, observation: Observation) -> Dict[str, Any]:
        data = observation.data if isinstance(observation, Observation) else observation
        if isinstance(data, (str, bytes)):
            try:
                data = _json.loads(data)
            except Exception:
                return {}
        out: Dict[str, Any] = {}
        for field, path in self.paths.items():
            cur = data
            ok = True
            for part in str(path).split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if ok:
                out[field] = cur
        return out


class RegexPerceptor(Perceptor):
    """Extract named regex groups into state fields. Declarative + deterministic
    (no code execution); optional per-field casts (e.g. int). Output is gated."""

    def __init__(self, pattern: str, schema: Optional[Dict[str, Any]] = None,
                 modality: str = "text", casts: Optional[Dict[str, Callable]] = None):
        self.pattern = pattern
        self._re = _re.compile(pattern)
        self.produces = list(self._re.groupindex)
        self.schema = dict(schema or {})
        self.modality = modality
        self.casts = dict(casts or {})

    def perceive(self, observation: Observation) -> Dict[str, Any]:
        data = observation.data if isinstance(observation, Observation) else observation
        m = self._re.search(str(data))
        if not m:
            return {}
        out = {k: v for k, v in m.groupdict().items() if v is not None}
        for k, cast in self.casts.items():
            if k in out:
                try:
                    out[k] = cast(out[k])
                except Exception:
                    pass
        return out


_EMIT_SYSTEM = ("You write the world's output. Use only the provided fields; "
                "reply with the text and nothing else.")


class LLMEmitter:
    """The world -> text boundary: ask an LLM to write output from state fields.

    Symmetric to TextPerceptor (text -> state). `template` is filled from the named
    `reads` state fields and used as the prompt; the LLM's reply is the emitted
    text. Deterministic with MockLLM. The output boundary that lets a world
    *answer* in natural language (perceive -> world -> emit)."""

    modality = "text"

    def __init__(self, llm: "BaseLLM", template: str, reads: List[str],
                 system: Optional[str] = None, schema: Optional[Dict[str, Any]] = None):
        self.llm = llm
        self.template = template
        self.reads = list(reads)
        self.system = system or _EMIT_SYSTEM
        self.schema = dict(schema or {})

    def emit(self, state: Dict[str, Any]) -> str:
        fields = {k: state.get(k) for k in self.reads}
        try:
            prompt = self.template.format(**fields)
        except Exception:
            prompt = self.template + "\n" + ", ".join(f"{k}={v}" for k, v in fields.items())
        return self.llm.ask(prompt, system=self.system)


class CodeEmitter:
    """Deterministic, verified-code output: `def emit(state) -> str|dict`, run in the
    sandbox. The mirror of CodePerceptor on the output side -- serializable, and runs
    server-side with no LLM. Declare a `schema` to have an EmissionGate contract-check
    structured output before it leaves the world."""

    modality = "text"

    def __init__(self, code: str, reads: Optional[List[str]] = None,
                 func_name: str = "emit", schema: Optional[Dict[str, Any]] = None):
        self.code = code
        self.reads = list(reads or [])
        self.func_name = func_name
        self.schema = dict(schema or {})

    def emit(self, state: Dict[str, Any]):
        from .sandbox import load_transition_code
        func = load_transition_code(self.code, self.func_name)
        ctx = {k: state.get(k) for k in self.reads} if self.reads else dict(state)
        return func(ctx)


class ToolRegistry:
    """A registry of tools the world may invoke to act on the outside. Handlers must
    be registered explicitly (no arbitrary execution); each may declare an args
    schema that is contract-checked before the handler runs."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, handler: Callable[[Dict[str, Any]], Any],
                 schema: Optional[Dict[str, Any]] = None) -> None:
        self._tools[name] = {"handler": handler, "schema": dict(schema or {})}

    def names(self) -> List[str]:
        return list(self._tools)

    def call(self, name: str, args: Dict[str, Any]) -> Any:
        if name not in self._tools:
            raise EmissionError(f"unknown tool {name!r}; registered: {self.names()}")
        spec = self._tools[name]["schema"]
        EmissionGate().check_schema(spec, dict(args))
        return self._tools[name]["handler"](dict(args))


class ToolEmitter:
    """The act-on-the-world boundary: verified code chooses a tool call
    `{name, args}` from state; if a ToolRegistry is given, the call is executed and
    its result attached. This is how a world (or brain) takes real actions."""

    modality = "action"

    def __init__(self, code: str, registry: Optional[ToolRegistry] = None,
                 reads: Optional[List[str]] = None, func_name: str = "choose_tool"):
        self.code = code
        self.registry = registry
        self.reads = list(reads or [])
        self.func_name = func_name

    def emit(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from .sandbox import load_transition_code
        func = load_transition_code(self.code, self.func_name)
        ctx = {k: state.get(k) for k in self.reads} if self.reads else dict(state)
        call = func(ctx)
        if not isinstance(call, dict) or "name" not in call:
            call = {"name": "noop", "args": {}}
        call.setdefault("args", {})
        if self.registry is not None:
            call["result"] = self.registry.call(call["name"], call["args"])
        return call


class EmissionGate:
    """Contract-check an emitted output before it leaves the world -- the mirror of
    PerceptionGate on the way out. `check` reads the emitter's declared `schema`;
    `check_schema` validates an explicit schema (used for tool args)."""

    def check(self, emitter: Any, output: Any) -> Any:
        return self.check_schema(getattr(emitter, "schema", {}) or {}, output)

    def check_schema(self, schema: Dict[str, Any], output: Any) -> Any:
        if not schema or not isinstance(output, dict):
            return output
        for key, value in output.items():
            spec = schema.get(key)
            if spec is None:
                continue
            typ, bounds = (spec if isinstance(spec, tuple) else (spec, None))
            if isinstance(typ, type) and not isinstance(value, typ):
                raise EmissionError(
                    f"output field {key!r} expected {typ.__name__}, got "
                    f"{type(value).__name__}")
            if bounds is not None and not (bounds[0] <= value <= bounds[1]):
                raise EmissionError(
                    f"output field {key!r}={value} out of range "
                    f"[{bounds[0]}, {bounds[1]}]")
        return dict(output)


_EXTRACT_SYSTEM = (
    "You extract structured fields from input. Reply with ONLY a JSON object "
    "containing exactly the requested fields and nothing else.")


def _extract_fields(llm: BaseLLM, text: str, produces: List[str], system: str,
                    images: Optional[List[str]] = None) -> Dict[str, Any]:
    """Ask an LLM to extract `produces` from text (optionally with images) as
    JSON; keep only owned fields. Shared by the text/audio/vision perceptors."""
    prompt = (
        f"Extract these fields as JSON: {produces}.\n\n"
        f"Input:\n{text}\n\n"
        f"Return only the JSON object with those keys.")
    reply = llm.ask(prompt, system=system, **({"images": images} if images else {}))
    data = extract_json(reply) or {}
    return {k: data[k] for k in produces if k in data}


class TextPerceptor(Perceptor):
    """Map free text to symbolic fields by asking an LLM to extract them as JSON.

    Deterministic with `MockLLM`. Only fields in `produces` are kept; anything
    else the model emits is dropped here and would be caught by the gate anyway.
    """

    modality = "text"

    def __init__(self, llm: BaseLLM, produces: List[str],
                 schema: Optional[Dict[str, Any]] = None, system: Optional[str] = None):
        self.llm = llm
        self.produces = list(produces)
        self.schema = dict(schema or {})
        self.system = system or _EXTRACT_SYSTEM

    def perceive(self, observation: Observation) -> Dict[str, Any]:
        return _extract_fields(self.llm, str(observation.data), self.produces, self.system)


class TranscriptPerceptor(Perceptor):
    """Audio -> transcript -> symbolic fields. (Phase 2.)

    Offline-friendly: an audio `Observation` may carry a transcript `str`
    directly, or `bytes`/path plus an injected `transcribe` callable (e.g. a
    Whisper-class ASR model). Field extraction reuses the text path. Ollama has
    no native audio, so transcription is the (optional) front of this pipe.
    """

    modality = "audio"

    def __init__(self, llm: BaseLLM, produces: List[str],
                 schema: Optional[Dict[str, Any]] = None, system: Optional[str] = None,
                 transcribe: Optional[Callable[[Any], str]] = None):
        self.llm = llm
        self.produces = list(produces)
        self.schema = dict(schema or {})
        self.system = system or _EXTRACT_SYSTEM
        self.transcribe = transcribe

    def perceive(self, observation: Observation) -> Dict[str, Any]:
        data = observation.data
        if isinstance(data, str):
            transcript = data
        elif self.transcribe is not None:
            transcript = self.transcribe(data)
        else:
            raise PerceptionError(
                "audio observation needs a transcript str or a transcribe= callable")
        return _extract_fields(self.llm, transcript, self.produces, self.system)


def image_to_b64(data: Any) -> str:
    """Encode image bytes, or the contents of a path, as base64 (for Ollama)."""
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
    else:
        with open(data, "rb") as fh:
            raw = fh.read()
    return base64.b64encode(raw).decode("ascii")


def sample_frames(frames: List[Any], k: int) -> List[Any]:
    """Pick `k` roughly evenly-spaced frames from a list (k>=1)."""
    if k <= 0:
        raise PerceptionError("k must be >= 1")
    if len(frames) <= k:
        return list(frames)
    step = (len(frames) - 1) / (k - 1) if k > 1 else 0
    return [frames[round(i * step)] for i in range(k)]


class VisionPerceptor(Perceptor):
    """Image / video frame -> symbolic fields via a vision LLM. (Phase 3.)

    The `Observation.data` is image bytes or a path (single frame). The image
    is base64-encoded and passed to the LLM via the additive `images=` channel;
    works with any `BaseLLM` (a vision-capable `OllamaLLM` live, `MockLLM` in
    tests). For a `video_segment`, sample frames with `sample_frames` and merge
    the per-frame deltas in the caller (last-seen wins), or perceive frames
    individually through `World.observe`.
    """

    modality = "image"

    def __init__(self, llm: BaseLLM, produces: List[str],
                 schema: Optional[Dict[str, Any]] = None, system: Optional[str] = None,
                 prompt_hint: str = "the image"):
        self.llm = llm
        self.produces = list(produces)
        self.schema = dict(schema or {})
        self.system = system or _EXTRACT_SYSTEM
        self.prompt_hint = prompt_hint

    def perceive(self, observation: Observation) -> Dict[str, Any]:
        return _extract_fields(self.llm, self.prompt_hint, self.produces,
                               self.system, images=[image_to_b64(observation.data)])
