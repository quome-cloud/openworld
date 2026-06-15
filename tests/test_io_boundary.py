"""Tests for the completed I/O boundary: perceptors, emitters, gate, memory."""

import pytest

from openworld import (CodeEmitter, CodeTransition, EmissionError, EmissionGate,
                       JSONPerceptor, MemoryStore, PerceptionError, PerceptionGate,
                       RegexPerceptor, ToolEmitter, ToolRegistry, World, from_spec,
                       to_spec, validate_spec)
from openworld.perceive import Observation


def test_json_perceptor_paths_and_gate():
    jp = JSONPerceptor(paths={"pri": "ticket.priority", "u": "ticket.user"},
                       schema={"pri": (int, (0, 9))})
    d = jp.perceive(Observation(modality="text", data={"ticket": {"priority": 7, "user": "a"}}))
    assert d == {"pri": 7, "u": "a"}
    assert JSONPerceptor(paths={"pri": "ticket.priority"},
                         schema={"pri": (int, (0, 9))}).perceive(
        Observation(modality="text", data='{"ticket":{"priority":3}}')) == {"pri": 3}  # JSON string
    with pytest.raises(PerceptionError):                     # gate rejects out-of-range
        bad = JSONPerceptor(paths={"pri": "p"}, schema={"pri": (int, (0, 5))})
        PerceptionGate().check(bad, bad.perceive(Observation(modality="text", data={"p": 99})))


def test_regex_perceptor_casts():
    rp = RegexPerceptor(r"load:\s*(?P<load>\d+)", casts={"load": int})
    assert rp.perceive(Observation(modality="text", data="cpu ok, load: 42")) == {"load": 42}
    assert rp.perceive(Observation(modality="text", data="nothing")) == {}


def test_code_emitter_and_emission_gate():
    ce = CodeEmitter(code='def emit(s):\n return {"score": s["a"] + s["b"]}',
                     reads=["a", "b"], schema={"score": (int, (0, 100))})
    assert ce.emit({"a": 3, "b": 4}) == {"score": 7}
    assert EmissionGate().check(ce, ce.emit({"a": 3, "b": 4})) == {"score": 7}
    bad = CodeEmitter(code='def emit(s):\n return {"score": 999}',
                      schema={"score": (int, (0, 100))})
    with pytest.raises(EmissionError):
        EmissionGate().check(bad, bad.emit({}))


def test_tool_emitter_executes_via_registry():
    log = []
    reg = ToolRegistry()
    reg.register("ping", lambda a: log.append(a["host"]) or "pong", schema={"host": str})
    te = ToolEmitter(code='def choose_tool(s):\n return {"name": "ping", "args": {"host": s["h"]}}',
                     registry=reg, reads=["h"])
    out = te.emit({"h": "example.com"})
    assert out["name"] == "ping" and out["result"] == "pong" and log == ["example.com"]
    with pytest.raises(EmissionError):                       # unknown tool rejected
        ToolEmitter(code='def choose_tool(s):\n return {"name": "nope", "args": {}}',
                    registry=reg).emit({})


def test_memory_semantic_beats_exact():
    m = MemoryStore()
    for cue, val in [("the capital of France", "Paris"),
                     ("the boiling point of water", "100C"),
                     ("the speed of light", "3e8")]:
        m.add(cue, val)
    q = "what is the capital city of France"
    assert m.recall(q, k=1)[0][1] == "Paris"                 # semantic finds it
    assert m.exact(q) is None                                # exact-key cannot


def test_world_with_new_io_round_trips():
    w = World(name="io", description="io", initial_state={"priority": 0, "out": 0},
              actions=["go"], transition=CodeTransition("def transition(s,a):\n return dict(s)"))
    w.perceptors = [JSONPerceptor(paths={"priority": "p"}, schema={"priority": (int, (0, 9))}),
                    RegexPerceptor(r"n:(?P<n>\d+)", casts={"n": int})]
    w.emit = [CodeEmitter(code='def emit(s):\n return {"out": s.get("priority",0)}',
                          reads=["priority"], schema={"out": (int, (0, 9))})]
    spec = to_spec(w)
    assert validate_spec(spec) == []
    assert [p["kind"] for p in spec["perception"]] == ["JSONPerceptor", "RegexPerceptor"]
    assert spec["emit"][0]["kind"] == "code"
    w2 = from_spec(spec, allow_code=True)
    kinds = [type(p).__name__ for p in w2.perceptors]
    assert "JSONPerceptor" in kinds and "RegexPerceptor" in kinds   # reconstructed, runnable
    jp = next(p for p in w2.perceptors if type(p).__name__ == "JSONPerceptor")
    assert jp.perceive(Observation(modality="text", data={"p": 4})) == {"priority": 4}
