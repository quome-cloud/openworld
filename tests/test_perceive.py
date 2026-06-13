"""Tests for the perception boundary (multimodal inputs -> symbolic state)."""

import pytest

from openworld import (
    Action, MockLLM, MockPerceptor, Observation, PerceptionError,
    PerceptionGate, TextPerceptor, TranscriptPerceptor, VisionPerceptor,
    World, image_to_b64, sample_frames,
)
from openworld.transition import FunctionTransition


def test_observation_validates_modality_and_hashes():
    obs = Observation("text", "patient is stable")
    assert obs.sha256 == Observation("text", "patient is stable").sha256
    assert obs.sha256 != Observation("text", "patient is critical").sha256
    assert Observation("image", b"\x00\x01").sha256  # bytes payload hashes too
    with pytest.raises(PerceptionError):
        Observation("smell", "nope")


def test_gate_accepts_owned_in_range_fields():
    p = MockPerceptor(produces=["hr"], deltas=[{"hr": 80}],
                      schema={"hr": (int, (0, 250))})
    assert PerceptionGate().check(p, {"hr": 80}) == {"hr": 80}


def test_gate_rejects_unowned_field():
    p = MockPerceptor(produces=["hr"], deltas=[{}])
    with pytest.raises(PerceptionError, match="does not"):
        PerceptionGate().check(p, {"bp": 120})


def test_gate_rejects_wrong_type_and_out_of_range():
    p = MockPerceptor(produces=["hr"], deltas=[{}], schema={"hr": (int, (0, 250))})
    with pytest.raises(PerceptionError, match="expected int"):
        PerceptionGate().check(p, {"hr": "fast"})
    with pytest.raises(PerceptionError, match="out of range"):
        PerceptionGate().check(p, {"hr": 999})


def test_text_perceptor_extracts_json_via_mock_llm():
    llm = MockLLM(['{"hr": 88, "note": "ignored"}'])
    p = TextPerceptor(llm, produces=["hr"], schema={"hr": int})
    assert p.perceive(Observation("text", "heart rate 88")) == {"hr": 88}  # owned only


def _counter_world():
    def step(state, action):
        s = dict(state)
        if action["name"] == "tick":
            s["count"] += s.get("hr", 0)
        return s
    return World(name="c", description="counter", initial_state={"count": 0, "hr": 0},
                 actions=["tick"], transition=FunctionTransition(step))


def test_observe_commits_delta_records_provenance_then_steps():
    world = _counter_world()
    p = MockPerceptor(produces=["hr"], deltas=[{"hr": 10}], schema={"hr": (int, (0, 250))})
    world.observe(Observation("text", "hr is 10"), p)
    assert world.state["hr"] == 10                      # perception committed to state
    assert world.perceptions[0]["delta"] == {"hr": 10}  # provenance recorded
    assert world.perceptions[0]["input_sha256"]
    world.step(Action("tick"))                          # dynamics run normally over it
    assert world.state["count"] == 10


def test_observe_supports_dict_of_perceptors_by_modality():
    world = _counter_world()
    perceptors = {"text": MockPerceptor(["hr"], [{"hr": 5}])}
    world.observe([Observation("text", "hr 5")], perceptors)
    assert world.state["hr"] == 5


def test_world_without_observe_is_unchanged():
    world = _counter_world()
    world.step(Action("tick"))
    assert world.state == {"count": 0, "hr": 0}
    assert "perceptions" not in world.__dict__   # nothing added unless observe() is used


def test_gate_rejection_surfaces_through_observe():
    world = _counter_world()
    bad = MockPerceptor(produces=["hr"], deltas=[{"hr": 999}], schema={"hr": (int, (0, 250))})
    with pytest.raises(PerceptionError):
        world.observe(Observation("text", "hr 999"), bad)
    assert world.state["hr"] == 0  # rejected percept never touched the state


# --- Phase 2: audio -------------------------------------------------------
def test_transcript_perceptor_accepts_transcript_string():
    llm = MockLLM(['{"hr": 92}'])
    p = TranscriptPerceptor(llm, produces=["hr"], schema={"hr": int})
    assert p.perceive(Observation("audio", "the patient's heart rate is 92")) == {"hr": 92}


def test_transcript_perceptor_uses_injected_transcriber_for_bytes():
    llm = MockLLM(['{"hr": 70}'])
    transcribe = lambda data: "resting, hr seventy"   # stand-in ASR
    p = TranscriptPerceptor(llm, produces=["hr"], transcribe=transcribe)
    assert p.perceive(Observation("audio", b"\x00\x01\x02")) == {"hr": 70}


def test_transcript_perceptor_needs_transcript_or_transcriber():
    p = TranscriptPerceptor(MockLLM(['{}']), produces=["hr"])
    with pytest.raises(PerceptionError, match="transcript"):
        p.perceive(Observation("audio", b"\x00\x01"))


# --- Phase 3: vision ------------------------------------------------------
def test_image_to_b64_handles_bytes_and_path(tmp_path):
    import base64
    assert image_to_b64(b"abc") == base64.b64encode(b"abc").decode()
    f = tmp_path / "frame.bin"
    f.write_bytes(b"xyz")
    assert image_to_b64(str(f)) == base64.b64encode(b"xyz").decode()


def test_sample_frames_picks_evenly_and_caps():
    assert sample_frames([0, 1, 2], 5) == [0, 1, 2]      # fewer than k -> all
    assert sample_frames(list(range(10)), 1) == [0]
    assert sample_frames(list(range(10)), 3) == [0, 4, 9]  # endpoints + middle (round-half-even)


def test_vision_perceptor_extracts_from_image_via_mock_llm():
    llm = MockLLM(['{"people": 2}'])
    p = VisionPerceptor(llm, produces=["people"], schema={"people": (int, (0, 100))})
    assert p.perceive(Observation("image", b"\x89PNGfake")) == {"people": 2}


def test_ollama_chat_attaches_images_additively(monkeypatch):
    # images must ride on the user message, not in the sampling options, and
    # absence must leave the payload unchanged.
    from openworld.llm import OllamaLLM
    captured = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"message": {"content": "ok"}}'

    def fake_urlopen(req, timeout=None):
        captured["payload"] = __import__("json").loads(req.data.decode())
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OllamaLLM(model="vis")
    llm.ask("describe", images=["B64DATA"])
    msgs = captured["payload"]["messages"]
    assert msgs[-1]["role"] == "user" and msgs[-1]["images"] == ["B64DATA"]
    assert "images" not in captured["payload"]["options"]   # not a sampling option
    captured.clear()
    llm.ask("plain")
    assert all("images" not in m for m in captured["payload"]["messages"])  # additive
