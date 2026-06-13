"""Multimodal inputs via the perception boundary - runnable, offline.

OpenWorld stays symbolic and verified at its core. Multimodal inputs
(text now; audio/video later) enter ONLY at a typed boundary: a Perceptor
resolves raw input to a partial symbolic state update, a PerceptionGate
contract-checks it, and only then does it touch the world - before any
verified dynamics run.

This demo: a tiny ICU bed whose state is symbolic (heart rate, a stability
flag). A free-text clinician note is *perceived* into the symbolic vitals,
the gate rejects an out-of-range reading, and then ordinary deterministic
dynamics step over the committed state.

    python tutorials/multimodal_perception.py            # offline (MockLLM)
"""

from openworld import (
    Action, MockLLM, Observation, PerceptionError, TextPerceptor, World,
)
from openworld.transition import FunctionTransition


# --- A symbolic world: dynamics are exact code over a JSON-serializable state.
def bed_dynamics(state, action):
    s = dict(state)
    if action["name"] == "monitor":
        s["minutes"] = s.get("minutes", 0) + 1
        s["stable"] = s.get("hr", 0) < 100      # a rule over the perceived vitals
    return s


def make_bed():
    return World(
        name="icu_bed",
        description="One ICU bed; vitals arrive as clinician notes, dynamics are code.",
        initial_state={"hr": 0, "stable": True, "minutes": 0},
        actions=["monitor"],
        transition=FunctionTransition(bed_dynamics),
    )


def main():
    bed = make_bed()

    # The perceptor: free text -> the symbolic field `hr`, bounded to a sane
    # physiological range. Offline we script the "vision/ASR/LLM" with MockLLM;
    # with a real model this is the same TextPerceptor, no other change.
    note_llm = MockLLM(['{"hr": 118}'])
    perceptor = TextPerceptor(note_llm, produces=["hr"], schema={"hr": (int, (20, 250))})

    note = "Pt tachycardic, HR up to 118, will keep monitoring."
    print(f"clinician note : {note!r}")
    bed.observe(Observation("text", note), perceptor)
    print(f"perceived state: hr={bed.state['hr']} (committed to symbolic state)")
    print(f"provenance     : {bed.perceptions[-1]}")

    # Verified dynamics run over the perceived state - exact code, not a guess.
    bed.step(Action("monitor"))
    print(f"after monitor  : stable={bed.state['stable']} minutes={bed.state['minutes']}")
    assert bed.state["hr"] == 118 and bed.state["stable"] is False

    # The gate is the trust layer: a perceptor that emits an out-of-range value
    # is rejected at the boundary; the bad reading never reaches the dynamics.
    bad_llm = MockLLM(['{"hr": 9000}'])
    bad = TextPerceptor(bad_llm, produces=["hr"], schema={"hr": (int, (20, 250))})
    try:
        bed.observe(Observation("text", "garbled telemetry hr 9000"), bad)
    except PerceptionError as exc:
        print(f"gate rejected  : {exc}")
    assert bed.state["hr"] == 118  # unchanged: rejected percept never committed

    print("\nThe core stayed symbolic and exact; perception is a separate, gated,")
    print("measured layer. Audio and video plug into the SAME boundary as new")
    print("Perceptor subclasses - additively, with no change to the world or its")
    print("dynamics.")


if __name__ == "__main__":
    main()
