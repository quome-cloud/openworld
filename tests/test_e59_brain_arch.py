"""Tests for E59: brain architecture search (LLM constant) + LLMEmitter."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "experiments"))
import e59_brain_arch as b                                  # noqa: E402

from openworld import (LLMEmitter, MockLLM, from_spec, to_spec,                # noqa: E402
                       validate_spec)


def test_retrieval_and_tree_help_with_constant_backbone():
    tasks = b.make_tasks(b.SEED)
    bare = b.eval_arch(tasks, "none", 1, False, seed=b.SEED)
    mem = b.eval_arch(tasks, "longterm", 1, False, seed=b.SEED)
    full = b.eval_arch(tasks, "longterm", 5, True, seed=b.SEED)
    assert mem > bare                       # retrieval helps
    assert full > mem                       # + best-of-N + verify helps more
    assert full - bare >= 0.3               # architecture gives a large lift


def test_best_of_n_needs_verify():
    tasks = b.make_tasks(b.SEED)
    # more drafts without a verifier doesn't help (it keeps draft[0])
    w1 = b.eval_arch(tasks, "longterm", 1, False, seed=b.SEED)
    w5_noverify = b.eval_arch(tasks, "longterm", 5, False, seed=b.SEED)
    w5_verify = b.eval_arch(tasks, "longterm", 5, True, seed=b.SEED)
    assert w5_noverify == w1                 # width alone (no verify) = no gain
    assert w5_verify > w5_noverify           # verify is what unlocks best-of-N


def test_search_recovers_richest_architecture():
    tasks = b.make_tasks(b.SEED)
    grid = [(m, w, v) for m in b.MEMS for w in b.WIDTHS for v in b.VERIFY]
    best = max(grid, key=lambda c: b.eval_arch(tasks, *c, seed=b.SEED))
    assert best == ("longterm", 5, True)


def test_brain_world_round_trips():
    brain = b.brain_world({"memory": "longterm", "width": 5, "verify": True})
    spec = to_spec(brain, card={"tags": ["brain"]})
    assert validate_spec(spec) == []
    w2 = from_spec(spec, allow_code=True)
    acts = ["tick", "conscious:think", "tick"]
    assert b._rollout(brain, acts) == b._rollout(w2, acts)
    assert "llm" == spec["emit"][0].get("kind")             # LLM emit channel recorded


def test_llm_emitter_with_mock():
    llm = MockLLM(responses=["Paris"])
    em = LLMEmitter(llm, reads=["question", "fact"],
                    template="Using {fact}, answer: {question}")
    out = em.emit({"question": "capital of France?", "fact": "France->Paris"})
    assert out == "Paris"
    assert "France->Paris" in llm.calls[0][-1]["content"]    # template was filled
