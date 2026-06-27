"""Hermetic unit tests for the multi-world dynamics rung (E120-E123): consensus vote, surprise detection,
the combined segmenter, the online causal monitor, and the replay-to-boundary resynthesis round-trip.

No ARC env: synthetic frames/tables/deltas only -> fast + deterministic. Run:
    ~/.arcv/bin/python -m pytest tests/test_e12x_multiworld.py -q
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
import openworld as O
import e121_surprise_regimes as E121
import e122_online_regimes as E122
import e123_online_resynth as E123


# ---- combine(): the segmenter (regression coverage for the step-0 drop bug) ----

def test_combine_always_preserves_step0():
    # a boundary within min_gap of 0 must NOT replace/drop the mandatory initial regime start
    assert E121.combine([0, 3], [0, 3])[0] == 0
    assert E121.combine([0, 2], [0, 2])[0] == 0
    assert E121.combine([0, 1], [0, 1]) == [0]            # near-zero boundary merges INTO regime 0


def test_combine_unions_level_and_surprise():
    b = E121.combine([0, 50], [0, 10, 30])
    assert b[0] == 0 and 10 in b and 30 in b and 50 in b


def test_combine_prefers_exact_levelup_on_collision():
    # surprise at 64, level-up at 66 (within min_gap) -> keep the exact level-up step
    b = E121.combine([0, 64], [0, 66])
    assert 66 in b and 64 not in b


def test_level_boundaries():
    assert E121.level_boundaries([0, 0, 1, 1, 2]) == [0, 2, 4]
    assert E121.level_boundaries([0, 0, 0]) == [0]


# ---- surprise detection ----

def test_detect_boundaries_finds_spikes():
    d = np.array([2, 3, 2, 500, 2, 3, 2, 2, 2, 2, 2, 2, 480, 2], dtype=float)   # spikes >min_gap apart
    b = E121.detect_boundaries(d)
    assert b[0] == 0 and 3 in b and 12 in b


def test_surprise_signals_delta_counts_changed_cells():
    f0 = np.zeros((64, 64), dtype=int)
    f1 = f0.copy(); f1[:1, :5] = 7                         # 5 cells changed
    f2 = f1.copy(); f2[10:30, 10:30] = 3                   # a big "reload"
    mask = np.zeros((64, 64), dtype=bool)
    _, _, _, delta, _ = E121.surprise_signals([f0, f1, f2], [[1], [1]], mask)
    assert delta[0] == 5 and delta[1] == 400


def test_score_recall_precision():
    levels = [0] * 10 + [1] * 10 + [2] * 5                 # level-ups at index 10 and 20
    sc = E121.score([0, 10, 20], levels)
    assert sc["recall"] == 1.0 and sc["precision"] == 1.0 and sc["level_ups"] == 2


# ---- E122 OnlineRegimeMonitor: causal, fires on a spike, not on noise ----

def test_online_monitor_fires_on_spike_causally():
    m = E122.OnlineRegimeMonitor()
    for _ in range(10):
        assert m.feed(5) is False                          # warmup of small in-regime deltas
    assert m.feed(3000) is True                            # a reload-sized spike -> regime change
    assert m.feed(5) is False                              # refractory: no immediate re-fire


def test_online_monitor_silent_on_noise():
    m = E122.OnlineRegimeMonitor()
    assert not any(m.feed(d) for d in [5, 6, 5, 7, 6, 5, 6, 5, 6, 7, 5, 6, 5, 6])


# ---- E120 ConsensusTransition vote ----

def _tbl_transition(tbl):
    def fn(state, action):
        nm = action.get("name") if isinstance(action, dict) else getattr(action, "name", action)
        return {"sig": tbl.get(state.get("sig"), {}).get(nm, state.get("sig"))}
    return O.FunctionTransition(fn)


def test_consensus_vote_majority_wins():
    A = _tbl_transition({"q0": {"s1": "qA"}})
    B = _tbl_transition({"q0": {"s1": "qB"}})
    C = _tbl_transition({"q0": {"s1": "qB"}})
    cons = O.ConsensusTransition([(A, 1.0), (B, 1.0), (C, 1.0)], mode="vote")
    out = cons.step({"sig": "q0"}, O.Action("s1"))
    assert out["sig"] == "qB"                              # 2 vs 1


# ---- E123 replay-to-boundary resynthesis: compose + round-trip ----

def test_phased_resynth_roundtrip_is_exact():
    # two regimes: q0-s1->q1-s1->q2 (regime 0), boundary at step 2, q2-s1->q3 (regime 1)
    regimes = [(0, {"q0": {"s1": "q1"}, "q1": {"s1": "q2"}}), (2, {"q2": {"s1": "q3"}})]
    w = E123.compose("x", regimes, "q0")
    fid = E123.roundtrip(w, [[1], [1], [1]], ["q0", "q1", "q2", "q3"])
    assert fid == 1.0


def test_phased_advances_irreversibly():
    regimes = [(0, {"q0": {"s1": "q1"}}), (1, {"q1": {"s1": "q2"}})]
    w = E123.compose("x", regimes, "q0")
    st = dict(w.initial_state)
    st = w.transition.step(st, O.Action("s1"))            # regime 0
    assert st["sig"] == "q1" and st["_phase"] == 0
    st = w.transition.step(st, O.Action("s1"))            # advanced to regime 1
    assert st["sig"] == "q2" and st["_phase"] == 1
