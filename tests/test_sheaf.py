"""Tests for sheaf gluing / consistency."""

from openworld import glue, is_consistent, localize_fault, majority_glue
from openworld.sheaf import nerve_betti1, obstruction_norm


COVER = {"a": ["x", "y"], "b": ["y", "z"], "c": ["z", "x"]}   # 3 agents, ring


def test_consistent_glues_to_unique_global():
    truth = {"x": 1.0, "y": 2.0, "z": 3.0}
    sec = {ag: {v: truth[v] for v in vs} for ag, vs in COVER.items()}
    assert is_consistent(COVER, sec)
    assert glue(COVER, sec) == truth


def test_inconsistent_detected_and_localized():
    truth = {"x": 1.0, "y": 2.0, "z": 3.0}
    sec = {ag: {v: truth[v] for v in vs} for ag, vs in COVER.items()}
    sec["b"]["y"] = 9.0                       # agent b misreports y
    sec["b"]["z"] = 9.0                       # ...and z
    assert not is_consistent(COVER, sec)
    assert obstruction_norm(COVER, sec) > 0
    assert localize_fault(COVER, sec)[0] == "b"


def test_majority_glue_corrects_a_fault():
    # each variable observed by 2 agents in the ring isn't enough to outvote;
    # use a redundant cover where each var has 3 observers
    cover = {"a": ["x", "y"], "b": ["x", "y"], "c": ["x", "y"], "d": ["x", "y"]}
    truth = {"x": 5.0, "y": 7.0}
    sec = {ag: dict(truth) for ag in cover}
    sec["d"] = {"x": 99.0, "y": -99.0}        # one faulty agent
    g = majority_glue(cover, sec)
    assert g == truth                          # majority recovers the truth
    # naive averaging is corrupted by the fault
    avg_x = sum(sec[a]["x"] for a in cover) / 4
    assert abs(avg_x - truth["x"]) > 1.0


def test_nerve_betti1_counts_cycles():
    assert nerve_betti1(COVER) == 1            # the ring a-b-c has one cycle
