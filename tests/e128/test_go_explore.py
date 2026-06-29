"""Offline, deterministic tests for the Go-Explore source-free solver.

ToyClickGame is the cleanest goal-as-PROCEDURE test: the win requires pressing A->B->C in order
(an ordered protocol -- exactly what static-goal methods cannot express and what defeats source-free
agents on the real walls). Go-Explore must discover that order. ToyGame is a navigation/collection
procedure. Both are deterministic, so results are reproducible with a fixed seed."""
from experiments.e128.go_explore import go_explore, identity_mask, cell_key
from experiments.e127 import perception
from tests.e127.toy_click import toy_click_factory, ToyClickGame
from tests.e127.toy import toy_factory, ToyGame
import numpy as np


def test_cracks_ordered_protocol_click_game():
    # win requires the procedure click A->B->C; Go-Explore should find it from scratch.
    res = go_explore(toy_click_factory, perception.candidate_actions, budget=8000, seed=0)
    assert res["win"] is True, res
    assert res["best_levels"] >= 1
    # verify the discovered action sequence ACTUALLY wins on the real game (replay-verify)
    g = ToyClickGame(); g.reset()
    for a in res["best_actions"]:
        g.step(a[0], a[1], a[2]) if a[0] == 6 else g.step(a[0])
    assert g.levels >= 1


def test_seeded_from_frontier_reaches_win_fast():
    # Seed with the first two correct presses (the 'banked frontier'); Go-Explore finishes level.
    seed = [[6, 2, 2], [6, 5, 4]]   # press A then B (phase 0->2); only C remains
    res = go_explore(toy_click_factory, perception.candidate_actions, budget=3000, seed=0,
                     seed_actions=seed)
    assert res["win"] is True and res["best_levels"] >= 1


def test_deterministic():
    a = go_explore(toy_click_factory, perception.candidate_actions, budget=8000, seed=0)
    b = go_explore(toy_click_factory, perception.candidate_actions, budget=8000, seed=0)
    assert a["best_actions"] == b["best_actions"] and a["win"] == b["win"]


def test_navigation_collection_procedure_toygame():
    # ToyGame: collect 3 gems by navigating -> level up. A larger budget (harder search).
    res = go_explore(toy_factory, perception.candidate_actions, budget=40000, seed=1)
    assert res["best_levels"] >= 1, res
    g = ToyGame(); g.reset()
    for a in res["best_actions"]:
        g.step(a[0], a[1], a[2]) if a[0] == 6 else g.step(a[0])
    assert g.levels >= 1


def test_identity_mask_flags_status_only():
    # under pure noop, ToyGame's status cell (0,0) changes every step -> masked; nothing else.
    g = ToyGame(); frames = [np.asarray(g.reset())]
    for _ in range(20):
        frames.append(np.asarray(g.step(7)))
    m = identity_mask(frames)
    assert m[0, 0] == True and m.sum() == 1
    # masking changes the cell key (status cell ignored)
    f = frames[-1]
    assert cell_key(f, m) != cell_key(f, None)
