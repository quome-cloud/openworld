"""Offline, deterministic tests for the Go-Explore source-free solver.

ToyClickGame is the cleanest goal-as-PROCEDURE test: the win requires pressing A->B->C in order
(an ordered protocol -- exactly what static-goal methods cannot express and what defeats source-free
agents on the real walls). Go-Explore must discover that order. ToyGame is a navigation/collection
procedure. Both are deterministic, so results are reproducible with a fixed seed."""
from experiments.e128.go_explore import go_explore, identity_mask, cell_key, object_cell
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


def test_mouse_clicks_are_in_the_action_space():
    # a click game (avail=[6]) must yield (6,x,y) click actions at inferred sprite targets
    g = ToyClickGame(); f = g.reset()
    acts = perception.candidate_actions(f, g.avail)
    assert g.avail == [6]
    assert acts and all(a[0] == 6 for a in acts)              # all clicks
    assert (6, 2, 2) in acts                                  # at button A (x=col,y=row)


def test_free_timer_masked_but_meaningful_counter_survives():
    # free-running timer: changes EVERY step -> masked (noise). meaningful counter: changes only on a
    # specific event -> NOT masked (kept in the representation).
    H = W = 8
    frames = []
    for t in range(20):
        f = np.zeros((H, W), dtype=int)
        f[0, 0] = (t % 15) + 1            # FREE-RUNNING timer: changes every step
        f[7, 7] = 1 if t >= 10 else 0     # MEANINGFUL counter: flips once, at t=10 (selective)
        frames.append(f)
    m = identity_mask(frames)
    assert m[0, 0] == True                # free timer masked
    assert m[7, 7] == False               # meaningful counter NOT masked -> stays in the cell rep
    # object-state cell distinguishes the counter state (the win-relevant change is preserved)
    before = object_cell(frames[0], m, levels=0)
    after = object_cell(frames[19], m, levels=0)
    assert before != after


def test_object_cell_distinguishes_procedure_phases():
    # the object-state rep must distinguish the ordered-protocol phases (else Go-Explore is blind)
    g = ToyClickGame(); f0 = g.reset()
    k0 = object_cell(f0, None, levels=0)
    g.step(6, 2, 2)                        # press A -> recolors -> different object-state
    k1 = object_cell(g.frame, None, levels=0)
    assert k0 != k1
