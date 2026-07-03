"""Tests for object-level macro-Go-Explore: avatar detection, object-directed macros, and that the
macro vocabulary cracks the procedural toys (ideally with FEWER real steps than micro)."""
from experiments.e128.macros import find_avatar, object_macros, macro_solve
from tests.e127.toy import toy_factory, ToyGame
from tests.e127.toy_click import toy_click_factory


def test_find_avatar_detects_cursor_and_directions():
    avatar, dir_map = find_avatar(toy_factory, None, [1, 2, 3, 4, 5, 7])
    assert avatar == 8                         # the cursor is the controllable avatar
    assert set(dir_map) <= {1, 2, 3, 4} and len(dir_map) >= 3   # learned direction displacements


def test_no_avatar_on_click_only_game():
    avatar, dir_map = find_avatar(toy_click_factory, None, [6])
    assert avatar is None and dir_map == {}


def test_macro_reach_collects_gems_and_wins_toygame():
    res = macro_solve(toy_factory, budget=20000, seed=0)
    assert res["win"] is True and res["best_levels"] >= 1
    g = ToyGame(); g.reset()
    for a in res["best_actions"]:
        g.step(a[0], a[1], a[2]) if a[0] == 6 else g.step(a[0])
    assert g.levels >= 1                        # replay-verified


def test_macro_clicks_crack_ordered_protocol():
    res = macro_solve(toy_click_factory, budget=6000, seed=0)
    assert res["win"] is True and res["best_levels"] >= 1


def test_object_macros_include_reach_and_click():
    g = ToyGame(); f = g.reset()
    m = object_macros(f, [1, 2, 3, 4, 6, 7], avatar=8, dir_map={1: (-1, 0), 2: (1, 0)})
    kinds = {x[0] for x in m}
    assert "reach" in kinds and "click" in kinds
