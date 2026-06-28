# tests/e127/test_perception.py
import numpy as np
from experiments.e127 import perception as P

def test_infer_click_targets_small_sprites_only():
    f = np.zeros((8, 8), dtype=int)
    f[2, 2] = 5                     # small sprite (size 1)
    f[5, 5] = 6                     # small sprite (size 1)
    f[0:1, 0:8] = 0                 # background row stays bg
    f[6:8, 0:8] = 3                 # a LARGE block (16 cells) -> not a click target
    t = P.infer_click_targets(f, max_size=4)
    assert (2, 2) in t and (5, 5) in t
    assert (6, 0) not in t          # big block excluded

def test_board_match_error_counts_and_exact():
    a = np.zeros((4, 4), dtype=int); b = np.zeros((4, 4), dtype=int)
    assert P.board_match_error(a, b)["exact"] is True
    b[1, 1] = 9
    e = P.board_match_error(a, b)
    assert e["exact"] is False and e["cells_wrong"] == 1 and e["cells_total"] == 16
    assert e["error_map"][1, 1] == True and e["error_map"].sum() == 1

def test_board_match_error_shape_mismatch_all_wrong():
    e = P.board_match_error(np.zeros((2, 2), dtype=int), np.zeros((4, 4), dtype=int))
    assert e["exact"] is False and e["cells_wrong"] == e["cells_total"]

def test_render_diff_marks_differences():
    a = np.zeros((3, 3), dtype=int); b = np.zeros((3, 3), dtype=int); b[0, 0] = 1
    s = P.render_diff(a, b)
    assert "X" in s and "board-match" in s

def test_candidate_actions_click_game():
    f = np.zeros((8, 8), dtype=int); f[2, 2] = 5
    acts = P.candidate_actions(f, avail=[6])
    assert (6, 2, 2) in acts        # click at x=col=2, y=row=2
    assert all(a[0] == 6 for a in acts)

def test_candidate_actions_directional_game():
    f = np.zeros((8, 8), dtype=int)
    acts = P.candidate_actions(f, avail=[1, 2, 3, 4, 7])
    assert (1, None, None) in acts and (7, None, None) in acts
    assert all(a[0] != 6 for a in acts)
