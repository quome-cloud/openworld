from e119 import macro


OJ = {"bg": 0, "objects": [{"id": 0, "color": 5, "centroid": (10, 20)},
                           {"id": 1, "color": 3, "centroid": (40, 8)}], "relations": []}


def test_compile_directional_and_repeat():
    assert macro.compile_macro(["a1", "a3 x2"], OJ, [1, 2, 3, 4]) == [(1,), (3,), (3,)]


def test_compile_click_resolves_object_centroid():
    # centroid is (row, col) = (y, x); click tuple is (6, x=col, y=row)
    assert macro.compile_macro(["click #0"], OJ, [6]) == [(6, 20, 10)]


def test_compile_drops_unresolvable_ops():
    # a5 not in avail -> dropped; click when 6 not in avail -> dropped; missing obj -> dropped
    assert macro.compile_macro(["a5", "a2", "click #9"], OJ, [1, 2, 3, 4]) == [(2,)]
    assert macro.compile_macro(["click #0"], OJ, [1, 2]) == []


def test_compile_caps_repeat_at_four():
    assert macro.compile_macro(["a1 x99"], OJ, [1]) == [(1,), (1,), (1,), (1,)]
