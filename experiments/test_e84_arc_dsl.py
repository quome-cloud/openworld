"""Unit tests for E84 ARC DSL primitives and program synthesis components.

Tests all 17 pure primitives, parameterized factories, Program class,
enumerate_programs, consistent(), and the corrupt-control behavior.

Run with: python3 experiments/test_e84_arc_dsl.py
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from openworld.arc_dsl.primitives import (
    rotate_90, rotate_180, rotate_270,
    flip_lr, flip_ud,
    transpose, antitranspose,
    gravity_down, gravity_up, gravity_right, gravity_left,
    crop_to_content,
    mirror_h, mirror_v,
    invert_colors,
    sort_rows,
    outline,
    make_recolor, make_translate, make_tile,
    connected_components, largest_object, symmetry_axis,
    PURE_PRIMS, get_parameterized_prims,
)
from openworld.arc_dsl.program import Program, enumerate_programs


# ---- helpers -------------------------------------------------------------------------------

def G(*rows):
    """Convenience: build a grid from rows of tuples/lists."""
    return [list(r) for r in rows]


# ---- test fixtures -------------------------------------------------------------------------

# Simple 3x3 test grid:
#  1 2 3
#  4 5 6
#  7 8 9
G3 = G([1, 2, 3], [4, 5, 6], [7, 8, 9])

# 2x2 identity test
G2 = G([1, 2], [3, 4])

# Grid with zeros (for gravity / crop tests)
GZ = G([0, 1, 0], [2, 0, 3], [0, 4, 0])  # sparse grid

# Grid for gravity tests
GCOL = G([1, 0, 0], [2, 0, 3], [0, 4, 0])

# Grid for invert_colors test
GC = G([0, 1, 9], [5, 0, 3])

# Simple symmetric grid
GSYM_H = G([1, 2, 1], [3, 4, 3], [1, 2, 1])  # both h and v symmetric
GSYM_V = G([1, 2, 1], [3, 4, 3], [5, 6, 5])  # v symmetric only

# Grid with interior (for outline test)
# 5x5 block of 1s — interior cells should be zeroed out
GBLOCK = G(
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
)
GBLOCK_OUTLINED = G(
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
)


# ---- Pure primitive tests ------------------------------------------------------------------

class TestRotations(unittest.TestCase):

    def test_rotate_90(self):
        # G3 rotated 90° CCW:
        # col 2 -> row 0: 3,6,9 -> [3,6,9] reversed? Actually np.rot90 k=1 CCW:
        # result[i,j] = original[j, n-1-i]
        result = rotate_90(G3)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 3)
        # Top row should be [3, 6, 9] (rightmost column bottom-to-top)
        self.assertEqual(result[0], [3, 6, 9])

    def test_rotate_180(self):
        result = rotate_180(G3)
        self.assertIsNotNone(result)
        # 180° rotation reverses the grid
        self.assertEqual(result[0], [9, 8, 7])
        self.assertEqual(result[2], [3, 2, 1])

    def test_rotate_270(self):
        result = rotate_270(G3)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        # 270° CCW = 90° CW: top row = left column top-to-bottom reversed
        # result[0] should be [7, 4, 1]
        self.assertEqual(result[0], [7, 4, 1])

    def test_rotate_90_then_270_is_identity(self):
        r = rotate_270(rotate_90(G3))
        self.assertEqual(r, G3)

    def test_rotate_180_twice_is_identity(self):
        r = rotate_180(rotate_180(G3))
        self.assertEqual(r, G3)

    def test_rotate_none_input(self):
        self.assertIsNone(rotate_90(None))


class TestFlips(unittest.TestCase):

    def test_flip_lr(self):
        result = flip_lr(G3)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], [3, 2, 1])
        self.assertEqual(result[2], [9, 8, 7])

    def test_flip_ud(self):
        result = flip_ud(G3)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], [7, 8, 9])
        self.assertEqual(result[2], [1, 2, 3])

    def test_flip_lr_twice_is_identity(self):
        r = flip_lr(flip_lr(G3))
        self.assertEqual(r, G3)

    def test_flip_ud_twice_is_identity(self):
        r = flip_ud(flip_ud(G3))
        self.assertEqual(r, G3)

    def test_flip_none_input(self):
        self.assertIsNone(flip_lr(None))
        self.assertIsNone(flip_ud(None))


class TestTranspose(unittest.TestCase):

    def test_transpose(self):
        result = transpose(G3)
        self.assertIsNotNone(result)
        # Transpose: rows become cols
        self.assertEqual(result[0], [1, 4, 7])
        self.assertEqual(result[1], [2, 5, 8])
        self.assertEqual(result[2], [3, 6, 9])

    def test_antitranspose(self):
        result = antitranspose(G3)
        self.assertIsNotNone(result)
        # Antitranspose: reflection along anti-diagonal
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 3)

    def test_transpose_twice_is_identity(self):
        r = transpose(transpose(G3))
        self.assertEqual(r, G3)

    def test_antitranspose_twice_is_identity(self):
        r = antitranspose(antitranspose(G3))
        self.assertEqual(r, G3)

    def test_transpose_non_square(self):
        g = G([1, 2, 3], [4, 5, 6])  # 2x3
        result = transpose(g)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)   # 3x2
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(result[0], [1, 4])
        self.assertEqual(result[2], [3, 6])


class TestGravity(unittest.TestCase):

    def test_gravity_down(self):
        g = G([1, 0], [0, 0], [0, 2])
        result = gravity_down(g)
        self.assertIsNotNone(result)
        # Col 0: [1, 0, 0] -> [0, 0, 1]; Col 1: [0, 0, 2] -> [0, 0, 2]
        self.assertEqual(result[2][0], 1)
        self.assertEqual(result[2][1], 2)
        self.assertEqual(result[0][0], 0)

    def test_gravity_up(self):
        g = G([0, 0], [1, 0], [0, 2])
        result = gravity_up(g)
        self.assertIsNotNone(result)
        # Col 0: [0, 1, 0] -> [1, 0, 0]; Col 1: [0, 0, 2] -> [2, 0, 0]
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][1], 2)

    def test_gravity_right(self):
        g = G([1, 0, 0], [0, 2, 0])
        result = gravity_right(g)
        self.assertIsNotNone(result)
        # Row 0: [1, 0, 0] -> [0, 0, 1]; Row 1: [0, 2, 0] -> [0, 0, 2]
        self.assertEqual(result[0][2], 1)
        self.assertEqual(result[1][2], 2)

    def test_gravity_left(self):
        g = G([0, 0, 1], [0, 2, 0])
        result = gravity_left(g)
        self.assertIsNotNone(result)
        # Row 0: [0, 0, 1] -> [1, 0, 0]; Row 1: [0, 2, 0] -> [2, 0, 0]
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[1][0], 2)

    def test_gravity_preserves_count(self):
        import numpy as np
        import numpy as np
        g = G([1, 0, 2], [0, 3, 0], [4, 0, 5])
        for fn in [gravity_down, gravity_up, gravity_right, gravity_left]:
            result = fn(g)
            self.assertIsNotNone(result)
            orig_nonzero = sum(c != 0 for row in g for c in row)
            res_nonzero = sum(c != 0 for row in result for c in row)
            self.assertEqual(orig_nonzero, res_nonzero, f"{fn.__name__} changed non-zero count")


class TestCropToContent(unittest.TestCase):

    def test_crop_basic(self):
        g = G([0, 0, 0], [0, 1, 0], [0, 0, 0])
        result = crop_to_content(g)
        self.assertIsNotNone(result)
        self.assertEqual(result, [[1]])

    def test_crop_rectangle(self):
        g = G([0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0])
        result = crop_to_content(g)
        self.assertIsNotNone(result)
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_crop_all_zeros_returns_none(self):
        g = G([0, 0], [0, 0])
        result = crop_to_content(g)
        self.assertIsNone(result)

    def test_crop_no_padding(self):
        # Grid already tight
        result = crop_to_content(G3)
        self.assertEqual(result, G3)


class TestMirror(unittest.TestCase):

    def test_mirror_h_doubles_width(self):
        g = G([1, 2], [3, 4])
        result = mirror_h(g)
        self.assertIsNotNone(result)
        self.assertEqual(len(result[0]), 4)  # double width
        self.assertEqual(result[0], [1, 2, 2, 1])  # [g | flip_lr(g)]

    def test_mirror_v_doubles_height(self):
        g = G([1, 2], [3, 4])
        result = mirror_v(g)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)  # double height
        self.assertEqual(result[0], [1, 2])
        self.assertEqual(result[3], [1, 2])  # flip_ud puts row 0 at bottom

    def test_mirror_h_symmetry(self):
        g = G([1, 2, 3])
        result = mirror_h(g)
        # Should be symmetric along the vertical center
        import numpy as np
        import numpy as np
        arr = result
        self.assertEqual(arr[0], arr[0][::-1])  # palindrome row


class TestInvertColors(unittest.TestCase):

    def test_zero_stays_zero(self):
        g = G([0, 1, 2], [9, 5, 0])
        result = invert_colors(g)
        self.assertIsNotNone(result)
        self.assertEqual(result[0][0], 0)  # 0 stays 0
        self.assertEqual(result[1][2], 0)  # 0 stays 0

    def test_nonzero_inverted(self):
        g = G([0, 1, 9], [5, 0, 3])
        result = invert_colors(g)
        self.assertIsNotNone(result)
        self.assertEqual(result[0][1], 8)   # 9 - 1 = 8
        self.assertEqual(result[0][2], 0)   # 9 - 9 = 0... but 9 is non-zero -> 0 (edge case)
        self.assertEqual(result[1][0], 4)   # 9 - 5 = 4
        self.assertEqual(result[1][2], 6)   # 9 - 3 = 6

    def test_invert_twice_except_9(self):
        # Applying invert twice: 0->0->0, c->(9-c)->(9-(9-c))=c for c not 9
        # For c=9: 9->0 (since 9-9=0) -> 0 stays 0 (not c), so NOT a true involution
        g = G([1, 2, 3], [4, 5, 6])
        result = invert_colors(invert_colors(g))
        # c=1->8->1: yes; c=9 would break but not in this grid
        self.assertEqual(result, g)


class TestSortRows(unittest.TestCase):

    def test_sort_rows_basic(self):
        g = G([3, 1, 2], [1, 2, 3], [2, 3, 1])
        result = sort_rows(g)
        self.assertIsNotNone(result)
        # Rows sorted lexicographically
        self.assertEqual(result[0], [1, 2, 3])
        self.assertEqual(result[1], [2, 3, 1])
        self.assertEqual(result[2], [3, 1, 2])

    def test_sort_rows_already_sorted(self):
        g = G([1, 2], [3, 4], [5, 6])
        result = sort_rows(g)
        self.assertEqual(result, g)


class TestOutline(unittest.TestCase):

    def test_outline_5x5_block(self):
        result = outline(GBLOCK)
        self.assertIsNotNone(result)
        self.assertEqual(result, GBLOCK_OUTLINED)

    def test_outline_single_cell(self):
        # Single cell: all neighbors missing -> not interior -> stays
        g = G([0, 0, 0], [0, 1, 0], [0, 0, 0])
        result = outline(g)
        self.assertIsNotNone(result)
        self.assertEqual(result[1][1], 1)

    def test_outline_preserves_border(self):
        result = outline(GBLOCK)
        # All border cells of GBLOCK should remain 1
        self.assertTrue(all(result[0][c] == 1 for c in range(5)))   # top row
        self.assertTrue(all(result[4][c] == 1 for c in range(5)))   # bottom row
        self.assertTrue(all(result[r][0] == 1 for r in range(5)))   # left col
        self.assertTrue(all(result[r][4] == 1 for r in range(5)))   # right col


# ---- Parameterized factory tests -----------------------------------------------------------

class TestMakeRecolor(unittest.TestCase):

    def test_recolor_replaces_color(self):
        g = G([1, 2, 1], [3, 1, 4])
        fn = make_recolor(1, 5)
        result = fn(g)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], [5, 2, 5])
        self.assertEqual(result[1], [3, 5, 4])

    def test_recolor_no_match(self):
        g = G([2, 3, 4])
        fn = make_recolor(1, 9)
        result = fn(g)
        self.assertEqual(result, g)

    def test_recolor_none_input(self):
        fn = make_recolor(1, 2)
        self.assertIsNone(fn(None))


class TestMakeTranslate(unittest.TestCase):

    def test_translate_down_one(self):
        g = G([1, 2], [3, 4], [5, 6])
        fn = make_translate(1, 0)
        result = fn(g)
        self.assertIsNotNone(result)
        # Row 0 should be zeros (vacated), row 1 should be old row 0
        self.assertEqual(result[0], [0, 0])
        self.assertEqual(result[1], [1, 2])
        self.assertEqual(result[2], [3, 4])

    def test_translate_right_one(self):
        g = G([1, 2, 3], [4, 5, 6])
        fn = make_translate(0, 1)
        result = fn(g)
        self.assertIsNotNone(result)
        self.assertEqual(result[0][0], 0)
        self.assertEqual(result[0][1], 1)
        self.assertEqual(result[0][2], 2)

    def test_translate_identity(self):
        # translate by (0,0) should not be in factories but test the function directly
        fn = make_translate(0, 0)
        result = fn(G3)
        self.assertEqual(result, G3)

    def test_translate_none_input(self):
        fn = make_translate(1, 0)
        self.assertIsNone(fn(None))


class TestMakeTile(unittest.TestCase):

    def test_tile_2x1_doubles_height(self):
        g = G([1, 2], [3, 4])
        fn = make_tile(2, 1)
        result = fn(g)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], [1, 2])
        self.assertEqual(result[2], [1, 2])  # repeated

    def test_tile_1x2_doubles_width(self):
        g = G([1, 2], [3, 4])
        fn = make_tile(1, 2)
        result = fn(g)
        self.assertIsNotNone(result)
        self.assertEqual(len(result[0]), 4)
        self.assertEqual(result[0], [1, 2, 1, 2])

    def test_tile_2x2_quadruples(self):
        g = G([1])
        fn = make_tile(2, 2)
        result = fn(g)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)

    def test_tile_none_input(self):
        fn = make_tile(2, 2)
        self.assertIsNone(fn(None))


# ---- Analysis function tests ---------------------------------------------------------------

class TestConnectedComponents(unittest.TestCase):

    def test_single_component(self):
        g = G([1, 1, 0], [1, 0, 0], [0, 0, 0])
        comps = connected_components(g)
        self.assertEqual(len(comps), 1)

    def test_two_components(self):
        g = G([1, 0, 2], [0, 0, 0], [3, 0, 4])
        comps = connected_components(g)
        self.assertEqual(len(comps), 4)  # each isolated cell is its own component

    def test_empty_grid(self):
        g = G([0, 0], [0, 0])
        comps = connected_components(g)
        self.assertEqual(len(comps), 0)

    def test_none_input(self):
        comps = connected_components(None)
        self.assertEqual(comps, [])


class TestLargestObject(unittest.TestCase):

    def test_largest_selected(self):
        # Two components: one big, one small
        g = G([1, 1, 0, 2], [1, 1, 0, 0], [0, 0, 0, 0])
        result = largest_object(g)
        self.assertIsNotNone(result)
        import numpy as np
        import numpy as np
        arr = [row for row in result]
        # Should contain only the 4-cell component (1s), not the single 2
        nonzero_vals = set(c for row in result for c in row if c != 0)
        self.assertIn(1, nonzero_vals)
        self.assertNotIn(2, nonzero_vals)

    def test_none_input(self):
        self.assertIsNone(largest_object(None))


class TestSymmetryAxis(unittest.TestCase):

    def test_both_symmetric(self):
        g = G([1, 2, 1], [3, 4, 3], [1, 2, 1])
        result = symmetry_axis(g)
        self.assertEqual(result, 'hv')

    def test_v_symmetric(self):
        g = G([1, 2, 1], [3, 4, 3], [5, 6, 5])
        result = symmetry_axis(g)
        self.assertEqual(result, 'v')

    def test_h_symmetric(self):
        g = G([1, 2, 3], [4, 5, 6], [1, 2, 3])
        result = symmetry_axis(g)
        self.assertEqual(result, 'h')

    def test_no_symmetry(self):
        result = symmetry_axis(G3)  # G3 has no symmetry
        self.assertIsNone(result)

    def test_none_input(self):
        self.assertIsNone(symmetry_axis(None))


# ---- Registry tests ------------------------------------------------------------------------

class TestPureRegistry(unittest.TestCase):

    def test_has_all_expected_keys(self):
        expected = {
            "rotate_90", "rotate_180", "rotate_270",
            "flip_lr", "flip_ud",
            "transpose", "antitranspose",
            "gravity_down", "gravity_up", "gravity_right", "gravity_left",
            "crop_to_content",
            "mirror_h", "mirror_v",
            "invert_colors",
            "sort_rows",
            "outline",
        }
        self.assertTrue(expected.issubset(set(PURE_PRIMS.keys())),
                        f"Missing keys: {expected - set(PURE_PRIMS.keys())}")

    def test_has_at_least_15_primitives(self):
        self.assertGreaterEqual(len(PURE_PRIMS), 15)

    def test_all_callables(self):
        for name, fn in PURE_PRIMS.items():
            self.assertTrue(callable(fn), f"{name} is not callable")


class TestGetParameterizedPrims(unittest.TestCase):

    def test_recolor_extracted(self):
        demos = [
            (G([1, 2], [3, 4]), G([5, 2], [3, 4]))
        ]
        prims = get_parameterized_prims(demos)
        self.assertIn("recolor_1_to_5", prims)

    def test_tile_always_present(self):
        demos = [(G([1]), G([1]))]
        prims = get_parameterized_prims(demos)
        self.assertIn("tile_2x1", prims)
        self.assertIn("tile_1x2", prims)
        self.assertIn("tile_2x2", prims)

    def test_translate_always_present(self):
        demos = [(G([1]), G([1]))]
        prims = get_parameterized_prims(demos)
        self.assertIn("translate_+1_+0", prims)
        self.assertIn("translate_-1_+0", prims)


# ---- Program and enumerate_programs tests --------------------------------------------------

class TestProgram(unittest.TestCase):

    def test_single_step(self):
        prog = Program([("flip_lr", flip_lr)])
        result = prog(G3)
        self.assertEqual(result, flip_lr(G3))

    def test_two_step(self):
        prog = Program([("rotate_90", rotate_90), ("flip_lr", flip_lr)])
        result = prog(G3)
        expected = flip_lr(rotate_90(G3))
        self.assertEqual(result, expected)

    def test_none_propagates(self):
        # A function that always returns None
        def bad_fn(g):
            return None
        prog = Program([("rotate_90", rotate_90), ("bad", bad_fn)])
        self.assertIsNone(prog(G3))

    def test_none_input_propagates(self):
        prog = Program([("rotate_90", rotate_90)])
        self.assertIsNone(prog(None))

    def test_len(self):
        prog = Program([("a", flip_lr), ("b", flip_ud)])
        self.assertEqual(len(prog), 2)

    def test_name(self):
        prog = Program([("rotate_90", rotate_90), ("flip_lr", flip_lr)])
        self.assertEqual(prog.name, "rotate_90 -> flip_lr")


class TestEnumeratePrograms(unittest.TestCase):

    def test_depth1_first(self):
        prims = {"a": flip_lr, "b": flip_ud}
        programs = list(enumerate_programs(prims, max_depth=1))
        self.assertEqual(len(programs), 2)
        self.assertEqual(programs[0].name, "a")
        self.assertEqual(programs[1].name, "b")

    def test_depth2(self):
        prims = {"a": flip_lr, "b": flip_ud}
        programs = list(enumerate_programs(prims, max_depth=2))
        # 2 (depth-1) + 4 (depth-2: aa, ab, ba, bb) = 6
        self.assertEqual(len(programs), 6)

    def test_depth1_order(self):
        # Depth-1 programs come before depth-2
        prims = {"a": flip_lr, "b": flip_ud}
        programs = list(enumerate_programs(prims, max_depth=2))
        # All depth-1 must precede depth-2
        self.assertTrue(all(len(p) == 1 for p in programs[:2]))
        self.assertTrue(all(len(p) == 2 for p in programs[2:]))

    def test_yields_programs(self):
        prims = {"a": flip_lr}
        for p in enumerate_programs(prims, max_depth=2):
            self.assertIsInstance(p, Program)


# ---- Consistent function tests -------------------------------------------------------------

class TestConsistent(unittest.TestCase):

    def _consistent(self, prog, demos):
        """Replicate consistent() locally."""
        import sys
        sys.path.insert(0, str(HERE.parent))
        from experiments.e84_arc_synthesis import consistent
        return consistent(prog, demos)

    def _import_consistent(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "e84", str(HERE / "e84_arc_synthesis.py")
        )
        mod = importlib.util.load_from_spec = None
        # simpler: just import directly
        import sys
        sys.path.insert(0, str(HERE))
        import e84_arc_synthesis as e84
        return e84.consistent

    def test_consistent_true(self):
        # flip_lr applied to G2 should match its output
        inp = G([1, 2], [3, 4])
        out = G([2, 1], [4, 3])  # flip_lr(inp)
        prog = Program([("flip_lr", flip_lr)])
        demos = [(inp, out)]
        # call consistent via import
        sys.path.insert(0, str(HERE))
        import e84_arc_synthesis as e84
        self.assertTrue(e84.consistent(prog, demos))

    def test_consistent_false(self):
        inp = G([1, 2], [3, 4])
        out = G([1, 2], [3, 4])  # identity, not flip_lr
        prog = Program([("flip_lr", flip_lr)])
        demos = [(inp, out)]
        sys.path.insert(0, str(HERE))
        import e84_arc_synthesis as e84
        self.assertFalse(e84.consistent(prog, demos))


# ---- Corrupt control test ------------------------------------------------------------------

class TestCorruptControl(unittest.TestCase):
    """Verify that shuffled demos prevent synthesis (the load-bearing control)."""

    def test_corrupt_demos_prevent_synthesis(self):
        """With demos that have shuffled outputs, no program from PURE_PRIMS should be
        consistent (assuming there are >=2 demos with different transformations).
        """
        # Create a task where flip_lr is the rule
        demos = [
            (G([1, 2], [3, 4]), G([2, 1], [4, 3])),  # flip_lr correct
            (G([5, 6], [7, 8]), G([6, 5], [8, 7])),  # flip_lr correct
        ]
        # Corrupt: swap outputs
        corrupt_demos = [(demos[0][0], demos[1][1]), (demos[1][0], demos[0][1])]

        # With correct demos, flip_lr should be consistent
        prog = Program([("flip_lr", flip_lr)])
        sys.path.insert(0, str(HERE))
        import e84_arc_synthesis as e84
        self.assertTrue(e84.consistent(prog, demos))
        # With corrupt demos, flip_lr should NOT be consistent
        self.assertFalse(e84.consistent(prog, corrupt_demos))


# ---- Integration smoke test ----------------------------------------------------------------

class TestIntegration(unittest.TestCase):

    def test_all_pure_prims_handle_g3(self):
        """All pure primitives should return a valid grid or None (not raise) on G3."""
        for name, fn in PURE_PRIMS.items():
            result = fn(G3)
            if result is not None:
                self.assertIsInstance(result, list, f"{name} returned non-list")
                self.assertGreater(len(result), 0, f"{name} returned empty list")
                self.assertIsInstance(result[0], list, f"{name} row is not list")

    def test_all_pure_prims_handle_none(self):
        """All pure primitives should return None when given None input."""
        for name, fn in PURE_PRIMS.items():
            result = fn(None)
            self.assertIsNone(result, f"{name} should return None for None input")

    def test_program_enumerate_runs_on_small_prims(self):
        """enumerate_programs should yield programs without error."""
        small_prims = {k: PURE_PRIMS[k] for k in ["rotate_90", "flip_lr"]}
        programs = list(enumerate_programs(small_prims, max_depth=2))
        self.assertGreater(len(programs), 0)
        # All should be runnable
        for p in programs[:10]:
            result = p(G3)
            # result can be None if a composed prim fails, that's OK


if __name__ == "__main__":
    unittest.main(verbosity=2)
