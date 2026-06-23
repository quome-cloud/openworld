"""
Geometric grid DSL for ARC-AGI program synthesis.

All primitives: Grid -> Grid where Grid = List[List[int]]
Pure primitives are parameter-free; parameterized primitives encode their params in the function name.
All handle exceptions by returning None (failed application).
"""
import numpy as np
from typing import List, Optional, Callable, Tuple
from itertools import product as iproduct

Grid = List[List[int]]
Primitive = Callable[[Grid], Optional[Grid]]


def _g(arr: np.ndarray) -> Grid:
    return arr.tolist()


def _a(g: Grid) -> np.ndarray:
    return np.array(g, dtype=int)


def _safe(f):
    def wrapper(g):
        if g is None:
            return None
        try:
            r = f(g)
            if r is None:
                return None
            if len(r) == 0 or len(r[0]) == 0:
                return None
            return r
        except Exception:
            return None
    return wrapper


# ---- Pure primitives -----------------------------------------------------------------------

@_safe
def rotate_90(g: Grid) -> Optional[Grid]:
    """Rotate 90 degrees counter-clockwise."""
    return _g(np.rot90(_a(g), k=1))


@_safe
def rotate_180(g: Grid) -> Optional[Grid]:
    """Rotate 180 degrees."""
    return _g(np.rot90(_a(g), k=2))


@_safe
def rotate_270(g: Grid) -> Optional[Grid]:
    """Rotate 90 degrees clockwise (270 CCW)."""
    return _g(np.rot90(_a(g), k=3))


@_safe
def flip_lr(g: Grid) -> Optional[Grid]:
    """Flip left-right (mirror horizontally)."""
    return _g(np.fliplr(_a(g)))


@_safe
def flip_ud(g: Grid) -> Optional[Grid]:
    """Flip up-down (mirror vertically)."""
    return _g(np.flipud(_a(g)))


@_safe
def transpose(g: Grid) -> Optional[Grid]:
    """Transpose (rows <-> cols)."""
    return _g(_a(g).T)


@_safe
def antitranspose(g: Grid) -> Optional[Grid]:
    """Antidiagonal transpose (flip along the anti-diagonal).
    Equivalent to rotate_90 then flip_lr, or flip both axes then transpose.
    """
    a = _a(g)
    # antitranspose: a[i,j] -> a[n-1-j, m-1-i]
    return _g(np.rot90(a, k=1)[:, ::-1])


@_safe
def gravity_down(g: Grid) -> Optional[Grid]:
    """Pull all non-zero cells downward (like gravity). Zero cells float to top."""
    a = _a(g)
    result = np.zeros_like(a)
    for col in range(a.shape[1]):
        nonzero = a[:, col][a[:, col] != 0]
        if len(nonzero) > 0:
            result[-len(nonzero):, col] = nonzero
    return _g(result)


@_safe
def gravity_up(g: Grid) -> Optional[Grid]:
    """Pull all non-zero cells upward. Zero cells sink to bottom."""
    a = _a(g)
    result = np.zeros_like(a)
    for col in range(a.shape[1]):
        nonzero = a[:, col][a[:, col] != 0]
        if len(nonzero) > 0:
            result[:len(nonzero), col] = nonzero
    return _g(result)


@_safe
def gravity_right(g: Grid) -> Optional[Grid]:
    """Pull all non-zero cells rightward. Zero cells float to left."""
    a = _a(g)
    result = np.zeros_like(a)
    for row in range(a.shape[0]):
        nonzero = a[row, :][a[row, :] != 0]
        if len(nonzero) > 0:
            result[row, -len(nonzero):] = nonzero
    return _g(result)


@_safe
def gravity_left(g: Grid) -> Optional[Grid]:
    """Pull all non-zero cells leftward. Zero cells drift to right."""
    a = _a(g)
    result = np.zeros_like(a)
    for row in range(a.shape[0]):
        nonzero = a[row, :][a[row, :] != 0]
        if len(nonzero) > 0:
            result[row, :len(nonzero)] = nonzero
    return _g(result)


@_safe
def crop_to_content(g: Grid) -> Optional[Grid]:
    """Crop to the bounding box of non-zero cells."""
    a = _a(g)
    rows_with_content = np.any(a != 0, axis=1)
    cols_with_content = np.any(a != 0, axis=0)
    if not np.any(rows_with_content) or not np.any(cols_with_content):
        return None
    r_min, r_max = np.where(rows_with_content)[0][[0, -1]]
    c_min, c_max = np.where(cols_with_content)[0][[0, -1]]
    cropped = a[r_min:r_max + 1, c_min:c_max + 1]
    if cropped.size == 0:
        return None
    return _g(cropped)


@_safe
def mirror_h(g: Grid) -> Optional[Grid]:
    """Tile with horizontal mirror: [grid | flip_lr(grid)] side by side."""
    a = _a(g)
    return _g(np.concatenate([a, np.fliplr(a)], axis=1))


@_safe
def mirror_v(g: Grid) -> Optional[Grid]:
    """Tile with vertical mirror: [grid; flip_ud(grid)] stacked."""
    a = _a(g)
    return _g(np.concatenate([a, np.flipud(a)], axis=0))


@_safe
def invert_colors(g: Grid) -> Optional[Grid]:
    """Map each cell: 0 stays 0; non-zero c -> (9 - c)."""
    a = _a(g)
    result = np.where(a != 0, 9 - a, 0)
    return _g(result)


@_safe
def sort_rows(g: Grid) -> Optional[Grid]:
    """Sort rows by their tuple value (ascending)."""
    rows = [tuple(row) for row in g]
    sorted_rows = sorted(rows)
    return [list(row) for row in sorted_rows]


@_safe
def outline(g: Grid) -> Optional[Grid]:
    """Keep only cells that are on the border of a connected object.

    A cell is 'interior' if all 4 of its neighbors exist, are non-zero, AND share the same value.
    Interior cells are set to 0.
    """
    a = _a(g)
    result = a.copy()
    rows, cols = a.shape
    for r in range(rows):
        for c in range(cols):
            if a[r, c] == 0:
                continue
            v = a[r, c]
            # Check all 4 neighbors
            neighbors = [
                (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)
            ]
            is_interior = True
            for nr, nc in neighbors:
                if not (0 <= nr < rows and 0 <= nc < cols):
                    is_interior = False
                    break
                if a[nr, nc] != v:
                    is_interior = False
                    break
            if is_interior:
                result[r, c] = 0
    return _g(result)


# ---- Parameterized primitive factories -----------------------------------------------------

def make_recolor(from_c: int, to_c: int) -> Primitive:
    """Replace color from_c with to_c everywhere."""
    @_safe
    def recolor(g: Grid) -> Optional[Grid]:
        a = _a(g)
        result = a.copy()
        result[a == from_c] = to_c
        return _g(result)
    recolor.__name__ = f"recolor_{from_c}_to_{to_c}"
    return recolor


def make_translate(dr: int, dc: int, fill: int = 0) -> Primitive:
    """Translate grid by (dr, dc), filling vacated cells with fill."""
    @_safe
    def translate(g: Grid) -> Optional[Grid]:
        a = _a(g)
        result = np.full_like(a, fill)
        rows, cols = a.shape
        # Source region
        src_r_start = max(0, -dr)
        src_r_end = min(rows, rows - dr)
        src_c_start = max(0, -dc)
        src_c_end = min(cols, cols - dc)
        # Destination region
        dst_r_start = max(0, dr)
        dst_r_end = min(rows, rows + dr)
        dst_c_start = max(0, dc)
        dst_c_end = min(cols, cols + dc)
        if src_r_end > src_r_start and src_c_end > src_c_start:
            result[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = \
                a[src_r_start:src_r_end, src_c_start:src_c_end]
        return _g(result)
    translate.__name__ = f"translate_{dr:+d}_{dc:+d}"
    return translate


def make_tile(nr: int, nc: int) -> Primitive:
    """Tile the grid nr times vertically, nc times horizontally."""
    @_safe
    def tile(g: Grid) -> Optional[Grid]:
        a = _a(g)
        return _g(np.tile(a, (nr, nc)))
    tile.__name__ = f"tile_{nr}x{nc}"
    return tile


# ---- Analysis functions (not transforms, used for synthesis) --------------------------------

def connected_components(g: Grid, connectivity: int = 4) -> List[Grid]:
    """Return each connected component (non-zero) as a separate grid (same size, zeros elsewhere)."""
    if g is None:
        return []
    try:
        a = _a(g)
        rows, cols = a.shape
        visited = np.zeros((rows, cols), dtype=bool)
        components = []

        def bfs(start_r, start_c, val):
            mask = np.zeros((rows, cols), dtype=bool)
            queue = [(start_r, start_c)]
            visited[start_r, start_c] = True
            mask[start_r, start_c] = True
            while queue:
                r, c = queue.pop(0)
                neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                if connectivity == 8:
                    neighbors += [(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]
                for nr, nc in neighbors:
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and a[nr, nc] == val:
                        visited[nr, nc] = True
                        mask[nr, nc] = True
                        queue.append((nr, nc))
            return mask

        for r in range(rows):
            for c in range(cols):
                if a[r, c] != 0 and not visited[r, c]:
                    mask = bfs(r, c, a[r, c])
                    comp = np.zeros_like(a)
                    comp[mask] = a[mask]
                    components.append(_g(comp))

        return components
    except Exception:
        return []


def largest_object(g: Grid) -> Optional[Grid]:
    """Return grid with only the largest connected component."""
    if g is None:
        return None
    try:
        comps = connected_components(g)
        if not comps:
            return None
        # Find component with most non-zero cells
        best = max(comps, key=lambda c: np.count_nonzero(_a(c)))
        return best
    except Exception:
        return None


def symmetry_axis(g: Grid) -> Optional[str]:
    """Detect if grid has symmetry: 'h', 'v', 'hv', or None.
    'h' = horizontal axis (flip_ud symmetric)
    'v' = vertical axis (flip_lr symmetric)
    'hv' = both
    """
    if g is None:
        return None
    try:
        a = _a(g)
        h_sym = np.array_equal(a, np.flipud(a))
        v_sym = np.array_equal(a, np.fliplr(a))
        if h_sym and v_sym:
            return 'hv'
        elif h_sym:
            return 'h'
        elif v_sym:
            return 'v'
        return None
    except Exception:
        return None


# ---- Registry of pure (parameter-free) primitives ------------------------------------------

PURE_PRIMS: dict = {
    "rotate_90": rotate_90,
    "rotate_180": rotate_180,
    "rotate_270": rotate_270,
    "flip_lr": flip_lr,
    "flip_ud": flip_ud,
    "transpose": transpose,
    "antitranspose": antitranspose,
    "gravity_down": gravity_down,
    "gravity_up": gravity_up,
    "gravity_right": gravity_right,
    "gravity_left": gravity_left,
    "crop_to_content": crop_to_content,
    "mirror_h": mirror_h,
    "mirror_v": mirror_v,
    "invert_colors": invert_colors,
    "sort_rows": sort_rows,
    "outline": outline,
}


def get_parameterized_prims(demos) -> dict:
    """Extract parameterized primitives from demo pairs (input->output).
    Returns dict of name -> fn.
    """
    prims = {}
    colors_seen = set()
    for inp, out in demos:
        for grid in [inp, out]:
            for row in grid:
                colors_seen.update(row)
    # recolor: try pairs of colors present in demos
    colors = sorted(colors_seen)
    for a in colors:
        for b in colors:
            if a != b:
                prims[f"recolor_{a}_to_{b}"] = make_recolor(a, b)
    # tile: small factors only
    for nr in [1, 2, 3]:
        for nc in [1, 2, 3]:
            if (nr, nc) != (1, 1):
                prims[f"tile_{nr}x{nc}"] = make_tile(nr, nc)
    # translate: bounded by [-2, 2]
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if (dr, dc) != (0, 0):
                prims[f"translate_{dr:+d}_{dc:+d}"] = make_translate(dr, dc)
    return prims
