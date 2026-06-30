"""E134 — K perception lenses for multi-modal world observation.

Each lens is `perceive(frame) -> dict` where frame is a 2D numpy int array.
LENSES is the registry; key_of(state) produces a canonical hashable tuple.

The lenses are designed for ARC-AGI-3 frames (typically 64x64 or smaller):
  objects   — connected components via objstate (position-centric)
  salience  — small/rare-color targets ranked for click-game use
  meter     — top-row counter/timer values (NOT masked; a timer-driven win is visible)
  symmetry  — h/v/d symmetry booleans + broken-cell count
  palette   — per-color cell counts (color-algebra)
  regions   — coarse 8x8 occupancy grid of dominant non-bg color per cell
"""

import os
import sys

import numpy as np

# Add the project root so experiments.e125.objstate is importable.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from experiments.e125.objstate import object_state as _object_state
except ImportError:                              # flat agent workspace: objstate.py sits alongside
    from objstate import object_state as _object_state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_int_array(frame):
    """Return a 2D numpy int array regardless of dtype."""
    return np.asarray(frame).astype(int)


def _bg(f):
    """Most-frequent color (= background)."""
    vals, counts = np.unique(f, return_counts=True)
    return int(vals[np.argmax(counts)])


# ---------------------------------------------------------------------------
# Lens 1: objects
# ---------------------------------------------------------------------------

def objects(frame):
    """Connected components → {"bg": int, "objects": [{color,size,y,x}, ...]}."""
    f = _to_int_array(frame)
    return _object_state(f.tolist())


# ---------------------------------------------------------------------------
# Lens 2: salience
# ---------------------------------------------------------------------------

def salience(frame):
    """Small (size<=16) non-bg components ranked by size then color-rarity.

    The status-bar / meter row (row 0) is excluded so that game-interior objects
    dominate; row-0 cells are handled by the 'meter' lens.
    Returns {"targets": [[y, x, size, color], ...]} sorted small/rare first.
    y is in FULL frame coordinates (row 0 of the sub-frame = row 1 of the frame).
    """
    f = _to_int_array(frame)

    # Color rarity = total pixel count in the FULL frame (fewer = rarer).
    vals, counts = np.unique(f, return_counts=True)
    color_counts = {int(v): int(c) for v, c in zip(vals, counts)}

    # Process the sub-frame (skip row 0 = status bar).
    sub = f[1:, :]
    state = _object_state(sub.tolist())

    small_objs = [o for o in state["objects"] if o["size"] <= 16]

    # Sort: size ascending, then color rarity ascending (rarer colors first in
    # ties — the rarest / smallest = most click-relevant).
    small_objs.sort(key=lambda o: (o["size"], color_counts.get(o["color"], 0)))

    # Translate sub-frame y-coordinates back to full-frame coordinates (+1).
    targets = [[o["y"] + 1, o["x"], o["size"], o["color"]] for o in small_objs]
    return {"targets": targets}


# ---------------------------------------------------------------------------
# Lens 3: meter
# ---------------------------------------------------------------------------

def meter(frame):
    """Status-bar / counter channel.

    Exposes the DISTINCT non-background values of the top row(s) as a sorted
    list so that timer/counter-driven win conditions are visible (not masked).
    Returns {"meter": [val, ...]} — non-empty when the top row has structure.
    """
    f = _to_int_array(frame)
    bg = _bg(f)
    top_row = f[0, :]
    vals = sorted({int(v) for v in top_row if int(v) != bg})
    return {"meter": vals}


# ---------------------------------------------------------------------------
# Lens 4: symmetry
# ---------------------------------------------------------------------------

def symmetry(frame):
    """Spatial symmetry descriptors.

    Returns {"h": bool, "v": bool, "d": bool, "broken": int}.
      h — horizontal (left-right) symmetry.
      v — vertical (top-bottom) symmetry.
      d — diagonal (transpose) symmetry (square frames only; False otherwise).
      broken — total symmetry-breaking cell count summed over the violated axes.
    """
    f = _to_int_array(frame)
    rows, cols = f.shape

    h_sym = bool(np.array_equal(f, np.fliplr(f)))
    v_sym = bool(np.array_equal(f, np.flipud(f)))
    d_sym = (rows == cols) and bool(np.array_equal(f, f.T))

    broken = 0
    if not h_sym:
        broken += int(np.sum(f != np.fliplr(f))) // 2
    if not v_sym:
        broken += int(np.sum(f != np.flipud(f))) // 2
    if not d_sym and rows == cols:
        broken += int(np.sum(f != f.T)) // 2

    return {"h": h_sym, "v": v_sym, "d": d_sym, "broken": broken}


# ---------------------------------------------------------------------------
# Lens 5: palette
# ---------------------------------------------------------------------------

def palette(frame):
    """Per-color cell counts.

    Returns {"hist": {color: count}} — the color-algebra signature.
    """
    f = _to_int_array(frame)
    vals, counts = np.unique(f, return_counts=True)
    hist = {int(v): int(c) for v, c in zip(vals, counts)}
    return {"hist": hist}


# ---------------------------------------------------------------------------
# Lens 6: regions
# ---------------------------------------------------------------------------

def regions(frame, G=8):
    """Coarse G×G occupancy grid.

    Divides the frame into G×G cells and records the dominant non-background
    color (or the bg color when the cell is all-bg).
    Returns {"grid": [[dominant_color, ...], ...]} — G rows of G ints.
    """
    f = _to_int_array(frame)
    bg = _bg(f)
    rows, cols = f.shape
    rh = rows / G
    rw = cols / G
    grid_out = []
    for gi in range(G):
        row_out = []
        for gj in range(G):
            r0 = int(gi * rh);  r1 = int((gi + 1) * rh)
            c0 = int(gj * rw);  c1 = int((gj + 1) * rw)
            cell = f[r0:r1, c0:c1]
            if cell.size == 0:
                row_out.append(bg)
                continue
            mask = cell != bg
            if not np.any(mask):
                row_out.append(bg)
                continue
            vs, cs = np.unique(cell[mask], return_counts=True)
            row_out.append(int(vs[np.argmax(cs)]))
        grid_out.append(row_out)
    return {"grid": grid_out}


# ---------------------------------------------------------------------------
# Canonical key
# ---------------------------------------------------------------------------

def key_of(state):
    """Convert any lens output dict to a canonical, hashable tuple.

    Recursively: dicts → sorted (k, v) tuples; lists/tuples → tuples;
    floats → rounded to 6 dp; numpy scalars → Python natives; all else as-is.
    Deterministic for equal inputs.
    """
    def _convert(v):
        if isinstance(v, dict):
            return tuple(sorted((_convert(k), _convert(val)) for k, val in v.items()))
        if isinstance(v, (list, tuple)):
            return tuple(_convert(x) for x in v)
        if isinstance(v, np.floating):
            return round(float(v), 6)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, float):
            return round(v, 6)
        return v

    return _convert(state)


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

LENSES = {
    "objects":   objects,
    "salience":  salience,
    "meter":     meter,
    "symmetry":  symmetry,
    "palette":   palette,
    "regions":   regions,
}
