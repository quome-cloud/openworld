"""Program representation: a sequence of primitives applied left-to-right."""
from typing import List, Tuple, Optional, Set
from itertools import product
from .primitives import Grid, Primitive


class Program:
    def __init__(self, steps: List[Tuple[str, Primitive]]):
        self.steps = steps
        self.name = " -> ".join(s[0] for s in steps)

    def __call__(self, g: Grid) -> Optional[Grid]:
        x = g
        for _, fn in self.steps:
            x = fn(x)
            if x is None:
                return None
        return x

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        return f"Program({self.name!r})"


# ---------------------------------------------------------------------------
# Equivalence-pruning: canonical-form filter for the enumerative synthesizer.
#
# A program is non-canonical if it contains a subsequence that is either:
#   (a) immediately redundant: f after its own inverse collapses to identity
#   (b) a longer spelling of a shorter-named primitive (rotate_90^3 = rotate_270)
#
# Pruning these before consistency checking avoids burning budget on no-ops.
# ---------------------------------------------------------------------------

# Pairs (a, b) such that applying b immediately after a is identity (wasted step).
_INVERSE_PAIRS: Set[Tuple[str, str]] = {
    ("rotate_90",  "rotate_270"),
    ("rotate_270", "rotate_90"),
    ("rotate_180", "rotate_180"),
    ("flip_lr",    "flip_lr"),
    ("flip_ud",    "flip_ud"),
    ("transpose",  "transpose"),
    ("antitranspose", "antitranspose"),
    ("invert_colors", "invert_colors"),
    ("sort_rows",  "sort_rows"),
    # gravity: no self-inverse (applying twice is NOT identity — it's idempotent)
    # crop_to_content: idempotent (same result on already-cropped grids), not an inverse pair
}

# Forbidden substrings: longer ways to say a shorter primitive.
# These appear as consecutive name sub-sequences anywhere in the program.
_FORBIDDEN_SUBSEQUENCES: List[Tuple[str, ...]] = [
    # rotate_90 × 3 = rotate_270 (shorter spelling)
    ("rotate_90", "rotate_90", "rotate_90"),
    # rotate_270 × 3 = rotate_90
    ("rotate_270", "rotate_270", "rotate_270"),
    # rotate_90 × 2 = rotate_180
    ("rotate_90", "rotate_90"),
    # rotate_270 × 2 = rotate_180
    ("rotate_270", "rotate_270"),
    # rotate_180 is its own two-step: but rotate_90+rotate_270=id is caught by inverse pairs
    # flip_lr after rotate_90 twice = flip_lr after rotate_180: not prunable without renaming
    # gravity: applying twice is idempotent (gravity_down∘gravity_down = gravity_down)
    ("gravity_down",  "gravity_down"),
    ("gravity_up",    "gravity_up"),
    ("gravity_left",  "gravity_left"),
    ("gravity_right", "gravity_right"),
    # crop_to_content is idempotent
    ("crop_to_content", "crop_to_content"),
    # mirror_h∘mirror_h would produce original but with doubled width twice — complex;
    # keep it simple: just prune exact self-repeats for idempotent-ish ops.
    ("mirror_h", "mirror_h"),
    ("mirror_v", "mirror_v"),
]


def is_canonical(names: Tuple[str, ...]) -> bool:
    """Return True if this sequence of primitive names is in canonical (non-redundant) form.

    A program is non-canonical if it contains:
    - Any adjacent (a, b) pair that is an identity (a ∘ b = id).
    - Any forbidden subsequence (longer spelling of a shorter primitive, or idempotent repeat).
    """
    # Check adjacent inverse pairs
    for a, b in zip(names, names[1:]):
        if (a, b) in _INVERSE_PAIRS:
            return False

    # Check forbidden subsequences (any length ≥ 2)
    for subseq in _FORBIDDEN_SUBSEQUENCES:
        k = len(subseq)
        for i in range(len(names) - k + 1):
            if names[i:i + k] == subseq:
                return False

    return True


def enumerate_programs(prims: dict, max_depth: int = 2):
    """Yield canonical programs in increasing length order.

    Filters non-canonical programs (inverse pairs, redundant subsequences) before
    yielding — avoids spending budget on programs equivalent to shorter ones.
    """
    names = list(prims)
    # depth 1: always canonical
    for n in names:
        yield Program([(n, prims[n])])
    # depth 2+: filter via is_canonical
    for d in range(2, max_depth + 1):
        for combo in product(names, repeat=d):
            if is_canonical(combo):
                yield Program([(n, prims[n]) for n in combo])
