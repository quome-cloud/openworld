"""E134 — Composite multi-perception fuse + fidelity-SELECT combiner.

composite_key  — concatenate all lens keys into one non-aliasing tuple.
fidelity       — measure Markov-consistency of a single lens over observed transitions.
select_lens    — ConsensusTransition(mode='select'): pick the most-consistent lens.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

try:
    from experiments.e134.perceptors import LENSES, key_of
except ImportError:                              # flat agent workspace: perceptors.py sits alongside
    from perceptors import LENSES, key_of


# ---------------------------------------------------------------------------
# Composite key
# ---------------------------------------------------------------------------

def composite_key(frame, lenses: Optional[Dict[str, Callable]] = None) -> tuple:
    """Concatenate key_of(fn(frame)) for every lens in sorted name order.

    Two frames differ in the composite key iff they differ under ANY lens.
    Using sorted order guarantees determinism regardless of dict-insertion order.
    """
    if lenses is None:
        lenses = LENSES
    result: list = []
    for name in sorted(lenses):
        fn = lenses[name]
        k = key_of(fn(frame))
        result.append(k)
    return tuple(result)


# ---------------------------------------------------------------------------
# Fidelity
# ---------------------------------------------------------------------------

def fidelity(transitions: List[Tuple], fn: Callable) -> float:
    """Measure the Markov-consistency of lens *fn* over observed transitions.

    transitions — list of (frame, action, next_frame).

    Build the table (key_of(fn(frame)), action) -> set of key_of(fn(next_frame)).
    Consistency = fraction of (key, action) pairs that map to exactly ONE next-key.

    Degeneracy penalty: if *fn* yields fewer than 2 distinct state-keys across all
    frames in *transitions*, return 0.0 (a lens that collapses everything is useless
    even if "consistent").
    """
    # Gather all keys from both source and target frames.
    table: dict = {}
    all_keys: set = set()

    for frame, action, next_frame in transitions:
        src_key = key_of(fn(frame))
        dst_key = key_of(fn(next_frame))
        all_keys.add(src_key)
        all_keys.add(dst_key)
        pair = (src_key, action)
        if pair not in table:
            table[pair] = set()
        table[pair].add(dst_key)

    # Degenerate check.
    if len(all_keys) < 2:
        return 0.0

    # Consistency = fraction of (key, action) pairs with exactly 1 next-key.
    if not table:
        return 0.0
    consistent = sum(1 for nexts in table.values() if len(nexts) == 1)
    return consistent / len(table)


# ---------------------------------------------------------------------------
# Select lens
# ---------------------------------------------------------------------------

def select_lens(
    transitions: List[Tuple],
    lenses: Optional[Dict[str, Callable]] = None,
) -> Tuple[str, Callable, float]:
    """Return the lens with the highest fidelity (SELECT, never average).

    Tie-break: among equal-fidelity lenses, prefer the one with MORE distinct
    state-keys (broader separation). Among still-equal lenses, the alphabetically
    first name wins (stable, reproducible).

    Returns (name, fn, score).
    """
    if lenses is None:
        lenses = LENSES

    best_name: Optional[str] = None
    best_fn: Optional[Callable] = None
    best_score: float = -1.0
    best_distinct: int = -1

    for name in sorted(lenses):
        fn = lenses[name]
        score = fidelity(transitions, fn)

        # Count distinct source keys (a proxy for state-space breadth).
        distinct = len({key_of(fn(frame)) for frame, _, _ in transitions})

        if (score > best_score) or (score == best_score and distinct > best_distinct):
            best_name = name
            best_fn = fn
            best_score = score
            best_distinct = distinct

    return best_name, best_fn, best_score
