"""E119 Phase 0 — deterministic probe: does any macro selection signal carry directional
information on the zero-reward procedure-walls? No LLM; reuses existing perception/search/DSL."""
import heapq
from collections import deque
import numpy as np
from e119 import slm, planner


def enumerate_predicates(frames):
    """All reach/count/align predicates over the colors and per-color counts observed in `frames`."""
    grids = [np.asarray(f).reshape(64, 64) for f in frames]
    colors = sorted({int(c) for g in grids for c in np.unique(g)})
    preds = [{"type": "reach", "color": c} for c in colors]
    for c in colors:
        for k in sorted({int((g == c).sum()) for g in grids}):
            for op in ("==", ">=", "<="):
                preds.append({"type": "count", "color": c, "op": op, "k": k})
    for i, a in enumerate(colors):
        for b in colors[i + 1:]:
            preds.append({"type": "align", "a": a, "b": b})
    return preds


def scan_satisfiable(preds, frames):
    """Subset of `preds` ever true on some frame in `frames`."""
    return [p for p in preds if slm.satisfiable(p, frames)]


def search_stats(game, candidates_fn, key_fn, budget, score_fn=None):
    """Instrumented mirror of planner.search_level. BFS if score_fn is None, else best-first.
    Returns exploration stats; frontier_exhausted=True means the reachable state space was
    fully explored within budget (no novelty headroom)."""
    game.reset(); base = game.levels
    seen = {key_fn(game.frame)}
    nodes = 0; max_depth = 0; solved = False
    if score_fn is None:
        frontier = deque([[]]); pop = frontier.popleft; push = frontier.append
    else:
        counter = 0; heap = [(-score_fn(game.frame), 0, [])]
        def pop(): return heapq.heappop(heap)[2]
        def push(seq):
            nonlocal counter
            counter += 1
            f, _, _ = planner._frame_after(game, seq)
            heapq.heappush(heap, (-score_fn(f), counter, seq))
        frontier = heap
    while frontier and nodes < budget["max_nodes"]:
        seq = pop()
        if len(seq) >= budget["max_depth"]:
            continue
        frame, _, _ = planner._frame_after(game, seq)
        for act in candidates_fn(frame):
            nodes += 1
            child = seq + [act]
            f2, levels2, _ = planner._frame_after(game, child)
            if levels2 > base:
                solved = True; break
            k = key_fn(f2)
            if k in seen:
                continue
            seen.add(k); push(child); max_depth = max(max_depth, len(child))
            if nodes >= budget["max_nodes"]:
                break
        if solved or nodes >= budget["max_nodes"]:
            break
    return {"nodes": nodes, "states": len(seen), "max_depth": max_depth,
            "frontier_exhausted": (len(frontier) == 0 and not solved), "solved": solved}
