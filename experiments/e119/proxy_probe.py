"""E119 Phase 0 — deterministic probe: does any macro selection signal carry directional
information on the zero-reward procedure-walls? No LLM; reuses existing perception/search/DSL."""
import heapq
from collections import deque
import numpy as np
from e119 import slm, planner, perceive, solve


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


def probe_game(game, budget, max_preds=20):
    """Probe one game: blind-search stats, predicate satisfiability, and the best depth/novelty
    gain from pursuing a satisfiable-but-false-at-start predicate (the directionality test)."""
    game.reset()
    trans = perceive.probe(game)
    frames = [t["before"] for t in trans] + [t["after"] for t in trans]
    mask = perceive.status_mask(frames)
    key_fn = lambda f, m=mask: perceive.state_key(f, m)
    cands = solve._candidates_fn(game, mask)
    start = trans[0]["before"]

    blind = search_stats(solve._PrefixGame(game, []), cands, key_fn, budget, None)

    preds = enumerate_predicates(frames)
    satisf = scan_satisfiable(preds, frames)
    gradient = [p for p in satisf if not slm.compile_predicate(p)(start)][:max_preds]

    best_depth_gain, best_novel_gain = 0, 0.0
    for p in gradient:
        score = lambda f, pp=p: 1.0 if slm.compile_predicate(pp)(f) else 0.0
        g = search_stats(solve._PrefixGame(game, []), cands, key_fn, budget, score)
        best_depth_gain = max(best_depth_gain, g["max_depth"] - blind["max_depth"])
        novel = max(0, g["states"] - blind["states"])
        best_novel_gain = max(best_novel_gain, (novel / blind["states"]) if blind["states"] else 0.0)

    avail = list(getattr(game, "avail", [1, 2, 3, 4, 5, 7]))
    modality = "click" if avail == [6] else ("dir" if 6 not in avail else "mixed")
    return {"game": getattr(game, "gid", type(game).__name__), "modality": modality,
            "n_satisfiable": len(satisf), "n_gradient": len(gradient), "blind": blind,
            "best_depth_gain": int(best_depth_gain), "best_novel_gain": round(best_novel_gain, 3),
            "novelty_headroom": not blind["frontier_exhausted"]}


def decide_go(rows, primary="g50t"):
    """GO iff the primary game shows either a non-flat subgoal proxy OR novelty headroom.
    Default the macro selection signal to novelty when both qualify (brainstorm decision)."""
    pr = next((r for r in rows if r["game"] == primary), None)
    if pr is None:
        return {"go": False, "signal": "none", "reason": f"primary {primary} missing from rows"}
    subgoal = pr["n_satisfiable"] >= 1 and (pr["best_depth_gain"] >= 2 or pr["best_novel_gain"] >= 0.10)
    novelty = bool(pr["novelty_headroom"])
    signal = "novelty" if novelty else ("subgoal" if subgoal else "none")
    return {"go": bool(novelty or subgoal), "signal": signal,
            "reason": (f"{primary}: subgoal={subgoal} (depth_gain={pr['best_depth_gain']}, "
                       f"novel_gain={pr['best_novel_gain']}, n_sat={pr['n_satisfiable']}), "
                       f"novelty_headroom={novelty}")}
