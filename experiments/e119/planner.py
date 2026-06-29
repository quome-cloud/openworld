"""Env-ground-truth search over the replay-only ARC env. Correctness via replay, never the model."""
import heapq
from collections import deque


def replay_levels(game, actions):
    """Replay an action list from reset(); return (max levels reached, done)."""
    game.reset(); base = game.levels; mx = base
    for act in actions:
        game.step(*act)
        if game.levels > mx: mx = game.levels
        if game.done: break
    return mx - base, game.done


def _frame_after(game, actions):
    game.reset()
    for act in actions:
        game.step(*act)
        if game.done: break
    return game.frame, game.levels, game.done


def search_level(game, candidates_fn, key_fn, budget, score_fn=None):
    """Find an action sequence that raises levels by >=1. BFS, or best-first when score_fn given.
    Each node is an action prefix; we replay it from reset() to expand (env is replay-only)."""
    game.reset(); base = game.levels
    start_frame = game.frame
    seen = {key_fn(start_frame)}
    nodes = 0
    if score_fn is None:
        frontier = deque([[]])
        pop = frontier.popleft
        push = frontier.append
    else:
        counter = 0
        heap = [(-score_fn(start_frame), 0, [])]
        def pop():
            return heapq.heappop(heap)[2]
        def push(seq):
            nonlocal counter
            counter += 1
            f, _, _ = _frame_after(game, seq)
            heapq.heappush(heap, (-score_fn(f), counter, seq))
        frontier = heap
    while frontier and nodes < budget["max_nodes"]:
        seq = pop()
        if len(seq) >= budget["max_depth"]:
            continue
        frame, _, _ = _frame_after(game, seq)
        for act in candidates_fn(frame):
            nodes += 1
            child = seq + [act]
            f2, levels2, done2 = _frame_after(game, child)
            if levels2 > base:
                return child
            k = key_fn(f2)
            if k in seen:
                continue
            seen.add(k)
            push(child)
            if nodes >= budget["max_nodes"]:
                break
    return None
