"""Best-of-N with BEHAVIORAL clustering and a tau abstention gate (spec law: route to the executor)."""
from collections import defaultdict


def best_of_n(sample_fn, behavior_fn, n, tau):
    clusters = defaultdict(list)   # behavior signature -> [candidates]
    drawn = 0
    need = -(-int(tau * n) // 1)   # ceil(tau*n): min cluster size to clear tau
    need = int(need) if need >= 1 else 1
    for _ in range(n):
        try:
            c = sample_fn()
        except StopIteration:
            break
        drawn += 1
        try:
            sig = behavior_fn(c)
        except Exception:
            continue               # ungradeable candidate is discarded, not fatal
        clusters[sig].append(c)
        if len(clusters[sig]) >= need:    # adaptive stop: this cluster already clears tau
            break
    if not clusters:
        return None, {"agreement": 0.0, "clusters": 0, "samples": drawn}
    best_sig = max(clusters, key=lambda s: len(clusters[s]))
    top = len(clusters[best_sig])
    agreement = top / drawn if drawn else 0.0
    winner = clusters[best_sig][0] if agreement >= tau else None
    return winner, {"agreement": agreement, "clusters": len(clusters), "samples": drawn}
