"""E133 EWM planning toolkit -- self-contained (stdlib only, no cross-imports), so it drops into an
audited source-free workspace as a SOLVER helper (reads no game code). It combines the prior approaches
for the EWM agent to use:

  * plan_in_model -- deep beam lookahead over ANY predict() you give it (your SYNTHESIZED generalizing
    model, or the tabular WorldSim). Pure: no real env, perfect backtracking, arbitrary horizon. This is
    how you find the exact action sequence (the HOW) once you've reasoned the win (the WHAT).
  * WorldSim       -- a default TABULAR predict (learns observed transitions). Fine for the explored
    neighbourhood; for transitions you have NOT observed, write your OWN predict() that GENERALIZES
    (object-relative: "avatar + UP -> avatar.y-1", "brush click -> stamp at orientation"), so a level-N
    model predicts level N+1 (ARC levels escalate the same mechanic).
  * salient_clicks -- salience-prioritised click targets (small components + rare colours first), to
    explore the win-relevant transitions fast instead of clicking everywhere.
  * _act / _replay_to -- exception-safe real-env step/replay (forward play + reach a state to verify).

Recipe: explore forward (salience-guided) -> synthesize a generalizing predict() + reason the win ->
plan_in_model(predict, ...) to deep-search for the achieving sequence -> VERIFY on the real env (levels
must rise) -> refine predict() on any sim-vs-real mismatch -> chain to the win.
"""


def value(start_levels, leaf_levels, leaf_key, seen):
    """Leaf score: level-delta dominates; novelty (unseen key) breaks ties. Compared as a tuple."""
    return (leaf_levels - start_levels, 0 if leaf_key in seen else 1)


def plan_in_model(predict, start_key, start_levels, actions_of, seen=None, depth=8, beam=8):
    """PURE deep beam lookahead over predict() -- no real env (so backtracking is free and depth is
    unbounded). Args:
      predict(state_key, action) -> (next_key, next_levels) | None   (None = unknown/frontier)
      actions_of(state_key) -> list of candidate actions ([a] or [6,x,y])
    Returns (best_plan, value_tuple, best_leaf_key): the FULL action path to the best-scoring reachable
    state. Commit the whole sub-plan (it's verified in the model), then check it on the real env."""
    seen = set(seen if seen is not None else [start_key])
    best_path, best_val, best_leaf = [], value(start_levels, start_levels, start_key, seen), start_key
    beam_nodes = [(start_key, start_levels, [])]
    visited = {start_key}
    for _ in range(depth):
        cands = []
        for key, lv, path in beam_nodes:
            for a in actions_of(key):
                nxt = predict(key, a)
                if nxt is None:
                    continue
                nk, nl = nxt
                v = value(start_levels, nl, nk, seen)
                np_ = path + [a]
                if v > best_val:
                    best_val, best_path, best_leaf = v, np_, nk
                if nk not in visited:
                    visited.add(nk)
                    cands.append((v, nk, nl, np_))
        if not cands:
            break
        cands.sort(key=lambda x: x[0], reverse=True)
        beam_nodes = [(k, lv, p) for (v, k, lv, p) in cands[:beam]]
    return best_path, best_val, best_leaf


class WorldSim:
    """Default TABULAR predict over OBSERVED transitions. Use `sim.predict` as plan_in_model's predict
    for the explored region; write your own generalizing predict() for the rest."""
    def __init__(self):
        self.trans = {}
        self.seen = set()

    def learn(self, key, action, next_key, next_levels):
        self.trans[(key, tuple(action))] = (next_key, int(next_levels))
        self.seen.add(key); self.seen.add(next_key)

    def predict(self, key, action):
        return self.trans.get((key, tuple(action)))


def salient_clicks(objects, max_size=16, top=12):
    """Salience-prioritised click candidates: small connected components, rarest colours first."""
    from collections import Counter
    cc = Counter(o["color"] for o in objects)
    small = [o for o in objects if o.get("size", 99) <= max_size]
    small.sort(key=lambda o: (o["size"], cc[o["color"]]))
    return [[6, int(o["x"]), int(o["y"])] for o in small[:top]]


def _act(env, a):
    """Exception-safe real-env step. Returns True if it stepped without raising and the env is not done."""
    try:
        env.step(6, a[1], a[2]) if a[0] == 6 else env.step(a[0])
    except Exception:
        return False
    return not getattr(env, "done", False)


def _replay_to(env, path):
    """Reset + replay path on the real env. Returns True if the whole path replayed cleanly."""
    try:
        env.reset()
    except Exception:
        return False
    for a in path:
        if not _act(env, a):
            return False
    return True
