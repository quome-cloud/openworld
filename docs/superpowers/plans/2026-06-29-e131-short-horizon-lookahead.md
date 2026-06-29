# E131 Short-Horizon Lookahead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A source-free ARC-AGI-3 solver that never infers a whole-level goal — instead it exhaustively looks **2-3 frames ahead** over the small action space, scores leaves by **level-delta (primary) + novelty (tie-break)**, commits the best first action, and recedes the horizon. The deterministic env + a **fast-forward transition cache** make the tiny tree (≤7³ leaves) essentially free.

**Architecture:** The insight is that the explosion is in the *horizon*, not the *branching*: per step it's ~4-7 actions, not billions. `lookahead.py` holds a `FrontierCache` (memoized `(state_key, action) → (next_key, next_levels)` + `path_to[key]` for replay) and `best_sequence` (beam-limited depth-d expansion using the cache where known, replaying the deterministic env where unknown). `solve_lookahead` is the receding-horizon driver. Reuses E130 `perception.extrospect` for state keys; no new perception.

**Tech Stack:** Python 3.9 (`~/.arcv/bin/python`), stdlib + numpy. Reuses `experiments/e130/perception.py`, `experiments/arc3_sandbox.py`, `scripts/capture_arc_run.py`, `scripts/autobank_sourcefree.py`.

**Research basis:** `papers/arc-3/research/ARC-AGI Short-Horizon Lookahead Plan.md` (committed). It grounds
this design and refines it: (i) the architecture *is* **Lazy Receding-Horizon A\*** — the FrontierCache
makes env interaction a lazily-evaluated edge (replay only on a cache miss), exactly LRA\*'s
edge-vs-graph-search trade-off; (ii) the leaf score is `level_delta + β·novelty` with a *small* `β` so
novelty only steers when level-delta is stagnant — our lexicographic `(level_delta, novelty)` tuple is
the `β→0` realization (kept: it is exactly "novelty breaks ties"); (iii) the orthogonal lever the
research flags as what made EWM/baseline1 succeed is a **closed-loop harness** (persistent workspace +
AST/type-check/unit-test execution feedback) — out of scope for E131's planner but noted as the next
direction if short-horizon lookahead alone stalls.

## Global Constraints

- **Source-free:** act only on the game stub / `SandboxGame` (`{frame, levels, win, avail, done}`); reason from frames; bank through the audit+replay+OpenWorld gate.
- **Deterministic replay only:** the env has no clone/checkpoint; to reach a state, `reset()` then replay its action path. Reuse ONE env (`arc.make` is slow). `path_to[key]` stores a replayable prefix.
- **State key = the masked object-state key** from `experiments/e130/perception.extrospect(...).key` — assumed Markov (the key determines dynamics), the same assumption E130's WorldModel makes.
- New code under `experiments/e131/` + `tests/e131/`; numpy allowed; no `openworld/` core changes.
- Run tests: `~/.arcv/bin/python -m pytest tests/e131/ -v`. Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: FrontierCache + leaf value

**Files:**
- Create: `experiments/e131/__init__.py`, `experiments/e131/lookahead.py`
- Test: `tests/e131/__init__.py`, `tests/e131/test_cache.py`

**Interfaces:**
- `value(frontier_levels, leaf_levels, leaf_key, seen) -> (int, int)` — returns `(level_delta, novelty)` = `(leaf_levels - frontier_levels, 0 if leaf_key in seen else 1)`. Compared as a tuple (level-delta dominates).
- class `FrontierCache` with:
  - `.seen: set` (state keys ever observed), `.trans: dict` (`(key,action) -> (next_key, next_levels)`), `.path_to: dict` (`key -> action_list` from reset).
  - `get(key, action) -> (next_key, next_levels) | None`
  - `put(key, action, next_key, next_levels, path_to_next)` — records the transition, adds `next_key` to `seen`, sets `path_to[next_key]` if absent.

- [ ] **Step 1: Write the failing test**

```python
# tests/e131/test_cache.py
from experiments.e131.lookahead import FrontierCache, value


def test_value_prefers_level_delta_then_novelty():
    seen = {("a",)}
    assert value(2, 3, ("b",), seen) == (1, 1)     # +1 level, novel
    assert value(2, 2, ("b",), seen) == (0, 1)     # no level, novel
    assert value(2, 2, ("a",), seen) == (0, 0)     # no level, seen
    assert value(2, 3, ("a",), seen) > value(2, 2, ("b",), seen)   # level-delta dominates novelty


def test_cache_roundtrip_and_seen():
    c = FrontierCache()
    assert c.get(("s0",), 1) is None
    c.put(("s0",), 1, ("s1",), 5, [[1]])
    assert c.get(("s0",), 1) == (("s1",), 5)
    assert ("s1",) in c.seen
    assert c.path_to[("s1",)] == [[1]]
    c.put(("s0",), 1, ("s1",), 5, [[9]])           # path_to not overwritten once set
    assert c.path_to[("s1",)] == [[1]]
```

- [ ] **Step 2: Run** `~/.arcv/bin/python -m pytest tests/e131/test_cache.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e131/__init__.py
# (empty package marker)
```
```python
# tests/e131/__init__.py
# (empty package marker)
```
```python
# experiments/e131/lookahead.py
"""Short-horizon lookahead: the explosion is in the HORIZON, not the branching. Per step there are
only ~4-7 actions, so don't infer a whole-level goal -- exhaustively look 2-3 frames ahead over the
deterministic env, score leaves by (level-delta, novelty), commit the best first action, recede.
The FrontierCache is the fast-forward memory: memoized transitions + a replayable path per state."""


def value(frontier_levels, leaf_levels, leaf_key, seen):
    """Leaf score: level-delta dominates; novelty breaks ties. Compared as a tuple."""
    return (leaf_levels - frontier_levels, 0 if leaf_key in seen else 1)


class FrontierCache:
    def __init__(self):
        self.seen = set()
        self.trans = {}        # (state_key, action) -> (next_key, next_levels)
        self.path_to = {}      # state_key -> action list from reset() that reaches it

    def get(self, key, action):
        return self.trans.get((key, action))

    def put(self, key, action, next_key, next_levels, path_to_next):
        self.trans[(key, action)] = (next_key, next_levels)
        self.seen.add(next_key)
        if next_key not in self.path_to:
            self.path_to[next_key] = [list(a) for a in path_to_next]
```

- [ ] **Step 4: Run** → 2 pass.
- [ ] **Step 5: Commit** `git add experiments/e131/__init__.py experiments/e131/lookahead.py tests/e131/__init__.py tests/e131/test_cache.py && git commit -m "E131 Task 1: FrontierCache + (level-delta, novelty) leaf value

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

---

### Task 2: Depth-d beam expansion (the lookahead core)

**Files:**
- Modify: `experiments/e131/lookahead.py` (add `_act`, `_replay_to`, `best_sequence`)
- Test: `tests/e131/test_lookahead.py`

**Interfaces:**
- `best_sequence(env, perceive, frontier_path, frontier_key, frontier_levels, avail, cache, depth=3, beam=4) -> (first_action | None, value_tuple)`:
  - Exhaustively expands action sequences up to `depth` from the frontier state. Each node uses `cache.get` when known; otherwise `_replay_to(env, frontier_path + suffix)` then `_act(env, a)` to discover + `cache.put` the transition (with `path_to_next = frontier_path + suffix + [a]`).
  - Tracks the best leaf by `value(frontier_levels, leaf_levels, leaf_key, cache.seen)`; returns the FIRST action of the best sequence and that value. Keeps only the top-`beam` nodes by value at each depth (beam>1 avoids greedy local optima). Returns `(None, (0,0))` if `avail` is empty.
- Helpers: `_act(env, a)` does `env.step(6,a[1],a[2]) if a[0]==6 else env.step(a[0])`, returns `not env.done`; actions are `[a]` or `[6,x,y]`. `_replay_to(env, path)` = `env.reset()` then replay `path`.

- [ ] **Step 1: Write the failing test**

```python
# tests/e131/test_lookahead.py
import numpy as np
from experiments.e131.lookahead import best_sequence, FrontierCache
from experiments.e130 import perception as P


class TwoStepGame:
    """Level rises ONLY after the ordered pair (action 1, then action 2). Neither alone helps, so a
    1-step greedy sees no signal but a depth-2 lookahead finds it via level-delta."""
    def __init__(self): self.s = 0; self.levels = 0; self.done = False; self.avail = [1, 2, 3]
    @property
    def frame(self):
        f = np.zeros((8, 8), dtype=int); f[0, self.s] = 5; return f
    def reset(self): self.s = 0; self.levels = 0; self.done = False; return self.frame
    def step(self, a, x=None, y=None):
        self.s = 1 if (a == 1 and self.s == 0) else (2 if (a == 2 and self.s == 1) else 0)
        if self.s == 2: self.levels = 1
        return self.frame


def test_depth2_finds_the_ordered_pair():
    g = TwoStepGame(); perceive = lambda fr: P.extrospect(fr, avail=[1, 2, 3])
    cache = FrontierCache()
    s = perceive(g.frame); cache.seen.add(s.key)
    first, val = best_sequence(g, perceive, [], s.key, 0, [1, 2, 3], cache, depth=2, beam=4)
    assert first == [1]            # the only first move that (via action 2) raises the level
    assert val[0] == 1             # a +1 level-delta leaf was found within the horizon


def test_returns_novelty_move_when_no_level_in_horizon():
    # depth 1: no pair reachable, so level-delta is 0 everywhere; it should still return a real action
    g = TwoStepGame(); perceive = lambda fr: P.extrospect(fr, avail=[1, 2, 3])
    cache = FrontierCache(); s = perceive(g.frame); cache.seen.add(s.key)
    first, val = best_sequence(g, perceive, [], s.key, 0, [1, 2, 3], cache, depth=1, beam=4)
    assert first is not None and val[0] == 0
```

- [ ] **Step 2: Run** → fail (`best_sequence` undefined).

- [ ] **Step 3: Implement** `_act`, `_replay_to`, and `best_sequence` (BFS over depth with the cache+replay discovery and beam pruning, as specified in Interfaces). State keys come from `perceive(env.frame).key` after a real step.

- [ ] **Step 4: Run** `~/.arcv/bin/python -m pytest tests/e131/test_lookahead.py -v` → 2 pass.
- [ ] **Step 5: Commit** the two files.

---

### Task 3: Receding-horizon driver + runner + sweep

**Files:**
- Modify: `experiments/e131/lookahead.py` (add `Result`, `solve_lookahead`)
- Create: `experiments/e131_lookahead.py`, `scripts/sweep_lookahead.sh`
- Test: `tests/e131/test_solve.py`

**Interfaces:**
- `solve_lookahead(env, perceive, seed_actions, win, depth=3, beam=4, budget=4000) -> Result` — replay `seed_actions`; loop: `best_sequence(...)` from the current state → commit its first action via `_act` → append to `actions`, update the frontier state/levels, mark `seen`; bank `best_levels`/`best_actions` on a level rise; stop on win, budget, `done`, or `K=20` consecutive cycles with no new state AND no level gain. Returns `Result(best_levels, best_actions, cycles, real_steps, cache_size)`.
- `experiments/e131_lookahead.py`: `solve(game, budget)` mirrors `experiments/e130_shu_cycle.py` `solve` — one `SandboxGame`, seed from `arc3_fullgame_sourcefree.json`, run `solve_lookahead`, write `scratch_arc/lh_<game>/solved.json` + `run_meta.json`, call `scripts/capture_arc_run.py <game> <wd> lookahead e131_lookahead.py`.
- `scripts/sweep_lookahead.sh`: pool over the unsolved walls, `lh_` prefix, banks via `SF_WD_PREFIX=lh_` autobank (mirrors `scripts/sweep_shu_cycle.sh`).

- [ ] **Step 1: Write the failing test**

```python
# tests/e131/test_solve.py
import numpy as np
from experiments.e131.lookahead import solve_lookahead
from experiments.e130 import perception as P
from tests.e131.test_lookahead import TwoStepGame


def test_solve_chains_to_the_win():
    g = TwoStepGame()
    res = solve_lookahead(g, lambda fr: P.extrospect(fr, avail=[1, 2, 3]),
                          seed_actions=[], win=1, depth=2, beam=4, budget=50)
    assert res.best_levels == 1
    assert res.best_actions[:2] == [[1], [2]]      # discovered + committed the ordered pair
```

- [ ] **Step 2: Run** → fail.
- [ ] **Step 3: Implement** `Result` + `solve_lookahead` in `lookahead.py`; write `experiments/e131_lookahead.py` and `scripts/sweep_lookahead.sh` (mirror the E130 `solve`/sweep structure exactly, swapping in `solve_lookahead`).
- [ ] **Step 4: Run** `~/.arcv/bin/python -m pytest tests/e131/ -v` → all pass. Smoke the runner import: `~/.arcv/bin/python -c "import experiments.e131_lookahead"`.
- [ ] **Step 5: Commit** the files (`chmod +x scripts/sweep_lookahead.sh`).

---

## Self-Review

**Spec coverage:** fast-forward memory → `FrontierCache` (Task 1); 2-3 frame exhaustive lookahead → `best_sequence` depth-d beam (Task 2); level-delta+novelty value → `value` (Task 1) used in Task 2; receding horizon + bank + capture → `solve_lookahead` + runner (Task 3). The real-game run is via `solve`/the sweep after Task 3 — measured honestly vs E130/E129.

**Placeholder scan:** none — all steps carry complete code or precise Interface specs.

**Type consistency:** state keys are `extrospect(...).key` tuples throughout; actions are `[a]`/`[6,x,y]` lists; `cache.path_to` stores reset-relative action lists; `value` returns a 2-tuple compared lexicographically everywhere.

**Honest risk (on the record):** a short horizon sees only a *local* value; if a win needs a long exact procedure with no level-delta or novelty signal within 2-3 frames, greedy/beam commit can stall (the same deceptive-wall failure mode as Go-Explore). The bet is that exhaustive short-horizon + level-delta catches near-term wins and chains locally-good moves; beam>1 mitigates local optima. Reported either way.
