# How OpenWorld solves tasks with zero training data

A common question: *"How can OpenWorld solve a task (e.g. cartpole-swingup, or
MiniGrid DoorKey) with **0 data**, when DreamerV3 needs tens of thousands of steps?"*

Short answer: **OpenWorld doesn't *learn* the dynamics — it's *given* them as verified
code. So there's nothing to train; it just plans.**

---

## The mechanism, in three steps

### 1. The "world model" is the dynamics, written as code
An OpenWorld world contains a `transition(state, action) -> state` function that *is*
the model. For cartpole, that's the standard physics equations of motion; for MiniGrid
DoorKey, it's the grid rules. Given a state and an action, it returns the next state —
**exactly**.

This code is hand-written, then **validated** (bit-for-bit against an independent
reference — e.g. cartpole vs. the Gym/Florian ODE, DoorKey vs. Farama `minigrid`). So
it's a faithful simulator of the *same* environment the learned models train on.

### 2. It solves by *planning*, not by acting-and-learning
With a correct model in hand, finding good actions is a **search** problem, run entirely
in imagination (no real environment needed):

- **Cartpole (continuous):** CEM-MPC (Cross-Entropy Method + Model-Predictive Control).
  Each step: sample hundreds of candidate action sequences, simulate every one forward
  through the verified model, score by a reward (pole upright + cart centered), keep the
  best, refit, execute the first action, replan. This *discovers* the swing-up maneuver —
  it is **not** a hardcoded solution.
- **MiniGrid (discrete):** breadth-first search over the verified transition to the goal,
  yielding the optimal plan (length 11 for DoorKey-6x6).

### 3. So "0 data" means "0 training transitions"
There is no learning phase. DreamerV3 starts knowing *nothing* about the dynamics and
must interact ~10k–150k times to *learn* a model before it can act well. OpenWorld skips
that because the model is already correct. The "intelligence" is split between
(a) the verified dynamics code and (b) a classical planner — **neither fits parameters
to data.**

---

## Worked examples

### Example 1 — A complete world from scratch (~15 lines, zero data)

The model is just a function. Here is an entire world — a reservoir you fill/drain —
plus a zero-data planner that finds how to reach exactly level 7:

```python
from collections import deque
from openworld import World, CodeTransition
from openworld.state import Action

CODE = '''
def transition(state, action):
    s = dict(state)
    if action["name"] == "fill":    s["level"] = min(10, s["level"] + 3)
    elif action["name"] == "drain": s["level"] = max(0,  s["level"] - 2)
    return s
'''
w = World(name="reservoir", description="a tank you fill (+3) or drain (-2)",
          initial_state={"level": 0}, actions=["fill", "drain"],
          rules=["fill adds 3 (cap 10); drain removes 2 (floor 0)"],
          transition=CodeTransition(CODE))

# Plan to exactly level 7 by breadth-first search over the verified transition:
seen, q = {0}, deque([({"level": 0}, [])])
while q:
    s, plan = q.popleft()
    if s["level"] == 7:
        print(plan); break                       # ['fill', 'fill', 'fill', 'drain']
    for a in w.actions:                           # 0 -> 3 -> 6 -> 9 -> 7
        ns = dict(w.transition.step(s, Action(a)))
        if ns["level"] not in seen:
            seen.add(ns["level"]); q.append((ns, plan + [a]))
```

No dataset, no training, no GPU — the plan is *computed* by searching a function. Swap
in any dynamics you can write (inventory, a state machine, an ODE) and the same pattern
solves it.

### Example 2 — The model is code you can *read* (MiniGrid DoorKey)

The DoorKey transition is plain, auditable Python (`experiments/minigrid_world.py`):

```python
# excerpt of MINIGRID_CODE
if name == "forward":
    nx, ny = s["x"] + dx, s["y"] + dy
    blocked = (... or (nx, ny) in walls
               or ((nx, ny) == door_pos and not s["door_open"])     # closed door blocks
               or ((nx, ny) == key_pos  and not s["has_key"]))      # key on ground blocks
    if not blocked:
        s["x"], s["y"] = nx, ny
elif name == "toggle":                                              # open the door...
    if (s["x"] + dx, s["y"] + dy) == door_pos and s["has_key"]:     # ...if holding the key
        s["door_open"] = True
```

Breadth-first search over it returns the **optimal 11-step plan** (grab key → open door
→ reach goal), with **0 training transitions**. And this isn't a toy re-implementation:
the transition is validated **bit-for-bit, 600/600 steps, against the real Farama
`minigrid`** (`bench/validate_minigrid.py`), so it's the *same* task DreamerV3 learns
from pixels — which needs ~10k interactions just to first succeed (experiment **E65**).

### Example 3 — Continuous control by planning (cartpole swing-up)

For physics, the "rules" are the equations of motion (`experiments/cartpole_world.py`):

```python
# excerpt: one Euler step of the standard cartpole ODE
temp  = (force + polemass_length * theta_dot**2 * sin(theta)) / total_mass
thacc = (g*sin(theta) - cos(theta)*temp) / (length * (4/3 - masspole*cos(theta)**2/total_mass))
s["theta"]     = theta + tau * theta_dot
s["theta_dot"] = theta_dot + tau * thacc
```

A planner (CEM-MPC) rolls candidate force-sequences through this verified model in
imagination, keeps the best, and executes — **swinging the pole up and balancing it 100%
of the time with zero data** (vs. a random controller at 0%). The swing-up maneuver is
*discovered* by search, not hardcoded (experiment **E67**).

### Example 4 — Zero data isn't only for *solving* — also for *prediction*

Because the transition is exact, **rollouts are bit-exact and have zero compounding
error** — you can ask "what happens if I take this sequence of actions?" and get the
right answer indefinitely:

```python
s = dict(w.initial_state)
for a in ["fill", "fill", "drain", "fill"]:
    s = dict(w.transition.step(s, Action(a)))   # 0 -> 3 -> 6 -> 4 -> 7, exactly
```

A *learned* next-state model (or an LLM asked to predict the next state) drifts after a
few steps; the verified code does not, by construction. OpenWorld's experiments quantify
this: verified code stays exact both in- and out-of-distribution, while the learned/LLM
proxy diverges (**E01/E10**).

---

## The honest trade-off (why this isn't a free lunch — and why we still test V-JEPA)

OpenWorld's zero-data solve works **only because the dynamics are cleanly writable as
code.** The cost didn't vanish — it *moved*:

| | Learned world model (Dreamer, V-JEPA) | OpenWorld (verified code) |
|---|---|---|
| Where knowledge lives | learned weights | hand-written, verified code |
| Up-front cost | collect data + train | write + verify the model |
| Result | approximate, after training | exact, instant, zero-shot |
| Guarantee | none | correctness (validated) |
| Works when… | you have data / can't write the rules | you can write the dynamics |

Where you *can* write the dynamics, verification buys an exact, instant, zero-shot
solution **plus a correctness guarantee**. Where you *can't* — messy real-world
pixels/video — the learned and perceptual models are your only option. That boundary is
exactly what the head-to-head experiments measure:

- **E65** (MiniGrid DoorKey): OpenWorld 0-shot vs. DreamerV3-from-pixels vs. V-JEPA.
- **E67** (cartpole-swingup): the same three species on continuous control — the learned
  models' home turf — where OpenWorld *still* solves with zero data.

The point is not "OpenWorld is smarter with no data." It's that the knowledge lives in
**verified code instead of learned weights**, and a planner turns a correct model into
correct actions for free.

---

*See `experiments/cartpole_world.py` / `experiments/e67_cartpole_bench.py` and
`experiments/minigrid_world.py` for the worlds, and
`docs/World Models_ Learned vs. Symbolic.md` for the broader genus/species framing.*
