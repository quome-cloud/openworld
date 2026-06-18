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
