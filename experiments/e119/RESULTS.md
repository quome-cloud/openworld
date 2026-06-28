# E119 — results & runbook (first real run)

Companion to `STRATEGY.md` (the design). This records the **first end-to-end run** of the
neuro-symbolic SLM solver against the live ARC-AGI-3 env with local Ollama models, the two
bugs that run surfaced, and the measured outcome.

Canonical results: `experiments/results/e119_slm_solver.json` (consolidated). Raw per-rung
files: `e119_rung_{search,slm_*}.json`; full 25-game map: `e119_reachability.json`; banked
verified solutions: `e119_logs/{vc33,lp85}_solved.json`.

## Headline

- **The harness works and solves levels.** Control (blind BFS + replay-verification, no model)
  solves **10/25 games** ≥1 level (click 4/6, dir 2/6, mixed 4/13).
- **SLM delta on the pilot = 0.** All four small models (`qwen2.5-coder:7b`, `qwen2.5:7b`,
  `gemma3`, `llama3.1:8b`) solve **exactly** what blind search solves, per game. The on-demand
  subgoal synthesis **neither helps nor hurts**.

This *confirms* E119's safety invariant ("the SLM only orders search; the env decides
correctness — a wrong/abstained subgoal costs speed, never a false solve") and *denies* its
bonus claim ("the model makes search faster") **on this pilot** — for the reasons below.

## Search vs SLM (levels per pilot game, budget 6000 nodes / depth 60)

| game | search | qwen2.5-coder:7b | qwen2.5:7b | gemma3 | llama3.1:8b |
|------|:---:|:---:|:---:|:---:|:---:|
| tn36 | 0 | 0 | 0 | 0 | 0 |
| ar25 | 0 | 0 | 0 | 0 | 0 |
| vc33 | 2 | 2 | 2 | 2 | 2 |
| lp85 | 1 | 1 | 1 | 1 | 1 |
| sk48 | 0 | 0 | 0 | 0 | 0 |
| **total** | **3** | **3** | **3** | **3** | **3** |

### Why delta = 0 (measured, not assumed)
1. The model frequently **abstains** (e.g. vc33: 6 samples didn't clear τ=0.5) → search runs
   unguided → identical to control.
2. When it commits, the predicate compiles to a **binary 0/1 frame score**, so best-first
   ordering over the *same* candidate set is flat-until-goal ≈ BFS. It never **prunes** the
   branching (click games have ~44 candidates/frame).
3. The reachable pilot levels (vc33 L1–2, lp85 L1) are **shallow enough that blind BFS already
   finds them** within budget — no headroom for a prior to add value.

## Reachability map (control rung, all 25 games)

10/25 solved ≥1 level: `vc33 2/7`, `lp85 1/8`, `r11l 1/6`, `ft09 1/6` (click); `ls20 1/7`,
`tu93 1/9` (dir); `sp80 1/6`, `lf52 1/10`, `su15 1/9`, `cd82 1/6` (mixed). Most cap at level 1;
only vc33 reaches level 2. See `e119_reachability.json`.

## Bugs found and fixed (both committed)
1. **Honest zero-solve tripped the verification assert.** `verified = … and reached > 0`
   marked a legitimate 0-solve "unverified", so any game the control couldn't solve aborted the
   whole pilot. Fixed via `_is_honest()` (driver). Commit `dc17a92`.
2. **Checkpoint-retaining `reset()` zeroed real solves.** The arc env's `reset()` restores the
   board to the current-level checkpoint but **retains `levels_completed`**, so `replay_levels`'
   delta `mx - base` collapsed to 0 on the reused env — a genuine ls20/vc33/lp85 solve was
   reported as 0. Fixed by verifying on a **fresh** env (mirrors `arc3_harness.replay`). Commit
   `266e6eb`. This was masking real solves; it is why the *first* sweep looked like 0-vs-0.

## How to run (validated on macOS, Python 3.13)

The Cowork runbook was broadly correct; corrections baked in below.

```bash
# 1. Env: arc_agi + arcengine ARE on PyPI now (the "download-only" note is stale).
python3 -m venv .venv && .venv/bin/pip install arc-agi arcengine numpy && .venv/bin/pip install -e .

# 2. Ollama: use the official app build (the Homebrew formula 0.30.7 ships without the
#    llama-server backend -> HTTP 500 on every model). brew install --cask ollama (or app).
ollama pull qwen2.5-coder:7b qwen2.5:7b gemma3   # llama3.1:8b too

# 3. Run (the harness lib is on scratch_arc/agent; experiments/ self-adds to path).
export PYTHONPATH="$PWD/scratch_arc/agent"
.venv/bin/python experiments/e119_slm_solver.py --mode search
.venv/bin/python experiments/e119_slm_solver.py --mode slm --model qwen2.5-coder:7b
# NOTE: do NOT `git worktree add` this branch — it is already checked out; run in place.
```

The env runs **locally** after a one-time ~1.3s Arcade init (`step()` sub-ms), so a 6000-node
search is ~30s/game.

## Next levers (to give the SLM a fair chance at a positive delta)
1. **Games with headroom** — sweep the harder reachable games / the 15 currently-0 games where
   blind BFS struggles, so ordering can break ties within budget.
2. **Dense scoring / pruning** — filter candidates or give a distance-to-goal gradient instead
   of binary 0/1 (the design intends this; the code does not do it yet).
3. **Beyond level 1** — needs a true-reset per level (fresh env) for the multi-level search loop,
   since `reset()` is checkpoint-based.
4. Minor: a malformed-JSON sample escapes `best_of_n` (only `behavior_fn` errors are discarded);
   one game errored on gemma3 in the pre-fix sweep. Discard `sample_fn` parse errors too.
