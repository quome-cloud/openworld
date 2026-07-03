---
license: mit
pretty_name: ARC-AGI-3 Source-Free Solving Traces (Hybrid World Models)
task_categories:
  - reinforcement-learning
  - other
tags:
  - arc-agi
  - agents
  - world-models
  - reasoning
  - reproducibility
  - source-free
  - claude
language:
  - en
size_categories:
  - n<1K
configs:
  - config_name: runs
    data_files: runs.jsonl
---

# ARC-AGI-3 Source-Free Solving Traces — Hybrid World Models

A reproducible, fully-annotated dataset of **agent and algorithmic attempts to solve interactive
ARC-AGI-3 games source-free**, produced by the OpenWorld "hybrid world models" pipeline. Each record is one
**run** (an attempt at one game by one method) carrying the exact **prompt**, a pointer to the full
structured **transcript** (every model message + tool call), the **model / version / effort** that produced
it, token usage and cost, host + git provenance, the **source-free integrity audit**, and a **verified
outcome** (replay against the real engine + a round-trip through an OpenWorld `World`).

The intent is to let others **see and reproduce** how these games were solved — and to isolate artifacts by
the exact (model, version, effort) tuple.

## Why "source-free"

The downloadable ARC-AGI-3 environment ships each game's Python source. An agent in the same process/dir can
read the win condition — *reading the answer key*. The invariant enforced here is: **the solver never reads
the game's source**. Two independent routes guarantee it, recorded per run in `fairness`:

- **`by-construction`** (agent tier): the agent runs in a process-isolated sandbox (`SandboxGame` pipe
  client); the game object and its source never exist in the agent's process or working dir.
- **`by-audit`** (cheap tier): a fixed, pixel-only search whose ~100 lines provably read only frames
  (statically verified — no `inspect.getsource` / `environment_files` / `spec_from_file_location`).

Stepping the env to *act* is the legitimate API (leaderboard agents do the same); only *reading source* is
the cheat. Every run records its audit result in `outcome.audit`.

## Routing (hybrid world models)

Each game is first attempted by the **cheap** tier (fast pixel-only frontier search). Games it does not
fully solve are routed to the **agent** tier (a live coding agent that discovers dynamics by acting and
reasons the win from frames). `tier` and `method` record which solved each game.

## Verification

A run's `outcome` is trustworthy because it is recomputed independently of the solver:

1. **Source-free audit** — `outcome.audit.clean` (see above).
2. **Real-engine replay** — the action trace is replayed from `reset()` in the real `arc_agi` engine and
   must raise `levels_completed` to the claimed depth (`outcome.replay_verified`).
3. **OpenWorld World round-trip** — the discovered masked-frame state graph is built into an OpenWorld
   `World` (`FunctionTransition` over the learned table + induced `CodeObjective` reward = levels); the
   solution is replayed through `world.step` and must reproduce the depth with **0 misses**, plus
   `validate_spec()==[]` and a renderable card (`outcome.openworld_roundtrip.pass`).

A run is a **full solve** (`outcome.full_solve`) only if audit-clean, replay-verified, round-trip-passing,
and `levels >= win`.

## Files

| File | Committed | Contents |
|------|-----------|----------|
| `runs.jsonl` | ✅ | One JSON record per run (the dataset index; see `SCHEMA.md`). |
| `prompts/<run_id>.md` | ✅ | The exact prompt given to the agent (agent runs only). |
| `solutions/<run_id>.json` | ✅ | The action trace the run produced (`[[a] | [6,x,y], ...]`). |
| `meta/<run_id>.json` | ✅ | Per-run sidecar written at launch (pre-outcome); source of `runs.jsonl`. |
| `transcripts/<run_id>.jsonl` | ⛔ gitignored | Full `claude -p` stream-json transcript (every message + tool call). Large → kept on disk for HuggingFace/object-storage upload. |

`run_id` = `<game>__<tier>__<UTC ISO-8601 timestamp>`, unique and immutable.

## Reproduce

```bash
# cheap tier (arc venv has arc_agi):  <arcv>/bin/python scripts/run_cheap_tier.py <game...>
# agent tier (pinned model + effort):  MODEL=claude-opus-4-8 EFFORT=high bash scripts/run_arc_agent_sandbox.sh <game> agent
# join verified outcomes -> runs.jsonl: <arcv>/bin/python scripts/finalize_traces.py
# bank deepest verified per game:       <arcv>/bin/python scripts/bank_from_runs.py
```

The full overnight pipeline is `scripts/sweep_routed.py`. Pipeline file SHA-256s are recorded in each
record's `pipeline` field so a run can be tied to the exact code that produced it.

## Benchmark

ARC-AGI-3: 64×64 grids, 16 colors; actions are directional `1..5,7` plus a click `ACTION6(x, y)` with
`x`=column, `y`=row in `0..63`. Environments are replay-deterministic. Reward = `levels_completed`.

## Limitations / honesty

The cheap tier is shallow (pixel search; often 0–1 levels). Depth comes from the agent tier. Some levels are
goal-as-*procedure* walls where the win is an ordered protocol no observation-only score expresses; a run
reaching a partial depth is reported as such (`full_solve=false`), never inflated. Records with `tier=cheap`
have no prompt/transcript (deterministic algorithm) — their reproducibility rests on the solver code SHA +
seed in `params`/`pipeline`.
