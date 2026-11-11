# Fable NetHack arm — BALROG NetHackChallenge, synthesis + classical search

LLM-free runtime agent for BALROG's `nle` environment (full NetHack), built
by offline world-model synthesis (Fable 5, max reasoning) + belief-state
planning. See `FABLE_NETHACK_REPORT.md` for method, results, source-leak
audit, and the memory experiment.

## Layout
- `nh_harness.py` — BALROG-identical env stack (vendored `balrog` package)
- `nh_common.py` — observation parsing, per-level Atlas, pathfinding,
  offline species/terrain tables (provenance disclosed)
- `nh_agent.py` — DiveAgent: layered policy (prompts → survival → food →
  combat → dig/descend → explore → doors → hidden search)
- `nh_memory.py` — cross-episode ledger (condition B), provenance-cited
- `nh_transitions.py` — streaming gzip JSONL transition logs (+decoder)
- `nh_runner.py` — episode runner (checkpoint, trajectory + frame capture)
- `run_suite.py` — condition A (seeds 1000+); `run_suite.py 2000 robustness`
- `run_memory.py` — condition B: 3 passes × 5 eps (seeds 4000/5000/6000)
- `run_explore_data.py` — labeled exploration dataset for the source-blind
  induction leg
- `render_animations.py` — GIFs with step/Dlvl/HP/action/memory overlays
- `dev_run.py` — single-episode dev harness (dev seeds 101+, disjoint)

## Repro
Requires the MiniHack arm's `pylib` (balrog-nle 0.9.0 stack) and vendored
`balrog` package (symlinked here in the working tree; copied paths in the
repo). Then:

    python3 run_suite.py            # condition A, official protocol
    python3 run_memory.py           # condition B
    python3 run_explore_data.py     # induction dataset
