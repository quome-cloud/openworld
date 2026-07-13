# MiniHack arm — world-model synthesis + classical search on BALROG

Code for the BALROG MiniHack suite run (92.5% vs 90.0% SOTA). See
`../../artifacts/minihack/FABLE_MINIHACK_REPORT.md` for the full report,
including the clean-observation audit, the memory-across-episodes
experiment, and the Lessons-for-NetHack section.

Runtime deps (not vendored here): `balrog-nle==0.9.0`, the balrog-ai
minihack fork, `gym==0.23`, `numpy<2`, plus the BALROG repo's
`balrog/` package (vendored locally at run time) for the exact wrapper
stack. `mh_harness.py` documents the env construction; the runners are
`run_suite.py` (condition A), `run_robustness.py`, `run_memory.py`
(condition B), `run_explore_data.py` (induction-leg dataset),
`render_animations.py` (GIFs from logged tty frames).
