# scikit-learn Bridge Tutorial - Plan

> Inline execution (executing-plans). Small, focused dev-rel content.

**Goal:** A bridge tutorial (`from_scikit_learn.md` + runnable `.py`) mapping OpenWorld to sklearn concepts, attacking three confusions (world != dataset, oracle what/why, verification != test accuracy), plus README banner + per-tutorial pointers.

**Spec:** docs/superpowers/specs/2026-06-12-sklearn-bridge-design.md

### Task 1: runnable companion `tutorials/from_scikit_learn.py`
- Toy "water tank" environment; rule fill +2 / drain -3 floored at 0 / wait.
- `oracle(level, action)` = ground truth.
- sklearn arm: DecisionTreeRegressor on transitions sampled over levels 0..30 (random policy); exact-match accuracy on in-range probes and 10x-scale probes -> memorizes in-range, collapses OOD.
- OpenWorld arm: `World` + `FunctionTransition` (offline; note compile() synthesizes this from rules); `step()`; verify exact vs oracle on the SAME probes incl. OOD; assert exact.
- Guard sklearn import; print side-by-side table; offline + deterministic.

### Task 2: `tutorials/from_scikit_learn.md` (~140 lines, house voice)
- Lead (fit/predict/score you know), Rosetta table, three myth-busters (oracle gets the most space), closing on when-to-use-which.

### Task 3: glue
- README "New to worlds? Start here" banner + table row + run command.
- One-line pointer atop each of the 5 existing .md tutorials.

### Task 4: verify + PR
- `python tutorials/from_scikit_learn.py` exits 0; `python -m pytest tests/ -q` unaffected; commit; PR.
