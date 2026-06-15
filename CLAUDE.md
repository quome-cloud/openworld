# OpenWorld — working notes for Claude

Zero-dependency Python framework for **symbolic, verified-code world models** with
a local Ollama backbone. These rules override default behavior.

## Branching & PRs (read this first)

- **Base every new experiment branch directly on `main`. Every PR targets `main`.**
- **Never stack a PR onto another experiment's branch.** Stacked PRs that target a
  sibling branch get *stranded*: when the parent branch merges to main first, the
  child merges into a dead branch and never reaches main. (This happened to E45
  real-repo, then to E45 next-token + E47 — each "merged" but absent from main.)
- The only cost of basing on main is a trivial collision in three spots, resolved
  at merge time: the `\NumExperiments` macro, the `EXPERIMENTS` list, and the
  registration calls in `scripts/make_paper_assets.py` `main()`. Fix those by hand
  in the PR; do not stack to avoid them.
- Concurrent cloud agents share this remote: **`git fetch` before pushing**, and
  never touch a branch checked out in another worktree.
- Commit/push only when asked. End commit messages with the Co-Authored-By line.

## Paper assets

- **All paper numbers/figures/tables come from `scripts/make_paper_assets.py`**,
  which reads `experiments/results/*.json` and writes `paper/numbers.tex`,
  `paper/figs/*`, `paper/tables/*`. **Never hand-edit `paper/numbers.tex`.**
- A new experiment ENN adds: its results JSON, an entry in `EXPERIMENTS`, a
  `fig_*`/`table_*` function + its call in `main()`, macros before the
  `numbers.tex` write, and a `\NumExperiments` bump (it is a count).
- Regenerate (`python scripts/make_paper_assets.py`) then compile with
  `tectonic main.tex` from `paper/` (no system latex). Check for undefined refs.
- **LaTeX control-sequence names are letters only — no digits** (`\LadderQwenSmall`,
  not `\ScaleQwen7`), or you get "Missing \begin{document}".

## Experiments

- Prefer **deterministic, offline, self-checking** experiments: fixed seeds,
  numpy baselines, `assert` the sign/shape of every claim. **Call `save_results`
  BEFORE the asserts** so a failed check never loses the run.
- Be honest: if a result is weak, flaky, or excluded, say so in the script and the
  paper (e.g. E45 excluded `incr`/`modk` with documented reasons). Don't tune to a
  desired number; fix the design or report the boundary.
- `experiments/common.py` holds shared worlds/helpers (`save_results`,
  `require_ollama`, the sprint world, stats helpers).

## Local Ollama gotchas

- Large models can carry a **huge default context** (qwen3-coder:30b defaults to
  256k → ~71 GB → swaps the GPU and times out). **Cap `num_ctx` (e.g. 8192)** via
  `OllamaLLM(options={"num_ctx": 8192})`; then a 30B fits VRAM and is fast.
- Give big/reasoning models long `timeout` (e.g. 1800s) and **wrap every LLM call
  in try/except** so one timeout is a miss, not a crashed run.
- Ollama on Metal is **not fully deterministic even with a fixed seed**; for
  synthesis use several attempts and keep the best (verified by reproduction).
- deepseek-r1's reasoning traces are impractical for trace-induction locally
  (excluded from E38). Run big-model experiments when the server is otherwise idle.

## Don'ts

- **Never reference CrewAI** anywhere in this codebase or docs.
- The **core library is zero-dependency**: `import openworld` and everything it
  pulls in (state, transitions, compose, spec, card, perceive, …) must use only
  the stdlib. Keep it that way.
- The **serve/CLI layer is the one exception**: `openworld/serve.py`,
  `openworld/cli.py`, `openworld/_tmux.py` may use `fastapi`/`uvicorn`/`click`/
  `rich` (declared in `pyproject.toml`). They are NOT imported by
  `openworld/__init__.py`, so the core import stays clean. Don't add other
  third-party runtime deps, and don't let serve/CLI deps leak into the core.
