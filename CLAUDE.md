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
- **One change → one branch → one PR.** Once a PR is merged, that branch is *done*:
  do **not** push more commits to it — a merged/closed PR ignores later pushes, so
  they never reach `main` (they "orphan"). Start a **fresh branch off updated
  `main`** for the next change. If you already pushed post-merge commits to a stale
  branch, cherry-pick them onto a fresh branch off `main` and open a new PR.
  (This bit us repeatedly: serve/README/view follow-ups merged at an earlier
  commit and had to be re-PR'd.)
- The human merges PRs quickly, so **finish and push a change before it's merged**,
  and when several changes touch the same file (`openworld/serve.py`,
  `papers/*/main.tex`, `papers/assets/*`), do them **one at a time**: open, wait for merge, branch off
  updated `main`. After a merge you can confirm with
  `git merge-base --is-ancestor <sha> origin/main`.
- Concurrent cloud agents share this remote: **`git fetch` before pushing**, and
  never touch a branch checked out in another worktree.
- Commit/push only when asked. End commit messages with the Co-Authored-By line.

## Paper assets

- **All paper numbers/figures/tables come from `scripts/make_paper_assets.py`**,
  which reads `experiments/results/*.json` and writes `papers/assets/numbers.tex`,
  `papers/assets/figs/*`, `papers/assets/tables/*`. **Never hand-edit
  `papers/assets/numbers.tex`.** The two papers (`papers/world-time-compute/`,
  `papers/framework/`) symlink `figs/tables/numbers.tex/refs.bib/sections` →
  `../assets/`, so the one pipeline is the single source of truth for both.
- A new experiment ENN adds: its results JSON, an entry in `EXPERIMENTS`, a
  `fig_*`/`table_*` function + its call in `main()`, macros before the
  `numbers.tex` write, and a `\NumExperiments` bump (it is a count).
- Regenerate (`python scripts/make_paper_assets.py`) then compile each paper with
  `tectonic main.tex` from its `papers/<name>/` dir (no system latex). Check for
  undefined refs.
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

## World-model specs, cards & serving

The portable artifact for a world model is a **spec** (`openworld/spec.py`):
`to_spec(world)` → JSON dict, `from_spec(spec, allow_code=False)` → a world,
`validate_spec(spec)` → list of problems (the publish gate). A complete spec
captures **every component of a world model** — keep all of these in sync when you
touch the format:

- `name`, `description`, `card` (`version`/`license`/`authors`/`tags`/`lineage`/`metrics`).
- `state_schema` (inferred type names) + `initial_state` (concrete values).
- `actions`, and `rules` — the declared natural-language contract.
- `transition` by `kind`: `code` (verified `CodeTransition`), `function`
  (`FunctionTransition` via `inspect.getsource`, else `lossy`), `phased`, `llm`,
  `composite`.
- **perception** (the perceive→world boundary): perceptors as `{kind, modality,
  produces, schema}`. `CodePerceptor` carries runnable `code` and round-trips +
  runs server-side with no LLM; Mock/Text/Vision are descriptor-only (not
  reconstructable from a spec).
- **emit** (the world→output boundary): `{modality, fields, report?}`.
- **objectives**: `{name, goal, weight?}`.
- `composite`: `children` (recursive specs), `bridges` (Route adds `on_cross`),
  `aggregators` (fn via `getsource`), `bindings`, `timescales`, `default_actions`,
  `agents`.
- `preview`: a numeric rollout `series` + a bounded state-transition `graph` (BFS
  from the initial state), computed once at serialize time for the card/view.

Rules:

- **Round-trip stays lossless**: `from_spec(to_spec(w), allow_code=True)` must
  reproduce `w`'s rollout. Un-serializable pieces (function transitions/aggregators
  without source, lambdas, callable phase triggers) are flagged `lossy` — never
  silently dropped.
- **`allow_code` is a trust gate, not a sandbox**: `from_spec` keeps embedded code
  inert unless `allow_code=True`; the serve registry needs it to run dynamics.
- Cards/exports live in `openworld/card.py`: `render_card` (one self-contained SVG,
  the "atlas" aesthetic), `render_gallery`, `to_reactflow` (`{nodes, edges,
  playground}`), `to_mermaid`. Keep the SVG card **self-contained** (no fetched
  resources; xmlns/`<a href>` are fine). The SVG card, the gallery, the live React
  Flow `/view`, the logo (`assets/logo.svg`), and the CLI banner share one design
  language (nested-worlds mark; blue/ochre/teal depth ramp; sensor/state/emit
  boundary) — keep them consistent.
- **Adding a spec component means touching all of:** serialize (`_*_to_spec`),
  reconstruct (`from_spec`/`_attach_io`), `validate_spec`, the card section + graph,
  `to_reactflow` + `to_mermaid`, and the serve `/metrics` + `/worlds/{name}` info.

## Build → optimize → deploy (CLI + server)

- `openworld serve <specs> --allow-code [--open]` runs a FastAPI multi-world
  registry (a stateless forward pass). Endpoints per world: `GET` `/worlds`,
  `/{name}` (info), `/spec`, `/state`, `/actions`, `/metrics`, `/card.svg`,
  `/mermaid`, `/reactflow`, `/view`; `POST` `/step`, `/predict` (batch),
  `/rollout`, `/observe`, `/run`; `WS /live`. Composites/bridges/perception serve
  transparently. The interactive `/view` is the primary world UI; `--open` lands on
  it.
- `openworld build/optimize` drive Claude Code to author/tune a spec: interactive
  tmux session if `tmux` is present, else **headless `claude -p`** (streamed
  progress), else scaffold + manual prompt. Never require tmux.
- Honest UX: stream long-running agent work (don't hide it behind a silent
  spinner); use **terminal-native ANSI/bright colors** (named colors adapt to the
  user's theme — hard-coded dark hex like `#1d4ed8` is unreadable on a black
  terminal).

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

- The **core library is zero-dependency**: `import openworld` and everything it
  pulls in (state, transitions, compose, spec, card, perceive, …) must use only
  the stdlib. Keep it that way.
- The **serve/CLI layer is the one exception**: `openworld/serve.py`,
  `openworld/cli.py`, `openworld/_tmux.py` may use `fastapi`/`uvicorn`/`click`/
  `rich` (declared in `pyproject.toml`). They are NOT imported by
  `openworld/__init__.py`, so the core import stays clean. Don't add other
  third-party runtime deps, and don't let serve/CLI deps leak into the core.
