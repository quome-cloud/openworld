# E127: Source-Simulated Solving via Differential CEGIS Engine Reconstruction

**Status:** design (brainstormed 2026-06-28; expert-reviewed by RL/MBRL + program-synthesis/verification lenses)
**Branch plan:** new branch (or git worktree) `e127-source-simulated` off updated `main` (per CLAUDE.md). E127 is all-new files (`experiments/e127/`), so it will not collide with the running source-free/codex sweeps except at the three known paper-asset merge spots.

## 1. One-paragraph summary

With **no access to the real game source**, reconstruct each ARC-AGI-3 game's *engine as code* purely from sandboxed play, then **certify it against the real environment** to a bounded error, and search the certified engine (as an OpenWorld `World`) to a win — replay-verified on the real env. Two models (Claude + OpenAI codex) reconstruct **independently**; their disagreement is used as a *signal for where to spend real-env labels* and as a *diversity source for counterexamples*, **not** as the acceptance gate. This defines a third protocol — **source-simulated** — strictly between *source-free-observational* (E124/E125) and *source-assisted* (reads `<game>.py`; the 24/25). The scientific payload: a falsifiable **equivalence-to-real certificate** obtained without source, plus a measurement of how much shared-model-prior bias inflates naive two-model agreement (the **A-vs-B-vs-real gap**).

## 2. Motivation / thesis

If a capable agent can **reduce an observable system to its (reconstructed) source**, the system becomes deterministic to that agent and solving collapses to search. We test whether *reliable* reconstruction is achievable without reading source, and at what real-env cost. Implication (sharpening the paper's existing cybersecurity reading): any observable system — including closed-source software — is, in principle, reducible to a deterministic model from interaction alone; the certificate quantifies *how reliably*.

## 3. What the expert review changed (record, so the plan honors it)

Two independent senior reviewers (model-based-RL world-models; program-synthesis/formal-verification) converged on the same load-bearing flaws of the naive "two-models-agree = unity" design. Adopted fixes:

1. **Convergence target = agreement with the REAL env**, not A≈B. Correlated pretraining priors make two LLMs capable of *identical* errors (folie à deux); a loop that only queries reality where A≠B stops checking reality exactly as the models correlate. The second model is demoted to a **diversity source** for counterexamples + a **sampling heuristic** for label spend. Report the **A-vs-B-vs-real gap** as a headline.
2. **Stateful latent-state engine**, not stateless `step(frame,action)→frame`. ARC-3 games carry hidden state (parity, budget counters, level index, "key unlocks new toggle", ghost buffers); a pure visible-frame function is provably misspecified (same frame+action → different next frame by history). Engine declares an inspectable latent `state` and is scored on **rollouts over ordered episodes**.
3. **`is_win(h, frame)` reads latent state** — frame-pair predicates cannot express goal-as-*procedure* (the dominant ARC-3 win type and the reason source-free stalls).
4. **Receding-horizon plan-then-verify-incrementally + ensemble pessimism** in search — best-first search over a learned engine hunts for the model's errors (model exploitation). Execute the first *k* steps on the real env, check each predicted frame, re-plan from the real frame on divergence (reuse E125 halt-on-sim-vs-real-mismatch). Only expand search nodes where engines agree; A≠B during search → real-env query.
5. **Disagreement-driven active exploration** — reversible shallow play cannot *identify* deep/irreversible dynamics (system-ID coverage). Pick actions maximizing predicted-frame disagreement / ensemble variance, plus explicit frontier + irreversible-event rewards.
6. **Mask is for state-identity hashing only**, never for the correctness comparison; the win/reward channel is never masked; the engine must **predict** "noisy" cells from latent state; report what the mask hides.
7. **Anti-gaming + termination:** keep-best monotone gate on held-out *real* accuracy; forbid lookup-table degeneracy (memorizing the corpus); report generalization gap `acc_T − acc_H`; on budget exhaustion emit the *measured bound*, never a binary "unified".
8. **Audit / honesty:** *enforce* no-source-read structurally (process/file-perm isolation + a read-manifest whose hashes never include `<game>.py`), not just promise it. Report real-env steps split by `{explore, adjudicate, verify}`; **solves-per-real-step** is the protocol's efficiency headline.
9. **Determinism probe first** (replay same actions twice from reset, diff) — if non-deterministic, the deterministic-engine premise is void for that game; report it.
10. **Rename** "adversarial unity" → **differential CEGIS reconstruction** (the two models cooperate; the real env is the verifier). "Adversarial" overclaims.

## 4. Protocol definitions (the paper's axis)

- **source-assisted** — agent reads `<game>.py` (win condition + dynamics + map). Humans can't read source → not human-comparable. (Our 24/25.)
- **source-free-observational** — only `SandboxGame` {frame, levels, win, avail, done}; synthesize a per-level `predict`, no reconstructed full engine (E124/E125). (Our 8/25.)
- **source-simulated (NEW, E127)** — no source; reconstruct + **certify against real** a full stateful engine, then search it. Certificate-bearing.

## 5. Architecture / data flow (per game)

```
Phase 0  Observe + probe (source-free)
  SandboxGame --reversible+frontier play--> ordered episodes E_obs (trajectories)
  determinism probe: same action seq x2 from reset -> diff  (record nondeterminism)
  mask M := cells action/state-independent (for state-IDENTITY hashing only)

Phase 1  Independent reconstruction (x2 models, isolated)
  model_A (Claude), model_B (codex) each emit a stateful Engine module from E_obs + action API
  (NEVER the real source; workdir audit-gated; read-manifest hashed)

Phase 2  Differential CEGIS against REAL  (loop)
  champion := best of {A,B} by held-out REAL rollout accuracy
  each round:
    (a) counterexample search (UNION of adversaries):
          - coverage-guided novelty search over reachable masked states
          - property falsifiers (determinism; no-op on invalid click; level-up => big delta;
            win channel never masked; color-range invariants)
          - A-vs-B disagreement proposals (diversity heuristic)
          - random differential rollout
    (b) adjudicate vs REAL: step the real SandboxGame on candidates (prefix-shared replays);
          any Engine != REAL is a true counterexample -> disjoint held-out / retrain split
    (c) revise each Engine (LLM resynth conditioned on counterexamples);
          accept only if monotone non-decreasing on disjoint held-out REAL set (keep-best);
          reject lookup-table degeneracy
    (d) active exploration: collect new episodes where A,B disagree most (epistemic frontier)
  terminate when (coverage >= target, per level) AND (acc_H >= 1-eps, Clopper-Pearson @ 1-delta)
    AND (generalization gap <= tau)  -> emit CERTIFICATE
    else on budget exhaustion -> emit MEASURED BOUND (not "unified")

Phase 3  Solve via certified engine
  certified Engine -> OpenWorld World (stateful CodeTransition over rollout; CodeObjective from is_win(h,.))
  receding-horizon search (best-first under ensemble pessimism) -> imagined plan
  execute first k steps on REAL; check predicted frame each step; re-plan from REAL on divergence
  bank ONLY if REAL levels_completed rises; log engine-vs-real divergence along the winning path
```

## 6. Components (files; each one focused; TDD)

- `experiments/e127/engine.py` — **stateful engine contract** + compile (SAFE_BUILTINS sandbox, reuse E125 pattern) + `rollout(engine, actions)` + `score_rollout(engine, episode, mask)` (graded per-cell + structural/object score) + `mask_for_identity(episodes)` + `engines_disagree(A, B, state, action, mask)` + lookup-table-degeneracy detector.
- `experiments/e127/reconstruct.py` — the **differential CEGIS loop** (`reconstruct(episodes, action_api, game, models, real_env, budget) -> {engine_src, certificate|bound, history, ab_gap}`), keep-best monotone gate, prompts (round-0 synth; counterexample-conditioned revise).
- `experiments/e127/probes.py` — **counterexample search** (novelty/coverage-guided + property falsifiers + A-vs-B diversity proposals + random differential) and **prefix-shared real-env adjudication**; coverage accounting.
- `experiments/e127/certify.py` — **held-out REAL accuracy + coverage + Clopper–Pearson bound + generalization gap**; emits the certificate object; per-level stratification.
- `experiments/e127/explore.py` — **disagreement/uncertainty-driven active exploration** (frontier + irreversible-event rewards) over SandboxGame; reuse E125 `explorer` where possible.
- `experiments/e127/world127.py` — certified stateful engine → OpenWorld `World` (`build_engine_world`) + `round_trip_ok`.
- `experiments/e127/solve.py` — **receding-horizon plan-then-verify-incrementally** under ensemble pessimism (reuse E125 halt-on-mismatch); replay-verify; emit solve record + divergence log.
- `experiments/e127/codex_iso.py` — **source-free codex runner** mirroring E125 `claude_iso.run(prompt, schema, model)` (codex `exec`, SandboxGame only, no source dir). [Or extend `claude_iso` to a `_runner` param.]
- `experiments/e127_source_simulated.py` — **harness**: per game observe → reconstruct → certify → solve → bank to `experiments/results/arc3_source_simulated.json`; audit-gated end to end.
- **Reuse:** `experiments/arc3_sandbox.py` (`SandboxGame`), `scripts/audit_sandbox.py`, E125 `explorer`/`claude_iso`/`verify` (SAFE_BUILTINS compile)/`world` patterns.

## 7. Key data structures

- **Engine module (model-authored):**
  ```python
  class Engine:
      def reset(self): ...                 # -> sets self.state (declared dict), returns initial frame
      def step(self, action): ...          # action=(kind,x,y); mutates self.state; returns 64x64 int grid
      def is_win(self, prev_frame): ...    # reads self.state (procedural progress) -> bool
      # self.state: an inspectable dict of declared latent variables (level idx, parity, budget, ...)
  ```
- **Episode:** ordered `[(action, frame), ...]` from `reset()` (trajectory, NOT shuffled triples).
- **Certificate:** `{eps, delta, n_holdout, coverage_per_level, generalization_gap, real_steps:{explore,adjudicate,verify}, ab_agreement, ab_vs_real_gap, nondeterministic:bool, mask_cells:int, mask_hidden_rate}`.

## 8. Results JSON (`experiments/results/arc3_source_simulated.json`)

Per game: rounds-to-certificate, the full certificate (§7), `solved` (bool, replay-verified), levels reached, solves-per-real-step, and the **A-vs-B-vs-real gap**. Plus an `ablation` block: single-model + differential-testing-vs-real (drop model B) on the same games.

## 9. Validation plan / controls

- **Positive (mechanism) control:** `ar25` first — already full *source-faithful and source-free*, so ground truth is known; confirm the certificate passes and the certified engine matches real to bound. (Note: ar25 success alone doesn't prove the *reconstruction* did the work — pair with the ablation + negative control below.)
- **Gap demonstration:** `dc22` — source-faithful solves, source-free doesn't; show source-simulated closes (some of) the gap.
- **Ablation:** single-model + differential-testing-vs-real (no model B). If it matches dual-model, the second model adds nothing — run it to find out.
- **Negative control:** stateless-engine ablation on `dc22` should *fail* where theory predicts (hidden state) — a method that can't fail isn't being tested.
- **Determinism probe** reported per game (premise check).

## 10. Honesty / audit guarantees

- Whole workdir `audit_sandbox`-gated: no `environment_files`, no `inspect.getsource`, no `spec_from_file_location`, no `<game>.py`. Read-manifest of every opened file (hashes); assert `<game>.py` hash never appears. Reconstruction model runs in a separate process/context with no game name or source in scope (structural isolation, reusing E125 Plan-2.5).
- Engine must be justified by T-evidence: flag any mechanic the engine encodes that has no support in the observed episodes ("you couldn't have known this from the data" → possible pretraining leakage).
- Real-env steps are budgeted and **reported by category**; a solve that needed enormous real-env stepping is disclosed as such (it is not cleanly "simulated").
- Built as OpenWorld `World` → `to_spec` → `preview.graph` is the map, `render_card` the atlas (CLAUDE.md), viewable in `openworld serve /view`.

## 11. Paper integration

- A **third protocol column** "source-simulated" in the main results matrix/figure.
- New metrics: certificate pass-rate, A-vs-B-vs-real gap, solves-per-real-step, rounds-to-certificate.
- Ablations: full-frame stateful vs stateless (negative control); dual-model vs single-model+differential-testing.
- Discussion: "reduce to source → deterministic" framed as the certificate; cybersecurity implication sharpened (reliable interaction-only reconstruction of closed systems).
- All numbers via `scripts/make_arc3_assets.py` (add macros + a figure fn + `main()` registration; bump `\NumExperiments`); never hand-edit `numbers.tex`.

## 12. Out of scope (YAGNI)

- K>2 bootstrapped engines (noted as a future improvement for richer epistemic variance) — start with the two named models + the single-model ablation.
- Full 25-game sweep — first prove the mechanism on `ar25` then `dc22`; generalize only after.
- A learned/neural world model — engines are **symbolic code** (the project's whole premise); no neural nets.
