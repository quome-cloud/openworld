# E119 — SLM neuro-symbolic ARC-AGI-3 solver: results & conclusion

This is the write-up of E119 as it would read in the paper. It states the question, the method,
what was measured, what it means, and — explicitly — what was *not* tested. The chronological lab
notebook (every step, bug, and decision) is in `PROGRESS.md`; the design is in `STRATEGY.md`.

## 1. Question

E119 ports the **"engineer the harness, not the model"** law — established on *static* tasks
(GSM8K / HumanEval, where a deterministic harness owns verification and small models only propose)
— to *interactive* world-model tasks (ARC-AGI-3, where a solve is an action sequence that raises
`levels_completed`, with no demonstration pairs). The concrete question:

> Can a set of small language models (≤ ~9B: qwen2.5 / qwen2.5-coder, llama3.1, gemma3) inside a
> verification-grounded harness solve these games — and **how much of any solving comes from the
> harness versus the model?**

The motivating prior is **E118**: a pure `qwen3-coder:30b` ReAct agent failed `ar25` even after
memory + 24 rounds + 32k context — evidence the bottleneck is harness/representation, not model
scale. E119 attacks the same question from the other side: not "scale the model," but "engineer the
harness and let small models only propose, gated by env replay."

## 2. Method

Every component routes the decision to a deterministic **executor** (env replay / a compiled
grader); the SLM only *proposes*, and a wrong or abstained proposal costs search time, never a false
solve. We measured the harness and the model contributions as separable **rungs**, at matched node
budget (`max_nodes=6000`, `max_depth=60`), on the 25 public ARC-AGI-3 games:

- **Control (harness only, no LLM):** deterministic perception (status-bar masking, masked-frame
  state keys, pixel-inferred click targets) + BFS over action prefixes, each prefix replayed from
  `reset()` (the env is ground truth). This isolates what the harness alone can do.
- **Subgoal rung (`--mode slm`):** the SLM synthesizes a goal predicate (`reach`/`count`/`align`)
  via best-of-N + behavioral clustering + τ-abstention; the predicate becomes a best-first
  search-ordering prior. Run across **four** models (qwen2.5-coder:7b, qwen2.5:7b, gemma3,
  llama3.1:8b) on a 5-game pilot.
- **Macro/procedure rung (`--mode macro`):** when search stalls, the SLM proposes a short *action
  procedure* (object/action-referential ops → compiled to primitives), best-of-N + abstention,
  ranked by the subgoal proxy, **banked only if a fresh-env replay raises a level**. Measured as a
  3-arm sweep — control / random-macro / SLM-macro — at matched budget, m=5 seeds, on the two
  procedure-walls a gating pre-experiment (Phase 0) showed carried the most directional signal
  (`tr87`, `re86`).

## 3. Results

**Finding 1 — the harness alone solves 10/25 games.** Control (zero LLM) completes ≥1 level on
**10 of 25** games (click 4/6, dir 2/6, mixed 4/13). Most cap at level 1; `vc33` reaches level 2.
This is the bulk of E119's solving capability, and it requires no model at all.

| | result |
|---|---|
| Harness-only reachability | **10/25 games ≥1 level** |

**Finding 2 — the subgoal rung adds nothing (delta = 0), across all four models.** On the pilot,
control solves 3 levels (`vc33`=2, `lp85`=1); **every** SLM solves *exactly the same* 3, per game.

| game | control | qwen2.5-coder:7b | qwen2.5:7b | gemma3 | llama3.1:8b |
|------|:---:|:---:|:---:|:---:|:---:|
| tn36 / ar25 / sk48 | 0 | 0 | 0 | 0 | 0 |
| vc33 | 2 | 2 | 2 | 2 | 2 |
| lp85 | 1 | 1 | 1 | 1 | 1 |
| **total** | **3** | **3** | **3** | **3** | **3** |

**Finding 3 — the macro/procedure rung also adds nothing, and the negative is *fair*.** On `tr87`
and `re86`, all three arms solve **0/5**:

| game | control | random-macro | SLM macro | SLM proposals valid |
|------|:---:|:---:|:---:|:---:|
| tr87 | 0/1 | 0/5 | 0/5 | 100% |
| re86 | 0/1 | 0/5 | 0/5 | 100% |

This negative survives the strongest scrutiny we could apply. A first macro sweep also returned
0/5, but per-call transcript logging revealed it was **confounded**: the proposer prompt never named
the available actions, so the model proposed *click* operations on directional games (`tr87`
`avail=[1,2,3,4]`, `re86` `[1,2,3,4,5]` — no click), and **54/60 samples compiled to empty**. After
the prompt named the valid actions, **100% of proposals were valid, varied procedures** (e.g.
`["a2","a3","a1",…]`, sweeps, repeats) — and the result was still 0/5. So even when the SLM proposes
*correct, diverse procedures every stall*, the macro slot solves nothing that blind search or a
seeded random-macro baseline does not.

## 4. Answer to the question — harness vs. model

**Essentially all of E119's measured solving comes from the harness; the SLM contributed no
measurable lift in any configuration we tested.** The harness alone solves 10/25 games. Adding an
SLM subgoal prior (4 models) changed the solved set by 0. Adding the design's primary slot for the
hard games — SLM-proposed procedures (after removing the proposal-validity confound) — changed it by
0, and did not beat an undirected random-macro baseline at matched budget.

The harness's **safety invariant held perfectly throughout**: across every rung, model, seed, and
the confounded and fair sweeps, the SLM never produced a single false solve — every banked level is
a fresh-env-replayable action sequence (the two banked solves re-verify 2/2). This is the
"verification-grounded" promise delivered: correctness is the env's, not the model's. But the
*value* the SLM was meant to add — making search faster or deeper — was **not observed**.

Combined with **E118** (a 30B coder agent, with memory + 24 rounds + 32k context, also failing
`ar25`), the two experiments triangulate the same conclusion from opposite directions: **scaling the
model up (E118) and adding SLM proposers to an engineered harness (E119) both fail to move the
needle.** The binding constraint on these interactive games is the harness's **search reach
(branching × depth) and the representation of the goal**, not the model. This is a *stronger* form
of the "engineer the harness, not the model" law for interactive world-model tasks: here the harness
is not merely sufficient — within the band and slots tested, the model is, to measurement, inert.

**Why the SLM can't help here (mechanism, evidenced).** The reachable levels the harness *does*
solve are shallow enough that blind BFS already finds them, leaving no headroom for a prior
(Finding 2). The levels it *doesn't* solve are **goal-as-procedure** — corroborated by the repo's
own E102/E103/E104, where atomic goals, Claude-generated compositional hypotheses, and
procedure-aware Bayesian + tropical-semiring planning *with perfect world models* scored 0 on these
walls. A Phase-0 probe found the SLM's subgoal proxy *was* directional (best-first reached +52 / +24
levels deeper on tr87 / re86) but **necessary-not-sufficient**: moving the search frontier deeper is
not the same as reaching a deep, specific procedural reward, and the fair macro sweep confirms it.

## 5. What was *not* tested (threats to validity — read before citing)

- **Scope of games:** the subgoal rung was measured on a 5-game pilot; the macro rung on the 2
  strongest-signal procedure-walls (`tr87`, `re86`). The harness reachability is the only all-25
  number. Negatives are reported on the games measured, not claimed universal.
- **Scope of models:** the subgoal rung used four ~7–8B models; the macro rung used **one**
  (qwen2.5-coder:7b) — by design (cost control: a clean negative on the first model means no
  diversity fan-out). The smaller end of the band (1.5–3B) and a macro-slot model-diversity sweep
  were **not** run.
- **The design's static-task gains did not transfer and were not reproduced here.** The source law's
  reported "+14 from sample+vote, +21 from model diversity, 0.75→0.97 abstention precision" are
  *static-task* numbers; on interactive ARC-AGI-3 the proposer machinery yielded **+0**. The
  best-of-N/abstention/diversity machinery is validated as **safe** (no false solves) but its
  *value* (faster/more solving) is **unproven** here — there was no positive signal for it to be
  precise about.
- **Slots not exercised:** per-slot model routing (coder vs. reasoning vs. tiny), adaptive sampling,
  and the explicitly-ablated self-review-repair / few-shot were not run.
- **One honest residual:** verification banks correctly on a fresh env, but per-call transcripts were
  only added late (they caught the Finding-3 confound); earlier rungs lack call-level transcripts.

## 6. Conclusion

On interactive ARC-AGI-3, a verification-grounded harness with small-model proposers behaves exactly
as the "engineer the harness, not the model" law predicts — in its strongest form. The harness
carries all measured solving (10/25 games, zero LLM); ≤9B SLMs, whether ordering search by a
synthesized subgoal (4 models) or proposing whole procedures (the design's primary slot, with valid
proposals), add **no measurable lift** and — by the replay invariant — never a false solve. With
E118's 30B-agent failure, the bottleneck is located in the **harness's search reach and goal
representation**, not in model capability or scale. The contribution of this work is therefore
twofold: a clean attribution (harness ≫ model on these tasks) and a method note — *log per-call
proposer transcripts*, because an aggregate score (k/m = 0) hid a proposal-validity bug that fully
explained a first, confounded negative.

**Honest next levers (hypotheses, not results), on a now-fair baseline:** a candidate *pruner* to
cut branching multiplicatively (the one width-bound game, `bp35`, b=190); *longer* macros than the
2–8-op cap; a *dense, non-binary* subgoal scorer (the current 0/1 predicate gives best-first no
gradient); and a true per-level reset for deep multi-level search. None is required to support the
conclusion above; each is a way to test whether the harness's reach — the located bottleneck — can be
extended.

## 7. Reproducibility & runbook (validated, macOS / Python 3.13)

Determinism splits by layer (full protocol in `STRATEGY.md` / the spec): the **env, control,
reachability, classification, Phase 0, and every banked solution are exact, machine-independent point
facts**; the **SLM arm is a distribution** (sampling + Metal nondeterminism) and is reported over
m≥5 seeds with the model + decoding options pinned in each results `env` block. Every banked solve is
re-verified by replaying its action sequence on a fresh env (`reverify.py`).

```bash
# 1. Env: arc-agi + arcengine are on PyPI (the "download-only" note was stale).
python3 -m venv .venv && .venv/bin/pip install arc-agi arcengine numpy && .venv/bin/pip install -e .
# 2. Ollama: use the official app build — the Homebrew formula 0.30.7 ships without the
#    llama-server backend (HTTP 500 on every model).  brew install --cask ollama
ollama pull qwen2.5-coder:7b qwen2.5:7b gemma3        # llama3.1:8b too
# 3. Run (harness lib on scratch_arc/agent; experiments/ self-adds to path). Do NOT git-worktree
#    this already-checked-out branch — run in place.
export PYTHONPATH="$PWD/scratch_arc/agent"
.venv/bin/python experiments/e119_slm_solver.py --mode search                       # harness only
.venv/bin/python experiments/e119_slm_solver.py --mode slm   --model qwen2.5-coder:7b   # subgoal rung
.venv/bin/python experiments/e119_proxy_probe.py                                    # Phase 0 gate
.venv/bin/python experiments/e119_macro_sweep.py tr87,re86                          # 3-arm macro sweep
```

**Artifacts.** Reachability: `experiments/results/e119_reachability.json`. Subgoal rung:
`e119_slm_solver.json` + `e119_rung_*.json`. Phase 0: `e119_proxy_probe.json`. Macro sweep:
`e119_macro_sweep.json`; per-run + per-call traces: `e119_logs/e119_runs.jsonl`,
`e119_logs/e119_call_traces.jsonl`. Banked verified solves: `e119_logs/{vc33,lp85}_solved.json`.

## Appendix — bugs found and fixed (each masked or corrupted a result)
1. **Honest zero-solve tripped the verification assert** (`verified = … and reached > 0` flagged a
   legitimate 0-solve as unverified, aborting the pilot). Fixed via `_is_honest()` — commit `dc17a92`.
2. **Checkpoint-retaining `reset()` zeroed real solves**: the arc env's `reset()` keeps
   `levels_completed`, so `replay_levels`' delta collapsed to 0 on a reused env; a genuine
   `ls20`/`vc33`/`lp85` solve read as 0. Fixed by verifying on a **fresh** env — commit `266e6eb`.
   This was masking real solves and is why the very first sweep looked like 0-vs-0.
3. **Proposal-validity confound** (Finding 3): the macro prompt omitted the available-action set, so
   the SLM proposed nonexistent click ops on directional games (54/60 empty). Fixed by naming valid
   actions in the prompt — commit `53846dd`. Caught only by per-call transcript logging.
