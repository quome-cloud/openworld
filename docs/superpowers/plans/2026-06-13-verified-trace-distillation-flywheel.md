# Plan: Verified-Trace Distillation Flywheel on OpenWorld-SWE-bench

**Status:** DRAFT for review (Anderson + Jim) — 2026-06-13
**Owner:** Anderson (evaluation/improvement half; complementary to Jim's composite-swe)
**Greenlit by Jim** 2026-06-12 ("amazing idea, definitely do it"; no collision; H100 + 70B teacher offered).

---

## Hypothesis

A small Qwen (1.5B/3B), LoRA-fine-tuned on **verified-passing** in-world repair traces, improves its **single-shot** solve rate — approaching or exceeding a **larger** base model's single-shot. That is the **param-efficiency** finding Jim framed: *outperform with less param count.* It is OpenWorld's training-free-world / verified-data answer to SIA's weight-update self-improvement loop.

## Why it's novel here

Everything merged in `quome-cloud/openworld` is benchmark + measurement + composite worlds. **Nobody has closed the loop into training.** The verifier (exact hidden tests) gives clean labels for free, so the worlds become a **training-data factory**, not just an exam.

## Success criteria

- **Primary (mechanism):** distilled-1.5B single-shot solve rate **>** base-1.5B single-shot, **significant by McNemar** on held-out instances.
- **Stretch (the headline):** distilled-1.5B single-shot **≥** base-7B single-shot — param-efficiency win.
- **Guardrail:** no regression on `pass_to_pass` (the fine-tune must not break already-working behavior).

## Non-goals (first pass)

- Full multi-round flywheel (do ONE distill pass first; iterate later).
- 70B teacher / H100 (local **7B teacher** first; scale only after a local lift exists).
- A production model. This is a mechanism demonstration.

## Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Teacher (trace generator) | qwen2.5:**7b** local now → 70B on H100 later | 7B solves enough to yield passing traces; 70B for data-rich scale |
| Student (fine-tuned) | qwen2.5:**1.5b**, then 3b | smallest model = strongest param-efficiency story + cheapest |
| PEFT method | **MLX LoRA** (Apple Silicon, 24GB) | local, $0, matches local-first ethos |
| Trace filter | **verified-passing only** (`fail_to_pass` all green, no `pass_to_pass` regression) | clean labels = the whole point |
| Train/test split | **by whole instance** | measures generalization, not memorization |
| Seeds | **vary per attempt** during harvest | avoids the pass@k verbatim-replay degeneration Jim flagged in PR #15 |
| Significance | existing multi-seed + **McNemar** (`bench.py`) | reuse what PR #13 shipped |

## Phases

### P0 — Trajectory logging (the one real gap) ⛔ blocker for everything
The runner currently saves only outcomes (`solved`/`attempts`/`saw_regression`) — it **discards the prompts and patch text** a LoRA needs.
- Add an opt-in `--log-traces <dir>` (or env flag) to the bench runner that records, per attempt: `{instance_id, seed, condition, attempt_idx, prompt, completion, patch, passed, fail_to_pass_passed, pass_to_pass_ok}`.
- Additive only — default off, existing result JSONs unchanged, swebench path untouched.
- **Deliverable:** JSONL trace sink; a unit test; one smoke run confirming traces land.

### P1 — Harvest verified traces
- Run the **in-world loop** with the 7B teacher across owsb-atomic + staged (+ contextbench later), `--log-traces` on, **seed varied per attempt**.
- **Ollama discipline (per CLAUDE.md):** cap `num_ctx` (e.g. 8192 — a big teacher defaults to a huge ctx and swaps the GPU), long `timeout` (≥1800s for big models), wrap every LLM call in try/except (a timeout is a miss, not a crash), and since Metal isn't deterministic, take several attempts and keep the verified-best.
- Filter to **passing** trajectories only. Each (prompt → passing patch) becomes a candidate training example. Seed variation yields multiple distinct passing traces per instance even from ~20 instances.
- **Deliverable:** `traces/verified/*.jsonl` + a count report (traces/instance, pass rate).

### P2 — Format + split
- Convert traces → SFT pairs in the **student's chat template** (input = world prompt + `last_errors` context; target = passing patch).
- **Split by instance** into train / held-out-eval. Record the split manifest (sha-pinned) so it's reproducible and contamination-free.
- **Deliverable:** `sft/train.jsonl`, `sft/heldout_instances.json`.

### P3 — LoRA fine-tune (local)
- `mlx_lm.lora` on qwen2.5-1.5b (small rank, early stop, eval-loss watch). Then 3b.
- **Deliverable:** LoRA adapter + training log; fused/exported model for eval.

### P4 — Eval + significance (the finding)
- Re-run the benchmark's **single-shot** condition on **held-out instances** for: base-1.5b, **distilled-1.5b**, base-7b (ceiling anchor). Multi-seed.
- Report solve rates + Wilson CIs + **McNemar** (distilled vs base). Check `pass_to_pass` guardrail.
- **Deliverable:** results JSON + a short writeup with the param-efficiency table.

### P5 — (optional) Iterate / scale
- True flywheel: distilled model generates new traces → retrain. And/or 70B teacher on the H100 for a data-rich run. Gate on P4 showing a real lift.

## Data-sufficiency plan

~20 atomic + staged instances is thin for a generalization claim. Mitigations, in order:
1. **Seed-varied harvest** → multiple passing (prompt,patch) traces per instance (more data without more instances).
2. **Generate more instances** via Jim's dataset factory (`openworld.bench` recipe + the #9 tutorial) for the *training* split; hold out the curated set for eval. (Ask Jim — this is his lane.)

## Risks & mitigations

- **Too few passing traces at small scale** → stronger teacher (7B/70B) + seed variation.
- **Memorization, not learning** → instance-level split; eval only on held-out.
- **Verbatim-replay inflating "lift"** → per-attempt seed variation (PR #15 issue).
- **LoRA overfit on small N** → low rank, early stop, held-out eval, expand instances.
- **Eval contamination** → never train on held-out instances; pin the split manifest.
- **Chat-template mismatch** → student's training template must match the eval harness's prompt format exactly.

## Repo conventions — per Jim's CLAUDE.md (added to main 2026-06-14, PR #32)

These now govern how we contribute; the plan is adjusted to fit:

- **Base the branch directly on CURRENT `main`; PR targets `main`; never stack.** Our local is stale (at `896e0a3`); `git fetch` + update local main to `origin/main` (`29a77ed`) and branch off that — NOT off our old `bench-significance` tip.
- **Zero-dependency framework — no third-party runtime deps in `openworld/*`.** ⚠️ MLX/torch is a third-party dep, so the LoRA-training code must NOT be imported by the framework core. Keep it **quarantined** as an experiment-side tool (e.g. `tooling/distill/` or an `experiments/eNN_*.py` harness that *shells out* to MLX). P0's trace-logging in `bench.py` is fine — it adds no dependency (just writes JSONL).
- **Paper integration = a real experiment ENN via `scripts/make_paper_assets.py`** (next free number, ~**E50**). That means: a results JSON in `experiments/results/`, an `EXPERIMENTS` entry, a `fig_*`/`table_*` fn + call in `main()`, macros before the `numbers.tex` write, and a `\NumExperiments` bump. **Never hand-edit `paper/numbers.tex`.** LaTeX macro names are letters-only (`\LadderQwenSmall`, not `\Qwen7`).
- **Experiment discipline:** deterministic/offline/self-checking, fixed seeds, `save_results` BEFORE asserts, and **be honest** about weak/excluded results in both script and paper. Shared helpers live in `experiments/common.py` (`save_results`, `require_ollama`, sprint world, stats).
- **Don'ts:** never reference CrewAI; no new runtime deps.

## Repo / branch hygiene

- Branch off **fresh** `main` (e.g. `distillation-flywheel`); PR → `main`; `git fetch` before any push (shared remote with Jim's cloud agents).
- P0 is additive (a flag) — won't alter existing results or the swebench path.
- Do **not** touch Jim's composite-swe / E-series files. Do **not** push without Anderson's OK.
- Heavy runs (harvest, training) go through `bgjob --notify quome`, never inline.

## Open questions for Jim

1. The 70B teacher — which exact model, and served where (H100 endpoint)?
2. Dataset factory — OK to generate N additional instances for the training split?

## First concrete step

Implement **P0** (trace logging) + a smoke harvest with the 7B teacher tonight, locally, so traces start existing. Everything else unblocks from there.
