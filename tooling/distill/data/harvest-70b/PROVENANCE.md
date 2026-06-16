# Verified-trace harvest — `llama3.1:70b` (frozen artifact)

Teacher-model traces for verified-trace distillation, pinned for reproducibility.

**This is a frozen snapshot, not regenerable.** Ollama is not byte-deterministic
even at a fixed seed (Metal/CUDA), so re-running the harvest will *not* reproduce
this file. Its purpose is to pin the exact traces the student is trained on, so the
*downstream* `format_traces.py → train → eval` pipeline is reproducible.

| | |
|---|---|
| file | `llama3.1-70b.traces.jsonl` — 452 records, 175 trials |
| teacher | `llama3.1:70b` (official Meta weights; **not** a `*-poisoned*` variant) |
| recipes | `owsb-atomic-v1` (20 instances × 5 seeds) + `owsb-staged-v1` (15 × 5) |
| harvested | 2026-06-16, `python -m openworld.bench <recipe> run --log-traces` on 2× A100 40GB |
| result | atomic single-shot 80% → in-world 95%; staged 13% → 75% |

**Per-line schema:** `instance_id, condition, attempt_idx, seed, model, system,
prompt, completion, patch, passed, fail_to_pass_failed, pass_to_pass_failed`.

The `prompt` fields are the bug reports + buggy module source from
`datasets/openworld-swebench*` — already in this repo; no new/proprietary inputs.

## Built with Llama

These are Llama 3.1 outputs. Under the Llama 3.1 Community License, any model
trained on them must include "Llama" at the start of its name and carry a
**"Built with Llama"** notice. Confirm against the current license text before
publishing the student model or this dataset.
