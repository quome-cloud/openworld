# Verified-trace distillation — pipeline & runbook

Tooling for the param-efficiency thesis: harvest verified (prompt→passing-patch) traces from a
**teacher** model, LoRA-fine-tune a small **student**, and test whether the student's single-shot
solve rate beats its own base. Plan: `docs/superpowers/plans/2026-06-13-verified-trace-distillation-flywheel.md`.
Findings so far: `LRN-openworld-teacher-scaling-not-the-lever-2026-06-15` (in the harness vault).

## Two roles — only one is trained

- **Teacher** = solves bugs to generate traces. **Pure inference, never trained.** Any platform
  (served via Ollama). 7B/14b run on a 24GB Mac; 70B needs a big GPU (H100).
- **Student** = the small model we **LoRA-fine-tune** on the traces, then eval. This is the only
  training step.

## ⚠️ PLATFORM SPLIT — READ THIS IF YOU ARE NOT ON A MAC

The scripts in this directory (`to_mlx_data.py`, `run_*_chain.sh`, and the training/eval calls in
them) use **Apple MLX (`mlx_lm`). MLX runs on Apple Silicon ONLY — it will not run on Linux/CUDA.**

The pipeline therefore has a platform-agnostic half and a Mac-only half:

| step | platform | notes |
|---|---|---|
| **harvest** (`openworld.bench … --log-traces`) | **any** | pure Python + Ollama HTTP; cross-platform |
| `format_traces.py` | **any** | reads traces JSONL → `train.jsonl` + `heldout_instances.json`; portable |
| `to_mlx_data.py` | Mac-only | converts to MLX data dir; MLX-specific |
| LoRA train (`mlx_lm lora`) | **Mac-only** | Apple MLX |
| `eval_heldout.py` | Mac-only **as written** | uses `mlx_lm` to load model/adapter; BUT the scoring (`openworld.swebench.run_instance_tests`) is pure/zero-dep and portable |

**If you are on a non-Mac box (e.g. Jim's CUDA/H100 machine), do ONE of:**

- **Option A — harvest only (recommended).** Run just the harvest (below), then ship the resulting
  `traces/<dir>/*.traces.jsonl` back to the Mac, which runs `format → train → eval`. The heavy GPU
  work (the 70B teacher) is exactly the part that *belongs* on your box; training a 1.5b–7b student
  is small and lives fine on the Mac. **No CUDA porting needed.**

- **Option B — full pipeline on CUDA (only if you must train there).** MLX is not portable, so:
  1. Run `harvest` then `format_traces.py` (both portable) → you have `sft/<v>/train.jsonl`
     (plain chat-format JSONL: `{"messages":[{role,content}...]}`) and `heldout_instances.json`.
  2. **Skip `to_mlx_data.py`** (MLX-only). Feed `train.jsonl` to a CUDA LoRA trainer — HuggingFace
     `peft` + `transformers` + `bitsandbytes` (4-bit). Mirror the hyperparams used here: LoRA rank 8,
     lr 1e-5, ~300 steps, seq len 2048, target the attention/MLP projections (MLX's `num_layers 8`
     ≈ apply LoRA to the last 8 transformer blocks; HF applies to all matched modules by default —
     either is fine for a first pass).
  3. For eval, **do not** reuse `eval_heldout.py` as-is (it imports `mlx_lm`). Reimplement the loop
     with your CUDA inference, but **reuse the scoring**: `from openworld.swebench import
     load_dataset, run_instance_tests, _base_prompt, extract_code, SYSTEM_PROMPT` — these are pure,
     zero-dependency, and platform-agnostic. Score = `run_instance_tests(patch, inst)["solved"]`.
     Emit the same result JSON shape (`base_solved`, `distilled_solved`, per-instance, McNemar
     discordant) so results stay comparable to the Mac runs.

## Pipeline (Mac path)

Env: **`.venv-distill`** (Python 3.12 + `mlx-lm`). The repo's main `.venv` is 3.14 and has **no
MLX** — don't use it for training. All commands run from the **repo root** with
`PYTHONPATH=<repo root>` (the `openworld` package lives at the root, and running a script by path
otherwise puts the script's own dir on `sys.path` instead — the `run_*_chain.sh` scripts already
`export PYTHONPATH`).

```bash
# 1. Harvest verified traces from the teacher (cross-platform; swap --models for your teacher)
for r in owsb-atomic-v1 owsb-staged-v1; do
  python -m openworld.bench recipes/$r.json run \
    --models <teacher-ollama-name> --seeds 5 --log-traces traces/harvest-<tag>
done
# Ollama discipline for big teachers (plan line 51): cap num_ctx (~8192 so it doesn't swap the
# GPU), use a long client timeout (>=1800s), keep the verified-best across seeds.

# 2. Traces -> SFT split (portable)
.venv-distill/bin/python tooling/distill/format_traces.py \
  traces/harvest-<tag>/*.traces.jsonl --out-dir sft/<v>

# 3. SFT -> MLX data (Mac-only)
.venv-distill/bin/python tooling/distill/to_mlx_data.py sft/<v>/train.jsonl --out sft/<v>/mlx-data

# 4. LoRA train the student (Mac-only). Students that fit a 24GB Mac: 1.5b, 3b, 7b (14b is tight).
PYTHONPATH=$PWD .venv-distill/bin/python -m mlx_lm lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train --data sft/<v>/mlx-data --adapter-path tooling/distill/adapters/<name> \
  --iters 300 --batch-size 1 --learning-rate 1e-5 --num-layers 8 --max-seq-length 2048 --save-every 100

# 5. Eval base vs distilled (Mac-only; runs both + McNemar in one shot)
PYTHONPATH=$PWD .venv-distill/bin/python tooling/distill/eval_heldout.py \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --heldout sft/<v>/heldout_instances.json \
  --adapter tooling/distill/adapters/<name> --out tooling/distill/eval/<name>.json
```

The chain scripts `run_v2_chain.sh` (full, incl. a harvest-wait) and `run_3b_chain.sh` (train+eval
only, student swap) wire steps together. `profile_difficulty.py` buckets instances into
base-solves / learnable-band / both-fail (uses the teacher's harvest traces, free).
