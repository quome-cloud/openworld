# ARC-3 agent run logs — preservation snapshots

Raw Claude Code stream-json `agent.log` + `solved.json` + `result.json` for every source-free
ARC-3 agent run (fl_ focused-final-level, sb_ Claude-SF, ge_/gm_ Go-Explore, su_ SHU-cycle).
Snapshotted out of the at-risk `scratch_arc/` (only 1 scratch file is git-tracked) so a scratch
clean can't lose them.

These are RAW traces. The HF-READY DATASET ALREADY EXISTS: scripts/capture_lib.py ->
experiments/results/arc3_traces/ (runs.jsonl + prompts/ + solutions/ + transcripts/ + meta/),
784 runs across all 25 games, committed. The sb_ Claude-SF runs are captured there.

DEFERRED BUILD (reuse capture_lib, do NOT rebuild):
- WIRE scripts/run_arc_agent_final_level.sh + scripts/sweep_final_level.sh + experiments/e130_shu_cycle.py
  to call capture_lib.append_run (as the sb_ runner already does) so fl_/su_ runs land in runs.jsonl.
- BACKFILL the existing fl_ runs (tu93 9/9, m0r0, ka59, ...) from these snapshots via
  capture_lib.summarize_transcript + append_run.

snapshots (belt-and-suspenders for the at-risk raw fl_ logs until the wiring lands):
- runs_2026-06-29.tar.gz  — focused-final-level (tu93 cracked 9/9) + sb_/ge_/gm_ runs to date
