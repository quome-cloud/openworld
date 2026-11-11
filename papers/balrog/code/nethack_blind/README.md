# Source-Blind NetHack Induction Arm

Crucial-experiment arm: play BALROG's NetHackChallenge with **no environment
source, no external game knowledge** — a world model induced purely from the
agent's own served observations, under continuous Popperian verification
(predict-before-observe possibility sets; anomalies force rule revision).

- `blind_env.py` — interface-only shim over the balrog wrapper (see quarantine audit in FABLE_NETHACK_BLIND_REPORT.md)
- `rules.json` — the world model: every rule carries evidence citations (episode, step) into `results/transitions/*.jsonl.gz`
- `anomalies.jsonl` — anomaly ledger with before/after resolutions
- `world_model.py` — predictive possibility sets + verification
- `policy_explore.py` — planner assembled from ledger rules + flagged EXPERIMENT hooks
- `tiles_learned.json` / `monsters_learned.json` — learned per-glyph tables
- `runner.py`, `run_batch.py`, `analyze.py`, `build_curves.py`, `render_anim.py`, `mine_replay.py` — infrastructure
- `results/` — per-episode transition logs (gzip JSONL), batch summaries, curves, frozen eval, animations

Report: `FABLE_NETHACK_BLIND_REPORT.md`.
