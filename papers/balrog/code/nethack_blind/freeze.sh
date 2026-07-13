#!/bin/bash
# Phase-3 code freeze: record md5s of everything that defines the agent.
cd /data/doh/teams/researchy/work/fable_nethack_blind
OUT=results/FREEZE_MD5S.txt
{
  echo "# Frozen at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  md5sum world_model.py policy_explore.py runner.py blind_env.py run_batch.py \
         rules.json tiles_learned.json monsters_learned.json
} > "$OUT"
cat "$OUT"
