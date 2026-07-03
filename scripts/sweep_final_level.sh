#!/usr/bin/env bash
# E129 -- FOCUSED FINAL-LEVEL source-free solver sweep (the lever to beat SOTA).
#
# Every UNDIRECTED method (random play, 4 goal-discovery methods, E127 reconstruction, micro + macro
# Go-Explore) stalls on ARC-3's procedural walls -- only an agent REASONING the win clears them. This
# sweep concentrates that proven win-reasoning on ONLY the unsolved final level of each near-full game,
# seeded from the agent's OWN deepest banked frontier (level N-1). Source-free + audit-gated + banked.
#
# Rate-limit aware: the Claude 5-hour window is shared, so this runs POOL=2 (not all-at-once) and waits
# out an exhausted window before starting (RESET_AT), then self-heals across rounds. Each round banks
# every fl_ gain into the source-free archive via the attestation gate (audit + replay + OpenWorld
# round-trip) and refreshes per-game telemetry in experiments/results/arc3_final_level.json.
#
#   caffeinate -i nohup bash scripts/sweep_final_level.sh > scratch_arc/fl_sweep.log 2>&1 &
#   env: POOL (2) ROUNDS (4) DELAY (600s between rounds) RESET_AT (unix; wait until then before round 1)
set -uo pipefail
ROOT=/Users/jim/Desktop/openworld
PY=/Users/jim/.arcv/bin/python                       # arc venv (numpy + arc_agi) for banking/telemetry
ARCH="$ROOT/experiments/results/arc3_fullgame_sourcefree.json"
TEL="$ROOT/experiments/results/arc3_final_level.json"
POOL="${POOL:-2}"; ROUNDS="${ROUNDS:-4}"; DELAY="${DELAY:-600}"
RESET_AT="${RESET_AT:-0}"                             # unix ts of the Claude 5-hour window reset (0=now)

# near-full first (shallowest gap -> highest chance), then the rest; only games NOT yet full source-free.
GAMES="m0r0 tu93 ka59 wa30 bp35 dc22 sc25 lf52 g50t su15 ls20 s5i5 sk48 sp80 tn36 vc33 r11l"

full_sf() { "$PY" -c "import json;d=json.load(open('$ARCH')).get('per_game',{}).get('$1',{});print(1 if (d.get('win') and d.get('levels',0)>=d.get('win')) else 0)" 2>/dev/null || echo 0; }

# wait out an exhausted Claude window so round 1 isn't wasted on rejected calls
NOW=$("$PY" -c "import time;print(int(time.time()))")
if [ "$RESET_AT" -gt "$NOW" ]; then
  WAIT=$(( RESET_AT - NOW + 120 ))
  echo "[fl-sweep] Claude window exhausted; sleeping ${WAIT}s until reset ($(date -r "$RESET_AT" '+%H:%M'))"
  sleep "$WAIT"
fi

echo "[fl-sweep] START $(date) pool=$POOL rounds=$ROUNDS delay=${DELAY}s"
for r in $(seq 1 "$ROUNDS"); do
  echo "[fl-sweep] === round $r/$ROUNDS $(date) ==="
  for g in $GAMES; do
    if [ "$(full_sf "$g")" = 1 ]; then echo "[fl-sweep] $g already full -- skip"; continue; fi
    while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$POOL" ]; do sleep 5; done
    echo "[fl-sweep] launch $g $(date '+%H:%M:%S')"
    bash "$ROOT/scripts/run_arc_agent_final_level.sh" "$g" > "$ROOT/scratch_arc/fl_${g}.out" 2>&1 &
  done
  wait
  echo "[fl-sweep] round $r: banking fl_ gains through the attestation gate"
  SF_WD_PREFIX=fl_ SF_ARCH="$ARCH" "$PY" "$ROOT/scripts/autobank_sourcefree.py" 2>&1 | grep -iE "sf-bank" || true
  # refresh per-game telemetry (frontier vs agent-best vs banked)
  "$PY" - "$ARCH" "$TEL" $GAMES <<'PY'
import json, os, sys
arch_path, tel_path = sys.argv[1], sys.argv[2]; games = sys.argv[3:]
ROOT = "/Users/jim/Desktop/openworld"
arch = json.load(open(arch_path)); pg = arch.get("per_game", {})
tel = {}
for g in games:
    banked = pg.get(g, {})
    sp = f"{ROOT}/scratch_arc/fl_{g}/solved.json"
    best = -1
    if os.path.exists(sp):
        try: best = int(json.load(open(sp)).get("levels", -1))
        except Exception: pass
    tel[g] = {"win": banked.get("win", 0), "banked_levels": banked.get("levels", 0),
              "agent_best": best, "full": bool(banked.get("win") and banked.get("levels", 0) >= banked.get("win", 0))}
n_full = sum(1 for v in tel.values() if v["full"])
out = {"experiment": "E129 focused final-level source-free solver", "per_game": tel,
       "n_full_sourcefree": arch.get("n_full_games"), "total_levels": arch.get("total_levels"),
       "total_possible": arch.get("total_possible")}
json.dump(out, open(tel_path, "w"), indent=1)
print(f"[fl-sweep] telemetry: source-free {arch.get('n_full_games')} full, "
      f"{arch.get('total_levels')}/{arch.get('total_possible')} levels")
PY
  [ "$r" -lt "$ROUNDS" ] && { echo "[fl-sweep] sleeping ${DELAY}s before next round"; sleep "$DELAY"; }
done
echo "[fl-sweep] DONE $(date)"
