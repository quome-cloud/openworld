#!/usr/bin/env bash
# FOCUSED FINAL-LEVEL source-free solver. Seeds from the Claude-SF banked frontier (level N-1) and
# dedicates the WHOLE budget to cracking ONLY the final level -- the wall reasoning-only agents stall
# on because they re-derive the whole game each run. Source-free: workdir holds only arc3_sandbox.py
# (no game code) + the OpenWorld object perceptor (objstate.py) + the macro-search tools (e128) as
# SOLVER helpers (they read no game source); every run is audited. The banked frontier is the agent's
# OWN prior source-free solution (not the answer key) -- legitimate to resume from, like best-keeper.
#   Usage: run_arc_agent_final_level.sh <game>
#   Env:   MODEL (default claude-opus-4-8), EFFORT (default high)
set -o pipefail
GAME="$1"
ROOT="/Users/jim/Desktop/openworld"
AGENT_PY="/Users/jim/.pyenv/versions/3.9.18/bin/python"
CLAUDE="/Users/jim/.local/bin/claude"
MODEL="${MODEL:-claude-opus-4-8}"; EFFORT="${EFFORT:-high}"
WD="$ROOT/scratch_arc/fl_$GAME"
mkdir -p "$WD"
cp "$ROOT/experiments/arc3_sandbox.py" "$WD/"                    # the only harness (no source)
cp "$ROOT/experiments/e125/objstate.py" "$WD/"                  # OpenWorld object perceptor (solver tool)
# the agent's OWN banked frontier (level N-1) to resume from
FR="$ROOT/scratch_arc/sb_$GAME/solved_best.json"; [ -f "$FR" ] || FR="$ROOT/scratch_arc/sb_$GAME/solved.json"
[ -f "$FR" ] && cp "$FR" "$WD/frontier.json" || echo '{"actions":[],"levels":0,"win":0}' > "$WD/frontier.json"
N=$("$AGENT_PY" -c "import json;d=json.load(open('$WD/frontier.json'));print(d.get('levels',0))")
W=$("$AGENT_PY" -c "import json;d=json.load(open('$WD/frontier.json'));print(d.get('win',0))")

cat > "$WD/TASK.md" <<TASK
You are solving the FINAL level of the interactive ARC-AGI-3 game **$GAME**, SOURCE-FREE (no game
code). You have already reached **level $N of $W** -- the action sequence that gets there is in
\`frontier.json\` (your own prior solution). Your ONLY job: discover what increments g.levels from
$N to $((N+1)) (and onward to $W). Do NOT re-derive the earlier levels.

Run python with: $AGENT_PY   (numpy; CANNOT import arc_agi).
Replay to the frontier:
    import json; from arc3_sandbox import SandboxGame
    fr = json.load(open("frontier.json"))["actions"]
    g = SandboxGame("$GAME"); g.reset()
    for a in fr: g.step(6,a[1],a[2]) if a[0]==6 else g.step(a[0])   # now at level $N
Tools (SOLVER helpers, not game source -- use them):
    from objstate import object_state, state_key   # OpenWorld object perceptor: frame -> {bg, objects[color,size,y,x]}
    # object_state(g.frame) gives you the objects (positions/colors) -- REASON the win over OBJECTS, not pixels.

How to crack the final level (REASON the win; it is an ordered PROCEDURE, not a state score):
1. From the frontier, perceive the objects. ARC-3 levels ESCALATE the same mechanic -- level $((N+1)) is
   almost certainly the level-$N mechanic, harder. Re-read your earlier levels' logic and EXTEND it.
2. Form a hypothesis for the win condition (what object configuration / ordered interaction raises
   g.levels). The env is DETERMINISTIC: test hypotheses by replaying from the frontier + branching --
   counterfactual probing is cheap and exact. When g.levels rises, you have found a win step; record
   the minimal action subsequence that caused it.
3. Chain to the win. SAVE often: write solved.json = {"game":"$GAME","actions":[...full sequence from
   reset...],"levels":M,"win":$W} whenever you reach a new deepest level M. Each action is [a] or [6,x,y].

EXECUTION DISCIPLINE: write code to .py FILES and run them with $AGENT_PY file.py (single-process; no
multiprocessing). Make ONE SandboxGame and reuse it via reset()+replay. DO NOT read game source or
import arc_agi -- every run is AUDITED; a tainted run is discarded.
TASK

cd "$WD"
"$CLAUDE" -p "$(cat TASK.md)" --model "$MODEL" --effort "$EFFORT" \
  --output-format stream-json --verbose --dangerously-skip-permissions \
  > "$WD/agent.log" 2> "$WD/agent.err"
echo "final-level agent finished for $GAME (was $N/$W)"
