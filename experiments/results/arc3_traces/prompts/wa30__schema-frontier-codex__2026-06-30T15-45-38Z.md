You are solving ARC-AGI-3 game **wa30**, SOURCE-FREE. Do not read game code or import arc_agi.
You already have a replay-verified Codex source-free frontier at **level 2 of 9** in `frontier.json`.

This is E137: CROSS-LEVEL PROCEDURAL SCHEMA INDUCTION. Your first input is `schema_packet.json`.
It was built only by replaying your own prior Codex source-free actions and observing frames/level counters.

Run python with: /Users/jim/.pyenv/versions/3.14.6/bin/python

Execution discipline:
- Write Python scripts to files in this directory and run them with `/Users/jim/.pyenv/versions/3.14.6/bin/python script.py`.
- Use only the public SandboxGame API: `reset()`, `step(a)`, and `step(6,x,y)`.
- There is no `g.replay`, no `hard_reset`, and no game source. If you need replay, write a tiny helper:
      def replay(g, actions):
          g.reset()
          for a in actions:
              g.step(6, a[1], a[2]) if a[0] == 6 else g.step(a[0])
- Keep one SandboxGame instance and reuse reset()+replay. Avoid multiprocessing.

Required workflow:
1. Read `schema_packet.json`. Inspect `solved_level_demos` and `candidate_schemas`.
2. Before free-form solving, choose or repair a schema that explains the solved demos. Treat the demos as
   within-game training examples: ARC levels escalate the same procedure.
3. Replay to the frontier:
      import json
      from arc3_sandbox import SandboxGame
      fr = json.load(open("frontier.json"))["actions"]
      g = SandboxGame("wa30")
      replay(g, fr)
4. Bind the schema roles on the frontier frame using:
      from objstate import object_state, state_key
      from ewm_toolkit import plan_in_model, WorldSim, salient_clicks, _act, _replay_to
      from ewm_toolkit import composite_key, select_lens, LENSES
   Use composite_key(frame), not a single object lens, when comparing states.
5. Execute the instantiated procedure. Use small source-free probes only to bind uncertain roles.
   If the level does not rise, explain which schema assumption failed, repair it from the counterexample,
   and try again. Do not restart from generic search until schema attempts are exhausted.
6. Save every deeper frontier immediately:
      solved.json = {"game":"wa30","actions":[...full actions from reset...],"levels":M,"win":9}
   Actions are [a] or [6,x,y]. A clean deeper solve will be audit/replay/OpenWorld banked.

Priority: getting any of ka59/su15/bp35/dc22/g50t/wa30 to full moves source-free toward beating 15/25.
Persist. The goal is not a beautiful theory; it is one more level-up from the current Codex frontier.
