You are solving ARC-AGI-3 game **su15**, SOURCE-FREE. Do not read game code or import arc_agi.
You have a replay-verified Codex source-free frontier at **level 4 of 9** in `frontier.json`.

This is E138: JUDGE-GUIDED SCHEMA TOURNAMENT.
Inputs:
- `schema_packet.json`: E137 solved-level demos, candidate action schemas, and goal-condition schemas.
- `judge_schema.py`: deterministic proposal ranker. It is not a solver and not a certifier.

Run python with: /Users/jim/.pyenv/versions/3.14.6/bin/python

Execution discipline:
- Write scripts to files and run them with `/Users/jim/.pyenv/versions/3.14.6/bin/python script.py`.
- Use only SandboxGame public API: `reset()`, `step(a)`, `step(6,x,y)`.
- There is no `g.replay`, no `hard_reset`, and no game source.
- If you need replay:
      def replay(g, actions):
          g.reset()
          for a in actions:
              g.step(6, a[1], a[2]) if a[0] == 6 else g.step(a[0])
- Keep one SandboxGame instance and reuse reset()+replay. Avoid multiprocessing.

Required workflow:
1. Read `schema_packet.json` and replay `frontier.json` to inspect the current frame.
2. Create at least 4 distinct proposal JSON files named `proposal_*.json`.
   Each proposal must be structured:
      {
        "proposal_id": "short-name",
        "schema_id": "candidate schema type/description/id",
        "goal_schema_id": "goal schema type/description/id",
        "hypothesis": "what level-up condition this instantiates",
        "role_bindings": {"role": "source-free visual/object evidence"},
        "probe_plan": [[6,x,y], [1], ...],
        "expected_deltas": ["what should change if right"],
        "fallback_repairs": ["specific repair from failed probe"],
        "confidence": 0.0
      }
3. Rank proposals before executing:
      /Users/jim/.pyenv/versions/3.14.6/bin/python judge_schema.py schema_packet.json tournament.json proposal_*.json
   Read `tournament.json`; execute the winner first.
4. Execute only small source-free probes/actions. If a probe fails, write `counterexample.json`,
   repair or add a new `proposal_repair_*.json`, rerank, and try again.
5. Save every deeper frontier immediately:
      solved.json = {"game":"su15","actions":[...full actions from reset...],"levels":M,"win":9}
   Actions are [a] or [6,x,y]. A clean deeper solve will be audit/replay/OpenWorld banked.

The judge/ranker allocates budget; replay and source-free audit are the only proof. Persist on the frontier.
