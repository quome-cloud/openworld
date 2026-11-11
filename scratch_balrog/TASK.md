You are an autonomous agent solving the BALROG Baba Is AI benchmark task **{TASK_ID}**. Work in this directory: /data/doh/teams/researchy/work/openworld/scratch_balrog

Run python with: python3  (baba and baba_harness are installed/present)

Harness (baba_harness.py, already here):
    from baba_harness import Game
    g = Game("{TASK_ID}"); g.reset()
    g.frame  -> 8x8x3 numpy int array (full grid state)
    g.levels, g.done, g.avail (action strings: up/right/down/left)
    g.step(a)          # a in ['up','right','down','left']
    g.agent_pos        # (x, y) of the agent
    g.get_ruleset_text()   # "key is win\nbaba is you"
    g.get_objects()        # {type: [(x,y),...]}
    g.get_win_positions()  # positions of WIN objects (per active ruleset)
    g.clone()          # deep copy for simulation
SUCCESS = g.levels increases. The env is randomized per reset(); replay only
works within the same episode (reset once, plan, execute).

Recipe (executable world model -- the OpenWorld way):
1. EXPLORE: gather (frame, action, next_frame) transitions; learn what each
   action does. Use g.get_ruleset_text() to read active rules. Use g.clone()
   to simulate actions without advancing the real game.
2. MODEL: write predict(frame, action) reproducing observed transitions exactly.
   Baba Is AI is turn-based and deterministic per episode: simulate with clone().
3. GOAL: REASON about the win condition -- what increases g.levels?
   Read the ruleset: "X is WIN" means reaching an X object wins.
   "X is STOP" means X blocks movement. Rule blocks are pushable.
   Form a hypothesis and TEST it with clone().step().
4. PLAN: find an action sequence that completes a level. Use BFS or A* over
   cloned states. Baba Is AI grids are 8x8; BFS is fast.
5. SAVE: write solved.json = {"task":"{TASK_ID}","actions":["up","right",...],"levels":1}

Think hard about the goal. Rule manipulation (push rule blocks to create/break
"X is WIN" or "X is STOP" rules) may be required for make_win/break_stop tasks.
