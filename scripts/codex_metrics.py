#!/usr/bin/env python3
"""Per-game telemetry for the SOURCE-FAITHFUL codex full-game sweep. Parses
scratch_arc/codex_<game>/{solved.json,agent.log} into experiments/results/codex_full_game.json so the run is
inspectable for issues (levels reached, errors, whether it read source, wall time, command count)."""
import json, sys, re, os
ROOT = "/Users/jim/Desktop/openworld"
g = sys.argv[1]
wd = f"{ROOT}/scratch_arc/codex_{g}"
out = f"{ROOT}/experiments/results/codex_full_game.json"
levels = win = 0
sj = f"{wd}/solved.json"
if os.path.exists(sj):
    try:
        d = json.load(open(sj)); levels = int(d.get("levels", 0)); win = int(d.get("win", 0))
    except Exception:
        pass
txt = open(f"{wd}/agent.log", errors="ignore").read() if os.path.exists(f"{wd}/agent.log") else ""
n_err = len(re.findall(r"Traceback|Error:|Exception|FAILED", txt))
read_src = bool(re.search(r"environment_files|" + re.escape(g) + r"\.py", txt))
n_cmds = txt.count("/bin/zsh -lc") + txt.count("/bin/bash -lc")
wall = 0
if os.path.exists(f"{wd}/agent.log") and os.path.exists(f"{wd}/TASK.md"):
    wall = int(os.path.getmtime(f"{wd}/agent.log") - os.path.getmtime(f"{wd}/TASK.md"))
rec = {"game": g, "levels": levels, "win": win, "full": bool(win and levels >= win),
       "n_errors": n_err, "read_source": read_src, "wall_s": max(0, wall), "n_cmds": n_cmds, "model": "gpt-5.5"}
allr = json.load(open(out)) if os.path.exists(out) else {}
allr[g] = rec
json.dump(allr, open(out, "w"), indent=2, sort_keys=True)
print(f"[codex_metrics] {g}: levels={levels}/{win} full={rec['full']} errors={n_err} "
      f"read_source={read_src} wall={rec['wall_s']}s cmds={n_cmds}")
