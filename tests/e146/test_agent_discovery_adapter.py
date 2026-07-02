"""Contract test for scripts/run_e146_agent_discovery.sh (the Claude EWM-agent discovery adapter).

Uses a stub `claude` CLI and a stub SandboxGame, so it validates the E146 discovery contract
(env vars -> frontier replay -> TASK.md -> solved.json harvest) with zero live-agent cost.
"""
import json
import os
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ADAPTER = REPO / "scripts" / "run_e146_agent_discovery.sh"

STUB_SANDBOX = '''\
class SandboxGame:
    def __init__(self, game):
        self.game = game
        self.levels = 0
        self.win = 2
        self.done = False
    def reset(self):
        self.levels = 0
    def step(self, a, x=None, y=None):
        self.levels += 1          # every action "gains" a level in the stub
    def close(self):
        pass
'''

# The stub agent ignores its arguments and writes a one-deeper solved.json into its cwd (the
# adapter's WD), exactly what a successful discovery session does.
STUB_CLAUDE = '''#!/usr/bin/env python3
import json
frontier = json.load(open("frontier.json"))
deeper = dict(frontier)
deeper["actions"] = list(frontier["actions"]) + [[1]]
deeper["levels"] = int(frontier.get("levels", 0)) + 1
json.dump(deeper, open("solved.json", "w"))
'''


def test_adapter_contract_end_to_end(tmp_path):
    # fake ROOT with the stub sandbox
    root = tmp_path / "root"
    (root / "experiments").mkdir(parents=True)
    (root / "experiments" / "arc3_sandbox.py").write_text(STUB_SANDBOX)

    # stub claude CLI
    claude = tmp_path / "claude"
    claude.write_text(STUB_CLAUDE)
    claude.chmod(claude.stat().st_mode | stat.S_IEXEC)

    # frontier at level 1 of 2
    stage = tmp_path / "stage0"
    (stage / "discovery").mkdir(parents=True)
    frontier = stage / "frontier.json"
    frontier.write_text(json.dumps({"game": "toy", "actions": [[1]], "levels": 1, "win": 2}))
    solved_out = stage / "discovery" / "solved.json"

    env = dict(os.environ)
    env.update({
        "ROOT": str(root), "CLAUDE": str(claude), "AGENT_TIMEOUT_S": "60",
        "E146_GAME": "toy", "E146_FRONTIER": str(frontier),
        "E146_STAGE_DIR": str(stage), "E146_SOLVED_OUT": str(solved_out),
    })
    proc = subprocess.run(["bash", str(ADAPTER)], env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr

    # the adapter harvested the deeper trace to E146_SOLVED_OUT
    assert solved_out.exists()
    deeper = json.loads(solved_out.read_text())
    assert deeper["levels"] == 2 and deeper["actions"] == [[1], [1]]

    # and the TASK it wrote is source-free + frontier-seeded
    task = (stage / "agent_discovery" / "TASK.md").read_text()
    assert "SOURCE-FREE" in task and "level 1 of 2" in task and "frontier.json" in task


def test_adapter_fails_loudly_without_contract_env(tmp_path):
    env = dict(os.environ)
    for k in ("E146_GAME", "E146_FRONTIER", "E146_STAGE_DIR", "E146_SOLVED_OUT"):
        env.pop(k, None)
    proc = subprocess.run(["bash", str(ADAPTER)], env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0
