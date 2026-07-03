import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e124 import codex_goalc

FRAMES = [np.zeros((64,64), dtype=int)]
API = "g.step(a); g.step(6,x,y)"

def _runner_returning(obj):
    def run(prompt, schema, model, game, **kw):
        return {"final": obj, "events": [{"type":"agent_message"}], "tainted": False,
                "raw": "RAW", "model_version": "gpt-5.5-test"}
    return run

def test_compile_parses_subgoals_and_macros(tmp_path):
    obj = {"subgoals":[{"name":"reach","predicate_src":"def predicate(frame):\n return frame.sum()>0"}],
           "macros":[[[1],[1],[6,12,30]]], "rationale":"go"}
    g = codex_goalc.compile_goal(FRAMES, API, "dyn", "ka59", 0, 0, n=1,
                                 traces_dir=str(tmp_path), _runner=_runner_returning(obj))
    assert not g.abstained and g.subgoals[0][0] == "reach" and g.macros == [[[1],[1],[6,12,30]]]

def test_compile_abstains_when_tainted(tmp_path):
    def run(prompt, schema, model, game, **kw):
        return {"final": {"subgoals":[],"macros":[],"rationale":""}, "events":[], "tainted": True,
                "raw":"", "model_version":""}
    g = codex_goalc.compile_goal(FRAMES, API, "dyn", "ka59", 0, 0, n=1,
                                 traces_dir=str(tmp_path), _runner=run)
    assert g.abstained

def test_compile_writes_telemetry(tmp_path):
    obj = {"subgoals":[],"macros":[],"rationale":"x"}
    codex_goalc.compile_goal(FRAMES, API, "dyn", "ka59", 0, 0, n=1,
                             traces_dir=str(tmp_path), _runner=_runner_returning(obj))
    assert os.path.exists(tmp_path/"calls.jsonl")
    import json
    rec = json.loads(open(tmp_path/"calls.jsonl").read().splitlines()[-1])
    assert rec["decision"] == "abstain"
