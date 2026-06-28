import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from e125 import claude_iso

FINAL = {"predict_src": "def predict(state, action):\n    return state, False",
         "goal_score_src": "def goal_score(state):\n    return 0.0", "rationale": "x"}

def test_extract_json_plain():
    assert claude_iso._extract_json(json.dumps(FINAL))["predict_src"].startswith("def predict")

def test_extract_json_in_fences_with_prose():
    text = "Here is my answer:\n```json\n" + json.dumps(FINAL) + "\n```\nDone."
    got = claude_iso._extract_json(text)
    assert got is not None and got["rationale"] == "x"

def test_extract_json_none_when_absent():
    assert claude_iso._extract_json("no json here") is None

def test_parse_result_wraps_claude_json_envelope():
    # claude --output-format json prints an envelope whose `result` holds the assistant text
    envelope = json.dumps({"type": "result", "is_error": False, "result": json.dumps(FINAL)})
    out = claude_iso.parse_result(envelope, "claude-opus-4-8", "g", events=[])
    assert out["final"]["predict_src"].startswith("def predict")
    assert out["tainted"] is False and out["model_version"] == "claude-opus-4-8"

def test_run_uses_injected_exec_and_returns_final():
    envelope = json.dumps({"type": "result", "is_error": False, "result": json.dumps(FINAL)})
    calls = {}
    def fake_exec(cmd, cwd, timeout):
        calls["cmd"] = cmd; calls["cwd"] = cwd
        return 0, envelope, ""
    out = claude_iso.run("PROMPT", {}, model="claude-opus-4-8", game="g", _exec=fake_exec)
    assert out["final"]["goal_score_src"].startswith("def goal_score")
    # isolation: tools disallowed, headless print mode
    flat = " ".join(calls["cmd"])
    assert "-p" in calls["cmd"] and "--disallowedTools" in calls["cmd"]
    assert "--dangerously-skip-permissions" not in calls["cmd"]   # tools must NOT be bypassed

def test_run_handles_exec_failure_gracefully():
    def boom_exec(cmd, cwd, timeout):
        raise TimeoutError("slow")
    out = claude_iso.run("P", {}, game="g", _exec=boom_exec)
    assert out["final"] is None and out["tainted"] is False
