import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from e124 import codex_iso

def test_audit_flags_game_source_read():
    events = [{"type": "exec", "command": "cat experiments/ka59.py"}]
    assert codex_iso.audit_events(events, "ka59") is True

def test_audit_flags_arc_agi_and_envfiles():
    assert codex_iso.audit_events([{"type": "exec", "command": "python -c 'import arc_agi'"}], "ka59") is True
    assert codex_iso.audit_events([{"type": "file_read", "path": "/x/environment_files/ka59/ka59.py"}], "ka59") is True

def test_audit_clean_when_no_source_touched():
    events = [{"type": "agent_message", "text": "here is the json"},
              {"type": "exec", "command": "ls"}]
    assert codex_iso.audit_events(events, "ka59") is False

def test_build_cmd_uses_schema_jsonl_readonly_cleandir():
    cmd = codex_iso.build_cmd("/t/p.txt", "/t/s.json", "/t/o.json", "/clean", "gpt-5.5")
    s = " ".join(cmd)
    assert cmd[0].endswith("codex") and "exec" in cmd
    assert "--output-schema" in cmd and "/t/s.json" in cmd
    assert "--json" in cmd and "-o" in cmd and "/t/o.json" in cmd
    assert "--cd" in cmd and "/clean" in cmd
    assert "read-only" in s and "-m" in cmd and "gpt-5.5" in cmd

def test_parse_events_extracts_final_and_version(tmp_path):
    # simulate the JSONL event stream + the -o final-message file codex writes
    events = [{"type":"token_count","info":{"model":"gpt-5.5-2026-05"}},
              {"type":"agent_message","text":"done"}]
    jsonl = "\n".join(json.dumps(e) for e in events)
    out = tmp_path/"o.json"; out.write_text(json.dumps({"subgoals":[],"macros":[],"rationale":"x"}))
    parsed = codex_iso.parse_output(jsonl, str(out))
    assert parsed["final"]["rationale"] == "x"
    assert "gpt-5.5" in parsed["model_version"]
