import os, sys
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
