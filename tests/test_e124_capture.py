import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import capture_lib

def test_codex_record_writes_jsonl_and_sidecars(tmp_path):
    rec = {"game":"ka59","level":0,"regime":0,"model":"gpt-5.5","model_version":"gpt-5.5-2026-05",
           "prompt":"PROMPT","raw":"RAW","events":[{"type":"agent_message"}],
           "parsed":{"subgoals":[]}, "decision":"commit", "tainted":False}
    rid = capture_lib.codex_record(str(tmp_path), rec)
    line = json.loads(open(tmp_path/"calls.jsonl").read().splitlines()[-1])
    assert line["run_id"] == rid and line["game"] == "ka59" and line["decision"] == "commit"
    assert (tmp_path/"prompts"/f"{rid}.txt").read_text() == "PROMPT"
    assert json.loads((tmp_path/"transcripts"/f"{rid}.json").read_text())["raw"] == "RAW"
