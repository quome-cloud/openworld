"""E148 strategy-space: parser + featurizer + PCA invariants on synthetic transcripts (CI-safe,
no dependence on the live arc3_traces capture)."""
import os, sys, json, gzip
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import e148_strategy_space as e


def _jsonl(tmp, rid, blocks):
    p = tmp / (rid + ".jsonl")
    with open(p, "w") as f:
        for b in blocks:
            f.write(json.dumps({"type": "assistant", "message": {"content": b}}) + "\n")
    return p


def test_parse_transcript_counts_tools_and_builds_doc(tmp_path):
    p = _jsonl(tmp_path, "toy", [
        [{"type": "text", "text": "I will build a simulator and verify by replay."}],
        [{"type": "tool_use", "name": "Bash", "input": {"command": "python sim.py  # beam search"}}],
        [{"type": "tool_use", "name": "Write", "input": {"file_path": "/x/parity_probe.py", "content": "x"}}],
        [{"type": "tool_use", "name": "Read", "input": {}}],
    ])
    r = e.parse_transcript(str(p))
    assert r["counts"]["n_bash"] == 1 and r["counts"]["n_write"] == 1 and r["counts"]["n_read"] == 1
    assert r["counts"]["n_scripts"] == 1                       # parity_probe.py
    assert "simulator" in r["doc"] and "beam" in r["doc"] and "parity_probe.py" in r["doc"]


def test_parse_codex_strips_the_echoed_prompt(tmp_path):
    # codex log: header + echoed TASK (must be stripped) + codex's own turn (kept)
    body = ("model: gpt-5.5\nuser\nYou must FULLY solve ... world model ... replay-verify ...\n"
            "codex\nI will write a lights-out simulator and search it.\n"
            "exec python sim.py\n")
    p = tmp_path / "g__agent-codex__t.codex.log.gz"
    gzip.open(p, "wt").write(body)
    r = e.parse_codex_log(str(p))
    assert "lights-out simulator" in r["doc"]                  # codex's own words kept
    assert "you must fully solve" not in r["doc"]              # echoed prompt stripped
    assert r["counts"]["n_bash"] == 1                          # one exec block


def test_lexicon_detects_strategy_families():
    assert e.re.search(e.LEXICON["simulate"], "i built a simulator to predict")
    assert e.re.search(e.LEXICON["search"], "run a beam search / bfs")
    assert e.re.search(e.LEXICON["mechanic"], "the lights-out parity ladder")
    assert e.re.search(e.LEXICON["memory"], "resume from my notes and toolkit")


def test_pca_is_deterministic_and_sign_fixed():
    rng = np.random.default_rng(0)
    Z = e.zscore(rng.normal(size=(40, 9)))
    c1, l1, v1 = e.pca2(Z)
    c2, l2, v2 = e.pca2(Z)
    assert np.allclose(c1, c2) and np.allclose(l1, l2)         # deterministic
    for k in range(2):                                          # sign convention: largest-|loading| positive
        assert l1[k][np.argmax(np.abs(l1[k]))] > 0
    assert v1[0] >= v1[1]                                       # variance ordered
