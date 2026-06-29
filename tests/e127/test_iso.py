from experiments.e127 import iso


def test_extract_json_plain():
    d = iso.extract_json('{"engine_src": "class Engine: pass", "rationale": "x"}')
    assert d["engine_src"] == "class Engine: pass"


def test_extract_json_with_fence_and_prose():
    txt = 'Here is my answer:\n```json\n{"engine_src": "class Engine:\\n    pass", "rationale": "ok {nested}"}\n```\nDone.'
    d = iso.extract_json(txt)
    assert "class Engine" in d["engine_src"] and d["rationale"] == "ok {nested}"


def test_extract_json_none_on_garbage():
    assert iso.extract_json("no json here") is None


def test_run_uses_injected_exec_no_real_llm():
    canned = '{"engine_src": "class Engine:\\n    def reset(self): return None", "rationale": "r"}'
    calls = {}
    def fake_exec(cmd, cwd, timeout):
        calls["cmd"] = cmd; calls["cwd"] = cwd
        return canned
    out = iso.run("PROMPT", model="claude", game="toy", _exec=fake_exec)
    assert "class Engine" in out["engine_src"] and out["rationale"] == "r"
    # isolation: claude invocation denies tools and loads no MCP servers
    joined = " ".join(calls["cmd"])
    assert "--disallowedTools" in joined and "--strict-mcp-config" in joined


def test_run_malformed_reply_returns_none_src():
    out = iso.run("PROMPT", model="codex", _exec=lambda c, w, t: "the model rambled, no json")
    assert out["engine_src"] is None
