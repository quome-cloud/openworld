"""Source-free model runner. Claude/codex PROPOSE engine source from observed play; they never read
game source. Isolation is structural: a clean tempdir with no source, all file/run tools denied, zero
MCP servers. `_exec` is injected in tests so no real LLM is called. The model only proposes; the
real-env certificate decides."""
import os, json, subprocess, tempfile

CLAUDE = os.path.expanduser("~/.local/bin/claude")
CODEX = os.path.expanduser("~/.local/bin/codex")
_DENY = "Bash,Read,Edit,Write,Glob,Grep,WebFetch,WebSearch,Task,NotebookEdit,MultiEdit,LS,NotebookRead"


def extract_json(text):
    """First brace-balanced JSON object containing 'engine_src', tolerating fences/prose."""
    if not text:
        return None
    i, n = 0, len(text)
    while i < n:
        start = text.find("{", i)
        if start == -1:
            return None
        depth = 0; instr = False; esc = False
        for j in range(start, n):
            ch = text[j]
            if instr:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    instr = False
            else:
                if ch == '"':
                    instr = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cand = text[start:j + 1]
                        try:
                            obj = json.loads(cand)
                            if isinstance(obj, dict) and "engine_src" in obj:
                                return obj
                        except Exception:
                            pass
                        break
        i = start + 1
    return None


def _default_exec(cmd, cwd, timeout):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout).stdout


def _cmd(model, prompt):
    if model == "claude":
        return [CLAUDE, "-p", prompt, "--output-format", "text",
                "--disallowedTools", _DENY, "--permission-mode", "default", "--strict-mcp-config"]
    if model == "codex":
        return [CODEX, "exec", "--skip-git-repo-check", "-m", "gpt-5.5", prompt]
    raise ValueError(f"unknown model {model}")


def run(prompt, model, game="", workdir=None, timeout=600, _exec=None):
    _exec = _exec or _default_exec
    wd = workdir or tempfile.mkdtemp(prefix=f"e127_{model}_{game}_")
    try:
        raw = _exec(_cmd(model, prompt), wd, timeout)
    except Exception:
        raw = ""
    obj = extract_json(raw) or {}
    return {"engine_src": obj.get("engine_src"), "rationale": obj.get("rationale", ""), "raw": raw}
