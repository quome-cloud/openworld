"""A source-free Claude proposal runner mirroring e124.codex_iso.run, for FunSearch model-diversity fallback.
Runs `claude -p` headless with tools DISALLOWED in a clean workdir (the model cannot read game source), instructs
strict-JSON output {predict_src, goal_score_src, rationale}, parses it, and returns the result.

Source-free isolation is STRUCTURAL, not audited:
  - `--disallowedTools` denies all file-read/run tools (Bash, Read, Grep, Glob, LS, NotebookRead, …).
  - `--permission-mode default` auto-denies any other tool request in headless mode.
  - `--strict-mcp-config` with no `--mcp-config` argument loads ZERO MCP servers,
    blocking inherited user-scope MCP servers that could otherwise read files.
  - The cwd is a clean tempdir containing no game source.
Because no tool can execute, `events` is empty BY DESIGN; `tainted` stays False for
contract parity with codex_iso (not because an audit occurred — with `--output-format json`
there is no event stream to audit). No --output-schema exists for claude, so JSON is
prompt-instructed and parsed leniently; a malformed reply just scores as a failed attempt.
Claude only PROPOSES; the gate decides."""
import os, sys, json, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from e124 import codex_iso

CLAUDE = os.path.expanduser("~/.local/bin/claude")
# tools that could read game source / run code -- denied for a pure proposal call
_DENY = "Bash,Read,Edit,Write,Glob,Grep,WebFetch,WebSearch,Task,NotebookEdit,MultiEdit,LS,NotebookRead"


def _extract_json(text):
    """Return the first JSON object containing 'predict_src' from text (tolerating ```json fences / prose).

    Uses a brace-balanced scanner: walks left-to-right tracking depth while skipping string
    literals (handling \\" escapes), so braces inside predict_src (dict literals, function
    bodies) do not truncate the candidate prematurely.  Handles (a) plain JSON, (b) ```json
    fences with prose before/after, (c) prose containing stray braces after the object.
    Single O(n) pass; no catastrophic backtracking.
    """
    if not text:
        return None
    i = 0
    n = len(text)
    while i < n:
        start = text.find('{', i)
        if start == -1:
            break
        depth = 0
        j = start
        end = None
        while j < n:
            ch = text[j]
            if ch == '"':
                # skip over a JSON string literal, honouring backslash escapes
                j += 1
                while j < n:
                    c2 = text[j]
                    if c2 == '\\':
                        j += 2          # skip escaped char (including \\")
                    elif c2 == '"':
                        j += 1
                        break
                    else:
                        j += 1
            elif ch == '{':
                depth += 1
                j += 1
            elif ch == '}':
                depth -= 1
                j += 1
                if depth == 0:
                    end = j
                    break
            else:
                j += 1
        if end is not None:
            candidate = text[start:end]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and "predict_src" in obj:
                    return obj
            except Exception:
                pass
        i = start + 1   # advance past the '{' we just tried and look for the next one
    return None


def parse_result(stdout, model, game, events=None):
    """Wrap `claude --output-format json` stdout into the runner return shape."""
    text = stdout or ""
    try:
        env = json.loads(text)
        if isinstance(env, dict) and "result" in env:
            text = env.get("result") or ""
    except Exception:
        pass                              # not an envelope -> treat stdout as the raw text
    final = _extract_json(text)
    # events is always [] (no event stream with --output-format json); tainted=False by structural design,
    # kept here for contract parity with codex_iso (not because an audit was performed).
    return {"final": final, "events": events or [], "tainted": codex_iso.audit_events(events or [], game),
            "model_version": model, "raw": stdout or ""}


def _default_exec(cmd, cwd, timeout):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def run(prompt, schema, model="claude-opus-4-8", game="", workdir=None, timeout=300, _exec=None):
    # schema is unused: claude has no --output-schema flag; JSON shape is enforced via the prompt instruction
    ex = _exec or _default_exec
    workdir = workdir or tempfile.mkdtemp(prefix="e125_claude_")   # clean dir: NO game source here
    os.makedirs(workdir, exist_ok=True)
    full = (prompt + "\n\nReturn ONLY a single JSON object with keys predict_src, goal_score_src, rationale. "
            "No prose, no markdown fences.")
    cmd = [CLAUDE, "-p", full, "--model", model, "--output-format", "json",
           "--permission-mode", "default", "--disallowedTools", _DENY, "--strict-mcp-config"]
    try:
        rc, out, err = ex(cmd, workdir, timeout)
    except Exception as e:
        return {"final": None, "events": [], "tainted": False, "model_version": model, "raw": "",
                "rc": None, "error": f"{type(e).__name__}: {e}"}
    result = parse_result(out, model, game, events=[])
    result["rc"] = rc
    result["stderr"] = err
    return result
