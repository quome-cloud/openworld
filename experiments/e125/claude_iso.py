"""A source-free Claude proposal runner mirroring e124.codex_iso.run, for FunSearch model-diversity fallback.
Runs `claude -p` headless with tools DISALLOWED in a clean workdir (the model cannot read game source), instructs
strict-JSON output {predict_src, goal_score_src, rationale}, parses it, and audits the event stream with the same
M0 audit as codex (a tainted call is discarded). No --output-schema exists for claude, so JSON is prompt-instructed
and parsed leniently; a malformed reply just scores as a failed attempt. Claude only PROPOSES; the gate decides."""
import os, sys, json, re, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from e124 import codex_iso

CLAUDE = os.path.expanduser("~/.local/bin/claude")
# tools that could read game source / run code -- denied for a pure proposal call
_DENY = "Bash,Read,Edit,Write,Glob,Grep,WebFetch,WebSearch,Task,NotebookEdit,MultiEdit"


def _extract_json(text):
    """Return the first JSON object containing 'predict_src' from text (tolerating ```json fences / prose)."""
    if not text:
        return None
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = fenced + re.findall(r"\{.*\}", text, re.S)
    for c in candidates:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if isinstance(obj, dict) and "predict_src" in obj:
            return obj
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
    return {"final": final, "events": events or [], "tainted": codex_iso.audit_events(events or [], game),
            "model_version": model, "raw": stdout or ""}


def _default_exec(cmd, cwd, timeout):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def run(prompt, schema, model="claude-opus-4-8", game="", workdir=None, timeout=300, _exec=None):
    ex = _exec or _default_exec
    workdir = workdir or tempfile.mkdtemp(prefix="e125_claude_")   # clean dir: NO game source here
    os.makedirs(workdir, exist_ok=True)
    full = (prompt + "\n\nReturn ONLY a single JSON object with keys predict_src, goal_score_src, rationale. "
            "No prose, no markdown fences.")
    cmd = [CLAUDE, "-p", full, "--model", model, "--output-format", "json",
           "--permission-mode", "default", "--disallowedTools", _DENY]
    try:
        rc, out, err = ex(cmd, workdir, timeout)
    except Exception as e:
        return {"final": None, "events": [], "tainted": False, "model_version": "", "raw": "",
                "error": f"{type(e).__name__}: {e}"}
    return parse_result(out, model, game, events=[])
