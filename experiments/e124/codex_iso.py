"""Source-free isolation for codex calls: audit the codex exec --json event stream for any read of game
source, and (Task 2) wrap the codex exec invocation. The read-only sandbox still permits disk reads, so the
audit is the real gate -- a tainted call is discarded (CLAUDE.md cardinal rule)."""
import json, re

# any of these appearing in a shell command or file path = a source read
_SOURCE_RE = re.compile(
    r"(arc_agi|environment_files|arcengine|experiments/[a-z0-9]{4}\.py|/[a-z0-9]{4}\.py\b)", re.I)


def audit_events(events, game):
    """Return True if any event reads game source (the game's <id>.py, arc_agi, environment_files)."""
    gid = re.escape(game)
    needles = re.compile(rf"({_SOURCE_RE.pattern}|{gid}\.py)", re.I)
    for e in events or []:
        blob = " ".join(str(e.get(k, "")) for k in ("command", "path", "cmd", "args", "text"))
        if needles.search(blob):
            # an agent_message merely mentioning the id is not a read; only exec/file events count
            if e.get("type") in ("agent_message", "reasoning", "token_count"):
                continue
            return True
    return False


import os, subprocess, tempfile, time
CODEX = os.path.expanduser("~/.local/bin/codex")


def build_cmd(prompt_file, schema_file, out_file, workdir, model):
    return [CODEX, "exec", "-m", model, "-s", "read-only", "--cd", workdir,
            "--skip-git-repo-check", "--output-schema", schema_file, "--json",
            "-o", out_file, "-", ]   # prompt piped on stdin (the '-' reads stdin)


def parse_output(jsonl_stdout, out_file):
    events = []
    for line in (jsonl_stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            pass
    ver = ""
    for e in events:
        info = e.get("info") or {}
        if isinstance(info, dict) and info.get("model"):
            ver = str(info["model"]); break
    final = None
    try:
        final = json.loads(open(out_file, encoding="utf-8").read())
    except Exception:
        final = None
    return {"final": final, "events": events, "model_version": ver,
            "raw": (open(out_file, encoding="utf-8").read() if os.path.exists(out_file) else "")}


def run(prompt, schema, model, game, workdir=None, timeout=300):
    workdir = workdir or tempfile.mkdtemp(prefix="e124_codex_")   # clean dir: NO game source here
    os.makedirs(workdir, exist_ok=True)
    pf = os.path.join(workdir, "_prompt.txt"); open(pf, "w").write(prompt)
    sf = os.path.join(workdir, "_schema.json"); open(sf, "w").write(json.dumps(schema))
    of = os.path.join(workdir, "_final.json")
    cmd = build_cmd(pf, sf, of, workdir, model)
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
        out = p.stdout
    except Exception as e:
        return {"final": None, "events": [], "tainted": False, "raw": "", "model_version": "",
                "error": f"{type(e).__name__}: {e}"}
    parsed = parse_output(out, of)
    parsed["tainted"] = audit_events(parsed["events"], game)
    return parsed
