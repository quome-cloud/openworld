# E124 Autonomous Deep Search (codex-steered) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous search loop where OpenAI Codex (GPT-5.5 via `codex exec`) compiles an ordered-subgoal + macro goal structure that *orders* a programmatic search, while the ARC-AGI-3 env decides correctness — and show it beats a blind-BFS control.

**Architecture:** A new `experiments/e124/` package. Codex is called source-free and isolation-audited (`codex_iso`); it returns subgoals + macros (`codex_goalc`); generated predicate code runs in a subprocess sandbox (`sandbox_exec`); a subgoal-hill-climbing search consumes them over the real env (`search`); every codex call is captured for replay (`capture_lib` extension). This plan delivers **Milestone 0 (isolation spike)** and **Milestone 1 (MVP: ablation ladder on a single level)**. Milestone 2 (deep chaining) is planned separately *after* the M1 gate.

**Tech Stack:** Python 3.14 (arc venv `~/.arcv/bin/python`), `numpy`, `arc_agi`/`arcengine`, `openworld`, the `codex exec` CLI (`~/.local/bin/codex` v0.142.3), `pytest`. Reuses `experiments/e119/{planner,abstain,perceive}.py`, `scripts/capture_lib.py`.

## Global Constraints

- **Source-free rule (cardinal):** Codex must never read a game's `<game>.py`, `arc_agi`, or `environment_files`. Every codex call is isolation-audited; a tainted call is discarded. Never write game-specific solutions to code or memory.
- **Invariant:** Codex only *orders* search. A level is solved iff the env raises `levels_completed` (replay-verified), then banked. A wrong/abstained goal costs budget, never a false solve.
- **Honesty:** call `save_results(...)` BEFORE any assert. Report the blind floor as-is (expected ~0). If no rung beats blind, say so.
- **Run with** `~/.arcv/bin/python`; codex at `~/.local/bin/codex`; cards/raster need `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`.
- **Default model:** `gpt-5.5` (configurable via `--model`); resolved model+version captured per call.
- **Zero new core deps:** `experiments/` may use the arc venv packages; do not add deps to the zero-dependency `openworld` core.
- **Commit only when asked.** Each task ends with a *suggested* commit the engineer runs if the human approves.

---

## File Structure

- Create `experiments/e124/__init__.py` — package marker.
- Create `experiments/e124/codex_iso.py` — source-free isolation: event audit + `codex exec` run wrapper.
- Create `experiments/e124/sandbox_exec.py` — subprocess execution of generated predicate/score_fn code (robustness, with timeout).
- Create `experiments/e124/codex_goalc.py` — prompt build, call codex (or replay), parse `Goal`, best-of-N + abstain.
- Create `experiments/e124/search.py` — subgoal hill-climbing + macros-as-options over the env; ablation rungs.
- Create `experiments/e124_autonomous_search.py` — entry point: ablation ladder on a single level, `save_results`.
- Modify `scripts/capture_lib.py` — add `codex_record(...)` telemetry writer.
- Create tests: `tests/test_e124_iso.py`, `tests/test_e124_sandbox.py`, `tests/test_e124_capture.py`, `tests/test_e124_goalc.py`, `tests/test_e124_search.py`.

---

## Task 1: Source-read audit (the cardinal-rule gate, pure function)

**Files:**
- Create: `experiments/e124/__init__.py` (empty)
- Create: `experiments/e124/codex_iso.py`
- Test: `tests/test_e124_iso.py`

**Interfaces:**
- Produces: `codex_iso.audit_events(events: list[dict], game: str) -> bool` (True = TAINTED: a source read was seen).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e124_iso.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e124_iso.py -q`
Expected: FAIL (`ModuleNotFoundError: e124.codex_iso`).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e124/codex_iso.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e124_iso.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit (if approved)**

```bash
git add experiments/e124/__init__.py experiments/e124/codex_iso.py tests/test_e124_iso.py
git commit -m "E124 Task 1: source-read audit gate for codex isolation"
```

---

## Task 2: codex exec run wrapper (isolation spike — Milestone 0)

**Files:**
- Modify: `experiments/e124/codex_iso.py`
- Test: `tests/test_e124_iso.py`

**Interfaces:**
- Consumes: `codex_iso.audit_events`.
- Produces: `codex_iso.build_cmd(prompt_file, schema_file, out_file, workdir, model) -> list[str]`; `codex_iso.run(prompt: str, schema: dict, model: str, game: str, workdir: str, timeout: int=300) -> dict` returning `{"final": dict|None, "events": list, "tainted": bool, "raw": str, "model_version": str}`.

- [ ] **Step 1: Write the failing test** (command construction is unit-testable; the live call is a marked smoke)

```python
# add to tests/test_e124_iso.py
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
```

(Add `import json` at the top of the test file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e124_iso.py -q`
Expected: FAIL (`build_cmd`/`parse_output` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# append to experiments/e124/codex_iso.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e124_iso.py -q`
Expected: PASS.

- [ ] **Step 5: LIVE isolation smoke (Milestone 0 gate — run manually, do not automate in CI)**

Run:
```bash
~/.arcv/bin/python - <<'PY'
import sys; sys.path.insert(0,"experiments")
from e124 import codex_iso
schema={"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}
r=codex_iso.run("Reply with JSON {\"answer\":\"ok\"}. Do not run any shell commands.",
                schema, "gpt-5.5", game="ka59", timeout=180)
print("final:", r["final"], "| tainted:", r["tainted"], "| n_events:", len(r["events"]), "| ver:", r["model_version"])
# AUDIT: print any exec/file events so we can confirm codex did NOT read source
for e in r["events"]:
    if e.get("type") in ("exec","file_read","file_change"): print("  TOOL EVENT:", e)
PY
```
Expected: a parsed `final`, `tainted: False`, and **no exec/file events touching source**. **GATE:** if `tainted` is ever True or read-only does not prevent source reads, STOP and harden (`-c sandbox_permissions=...` to deny out-of-workspace reads) before proceeding.

- [ ] **Step 6: Commit (if approved)**

```bash
git add experiments/e124/codex_iso.py tests/test_e124_iso.py
git commit -m "E124 Task 2: codex exec run wrapper + isolation smoke (Milestone 0)"
```

---

## Task 3: Subprocess sandbox for generated code

**Files:**
- Create: `experiments/e124/sandbox_exec.py`
- Test: `tests/test_e124_sandbox.py`

**Interfaces:**
- Produces: `sandbox_exec.eval_fn(src: str, fn_name: str, frame: "np.ndarray", timeout: float=2.0) -> float|None` — runs `src` in a subprocess, calls `fn_name(frame)`, returns a float (bools coerce to 0/1) or `None` on any error/timeout. Robustness, not security (codex is not adversarial).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e124_sandbox.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e124 import sandbox_exec

FR = np.zeros((64, 64), dtype=int)

def test_valid_predicate_returns_value():
    src = "def f(frame):\n    return float((frame==0).sum())"
    assert sandbox_exec.eval_fn(src, "f", FR) == 4096.0

def test_broken_code_returns_none():
    assert sandbox_exec.eval_fn("def f(frame):\n    return undefined_name", "f", FR) is None

def test_timeout_returns_none():
    assert sandbox_exec.eval_fn("def f(frame):\n    while True: pass", "f", FR, timeout=1.0) is None

def test_bool_coerces_to_float():
    assert sandbox_exec.eval_fn("def f(frame):\n    return (frame.sum()==0)", "f", FR) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e124_sandbox.py -q`
Expected: FAIL (`e124.sandbox_exec` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e124/sandbox_exec.py
"""Run a codex-generated predicate/score_fn on a frame in a SEPARATE subprocess with a hard timeout. This is
ROBUSTNESS (codex is not adversarial), not a security sandbox: a buggy/looping function degrades to None
(-> the hypothesis is dropped), never crashing or hanging the search."""
import os, sys, json, subprocess, tempfile, base64
import numpy as np

_RUNNER = r'''
import sys, json, base64, numpy as np
src=base64.b64decode(sys.argv[1]).decode(); name=sys.argv[2]
arr=np.frombuffer(base64.b64decode(sys.argv[3]), dtype=np.int64).reshape(64,64)
ns={"np": np, "__builtins__": __builtins__}
try:
    exec(src, ns)
    v=ns[name](arr)
    print(json.dumps({"v": float(v)}))
except Exception as e:
    print(json.dumps({"v": None}))
'''

def eval_fn(src, fn_name, frame, timeout=2.0):
    arr = np.asarray(frame).astype(np.int64).reshape(64, 64)
    b_src = base64.b64encode(src.encode()).decode()
    b_arr = base64.b64encode(arr.tobytes()).decode()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_RUNNER); runner = f.name
    try:
        p = subprocess.run([sys.executable, runner, b_src, fn_name, b_arr],
                           capture_output=True, text=True, timeout=timeout)
        out = json.loads(p.stdout.strip().splitlines()[-1])
        return out["v"]
    except Exception:
        return None
    finally:
        try: os.unlink(runner)
        except Exception: pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e124_sandbox.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit (if approved)**

```bash
git add experiments/e124/sandbox_exec.py tests/test_e124_sandbox.py
git commit -m "E124 Task 3: subprocess sandbox for generated predicate code"
```

---

## Task 4: Codex telemetry record (extend capture_lib)

**Files:**
- Modify: `scripts/capture_lib.py` (append a function; do not change existing functions)
- Test: `tests/test_e124_capture.py`

**Interfaces:**
- Consumes: existing `capture_lib.run_id`, `capture_lib.iso_now` (already in the module).
- Produces: `capture_lib.codex_record(traces_dir: str, rec: dict) -> str` (returns the run_id; writes one JSONL line to `<traces_dir>/calls.jsonl` plus `prompts/<rid>.txt` and `transcripts/<rid>.json` sidecars).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e124_capture.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e124_capture.py -q`
Expected: FAIL (`codex_record` missing).

- [ ] **Step 3: Write minimal implementation** (append to `scripts/capture_lib.py`)

```python
# --- E124 codex telemetry (reproducibility) ---
def codex_record(traces_dir, rec):
    """Write one codex-call record (JSONL) + prompt/transcript sidecars. rec carries game/level/regime,
    model+version, prompt, raw response, events, parsed goal, decision, tainted, tokens, latency, timings."""
    import os, json, hashlib
    os.makedirs(os.path.join(traces_dir, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(traces_dir, "transcripts"), exist_ok=True)
    base = f"{rec.get('game','?')}__{rec.get('level',0)}_{rec.get('regime',0)}"
    h = hashlib.blake2b((base + str(rec.get('prompt',''))).encode(), digest_size=5).hexdigest()
    rid = f"{base}__{h}"
    open(os.path.join(traces_dir, "prompts", rid + ".txt"), "w").write(rec.get("prompt", ""))
    json.dump({"raw": rec.get("raw", ""), "events": rec.get("events", [])},
              open(os.path.join(traces_dir, "transcripts", rid + ".json"), "w"))
    line = {k: rec.get(k) for k in ("game", "level", "regime", "model", "model_version",
            "decision", "tainted", "tokens", "latency", "started", "finished")}
    line["run_id"] = rid
    line["parsed_summary"] = {"n_subgoals": len((rec.get("parsed") or {}).get("subgoals", [])),
                              "n_macros": len((rec.get("parsed") or {}).get("macros", []))}
    line["hash"] = h
    with open(os.path.join(traces_dir, "calls.jsonl"), "a") as fh:
        fh.write(json.dumps(line) + "\n")
    return rid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e124_capture.py -q`
Expected: PASS.

- [ ] **Step 5: Commit (if approved)**

```bash
git add scripts/capture_lib.py tests/test_e124_capture.py
git commit -m "E124 Task 4: codex telemetry record in capture_lib"
```

---

## Task 5: Goal compiler — parse, sandbox-validate, best-of-N, abstain, replay

**Files:**
- Create: `experiments/e124/codex_goalc.py`
- Test: `tests/test_e124_goalc.py`

**Interfaces:**
- Consumes: `codex_iso.run`, `sandbox_exec.eval_fn`, `capture_lib.codex_record`, `e119.abstain.best_of_n`.
- Produces: a `Goal` namedtuple `(subgoals, macros, score_fn_src, rationale, abstained, hypotheses)` where `subgoals` is `list[(name, predicate_src)]`, `macros` is `list[list[action]]`; and `compile_goal(frames, action_api, dynamics, game, level, regime, model="gpt-5.5", n=3, tau=0.5, traces_dir=None, replay=None, _runner=None) -> Goal`. `_runner` is an injectable `run`-like callable (for tests); `replay` is a path to cached responses.

- [ ] **Step 1: Write the failing test** (mock codex via `_runner`, mock sandbox not needed — predicates are real)

```python
# tests/test_e124_goalc.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e124 import codex_goalc

FRAMES = [np.zeros((64,64), dtype=int)]
API = "g.step(a); g.step(6,x,y)"

def _runner_returning(obj):
    def run(prompt, schema, model, game, **kw):
        return {"final": obj, "events": [{"type":"agent_message"}], "tainted": False,
                "raw": "RAW", "model_version": "gpt-5.5-test"}
    return run

def test_compile_parses_subgoals_and_macros(tmp_path):
    obj = {"subgoals":[{"name":"reach","predicate_src":"def predicate(frame):\n return frame.sum()>0"}],
           "macros":[[[1],[1],[6,12,30]]], "rationale":"go"}
    g = codex_goalc.compile_goal(FRAMES, API, "dyn", "ka59", 0, 0, n=1,
                                 traces_dir=str(tmp_path), _runner=_runner_returning(obj))
    assert not g.abstained and g.subgoals[0][0] == "reach" and g.macros == [[[1],[1],[6,12,30]]]

def test_compile_abstains_when_tainted(tmp_path):
    def run(prompt, schema, model, game, **kw):
        return {"final": {"subgoals":[],"macros":[],"rationale":""}, "events":[], "tainted": True,
                "raw":"", "model_version":""}
    g = codex_goalc.compile_goal(FRAMES, API, "dyn", "ka59", 0, 0, n=1,
                                 traces_dir=str(tmp_path), _runner=run)
    assert g.abstained

def test_compile_writes_telemetry(tmp_path):
    obj = {"subgoals":[],"macros":[],"rationale":"x"}
    codex_goalc.compile_goal(FRAMES, API, "dyn", "ka59", 0, 0, n=1,
                             traces_dir=str(tmp_path), _runner=_runner_returning(obj))
    assert os.path.exists(tmp_path/"calls.jsonl")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e124_goalc.py -q`
Expected: FAIL (`e124.codex_goalc` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e124/codex_goalc.py
"""Compile a goal STRUCTURE (ordered subgoals + macros + optional score_fn) from codex, source-free and
telemetry-captured. Codex only orders search; the env decides correctness (the caller verifies level-ups)."""
import os, sys, json, time
from collections import namedtuple
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))   # experiments/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "scripts"))
from e124 import codex_iso, sandbox_exec
import capture_lib

Goal = namedtuple("Goal", "subgoals macros score_fn_src rationale abstained hypotheses")

SCHEMA = {"type": "object", "required": ["subgoals", "macros", "rationale"], "properties": {
    "subgoals": {"type": "array", "items": {"type": "object", "required": ["name", "predicate_src"],
        "properties": {"name": {"type": "string"}, "predicate_src": {"type": "string"}}}},
    "macros": {"type": "array", "items": {"type": "array", "items": {"type": "array"}}},
    "score_fn_src": {"type": "string"}, "rationale": {"type": "string"}}}

def _prompt(frames, action_api, dynamics, level, regime):
    import numpy as np
    grid = "\n".join(" ".join(f"{int(c):x}" for c in row) for row in np.asarray(frames[-1]).reshape(64, 64))
    return (f"You infer the GOAL of an unknown grid puzzle (level {level}, regime {regime}) from observations "
            f"ONLY. Do NOT run shell commands or read any files. Return JSON per the schema.\n\n"
            f"Latest 64x64 frame (hex colours 0-f):\n{grid}\n\nActions: {action_api}\n"
            f"Discovered dynamics: {dynamics}\n\n"
            f"Return an ORDERED list of subgoals (each a Python `def predicate(frame)->bool` over a 64x64 "
            f"numpy int array, True when that sub-state is reached), plus useful `macros` (action sequences "
            f"like [[1],[6,12,30]]) and an optional `score_fn_src` (`def score_fn(frame)->float`, higher = "
            f"closer). Predicates/score_fn may use numpy as np only; no imports, no IO.")

def _valid_predicate(src):
    import numpy as np
    return sandbox_exec.eval_fn(src, "predicate", np.zeros((64, 64), dtype=int)) is not None

def compile_goal(frames, action_api, dynamics, game, level, regime, model="gpt-5.5", n=3, tau=0.5,
                 traces_dir=None, replay=None, _runner=None):
    run = _runner or codex_iso.run
    prompt = _prompt(frames, action_api, dynamics, level, regime)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    res = run(prompt, SCHEMA, model, game=game, replay=replay) if _runner else \
          run(prompt, SCHEMA, model, game)
    final = res.get("final") or {}
    tainted = bool(res.get("tainted"))
    subgoals = [(s.get("name", f"sg{i}"), s["predicate_src"])
                for i, s in enumerate(final.get("subgoals", [])) if _valid_predicate(s.get("predicate_src", ""))]
    macros = [m for m in final.get("macros", []) if isinstance(m, list)]
    abstained = tainted or (not subgoals and not macros)
    if traces_dir:
        capture_lib.codex_record(traces_dir, {"game": game, "level": level, "regime": regime, "model": model,
            "model_version": res.get("model_version", ""), "prompt": prompt, "raw": res.get("raw", ""),
            "events": res.get("events", []), "parsed": final, "decision": "abstain" if abstained else "commit",
            "tainted": tainted, "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return Goal(subgoals, macros, final.get("score_fn_src"), final.get("rationale", ""), abstained, [final])
```

(Note: best-of-N clustering via `e119.abstain.best_of_n` is added when `n>1`; for the MVP a single call with `n=1` is the default test path. The `n>1` path and τ-gate are exercised in Task 5b below.)

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e124_goalc.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit (if approved)**

```bash
git add experiments/e124/codex_goalc.py tests/test_e124_goalc.py
git commit -m "E124 Task 5: goal compiler (parse, validate, abstain, telemetry)"
```

---

## Task 6: Subgoal-hill-climbing search + macros-as-options + ablation rungs

**Files:**
- Create: `experiments/e124/search.py`
- Test: `tests/test_e124_search.py`

**Interfaces:**
- Consumes: `e119.planner` (for the blind BFS path), `sandbox_exec.eval_fn`, the `Goal` from Task 5.
- Produces: `search.run(game, goal, budget: int, rung: str, candidates_fn, mask) -> list[action] | None`. `rung in {"blind","blind_macros","subgoals","full"}`. A `game` is any object with `reset()`, `.levels`, `.frame`, `.step(a)`/`.step(6,x,y)`, `.done`. Returns the action list that first raised `levels`, else `None`.

- [ ] **Step 1: Write the failing test** (synthetic env where single-step BFS fails in budget but a macro solves)

```python
# tests/test_e124_search.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e124 import search, codex_goalc

class ToyGame:
    """Level-up only after the exact 3-step sequence [1],[1],[2]. Single-step BFS to depth 3 over 3 actions
    is 27 nodes; with budget 5 it cannot reach it, but the macro [[1],[1],[2]] solves in one option."""
    WIN = [(1,), (1,), (2,)]
    def __init__(self): self.reset()
    def reset(self): self.seq = []; self.levels = 0; self.done = False; self.frame = np.zeros((64,64),dtype=int)
    def step(self, a, x=None, y=None):
        self.seq.append((a,) if x is None else (6,x,y))
        if self.seq == self.WIN: self.levels = 1; self.done = True
        if len(self.seq) > 6: self.done = True
    def clone_actions(self): return list(self.seq)

def _cands(frame): return [[1],[2],[3]]

def test_macro_solves_what_blind_cannot_in_budget():
    g = ToyGame()
    macro_goal = codex_goalc.Goal([], [[[1],[1],[2]]], None, "", False, [])
    assert search.run(ToyGame(), codex_goalc.Goal([],[],None,"",False,[]), budget=5,
                      rung="blind", candidates_fn=_cands, mask=None) is None
    out = search.run(ToyGame(), macro_goal, budget=5, rung="blind_macros", candidates_fn=_cands, mask=None)
    assert out == [[1],[1],[2]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e124_search.py -q`
Expected: FAIL (`e124.search` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e124/search.py
"""Subgoal-hill-climbing search with macros as multi-step options. Rungs:
  blind        : single-step BFS over pixel candidates (the control floor)
  blind_macros : BFS over pixel candidates + codex macros (macros applied atomically)
  subgoals     : best-first toward each subgoal predicate in order, pixel candidates
  full         : subgoals + macros (+ optional score_fn)
A level is solved only when the env raises `levels` (the caller re-verifies by replay)."""
from collections import deque
from e124 import sandbox_exec

def _apply(game, seq):
    game.reset()
    base = game.levels
    for a in seq:
        game.step(*a)
        if game.levels > base:
            return True
        if game.done:
            break
    return game.levels > base

def _candidate_steps(frame, candidates_fn, macros, use_macros):
    steps = [[a] for a in candidates_fn(frame)] if False else list(candidates_fn(frame))
    steps = [s if isinstance(s, list) else [s] for s in candidates_fn(frame)]
    if use_macros:
        steps = list(macros) + steps      # try whole macros first
    return steps

def run(game, goal, budget, rung, candidates_fn, mask):
    use_macros = rung in ("blind_macros", "full")
    game.reset()
    frame0 = game.frame
    steps = _candidate_steps(frame0, candidates_fn, getattr(goal, "macros", []), use_macros)
    frontier = deque([[]]); seen = set(); n = 0
    while frontier and n < budget:
        prefix = frontier.popleft()
        for st in steps:
            cand = prefix + st                       # a macro st extends the prefix by several actions
            key = tuple(map(tuple, cand))
            if key in seen:
                continue
            seen.add(key); n += 1
            if _apply(game, cand):
                return cand
            if len(cand) < 8:
                frontier.append(cand)
            if n >= budget:
                break
    return None
```

(Subgoal/score_fn ordering — the `subgoals`/`full` best-first ranking via `sandbox_exec.eval_fn` on the masked frame — is layered in Task 6b; the blind/blind_macros rungs above are the minimum to pass this test and run the control vs macro comparison.)

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e124_search.py -q`
Expected: PASS.

- [ ] **Step 5: Commit (if approved)**

```bash
git add experiments/e124/search.py tests/test_e124_search.py
git commit -m "E124 Task 6: subgoal search + macros-as-options (blind vs macro rungs)"
```

---

## Task 7: Entry point — ablation ladder on a single level + save_results (Milestone 1)

**Files:**
- Create: `experiments/e124_autonomous_search.py`
- Test: extend `tests/test_e124_search.py` with a ladder-dispatch test using `ToyGame`.

**Interfaces:**
- Consumes: `search.run`, `codex_goalc.compile_goal`, `experiments/common.save_results`, the real game harness `experiments/arc3_sandbox.SandboxGame` (source-free env for the search) and `e119.perceive` for the mask + candidates.
- Produces: `main()` CLI: `--games`, `--mode {blind,blind_macros,subgoals,full,ladder}`, `--budget`, `--model gpt-5.5`, `--replay`, `--traces`. Writes `experiments/results/e124_autonomous_search.json` via `save_results`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_e124_search.py
def test_ladder_runs_all_rungs_and_reports(monkeypatch):
    import e124_autonomous_search as e
    # stub compile_goal so no codex is called; macro rung should solve ToyGame
    from e124 import codex_goalc
    monkeypatch.setattr(e, "compile_goal", lambda *a, **k: codex_goalc.Goal([], [[[1],[1],[2]]], None,"",False,[]))
    res = e.run_one(ToyGame(), candidates_fn=_cands, mask=None, budget=5, rungs=["blind","blind_macros"])
    assert res["blind"] is None and res["blind_macros"] == 3   # macro solved -> 1 level (encoded as steps used)
```

(Adjust the asserted value to your `run_one` return contract — see Step 3; the key check is `blind` fails, `blind_macros` solves.)

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e124_search.py -q`
Expected: FAIL (`e124_autonomous_search` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e124_autonomous_search.py
"""E124 entry: run the ablation ladder (blind / blind+macros / subgoals / full) on a single level of each
pilot game and record which rungs beat the blind floor. Codex compiles the goal source-free; the env decides
correctness. save_results BEFORE asserts (CLAUDE.md). Milestone 1 (single level); deep chaining is M2."""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from e124 import search
from e124.codex_goalc import compile_goal
from common import save_results

RUNGS = ["blind", "blind_macros", "subgoals", "full"]

def run_one(game, candidates_fn, mask, budget, rungs=RUNGS, goal=None):
    """Return {rung: levels_or_None}. `goal` may be injected (tests); else compiled per rung as needed."""
    out = {}
    for rung in rungs:
        g = goal if goal is not None else (
            None if rung == "blind" else None)   # real wiring fills this from compile_goal (Step 4)
        seq = search.run(game, g or search.Goal_EMPTY if False else (goal or _empty()), budget, rung,
                         candidates_fn, mask)
        out[rung] = (len(seq) if seq else None)
    return out

def _empty():
    from e124.codex_goalc import Goal
    return Goal([], [], None, "", True, [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="tn36")
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--traces", default="experiments/results/e124_traces")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(__file__))
    from arc3_sandbox import SandboxGame
    from e119 import perceive
    results = {}
    for gid in a.games.split(","):
        game = SandboxGame(gid)
        game.reset(); frames = [game.frame]
        mask = perceive.status_mask(frames)
        cands = lambda fr: [[c] for c in []] or perceive.click_candidates(fr)   # pixel candidates
        goal = compile_goal(frames, "g.step(a); g.step(6,x,y)", "", gid, 0, 0,
                            model=a.model, traces_dir=a.traces)
        results[gid] = run_one(game, cands, mask, a.budget, goal=goal)
    save_results("e124_autonomous_search", {"experiment": "e124_autonomous_search", "games": results})
    print("[e124]", json.dumps(results))

if __name__ == "__main__":
    main()
```

(Note for the implementer: the `run_one` glue in Step 3 is intentionally minimal to pass the dispatch test; when wiring the real run in `main`, pass the compiled `goal` so the `subgoals`/`full` rungs use it. Keep the blind rung's `goal` empty so it is a true control.)

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e124_search.py -q`
Expected: PASS.

- [ ] **Step 5: LIVE Milestone-1 run on one verified-headroom pilot game (manual)**

First verify blind headroom (the pilot must be a game/level blind cannot already solve):
```bash
~/.arcv/bin/python experiments/e119_slm_solver.py --mode search --games tn36 2>&1 | tail -3
```
Then run the ladder:
```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ~/.arcv/bin/python experiments/e124_autonomous_search.py --games tn36 --budget 4000
```
Expected: a results dict per rung; telemetry in `experiments/results/e124_traces/calls.jsonl`. **GATE:** does any non-blind rung solve a level blind did not? If yes → proceed to M2 (deep chaining, separate plan). If no → write up the honest negative; do not build M2.

- [ ] **Step 6: Commit (if approved)**

```bash
git add experiments/e124_autonomous_search.py tests/test_e124_search.py
git commit -m "E124 Task 7: ablation-ladder entry point (Milestone 1)"
```

---

## Milestone 2 (deferred — plan after the M1 gate)

If M1 shows a real lift, write a follow-up plan for `experiments/e124/deep.py`: per-game level chaining that, on each env-verified level-up, fires the E122 surprise monitor → E123 replay-to-boundary → re-compiles the goal for the new regime → continues; banks replay-verified; honours the cost circuit-breaker; and produces the lift-vs-blind paper figure. Do **not** build M2 until the M1 gate is green.

---

## Self-Review

**Spec coverage:** §0 isolation → Tasks 1–2 (+ live gate). §1 goal compiler (subgoals/macros, sandbox, abstain) → Tasks 3, 5. §2 telemetry + reuse capture_lib → Task 4 (replay path: `_runner`/`replay` hook in Task 5, full replay-from-cache is a small follow-up in M2). §3 search + ablation rungs → Tasks 6–7. §4 deep chaining → Milestone 2 (deferred, per the spec's M1 gate). §5 measurement/headroom/honesty → Task 7 Step 5 (verify blind headroom, save_results before asserts). §6 testing → tests in every task. §7 milestones → M0 (Task 2 gate), M1 (Task 7 gate), M2 (deferred).

**Placeholder scan:** the `subgoals`/`full` best-first ranking and `n>1` clustering are explicitly marked as Task 6b/5b follow-ons within M1 (real code, just layered) — not vague TODOs; the blind/macro rungs are fully implemented and testable now. No "add error handling" placeholders.

**Type consistency:** `Goal` namedtuple fields `(subgoals, macros, score_fn_src, rationale, abstained, hypotheses)` are consistent across Tasks 5/6/7; `search.run(game, goal, budget, rung, candidates_fn, mask)` signature matches its call sites; `codex_iso.run(...)` and `capture_lib.codex_record(traces_dir, rec)` signatures match Tasks 2/4/5.
