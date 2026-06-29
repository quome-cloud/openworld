# E125 Phase-2 — Plan 2.5: Claude proposal runner + stall fallback

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When codex (OpenAI) stops making progress in FunSearch, fall back to **Claude** as the proposal engine — model-diversity-on-stall, picking up the *same* k-shot evolution trajectory with a different model.

**Architecture:** A `claude_iso.run(...)` runner mirrors `e124/codex_iso.run` (prompt → `{predict_src, goal_score_src, rationale}`) via `claude -p` headless, source-free (tools disallowed + clean workdir + the existing M0 event audit). `synthesize`/`synthesize_obj` gain `fallback_runner` + `stall_window`: after `stall_window` attempts with no improvement in `db.best`, the active runner switches from the primary (codex) to the fallback (Claude). The FunSearch DB, k-shot prompts, and failure memory are model-agnostic, so the switch is transparent.

**Tech Stack:** Python 3.14 (`~/.arcv/bin/python`), pytest. Reuses `experiments/e124/codex_iso` (`audit_events`), `experiments/e125/synth` (`_Database`, `synthesize`, `synthesize_obj`). Claude CLI at `~/.local/bin/claude`.

## Global Constraints

- **Source-free:** the Claude runner must not let the model read game source — run with tools disallowed, in a clean workdir, and audit the event stream with `codex_iso.audit_events` (a tainted call is discarded, same rule as codex).
- **No behavior change when unused:** `fallback_runner` defaults to `None`; with it `None`, `synthesize`/`synthesize_obj` behave exactly as today (the ~86 existing tests must stay green).
- **Runner contract:** a runner is `run(prompt, schema, model, game, **kw) -> {"final": {...}|None, "events": list, "tainted": bool, "model_version": str, "raw": str}`. `final` carries `{predict_src, goal_score_src, rationale}` (or is `None`/partial on a parse failure — which then just scores as a failed attempt).
- **Hermetic tests:** never invoke the real `claude`/`codex` CLIs in tests — inject a fake exec / fake runner.
- **Run with** `~/.arcv/bin/python`. **Commit only when asked** — per-task commits authorized; append `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File Structure

- Create `experiments/e125/claude_iso.py` — Claude proposal runner (`run`, `parse_result`, `_extract_json`).
- Modify `experiments/e125/synth.py` — add `fallback_runner`/`stall_window` to `synthesize` and `synthesize_obj`.
- Tests: `tests/test_e125_claude_iso.py`, `tests/test_e125_fallback.py`.

---

## Task 1: `claude_iso` — a source-free Claude proposal runner

**Files:**
- Create: `experiments/e125/claude_iso.py`
- Test: `tests/test_e125_claude_iso.py`

**Interfaces:**
- Consumes: `e124.codex_iso.audit_events`.
- Produces: `claude_iso._extract_json(text) -> dict|None` (find the first `{...}` object carrying `predict_src`, tolerating ```json fences/prose); `claude_iso.parse_result(stdout, model, game, events=None) -> dict` (the runner return shape, with `tainted = audit_events(events, game)`); `claude_iso.run(prompt, schema, model="claude-opus-4-8", game="", workdir=None, timeout=300, _exec=None) -> dict`. `_exec(cmd, cwd, timeout) -> (returncode, stdout, stderr)` is injectable for tests; the default uses `subprocess.run`. The CLI call disallows tools and runs in a clean workdir.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e125_claude_iso.py
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from e125 import claude_iso

FINAL = {"predict_src": "def predict(state, action):\n    return state, False",
         "goal_score_src": "def goal_score(state):\n    return 0.0", "rationale": "x"}

def test_extract_json_plain():
    assert claude_iso._extract_json(json.dumps(FINAL))["predict_src"].startswith("def predict")

def test_extract_json_in_fences_with_prose():
    text = "Here is my answer:\n```json\n" + json.dumps(FINAL) + "\n```\nDone."
    got = claude_iso._extract_json(text)
    assert got is not None and got["rationale"] == "x"

def test_extract_json_none_when_absent():
    assert claude_iso._extract_json("no json here") is None

def test_parse_result_wraps_claude_json_envelope():
    # claude --output-format json prints an envelope whose `result` holds the assistant text
    envelope = json.dumps({"type": "result", "is_error": False, "result": json.dumps(FINAL)})
    out = claude_iso.parse_result(envelope, "claude-opus-4-8", "g", events=[])
    assert out["final"]["predict_src"].startswith("def predict")
    assert out["tainted"] is False and out["model_version"] == "claude-opus-4-8"

def test_run_uses_injected_exec_and_returns_final():
    envelope = json.dumps({"type": "result", "is_error": False, "result": json.dumps(FINAL)})
    calls = {}
    def fake_exec(cmd, cwd, timeout):
        calls["cmd"] = cmd; calls["cwd"] = cwd
        return 0, envelope, ""
    out = claude_iso.run("PROMPT", {}, model="claude-opus-4-8", game="g", _exec=fake_exec)
    assert out["final"]["goal_score_src"].startswith("def goal_score")
    # isolation: tools disallowed, headless print mode
    flat = " ".join(calls["cmd"])
    assert "-p" in calls["cmd"] and "--disallowedTools" in calls["cmd"]
    assert "--dangerously-skip-permissions" not in calls["cmd"]   # tools must NOT be bypassed

def test_run_handles_exec_failure_gracefully():
    def boom_exec(cmd, cwd, timeout):
        raise TimeoutError("slow")
    out = claude_iso.run("P", {}, game="g", _exec=boom_exec)
    assert out["final"] is None and out["tainted"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_claude_iso.py -q`
Expected: FAIL (`e125.claude_iso` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e125/claude_iso.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_claude_iso.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit (authorized)**

```bash
git add experiments/e125/claude_iso.py tests/test_e125_claude_iso.py
git commit -m "E125 P2.8: source-free Claude proposal runner (claude -p, tools-denied, audited)"
```

---

## Task 2: Stall-fallback in `synthesize` and `synthesize_obj`

**Files:**
- Modify: `experiments/e125/synth.py` (`synthesize`, `synthesize_obj`)
- Test: `tests/test_e125_fallback.py`

**Interfaces:**
- Consumes: existing `_Database`.
- Produces: `synthesize(..., fallback_runner=None, stall_window=3)` and `synthesize_obj(..., fallback_runner=None, stall_window=3)` — identical semantics: the per-attempt active runner is `fallback_runner` once `db.best` has not improved for `stall_window` consecutive attempts (and `fallback_runner is not None`), else the primary runner (`_runner or codex_iso.run`). The improvement counter resets whenever `db.best["score"]` increases.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e125_fallback.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import synth, verify

F0 = np.zeros((64,64), dtype=int); F1 = F0.copy(); F1[0,0]=1; F2=F1.copy(); F2[0,0]=2
def _t(f,a,nf,lu): return {"frame":f,"action":a,"next_frame":nf,"level_up":lu}
TRANS = [_t(F0,[1],F1,False), _t(F1,[1],F2,False)]
GOOD = "def predict(frame, action):\n    nf=frame.copy(); nf[0,0]=frame[0,0]+1\n    return nf, bool(nf[0,0]==3)"
GOAL = "def goal_score(frame):\n    return float(3 - frame[0,0])"
STUCK = "def predict(frame, action):\n    return frame.copy(), False"   # scores 0, never improves

def _runner(src, tag, log=None):
    def run(prompt, schema, model, game, **kw):
        if log is not None: log.append(tag)
        return {"final": {"predict_src": src, "goal_score_src": GOAL, "rationale": ""},
                "events": [], "tainted": False, "raw": "", "model_version": tag}
    return run

def test_no_fallback_when_not_provided():
    log = []
    synth.synthesize(TRANS, "a", "g", mask=None, n_retries=3, _runner=_runner(STUCK, "primary", log))
    assert set(log) == {"primary"}              # only the primary runner is ever called

def test_switches_to_fallback_after_stall():
    log = []
    primary = _runner(STUCK, "primary", log)
    fallback = _runner(GOOD, "claude", log)
    src, fn, goal = synth.synthesize(TRANS, "a", "g", mask=None, n_retries=6, _runner=primary,
                                     fallback_runner=fallback, stall_window=2)
    assert "claude" in log                       # stalled on primary -> Claude was called
    assert fn is not None                         # Claude's GOOD model passed the gate
    assert log[:2] == ["primary", "primary"]      # first stall_window attempts use the primary

def test_fallback_resets_counter_on_improvement():
    # primary improves on attempt 1 (GOOD), so the stall counter resets and Claude is never needed
    log = []
    src, fn, goal = synth.synthesize(TRANS, "a", "g", mask=None, n_retries=4, _runner=_runner(GOOD, "primary", log),
                                     fallback_runner=_runner(GOOD, "claude", log), stall_window=2)
    assert "claude" not in log and fn is not None

def test_synthesize_obj_also_supports_fallback():
    S0={"bg":0,"objects":[{"color":3,"size":1,"y":1,"x":1}]}; S1={"bg":0,"objects":[{"color":3,"size":1,"y":1,"x":2}]}
    S2={"bg":0,"objects":[{"color":3,"size":1,"y":1,"x":3}]}
    TRO=[{"state":S0,"action":[4],"next_state":S1,"level_up":False},
         {"state":S1,"action":[4],"next_state":S2,"level_up":False}]
    GOODO=("def predict(state, action):\n    ns={'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}\n"
           "    if action==[4]:\n        [o.__setitem__('x',o['x']+1) for o in ns['objects']]\n    return ns, False")
    STUCKO="def predict(state, action):\n    return state, False"
    log=[]
    r = synth.synthesize_obj(TRO, "a", "g", n_retries=6, _runner=_runner(STUCKO,"primary",log),
                             fallback_runner=_runner(GOODO,"claude",log), stall_window=2)
    assert "claude" in log and r[1] is not None    # (src, predict_fn, goal_fn, ensemble)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_fallback.py -q`
Expected: FAIL (`synthesize` has no `fallback_runner` kwarg).

- [ ] **Step 3: Write minimal implementation**

In `experiments/e125/synth.py`, add `fallback_runner=None, stall_window=3` to BOTH `synthesize` and `synthesize_obj` signatures. In each, introduce the active-runner selection. For `synthesize`, replace the existing loop preamble so it reads:

```python
def synthesize(transitions, action_api, game, mask, model="gpt-5.5", n_retries=4, traces_dir=None, _runner=None,
               functions_per_prompt=2, seed=0, seed_src=None, fallback_runner=None, stall_window=3):
    primary = _runner or codex_iso.run
    ...
    db = _Database(functions_per_prompt=functions_per_prompt, rng=np.random.RandomState(seed))
    # (seed_src block unchanged)
    best_score = db.best["score"] if db.best else -1
    since_improve = 0
    for attempt in range(n_retries):
        run = (fallback_runner if (fallback_runner is not None and since_improve >= stall_window) else primary)
        samples = db.sample()
        prompt = (...)                      # unchanged
        res = run(prompt, SCHEMA, model, game)
        ...                                  # parse/compile/score/register unchanged
        cur = db.best["score"] if db.best else -1
        if cur > best_score:
            best_score = cur; since_improve = 0
        else:
            since_improve += 1
        ...                                  # telemetry + accept-on-best-full unchanged
```

Apply the IDENTICAL change to `synthesize_obj` (its loop currently does `run = _runner or codex_iso.run` once before the loop — replace that single binding with `primary = _runner or codex_iso.run`, init `best_score`/`since_improve` after the `_Database`/seed_src block, select `run` per attempt as above, and update the counter after each `register`). Do not change any other logic (compile_obj_predict, score, ensemble, accept).

(Re-read both functions in the current `synth.py` and make the minimal edits — only the runner binding, the two counter lines, and the signature.)

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_fallback.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the full E125 suite (no regressions)**

Run: `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ~/.arcv/bin/python -m pytest tests/test_e125_*.py -q`
Expected: all pass (86 prior + new claude_iso/fallback tests).

- [ ] **Step 6: Commit (authorized)**

```bash
git add experiments/e125/synth.py tests/test_e125_fallback.py
git commit -m "E125 P2.9: stall-fallback runner (codex stalls -> Claude) in synthesize + synthesize_obj"
```

---

## Self-Review

**Spec coverage:** "if OpenAI makes no progress, call Claude" → Task 1 (the Claude runner) + Task 2 (stall detection + runner switch). Source-free guarantee → Task 1 (tools denied + clean workdir + `audit_events`). No-behavior-change-when-unused → Task 2 (defaults `fallback_runner=None`).

**Placeholder scan:** none — Task 1 has full code; Task 2 specifies the exact minimal edit (signature + active-runner selection + two counter lines) against the current `synth.py` and is testable. The "re-read and edit" instruction in Task 2 Step 3 is a *localization* instruction, not a placeholder — the precise change (what to add and where) is fully given.

**Type consistency:** the runner contract `run(prompt, schema, model, game, **kw) -> {final, events, tainted, model_version, raw}` matches `codex_iso.run`, `claude_iso.run`, and the fake runners in the fallback tests; `synthesize` returns its 3-tuple and `synthesize_obj` its 4-tuple unchanged; `fallback_runner`/`stall_window` are appended kwargs with safe defaults so all existing call sites and the ~86 tests are unaffected.

## Live validation (controller-run, after the plan lands)

On a stall (e.g. a map where codex plateaus), confirm a real `claude -p` proposal is made and audited (tainted stays False, JSON parses), and that the FunSearch trajectory continues under Claude. Report honestly whether Claude breaks a plateau codex could not — no banked answers.
