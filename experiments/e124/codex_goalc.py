"""Compile a goal STRUCTURE (ordered subgoals + macros + optional score_fn) from codex, source-free and
telemetry-captured. Codex only orders search; the env decides correctness (the caller verifies level-ups)."""
import os, sys, json, time
from collections import namedtuple
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))   # experiments/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "scripts"))
from e124 import codex_iso, sandbox_exec
import capture_lib

Goal = namedtuple("Goal", "subgoals macros score_fn_src rationale abstained hypotheses")

# OpenAI strict-structured-output compliant schema: every object has additionalProperties:false,
# every property is listed in required (nullable for optional fields).
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["subgoals", "macros", "score_fn_src", "rationale"],
    "properties": {
        "subgoals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "predicate_src"],
                "properties": {
                    "name": {"type": "string"},
                    "predicate_src": {"type": "string"}
                }
            }
        },
        "macros": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"}
                }
            }
        },
        "score_fn_src": {"type": ["string", "null"]},
        "rationale": {"type": "string"}
    }
}


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
    if _runner:
        res = run(prompt, SCHEMA, model, game, replay=replay)
    else:
        res = run(prompt, SCHEMA, model, game)
    final = res.get("final") or {}
    tainted = bool(res.get("tainted"))
    subgoals = [(s.get("name", f"sg{i}"), s["predicate_src"])
                for i, s in enumerate(final.get("subgoals", []))
                if _valid_predicate(s.get("predicate_src", ""))]
    macros = [m for m in final.get("macros", []) if isinstance(m, list)]
    abstained = tainted or (not subgoals and not macros)
    if traces_dir:
        capture_lib.codex_record(traces_dir, {
            "game": game, "level": level, "regime": regime, "model": model,
            "model_version": res.get("model_version", ""), "prompt": prompt,
            "raw": res.get("raw", ""), "events": res.get("events", []),
            "parsed": final, "decision": "abstain" if abstained else "commit",
            "tainted": tainted, "started": started,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S")
        })
    return Goal(subgoals, macros, final.get("score_fn_src"), final.get("rationale", ""),
                abstained, [final])
