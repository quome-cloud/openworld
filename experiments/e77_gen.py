"""E77 (generation) - a coding WORLD family: LLM-authored function-implementation tasks,
each a verified world (spec + hidden pytest oracle), for the world-time-compute realism
check on coding.

A coding task is the cleanest verified-code world we have: state = the function under
implementation, the transition is execution, and the ORACLE is the tests passing. We have
an LLM author diverse tasks (prompt + reference solution + asserts), then VERIFY each in a
sandboxed subprocess (the reference solution must pass its own tests). Only verified tasks
enter the family -- the test suite is ground truth, exactly the OpenWorld stance.

This stage is offline w.r.t. GPU (authoring via Gemini REST + local subprocess verification).
Writes experiments/results/e77_artifacts/tasks.jsonl. Reads GEMINI_API_KEY from .env.
"""

import json
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "experiments" / "results" / "e77_artifacts"
GEN_MODEL = "gemini-2.5-flash"
TOPICS = ["string manipulation", "list/array processing", "dictionaries and counting",
          "basic math and number theory", "recursion", "sorting and searching",
          "parsing simple formats", "matrix/grid operations", "intervals and ranges",
          "stack/queue logic", "greedy selection", "simple dynamic programming"]
PER_TOPIC = 30          # scaled run
TIMEOUT_S = 10
SEED = 77


def load_key():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("GEMINI_API_KEY=") and not line.strip().startswith("#"):
                return line.split("=", 1)[1].strip()
    import os
    return os.environ.get("GEMINI_API_KEY", "")


def gemini_code(prompt, key, retries=4):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEN_MODEL}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": 0.8, "maxOutputTokens": 2048,
                                    "thinkingConfig": {"thinkingBudget": 0}}}
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read().decode("utf-8"))
            return d["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            return None
        except Exception:  # noqa: BLE001
            return None
    return None


GEN_PROMPT = """Create ONE self-contained Python function-implementation coding task on the topic: {topic}.
Return ONLY a JSON object (no markdown fences) with exactly these keys:
- "name": short snake_case task id
- "prompt": the function signature and a clear docstring describing what to implement (NO solution code)
- "solution": the complete correct function definition (the reference implementation)
- "tests": a list of 4-6 Python assert statements that call the function and check outputs
The function must be pure (no I/O, no randomness, stdlib only). Make it non-trivial but solvable.
Example shape: {{"name":"...","prompt":"def f(x):\\n    \\"\\"\\"...\\"\\"\\"","solution":"def f(x):\\n    return ...","tests":["assert f(1)==2","..."]}}"""


def parse_task(txt):
    if not txt:
        return None
    s, e = txt.find("{"), txt.rfind("}")
    if s < 0 or e < 0:
        return None
    try:
        t = json.loads(txt[s:e + 1])
    except Exception:  # noqa: BLE001
        return None
    if not all(k in t for k in ("name", "prompt", "solution", "tests")) or not t["tests"]:
        return None
    return t


def verify(task):
    """The reference solution must pass its own tests in a sandboxed subprocess."""
    program = task["solution"] + "\n\n" + "\n".join(task["tests"]) + "\nprint('OK')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as f:
        f.write(program)
        f.flush()
        try:
            r = subprocess.run([sys.executable, f.name], capture_output=True,
                               text=True, timeout=TIMEOUT_S)
            return r.returncode == 0 and "OK" in r.stdout
        except Exception:  # noqa: BLE001
            return False


def main():
    ART.mkdir(parents=True, exist_ok=True)
    key = load_key()
    if not key:
        raise SystemExit("Set GEMINI_API_KEY in .env")
    rng = random.Random(SEED)
    tasks, attempts, verified = [], 0, 0
    seen_names = set()
    for topic in TOPICS:
        for _ in range(PER_TOPIC):
            attempts += 1
            t = parse_task(gemini_code(GEN_PROMPT.format(topic=topic), key))
            time.sleep(0.5)
            if not t:
                continue
            if not verify(t):
                continue
            name = t["name"]
            if name in seen_names:
                name = f"{name}_{rng.randint(100, 999)}"
            seen_names.add(name)
            tasks.append({"name": name, "topic": topic, "prompt": t["prompt"],
                          "solution": t["solution"], "tests": t["tests"]})
            verified += 1
            print(f"  [ok] {topic:28} -> {name}", flush=True)
    (ART / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    print(f"[e77-gen] verified {verified}/{attempts} tasks (model {GEN_MODEL}) -> {ART/'tasks.jsonl'}")


if __name__ == "__main__":
    main()
