"""E81: composite programming worlds -- generalizing programming principles across language frames.

Each COMPOSITE world is one programming problem; its SUB-WORLDS are that problem realized in N
languages (Python, C++, Java, JS, Go), each a verified code world (transition = the language's
tests executing). A language is a reference FRAME: the same computational principle in different
coordinates. World-time compute traversing the language-frames should learn the frame-invariant
principle (cross-language transfer) and, across problems, generalize programming itself.

Substrate: HumanEval-X (zai-org/humaneval-x) -- 164 problems x 5 languages, each with a canonical
solution (the trajectory to traverse) and tests (the oracle). This file:
  - build_worlds(): the composite structure {problem: {lang: {prompt, declaration, solution, test}}}
  - run_one(lang, program): execute a full program (subprocess, timeout) -> pass/fail
  - validate(): canonical solution + test should PASS per sub-world (the E68-style validation rate)

Phase 1-2 here (structure + verified execution). Phase 3 (world-time compute over the frames)
is e81_wtc.py. Execution needs language runtimes (python3, g++, node, go, javac/java) -- run on
a box. Offline, the structure builds/validates with no runtimes.
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

import urllib.request

HERE = Path(__file__).resolve().parent
LANGS = ["python", "cpp", "java", "js", "go"]
HX = "https://huggingface.co/datasets/zai-org/humaneval-x/resolve/main/data"


def build_worlds(cache=None):
    """{problem_id: {lang: {prompt, declaration, canonical_solution, test, [import,test_setup]}}}."""
    cache = Path(cache) if cache else (HERE / "results" / "e81_progworlds.json")
    if cache.exists():
        return json.loads(cache.read_text())
    data = {}
    for lg in LANGS:
        raw = urllib.request.urlopen(f"{HX}/{lg}/data/humaneval.jsonl", timeout=60).read().decode()
        data[lg] = {r["task_id"].split("/")[-1]: r
                    for r in (json.loads(x) for x in raw.splitlines() if x.strip())}
    common = sorted(set.intersection(*[set(data[lg]) for lg in LANGS]), key=int)
    keep = ("prompt", "declaration", "canonical_solution", "test", "import", "test_setup")
    comp = {p: {lg: {k: data[lg][p].get(k, "") for k in keep} for lg in LANGS} for p in common}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(comp))
    return comp


# ---- per-language program assembly (HumanEval-X conventions) + execution ----

def _entry_point(prompt, declaration):
    m = re.search(r"def\s+(\w+)\s*\(", prompt) or re.search(r"(\w+)\s*\(", declaration)
    return m.group(1) if m else "candidate"


def assemble(lang, sw, body):
    """Full runnable program: sub-world `sw` with the function `body` (solution or generation)."""
    if lang == "python":
        ep = _entry_point(sw["prompt"], sw["declaration"])
        return f"{sw['prompt']}{body}\n{sw['test']}\ncheck({ep})\n"
    if lang == "cpp":
        return f"{sw['declaration']}{body}\n{sw['test']}\n"
    if lang == "js":
        return f"{sw['declaration']}{body}\n{sw['test']}\n"
    if lang == "java":
        return f"{sw['declaration']}{body}\n{sw['test']}\n"
    if lang == "go":
        return f"{sw.get('import','')}{sw['declaration']}{body}\n{sw.get('test_setup','')}{sw['test']}\n"
    raise ValueError(lang)


def run_one(lang, program, timeout=20):
    """Compile/run a full program; return (passed: bool, info: str)."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        try:
            if lang == "python":
                (d / "m.py").write_text(program)
                r = subprocess.run(["python3", "m.py"], cwd=d, capture_output=True,
                                   timeout=timeout, text=True)
            elif lang == "js":
                (d / "m.js").write_text(program)
                r = subprocess.run(["node", "m.js"], cwd=d, capture_output=True,
                                   timeout=timeout, text=True)
            elif lang == "cpp":
                (d / "m.cpp").write_text(program)
                c = subprocess.run(["g++", "-std=c++17", "-O1", "m.cpp", "-o", "m"], cwd=d,
                                   capture_output=True, timeout=timeout, text=True)
                if c.returncode:
                    return False, "compile:" + c.stderr[-300:]
                r = subprocess.run(["./m"], cwd=d, capture_output=True, timeout=timeout, text=True)
            elif lang == "go":
                (d / "m_test.go").write_text(program)
                r = subprocess.run(["go", "test", "./..."], cwd=d, capture_output=True,
                                   timeout=timeout, text=True)
            elif lang == "java":
                (d / "Main.java").write_text(program)
                c = subprocess.run(["javac", "Main.java"], cwd=d, capture_output=True,
                                   timeout=timeout, text=True)
                if c.returncode:
                    return False, "compile:" + c.stderr[-300:]
                r = subprocess.run(["java", "Main"], cwd=d, capture_output=True,
                                   timeout=timeout, text=True)
            else:
                return False, "unknown lang"
            return r.returncode == 0, (r.stderr or r.stdout)[-200:]
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except FileNotFoundError as e:
            return False, f"runtime-missing:{e}"


def validate(comp, n=None):
    """Canonical solution + test should PASS for each sub-world (E68-style validation rate)."""
    pids = list(comp)[:n] if n else list(comp)
    by_lang = {lg: [0, 0] for lg in LANGS}
    for p in pids:
        for lg in LANGS:
            sw = comp[p][lg]
            ok, _ = run_one(lg, assemble(lg, sw, sw["canonical_solution"]))
            by_lang[lg][0] += int(ok)
            by_lang[lg][1] += 1
    return {lg: {"pass": v[0], "n": v[1], "rate": round(v[0] / v[1], 3) if v[1] else None}
            for lg, v in by_lang.items()}


if __name__ == "__main__":
    import sys
    comp = build_worlds()
    print(f"composite programming worlds: {len(comp)} (each {len(LANGS)} language sub-worlds: {LANGS})")
    sizes = {lg: sum(1 for p in comp if comp[p][lg]["canonical_solution"]) for lg in LANGS}
    print("sub-worlds with reference solution:", sizes)
    if "--validate" in sys.argv:  # needs language runtimes (run on a box)
        n = 100
        print(f"validating canonical solutions on {n} composites (needs runtimes)...")
        print(json.dumps(validate(comp, n=n), indent=2))
