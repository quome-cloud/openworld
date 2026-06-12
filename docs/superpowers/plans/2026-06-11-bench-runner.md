# openworld.bench Runner + Recipe Standard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sub-project 1 of the dataset-factory spec: one recipe file + one command (`python -m openworld.bench <recipe> <cmd>`) per dataset, a frozen result schema, auto-generated dataset cards, and the two existing datasets retrofitted onto it.

**Architecture:** A single flat module `openworld/bench.py` (repo style: flat modules) owning recipe loading, the validation gate, paired evaluation, result writing, and card generation. Per-dataset `run_comparison.py` scripts are deleted; their logic (incl. `wilson_ci`) moves into bench. Recipes live in `recipes/*.json`.

**Tech Stack:** stdlib only (json, hashlib, argparse, subprocess, urllib). Python 3.9.

**Spec deviation (recorded):** the spec sketch said `recipes/<dataset>.yaml`; recipes are JSON because the repo is zero-dependency on Python 3.9 (no PyYAML, no tomllib). Same fields, same semantics.

**Scope note:** unifying the *contextbench* runner is deferred (its conditions are with/without-context, not single-shot/in-world; it needs a condition abstraction that is YAGNI here). Only the two swebench datasets are retrofitted.

**Base branch:** branch off `staged-on-main` (PR #7) until it merges; rebase onto main after.

---

### Task 1: Recipe schema + loader + the two recipe files

**Files:**
- Create: `openworld/bench.py`
- Create: `recipes/owsb-atomic-v1.json`
- Create: `recipes/owsb-staged-v1.json`
- Create: `tests/test_bench.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bench.py`:

```python
"""Tests for openworld.bench: recipes, gate, paired evaluation, cards."""

import json

import pytest

from openworld.bench import RecipeError, load_recipe

ATOMIC_RECIPE = "recipes/owsb-atomic-v1.json"
STAGED_RECIPE = "recipes/owsb-staged-v1.json"


def test_load_recipe_resolves_paths_and_defaults():
    recipe = load_recipe(ATOMIC_RECIPE)
    assert recipe["schema_version"] == 1
    assert recipe["dataset"]["name"] == "owsb-atomic"
    # paths are resolved to absolute paths under the repo root
    assert recipe["dataset"]["path"].is_absolute()
    assert recipe["dataset"]["path"].name == "tasks.jsonl"
    assert recipe["generator"]["builder"].is_absolute()
    assert recipe["eval"]["budget"] == 4
    assert "qwen2.5:7b" in recipe["eval"]["models"]


def test_load_recipe_staged():
    recipe = load_recipe(STAGED_RECIPE)
    assert recipe["dataset"]["name"] == "owsb-staged"
    assert recipe["dataset"]["path"].exists()


def test_load_recipe_rejects_wrong_schema_version(tmp_path):
    bad = tmp_path / "r.json"
    bad.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(RecipeError, match="schema_version"):
        load_recipe(bad)


def test_load_recipe_rejects_missing_section(tmp_path):
    bad = tmp_path / "r.json"
    bad.write_text(json.dumps({"schema_version": 1, "dataset": {
        "name": "x", "version": "v1", "description": "d", "path": "nope.jsonl"
    }}), encoding="utf-8")
    with pytest.raises(RecipeError, match="generator"):
        load_recipe(bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bench.py -v`
Expected: collection error — no module `openworld.bench`.

- [ ] **Step 3: Implement**

Create `recipes/owsb-atomic-v1.json`:

```json
{
  "schema_version": 1,
  "dataset": {
    "name": "owsb-atomic",
    "version": "v1",
    "description": "20 atomic single-defect SWE program-repair instances, each with an explicit world-model spec and a pass_to_pass regression suite.",
    "path": "datasets/openworld-swebench/tasks.jsonl"
  },
  "generator": {
    "type": "hand",
    "builder": "datasets/openworld-swebench/build_tasks.py",
    "seed": null
  },
  "harness": {
    "result_schema_version": 1
  },
  "eval": {
    "models": ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "llama3.2"],
    "budget": 4,
    "temperature": 0.2,
    "seed": 41
  },
  "artifacts": {
    "tasks_jsonl_sha256": null
  }
}
```

Create `recipes/owsb-staged-v1.json` (same shape):

```json
{
  "schema_version": 1,
  "dataset": {
    "name": "owsb-staged",
    "version": "v1",
    "description": "15 two-stage program-repair instances: the issue-visible fix surfaces a latent second failing test, so feedback-driven iteration is required.",
    "path": "datasets/openworld-swebench-staged/tasks.jsonl"
  },
  "generator": {
    "type": "hand",
    "builder": "datasets/openworld-swebench-staged/build_tasks.py",
    "seed": null
  },
  "harness": {
    "result_schema_version": 1
  },
  "eval": {
    "models": ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b"],
    "budget": 4,
    "temperature": 0.2,
    "seed": 41
  },
  "artifacts": {
    "tasks_jsonl_sha256": null
  }
}
```

Create `openworld/bench.py`:

```python
"""The benchmark recipe runner: one recipe file + one command per dataset.

A recipe (recipes/*.json, schema_version 1) pins everything needed to rebuild
a dataset, validate it, run the paired single-shot vs in-world evaluation,
and emit a dataset card:

    python -m openworld.bench recipes/owsb-atomic-v1.json build
    python -m openworld.bench recipes/owsb-atomic-v1.json validate
    python -m openworld.bench recipes/owsb-atomic-v1.json run --mock
    python -m openworld.bench recipes/owsb-atomic-v1.json card
    python -m openworld.bench recipes/owsb-atomic-v1.json all --mock

Recipes are JSON (not YAML) because the framework is zero-dependency on
Python 3.9. Results are written one file per (model, recipe) in a frozen
result schema so runs stay comparable across datasets and time.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent

RECIPE_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1

_REQUIRED_SECTIONS = ("dataset", "generator", "harness", "eval", "artifacts")
_PATH_FIELDS = (("dataset", "path"), ("generator", "builder"))


class RecipeError(ValueError):
    """A recipe file is malformed or inconsistent."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_recipe(path) -> Dict[str, Any]:
    """Load and structurally validate a recipe; resolve paths to absolute."""
    path = Path(path)
    recipe = json.loads(path.read_text(encoding="utf-8"))
    if recipe.get("schema_version") != RECIPE_SCHEMA_VERSION:
        raise RecipeError(
            f"{path}: schema_version must be {RECIPE_SCHEMA_VERSION}, "
            f"got {recipe.get('schema_version')!r}"
        )
    for section in _REQUIRED_SECTIONS:
        if section not in recipe:
            raise RecipeError(f"{path}: missing section {section!r}")
    for fld in ("name", "version", "description", "path"):
        if not recipe["dataset"].get(fld):
            raise RecipeError(f"{path}: dataset.{fld} is required")
    for fld in ("models", "budget"):
        if fld not in recipe["eval"]:
            raise RecipeError(f"{path}: eval.{fld} is required")
    for section, fld in _PATH_FIELDS:
        if recipe[section].get(fld):
            recipe[section][fld] = (ROOT / recipe[section][fld]).resolve()
    recipe["_recipe_path"] = path.resolve()
    recipe["_recipe_sha256"] = sha256_file(path)
    return recipe


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion. Returns (low, high)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bench.py -v` — all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add openworld/bench.py recipes/ tests/test_bench.py
git commit -m "Add openworld.bench recipe schema v1 with recipes for both swebench datasets"
```

---

### Task 2: The validate command (gate v1)

**Files:**
- Modify: `openworld/bench.py` (append)
- Modify: `tests/test_bench.py` (append)

- [ ] **Step 1: Write the failing tests** (append; extend the bench import with `validate_dataset`)

```python
def test_validate_atomic_dataset_passes():
    report = validate_dataset(load_recipe(ATOMIC_RECIPE))
    assert report["ok"] is True
    assert report["n_instances"] == 20
    failed = [c for c in report["checks"] if not c["ok"]]
    assert failed == []


def test_validate_staged_dataset_passes():
    report = validate_dataset(load_recipe(STAGED_RECIPE))
    assert report["ok"] is True
    assert report["n_instances"] == 15


def test_validate_catches_artifact_drift(tmp_path):
    recipe = load_recipe(ATOMIC_RECIPE)
    recipe["artifacts"]["tasks_jsonl_sha256"] = "0" * 64  # wrong on purpose
    report = validate_dataset(recipe)
    assert report["ok"] is False
    assert any("sha256" in c["name"] and not c["ok"] for c in report["checks"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bench.py -v`
Expected: ImportError for `validate_dataset`; earlier tests still pass.

- [ ] **Step 3: Implement** (append to `openworld/bench.py`)

```python
def validate_dataset(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """Gate v1: every check the dataset must pass before results count.

    - instance ids unique and non-empty dataset
    - reference solves both suites on every instance
    - buggy fails ALL fail_to_pass and passes ALL pass_to_pass
    - stored world.initial_state matches recomputation
    - tasks.jsonl sha256 matches recipe.artifacts (when frozen)
    """
    from .swebench import initial_world_state, load_dataset, run_instance_tests

    instances = load_dataset(recipe["dataset"]["path"])
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    ids = [i.instance_id for i in instances]
    check("nonempty", len(instances) > 0, f"{len(instances)} instances")
    check("unique-ids", len(ids) == len(set(ids)))

    for inst in instances:
        ref = run_instance_tests(inst.reference_source, inst)
        check(f"oracle:{inst.instance_id}", ref["solved"],
              "; ".join(ref["fail_to_pass"]["errors"][:1] + ref["pass_to_pass"]["errors"][:1]))
        buggy = run_instance_tests(inst.buggy_source, inst)
        check(f"bug-real:{inst.instance_id}",
              buggy["fail_to_pass"]["passed"] == 0 and buggy["pass_to_pass"]["failed"] == 0)
        recomputed = initial_world_state(inst)
        stored = inst.world.get("initial_state", {})
        drift = [k for k in ("fail_to_pass_passed", "fail_to_pass_failed",
                             "pass_to_pass_passed", "pass_to_pass_failed",
                             "attempts", "solved", "source")
                 if stored.get(k) != recomputed[k]]
        check(f"initial-state:{inst.instance_id}", not drift, ",".join(drift))

    frozen = recipe["artifacts"].get("tasks_jsonl_sha256")
    if frozen:
        actual = sha256_file(recipe["dataset"]["path"])
        check("artifact-sha256", actual == frozen,
              f"recipe={frozen[:12]}.. actual={actual[:12]}..")

    return {
        "dataset": recipe["dataset"]["name"],
        "n_instances": len(instances),
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bench.py -v` — all PASS (this re-runs both full gates; ~10s is normal, the suites fork).

- [ ] **Step 5: Commit**

```bash
git add openworld/bench.py tests/test_bench.py
git commit -m "bench: validate command — gate v1 over any recipe'd dataset"
```

---

### Task 3: Paired evaluation + frozen result schema

**Files:**
- Modify: `openworld/bench.py` (append)
- Modify: `tests/test_bench.py` (append)

- [ ] **Step 1: Write the failing tests** (append; extend bench import with `evaluate, mock_factory, summarize`)

```python
def test_evaluate_mock_oracle_second_try(tmp_path):
    recipe = load_recipe(ATOMIC_RECIPE)
    result = evaluate(recipe, model="mock", llm_factory=mock_factory,
                      budget=4, mock=True, results_dir=tmp_path)
    out = tmp_path / "mock.json"
    assert out.exists()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["result_schema_version"] == 1
    assert saved["dataset"] == "owsb-atomic"
    assert saved["recipe_sha256"] == recipe["_recipe_sha256"]
    assert saved["mock"] is True
    assert saved["n_instances"] == 20
    assert len(saved["rows"]) == 20
    row = saved["rows"][0]
    assert set(row) == {"instance_id", "single_shot", "in_world"}
    s = saved["summary"]
    assert s["single_shot_pass_at_1"] == 0.0
    assert s["in_world_pass_at_budget"] == 1.0
    assert s["delta"] == 1.0
    assert s["mean_attempts_when_solved"] == 2.0
    assert result["summary"] == s


def test_summarize_handles_no_solves():
    rows = [{"instance_id": "x",
             "single_shot": {"solved": False, "solved_first_attempt": False, "attempts": 1},
             "in_world": {"solved": False, "solved_first_attempt": False, "attempts": 4}}]
    s = summarize(rows, budget=4)
    assert s["mean_attempts_when_solved"] is None
    assert s["delta"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bench.py -v` — ImportError for `evaluate`.

- [ ] **Step 3: Implement** (append to `openworld/bench.py`)

```python
def mock_factory(instance):
    """Offline smoke model: garbage first, the oracle patch second."""
    from .llm import MockLLM

    return MockLLM([
        "I think the bug is in the loop, roughly.",
        f"```python\n{instance.reference_source}\n```",
    ])


def _model_slug(model: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in model)


def _ollama_digest(model: str) -> Optional[str]:
    """Best-effort digest lookup from a local Ollama; None when unavailable."""
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as fh:
            tags = json.loads(fh.read().decode("utf-8"))
        for entry in tags.get("models", []):
            if entry.get("name") == model or entry.get("model") == model:
                return entry.get("digest")
    except Exception:
        return None
    return None


def summarize(rows: List[Dict[str, Any]], budget: int) -> Dict[str, Any]:
    n = len(rows)
    single = sum(r["single_shot"]["solved"] for r in rows)
    world_first = sum(r["in_world"]["solved_first_attempt"] for r in rows)
    world_budget = sum(r["in_world"]["solved"] for r in rows)
    solved_attempts = [r["in_world"]["attempts"] for r in rows if r["in_world"]["solved"]]
    return {
        "n_instances": n,
        "budget": budget,
        "single_shot_pass_at_1": single / n,
        "single_shot_pass_at_1_ci": list(wilson_ci(single, n)),
        "in_world_pass_at_1": world_first / n,
        "in_world_pass_at_1_ci": list(wilson_ci(world_first, n)),
        "in_world_pass_at_budget": world_budget / n,
        "in_world_pass_at_budget_ci": list(wilson_ci(world_budget, n)),
        "delta": (world_budget - single) / n,
        "mean_attempts_when_solved": (
            sum(solved_attempts) / len(solved_attempts) if solved_attempts else None
        ),
    }


def evaluate(recipe, model, llm_factory, budget=None, mock=False,
             results_dir=None) -> Dict[str, Any]:
    """Run the paired ablation for one model; write one frozen-schema file."""
    from datetime import datetime

    from .swebench import load_dataset, solve_in_world, solve_single_shot

    instances = load_dataset(recipe["dataset"]["path"])
    budget = budget or recipe["eval"]["budget"]
    rows = []
    for inst in instances:
        single = solve_single_shot(inst, llm_factory(inst))
        in_world = solve_in_world(inst, llm_factory(inst), budget=budget)
        rows.append({"instance_id": inst.instance_id,
                     "single_shot": single, "in_world": in_world})
        print(f"  {inst.instance_id}: "
              f"single-shot={'Y' if single['solved'] else 'n'} "
              f"in-world={'Y' if in_world['solved'] else 'n'} "
              f"({in_world['attempts']} att)")
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "dataset": recipe["dataset"]["name"],
        "dataset_version": recipe["dataset"]["version"],
        "recipe_sha256": recipe["_recipe_sha256"],
        "tasks_jsonl_sha256": sha256_file(recipe["dataset"]["path"]),
        "model": model,
        "model_digest": None if mock else _ollama_digest(model),
        "mock": bool(mock),
        "budget": budget,
        "n_instances": len(rows),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "summary": summarize(rows, budget),
    }
    results_dir = Path(results_dir) if results_dir else (
        recipe["dataset"]["path"].parent / "results")
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"{_model_slug(model)}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[saved] {out}")
    return result


def markdown_table(results: List[Dict[str, Any]]) -> str:
    budget = results[0]["budget"] if results else 0
    lines = [
        f"| model | single-shot pass@1 | in-world pass@1 | in-world pass@{budget} "
        f"| Δ (pass@{budget} − SS) | mean attempts |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        s = r["summary"]
        mean_att = (f"{s['mean_attempts_when_solved']:.1f}"
                    if s["mean_attempts_when_solved"] is not None else "—")
        lines.append(
            f"| {r['model']} | {s['single_shot_pass_at_1']:.0%} | "
            f"{s['in_world_pass_at_1']:.0%} | {s['in_world_pass_at_budget']:.0%} | "
            f"{s['delta']:+.0%} | {mean_att} |")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bench.py -v` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add openworld/bench.py tests/test_bench.py
git commit -m "bench: paired evaluation with frozen result schema v1"
```

---

### Task 4: Dataset card generation

**Files:**
- Modify: `openworld/bench.py` (append)
- Modify: `tests/test_bench.py` (append)

- [ ] **Step 1: Write the failing test** (append; extend bench import with `write_card`)

```python
def test_card_contains_provenance_and_gate(tmp_path):
    recipe = load_recipe(ATOMIC_RECIPE)
    report = validate_dataset(recipe)
    card_path = write_card(recipe, report, out=tmp_path / "CARD.md")
    card = card_path.read_text(encoding="utf-8")
    assert "# owsb-atomic v1" in card
    assert "hand" in card                      # generator type
    assert "20 instances" in card
    assert "all checks passed" in card
    assert recipe["_recipe_sha256"][:12] in card
    assert "Tier 0" in card and "Tier 2" in card
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench.py::test_card_contains_provenance_and_gate -v`
Expected: ImportError for `write_card`.

- [ ] **Step 3: Implement** (append to `openworld/bench.py`)

```python
def write_card(recipe, validate_report, out=None) -> Path:
    """Auto-generate the dataset card from the recipe + gate report."""
    from datetime import datetime

    ds, gen = recipe["dataset"], recipe["generator"]
    rel_recipe = recipe["_recipe_path"].relative_to(ROOT)
    gate_line = ("all checks passed"
                 if validate_report["ok"]
                 else "GATE FAILING: " + "; ".join(
                     c["name"] for c in validate_report["checks"] if not c["ok"]))
    frozen = recipe["artifacts"].get("tasks_jsonl_sha256") or "(not frozen)"
    card = f"""# {ds['name']} {ds['version']}

{ds['description']}

*Auto-generated by `python -m openworld.bench {rel_recipe} card` on
{datetime.now().isoformat(timespec='seconds')}. Do not edit by hand.*

## Provenance

| | |
|---|---|
| generator | {gen['type']} (`{gen['builder'].relative_to(ROOT)}`) |
| generator seed | {gen.get('seed')} |
| recipe | `{rel_recipe}` (sha256 `{recipe['_recipe_sha256'][:12]}…`) |
| tasks.jsonl sha256 (frozen) | `{frozen}` |

## Gate

{validate_report['n_instances']} instances; {gate_line}.
The gate enforces: unique ids; the reference source solves both hidden
suites; the buggy source fails every `fail_to_pass` test and passes every
`pass_to_pass` test; the stored world `initial_state` matches recomputation;
the artifact hash matches the recipe when frozen.

## Evaluation protocol

Paired ablation per instance: the same model single-shot (one completion,
no feedback) and in-world (iterative `submit_patch` against exact dynamics,
budget {recipe['eval']['budget']}). Default ladder: {', '.join(recipe['eval']['models'])}.
Per-instance paired records are always saved so exact tests (e.g. McNemar)
remain possible.

## Reproducibility tiers

- **Tier 0 — structural:** the mock path runs in pytest on every commit
  (`python -m openworld.bench {rel_recipe} run --mock`).
- **Tier 1 — artifact:** `python -m openworld.bench {rel_recipe} build`
  regenerates `tasks.jsonl` byte-identically from the pinned builder.
- **Tier 2 — statistical:** rerunning with the same Ollama model digests
  reproduces results within the stated Wilson CIs; absolute rates are
  specific to the pinned quantized snapshots.
"""
    out = Path(out) if out else ds["path"].parent / "CARD.md"
    out.write_text(card, encoding="utf-8")
    print(f"[saved] {out}")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bench.py -v` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add openworld/bench.py tests/test_bench.py
git commit -m "bench: auto-generated dataset cards from recipe + gate report"
```

---

### Task 5: CLI (`__main__`) with build/validate/run/card/all

**Files:**
- Modify: `openworld/bench.py` (append)

- [ ] **Step 1: Implement the CLI** (append to `openworld/bench.py`)

```python
def cmd_build(recipe, freeze: bool = False) -> None:
    """Re-run the pinned builder; verify (or --freeze) the artifact hash."""
    import subprocess
    import sys

    builder = recipe["generator"]["builder"]
    subprocess.run([sys.executable, str(builder)], check=True)
    actual = sha256_file(recipe["dataset"]["path"])
    frozen = recipe["artifacts"].get("tasks_jsonl_sha256")
    if freeze:
        raw = json.loads(recipe["_recipe_path"].read_text(encoding="utf-8"))
        raw["artifacts"]["tasks_jsonl_sha256"] = actual
        recipe["_recipe_path"].write_text(
            json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        print(f"[frozen] artifacts.tasks_jsonl_sha256 = {actual[:12]}…")
    elif frozen and frozen != actual:
        raise SystemExit(
            f"artifact drift: recipe pins {frozen[:12]}… but build produced "
            f"{actual[:12]}… (rerun with --freeze to accept)")


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m openworld.bench",
                                     description=__doc__)
    parser.add_argument("recipe", help="path to a recipes/*.json file")
    parser.add_argument("command", choices=["build", "validate", "run", "card", "all"])
    parser.add_argument("--mock", action="store_true",
                        help="offline smoke run with a scripted MockLLM")
    parser.add_argument("--models", nargs="*", default=None,
                        help="override recipe eval.models")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--freeze", action="store_true",
                        help="with build: write the artifact hash into the recipe")
    args = parser.parse_args(argv)

    recipe = load_recipe(args.recipe)
    report = None

    if args.command in ("build", "all"):
        cmd_build(recipe, freeze=args.freeze)
    if args.command in ("validate", "all"):
        report = validate_dataset(recipe)
        bad = [c for c in report["checks"] if not c["ok"]]
        print(f"[gate] {report['dataset']}: {report['n_instances']} instances, "
              f"{'OK' if report['ok'] else 'FAILED'}")
        for c in bad:
            print(f"  FAIL {c['name']} {c['detail']}")
        if bad and args.command == "validate":
            return 1
        if bad:
            raise SystemExit("gate failed; not running evaluation")
    if args.command in ("run", "all"):
        results = []
        if args.mock:
            results.append(evaluate(recipe, "mock", mock_factory,
                                    budget=args.budget, mock=True))
        else:
            from .llm import OllamaConnectionError, OllamaLLM

            for model in (args.models or recipe["eval"]["models"]):
                llm = OllamaLLM(model=model,
                                temperature=recipe["eval"].get("temperature", 0.2),
                                options={"seed": recipe["eval"].get("seed", 41)})
                try:
                    llm.ask("Reply with OK.")
                except OllamaConnectionError as exc:
                    raise SystemExit(f"model {model!r} unavailable: {exc}")
                print(f"[{model}] budget {args.budget or recipe['eval']['budget']}")
                results.append(evaluate(recipe, model,
                                        lambda inst, _llm=llm: _llm,
                                        budget=args.budget))
        print("\n" + markdown_table(results))
    if args.command in ("card", "all"):
        if report is None:
            report = validate_dataset(recipe)
        write_card(recipe, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke the CLI end-to-end (mock)**

```bash
python -m openworld.bench recipes/owsb-atomic-v1.json validate
python -m openworld.bench recipes/owsb-staged-v1.json all --mock
```

Expected: gate OK for both; staged `all` rebuilds tasks.jsonl (no drift since
hash not yet frozen), runs the mock (table row: 0% / 0% / 100% / +100% / 2.0),
and writes `datasets/openworld-swebench-staged/CARD.md`.

CAUTION: `run` overwrites `results/<model>.json`. The staged dataset has a
committed legacy `results/comparison.json` — the new writer uses per-model
filenames so it is NOT touched; verify with `git status` that only `CARD.md`
(and a gitignored results file) appeared.

- [ ] **Step 3: Freeze both artifacts**

```bash
python -m openworld.bench recipes/owsb-atomic-v1.json build --freeze
python -m openworld.bench recipes/owsb-staged-v1.json build --freeze
python -m openworld.bench recipes/owsb-atomic-v1.json validate
python -m openworld.bench recipes/owsb-staged-v1.json validate
```

Expected: both recipes now pin sha256; both validates print OK (artifact-sha256 check now active).

- [ ] **Step 4: Gitignore staged mock results, generate and commit the cards**

The staged dataset dir has no `.gitignore`, and its legacy
`results/comparison.json` (the original E29 run cited by the paper) must stay
tracked while new per-model result files stay out of git. Create
`datasets/openworld-swebench-staged/.gitignore` with exactly:

```
results/*
!results/comparison.json
```

(Already-tracked files are unaffected by gitignore, so `comparison.json`
stays; the negation documents intent.)

```bash
python -m openworld.bench recipes/owsb-atomic-v1.json card
python -m openworld.bench recipes/owsb-staged-v1.json card
git add openworld/bench.py recipes/ datasets/openworld-swebench/CARD.md \
        datasets/openworld-swebench-staged/CARD.md \
        datasets/openworld-swebench-staged/.gitignore
git commit -m "bench: CLI with build/validate/run/card; freeze both dataset artifacts; emit cards"
```

---

### Task 6: Retire the per-dataset runners + docs + full verification

**Files:**
- Delete: `datasets/openworld-swebench/run_comparison.py`
- Delete: `datasets/openworld-swebench-staged/run_comparison.py`
- Modify: `datasets/openworld-swebench/README.md` (Running section)
- Modify: `datasets/openworld-swebench-staged/README.md` (Files/Running sections)

- [ ] **Step 1: Delete the runners and update the READMEs**

```bash
git rm datasets/openworld-swebench/run_comparison.py \
       datasets/openworld-swebench-staged/run_comparison.py
```

In each README, replace the `run_comparison.py` commands with the bench
equivalents and mention the recipe file, e.g. for atomic:

```markdown
## Running

All operations go through the recipe (`recipes/owsb-atomic-v1.json`):

```bash
python -m openworld.bench recipes/owsb-atomic-v1.json run --mock   # offline smoke
python -m openworld.bench recipes/owsb-atomic-v1.json run          # Ollama ladder
python -m openworld.bench recipes/owsb-atomic-v1.json all --mock   # build+validate+run+card
```

Results land in `results/<model>.json` (frozen result schema v1, one file
per model); the dataset card is `CARD.md`.
```

(Equivalent edit in the staged README, citing `recipes/owsb-staged-v1.json`;
keep its legacy-results note: `results/comparison.json` is the original E29
run cited by the paper.)

- [ ] **Step 2: Repo-wide reference sweep**

Run: `grep -rn "run_comparison" --include="*.py" --include="*.md" . | grep -v ".git/" | grep -v contextbench`
Expected: no hits outside `docs/` history (specs/plans/reviews are historical
records — leave them) and the paper (which describes the E28/E29 provenance —
leave it). Fix any live reference (e.g. the root README) to use bench.

- [ ] **Step 3: Full verification**

```bash
python -m pytest tests/ -q                                  # everything passes
python -m openworld.bench recipes/owsb-atomic-v1.json build # no drift vs frozen hash
python -m openworld.bench recipes/owsb-staged-v1.json build
python -m openworld.bench recipes/owsb-atomic-v1.json run --mock
git status --short                                          # only intended changes
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "bench: retire per-dataset runners; datasets now run via recipes"
```
