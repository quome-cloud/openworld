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
