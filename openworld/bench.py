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
