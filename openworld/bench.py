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


def harness_kind(recipe: Dict[str, Any]) -> str:
    """The harness dispatch kind; defaults to the original swebench ablation."""
    return recipe.get("harness", {}).get("kind", "swebench")


def validate_dataset(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """Gate v1: every check the dataset must pass before results count.

    - instance ids unique and non-empty dataset
    - reference solves both suites on every instance
    - buggy fails ALL fail_to_pass and passes ALL pass_to_pass
    - stored world.initial_state matches recomputation
    - tasks.jsonl sha256 matches recipe.artifacts (when frozen)

    For harness.kind == "contextbench" the unit of iteration is a
    ContextBenchInstance whose `.task` IS a swebench instance, so the same
    per-task gate runs against `inst.task`. The contextbench world only stores
    a `source/attempts/solved` initial state (no per-suite counts), so the
    initial-state drift check is run over exactly the keys present in the
    stored state — still meaningful, just narrower than swebench's.
    """
    from .swebench import initial_world_state, run_instance_tests

    kind = harness_kind(recipe)
    if kind == "contextbench":
        from .contextbench import load_dataset as load_cb
        cb_instances = load_cb(recipe["dataset"]["path"])
        instances = [c.task for c in cb_instances]
        ids = [c.instance_id for c in cb_instances]
    else:
        from .swebench import load_dataset
        instances = load_dataset(recipe["dataset"]["path"])
        ids = [i.instance_id for i in instances]

    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

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
        # Gate over the keys the stored state actually carries (swebench stores
        # the full per-suite counts; contextbench stores source/attempts/solved).
        drift = [k for k in ("fail_to_pass_passed", "fail_to_pass_failed",
                             "pass_to_pass_passed", "pass_to_pass_failed",
                             "attempts", "solved", "source")
                 if k in stored and stored.get(k) != recomputed[k]]
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


def _oracle_source(instance) -> str:
    """The reference patch for either a swebench or contextbench instance."""
    # ContextBenchInstance wraps a SWEBenchInstance in `.task`.
    task = getattr(instance, "task", instance)
    return task.reference_source


def mock_factory(instance, seed=None):
    """Offline smoke model: garbage first, the oracle patch second.

    For contextbench the model is asked once per condition (no iteration), so
    the oracle must be the FIRST response — the leading garbage line would make
    the without/with-context single-shot fail. `solve_with/without_context`
    extract the first code block, so we lead with the oracle for those.
    """
    from .llm import MockLLM

    oracle = _oracle_source(instance)
    if getattr(instance, "task", None) is not None:  # contextbench: single ask
        return MockLLM([f"```python\n{oracle}\n```"])
    return MockLLM([
        "I think the bug is in the loop, roughly.",
        f"```python\n{oracle}\n```",
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


def _mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar (sign-test) p-value over discordant pairs.

    b, c are the discordant counts (one condition solved, the other didn't).
    Exact binomial under H0 p=0.5 — zero-dependency, valid for small samples.
    """
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def summarize(rows: List[Dict[str, Any]], budget: int) -> Dict[str, Any]:
    n = len(rows)  # total paired trials = n_instances x n_seeds
    single = sum(r["single_shot"]["solved"] for r in rows)
    world_first = sum(r["in_world"]["solved_first_attempt"] for r in rows)
    world_budget = sum(r["in_world"]["solved"] for r in rows)
    solved_attempts = [r["in_world"]["attempts"] for r in rows if r["in_world"]["solved"]]
    # Paired McNemar: in-world(budget) vs single-shot over every trial.
    iw_wins = sum(1 for r in rows
                  if r["in_world"]["solved"] and not r["single_shot"]["solved"])
    ss_wins = sum(1 for r in rows
                  if r["single_shot"]["solved"] and not r["in_world"]["solved"])
    return {
        "n_instances": len({r["instance_id"] for r in rows}),
        "n_seeds": len({r.get("seed", 0) for r in rows}),
        "n_trials": n,
        "budget": budget,
        "single_shot_pass_at_1": single / n,
        "single_shot_pass_at_1_ci": list(wilson_ci(single, n)),
        "in_world_pass_at_1": world_first / n,
        "in_world_pass_at_1_ci": list(wilson_ci(world_first, n)),
        "in_world_pass_at_budget": world_budget / n,
        "in_world_pass_at_budget_ci": list(wilson_ci(world_budget, n)),
        "delta": (world_budget - single) / n,
        "mcnemar_in_world_wins": iw_wins,
        "mcnemar_single_shot_wins": ss_wins,
        "mcnemar_p_value": _mcnemar_p(iw_wins, ss_wins),
        "mean_attempts_when_solved": (
            sum(solved_attempts) / len(solved_attempts) if solved_attempts else None
        ),
    }


def summarize_contextbench(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summary for the with-context vs without-context ablation.

    `rows` carry {instance_id, seed, without, with} where without/with each hold
    at least "solved". Reports both pass@1 rates + Wilson CIs, their delta, and
    an exact paired McNemar over the discordant trials (reusing `_mcnemar_p`).
    """
    n = len(rows)  # total paired trials = n_instances x n_seeds
    without = sum(r["without"]["solved"] for r in rows)
    with_ = sum(r["with"]["solved"] for r in rows)
    # Paired McNemar: with-context vs without-context over every trial.
    wc_wins = sum(1 for r in rows
                  if r["with"]["solved"] and not r["without"]["solved"])
    woc_wins = sum(1 for r in rows
                   if r["without"]["solved"] and not r["with"]["solved"])
    return {
        "n_instances": len({r["instance_id"] for r in rows}),
        "n_seeds": len({r.get("seed", 0) for r in rows}),
        "n_trials": n,
        "without_context_pass_at_1": without / n if n else 0.0,
        "without_context_pass_at_1_ci": list(wilson_ci(without, n)),
        "with_context_pass_at_1": with_ / n if n else 0.0,
        "with_context_pass_at_1_ci": list(wilson_ci(with_, n)),
        "delta": (with_ - without) / n if n else 0.0,
        "mcnemar_with_context_wins": wc_wins,
        "mcnemar_without_context_wins": woc_wins,
        "mcnemar_p_value": _mcnemar_p(wc_wins, woc_wins),
    }


def evaluate_contextbench(recipe, model, llm_factory, mock=False,
                          results_dir=None, seeds=None) -> Dict[str, Any]:
    """Run the with-context vs without-context ablation for one model.

    Result-file shape is kept parallel to the swebench `evaluate` (same
    top-level keys) so results stay comparable across harness kinds.
    `budget` is carried (from recipe.eval.budget) but unused — contextbench has
    no iteration budget; the field exists only for schema parity.
    """
    from datetime import datetime

    from .contextbench import (
        load_dataset, solve_with_context, solve_without_context,
    )

    instances = load_dataset(recipe["dataset"]["path"])
    budget = recipe["eval"].get("budget")  # unused; schema parity only
    seeds = list(seeds) if seeds else [recipe["eval"].get("seed", 41)]
    rows = []
    for inst in instances:
        for sd in seeds:
            without = solve_without_context(inst, llm_factory(inst, sd))
            with_ = solve_with_context(inst, llm_factory(inst, sd))
            rows.append({"instance_id": inst.instance_id, "seed": sd,
                         "without": without, "with": with_})
            tag = f" seed={sd}" if len(seeds) > 1 else ""
            print(f"  {inst.instance_id}{tag}: "
                  f"without-context={'Y' if without['solved'] else 'n'} "
                  f"with-context={'Y' if with_['solved'] else 'n'}")
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
        "seeds": seeds,
        "n_seeds": len(seeds),
        "n_instances": len(instances),
        "n_trials": len(rows),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "summary": summarize_contextbench(rows),
    }
    results_dir = Path(results_dir) if results_dir else (
        recipe["dataset"]["path"].parent / "results")
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"{_model_slug(model)}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[saved] {out}")
    return result


def evaluate(recipe, model, llm_factory, budget=None, mock=False,
             results_dir=None, seeds=None, trace_sink=None) -> Dict[str, Any]:
    """Run the paired ablation for one model; write one frozen-schema file.

    With multiple `seeds`, each instance is run once per seed (n_instances x
    n_seeds paired trials), tightening the CIs and powering the McNemar test.
    `llm_factory(instance, seed)` builds the model for a given seed.

    If `trace_sink` is given, each attempt's full (prompt, completion, patch,
    pass/fail) record is forwarded to it, annotated with `seed` and `model` —
    the verified-trace harvest for distillation. Default None changes nothing.
    """
    from datetime import datetime

    from .swebench import load_dataset, solve_in_world, solve_single_shot

    instances = load_dataset(recipe["dataset"]["path"])
    budget = budget or recipe["eval"]["budget"]
    seeds = list(seeds) if seeds else [recipe["eval"].get("seed", 41)]
    rows = []
    for inst in instances:
        for sd in seeds:
            sink = None
            if trace_sink is not None:
                sink = lambda rec, _sd=sd: trace_sink(
                    {**rec, "seed": _sd, "model": model})
            single = solve_single_shot(inst, llm_factory(inst, sd), trace_sink=sink)
            in_world = solve_in_world(inst, llm_factory(inst, sd), budget=budget,
                                      trace_sink=sink)
            rows.append({"instance_id": inst.instance_id, "seed": sd,
                         "single_shot": single, "in_world": in_world})
            tag = f" seed={sd}" if len(seeds) > 1 else ""
            print(f"  {inst.instance_id}{tag}: "
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
        "seeds": seeds,
        "n_seeds": len(seeds),
        "n_instances": len(instances),
        "n_trials": len(rows),
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
    ds, gen = recipe["dataset"], recipe["generator"]
    rel_recipe = recipe["_recipe_path"].relative_to(ROOT)
    gate_line = ("all checks passed"
                 if validate_report["ok"]
                 else "GATE FAILING: " + "; ".join(
                     c["name"] for c in validate_report["checks"] if not c["ok"]))
    frozen = recipe["artifacts"].get("tasks_jsonl_sha256") or "(not frozen)"
    kind = harness_kind(recipe)
    if kind == "contextbench":
        protocol = f"""## Evaluation protocol

With-context vs without-context ablation per instance: the same model solves
the repair task once with no examples (without-context) and once with a short
history of related, already-solved bugs prepended (with-context). The axis is
in-context learning — does showing prior solved examples help the model
transfer the fix pattern to a new module? Scoring is identical to swebench:
solved = zero failures in both hidden suites. Default ladder: {', '.join(recipe['eval']['models'])}.
Per-instance paired records are always saved so exact tests (e.g. McNemar)
remain possible. (`eval.budget` is a schema-parity placeholder; contextbench
has no iteration budget.)"""
    else:
        protocol = f"""## Evaluation protocol

Paired ablation per instance: the same model single-shot (one completion,
no feedback) and in-world (iterative `submit_patch` against exact dynamics,
budget {recipe['eval']['budget']}). Default ladder: {', '.join(recipe['eval']['models'])}.
Per-instance paired records are always saved so exact tests (e.g. McNemar)
remain possible."""
    card = f"""# {ds['name']} {ds['version']}

{ds['description']}

*Auto-generated by `python -m openworld.bench {rel_recipe} card`.
Do not edit by hand; regeneration is deterministic for a frozen recipe.*

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

{protocol}

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


def _is_contextbench_summary(summary: Dict[str, Any]) -> bool:
    return "with_context_pass_at_1" in summary


def markdown_table(results: List[Dict[str, Any]]) -> str:
    if results and _is_contextbench_summary(results[0]["summary"]):
        return _markdown_table_contextbench(results)
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


def _markdown_table_contextbench(results: List[Dict[str, Any]]) -> str:
    lines = [
        "| model | without-context pass@1 | with-context pass@1 "
        "| Δ (with − without) | McNemar p |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        s = r["summary"]
        lines.append(
            f"| {r['model']} | {s['without_context_pass_at_1']:.0%} | "
            f"{s['with_context_pass_at_1']:.0%} | {s['delta']:+.0%} | "
            f"{s['mcnemar_p_value']:.3f} |")
    return "\n".join(lines)


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
    parser.add_argument("--seeds", type=int, default=None,
                        help="run N seeds per instance for multi-trial "
                             "significance (McNemar); overrides recipe eval.seeds")
    parser.add_argument("--freeze", action="store_true",
                        help="with build: write the artifact hash into the recipe")
    parser.add_argument("--log-traces", default=None, metavar="DIR",
                        help="with run (swebench kind): append every attempt's "
                             "(prompt, completion, patch, pass/fail) to "
                             "DIR/<model>.traces.jsonl for verified-trace "
                             "distillation. Does not alter results output.")
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
        kind = harness_kind(recipe)
        base_seed = recipe["eval"].get("seed", 41)
        if args.seeds:
            seeds = [base_seed + i for i in range(args.seeds)]
        else:
            seeds = recipe["eval"].get("seeds") or [base_seed]

        def _run(model, factory, mock):
            if kind == "contextbench":
                return evaluate_contextbench(recipe, model, factory,
                                             mock=mock, seeds=seeds)
            sink, fh = None, None
            if args.log_traces:
                tdir = Path(args.log_traces)
                tdir.mkdir(parents=True, exist_ok=True)
                fh = open(tdir / f"{_model_slug(model)}.traces.jsonl",
                          "a", encoding="utf-8")
                sink = lambda rec, _fh=fh: (_fh.write(json.dumps(rec) + "\n"),
                                            _fh.flush())
            try:
                return evaluate(recipe, model, factory, budget=args.budget,
                                mock=mock, seeds=seeds, trace_sink=sink)
            finally:
                if fh is not None:
                    fh.close()

        if args.mock:
            results.append(_run("mock", mock_factory, True))
        else:
            from .llm import OllamaConnectionError, OllamaLLM

            models = args.models or recipe["eval"]["models"]
            slugs = [_model_slug(m) for m in models]
            if len(set(slugs)) != len(slugs):
                raise SystemExit(
                    f"model names collide after filename slugging: {models}")
            temp = recipe["eval"].get("temperature", 0.2)
            for model in models:
                def _factory(inst, sd, _m=model, _t=temp):
                    # Cap context (a 70B's huge default num_ctx swaps the GPU)
                    # and give the big model a long timeout for the harvest run.
                    return OllamaLLM(model=_m, temperature=_t, timeout=1800,
                                     options={"seed": sd, "num_ctx": 8192})
                try:
                    _factory(None, seeds[0]).ask("Reply with OK.")
                except OllamaConnectionError as exc:
                    raise SystemExit(f"model {model!r} unavailable: {exc}")
                print(f"[{model}] budget {args.budget or recipe['eval']['budget']} "
                      f"seeds {seeds}")
                results.append(_run(model, _factory, False))
        print("\n" + markdown_table(results))
    if args.command in ("card", "all"):
        if report is None:
            report = validate_dataset(recipe)
        write_card(recipe, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
