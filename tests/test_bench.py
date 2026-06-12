"""Tests for openworld.bench: recipes, gate, paired evaluation, cards."""

import json

import pytest

from openworld.bench import RecipeError, evaluate, load_recipe, mock_factory, summarize, validate_dataset, write_card

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
    assert set(row) == {"instance_id", "seed", "single_shot", "in_world"}
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
