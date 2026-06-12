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
