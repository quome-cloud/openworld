import json

from experiments.e145.sourcefree_controller import (
    eligible_solution_paths,
    is_sourcefree_eligible_run,
)


def good_record(**overrides):
    record = {
        "game": "toy",
        "source_free": True,
        "memory_tainted": False,
        "solution_file": "solutions/toy__agent__ok.json",
        "knowledge_audit": {"clean": True},
        "outcome": {
            "levels": 2,
            "replay_verified": True,
            "audit": {"clean": True},
        },
    }
    record.update(overrides)
    return record


def test_sourcefree_eligibility_accepts_clean_verified_run():
    assert is_sourcefree_eligible_run(good_record())


def test_sourcefree_eligibility_rejects_tainted_or_unverified():
    assert not is_sourcefree_eligible_run(good_record(memory_tainted=True))
    assert not is_sourcefree_eligible_run(good_record(source_free=False))
    assert not is_sourcefree_eligible_run(good_record(outcome={"levels": 2, "replay_verified": False}))
    assert not is_sourcefree_eligible_run(
        good_record(outcome={"levels": 2, "replay_verified": True, "audit": {"clean": False}})
    )


def test_eligible_solution_paths_uses_runs_jsonl_allowlist(tmp_path):
    (tmp_path / "solutions").mkdir()
    good = tmp_path / "solutions" / "toy__agent__ok.json"
    bad = tmp_path / "solutions" / "toy__agent__bad.json"
    good.write_text(json.dumps({"game": "toy", "levels": 2, "actions": [[1]]}))
    bad.write_text(json.dumps({"game": "toy", "levels": 3, "actions": [[2]]}))
    records = [
        good_record(),
        good_record(solution_file="solutions/toy__agent__bad.json", memory_tainted=True),
    ]
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")

    assert eligible_solution_paths(tmp_path, games=["toy"]) == [good]

