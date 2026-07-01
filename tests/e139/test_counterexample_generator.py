from experiments.e139.counterexample_generator import generate_from_counterexample


def test_generates_terminal_meter_suffixes():
    counterexample = {
        "proposal_id": "failed-meter",
        "executed": [[3], [4], [2]],
        "final_summary": {
            "levels": 5,
            "done": False,
            "cursor": {"zero": {"x": 49, "y": 25}},
            "palette": {"4": 62},
            "small": [
                {"c": 4, "n": 13, "x": 49, "y": 22},
                {"c": 4, "n": 9, "x": 4, "y": 63},
            ],
        },
    }

    proposals = generate_from_counterexample(counterexample)
    ids = {p["proposal_id"] for p in proposals}

    assert "gen-terminal-click-square-after-meter" in ids
    assert "gen-terminal-click-c4-remnant-4-63" in ids
    assert all(p["probe_plan"][:3] == [[3], [4], [2]] for p in proposals)
