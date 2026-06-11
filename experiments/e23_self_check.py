"""E23 - Oracle-free error detection via cross-program disagreement (Q4).

A practitioner has no hand-written oracle; they CAN synthesize the dynamics
several times. This experiment tests whether disagreement with the ensemble
majority predicts ground-truth error, using the 16 accepted programs per
world already stored by E2 (Qwen 7B/3B, 5 each) and E16 (Llama/Gemma, 3
each): on random probe states, flag a program's transition when it differs
from the ensemble mode, and score that flag against the oracle.

Fully offline. Metrics: flag precision/recall against true errors, and the
Spearman correlation between a program's ensemble-agreement rate and its
true probe accuracy (can agreement RANK artifact quality without an oracle?).
"""

import json
import random
from pathlib import Path

from openworld import WorldState
from openworld.state import Action
from openworld.transition import CodeTransition

from common import RESULTS_DIR, WORLD_SPECS, save_results, spearman, wilson_ci

N_PROBES = 30
SEED = 41


def random_probes(spec_name, rng):
    spec = WORLD_SPECS[spec_name]
    probes = []
    for _ in range(N_PROBES):
        if spec_name == "sprint":
            state = {"backlog": rng.randint(0, 12), "shipped": rng.randint(0, 12),
                     "bugs": rng.randint(0, 10), "debt": rng.randint(0, 10)}
        elif spec_name == "orchard":
            state = {"apples": rng.randint(0, 12),
                     "harvested": {"alice": rng.randint(0, 6)}}
        else:  # triage
            state = {"tick": rng.randint(0, 15),
                     "critical_waiting": rng.randint(0, 5),
                     "moderate_waiting": rng.randint(0, 8),
                     "treated": rng.randint(0, 10),
                     "deteriorated": rng.randint(0, 4),
                     "outcomes": rng.randint(-5, 20),
                     "spend": rng.randint(0, 20)}
        probes.append((state, Action(rng.choice(spec["actions"]), agent="alice")))
    return probes


def predict(code, state, action):
    try:
        return json.dumps(dict(CodeTransition(code).step(WorldState(state), action)),
                          sort_keys=True, default=str)
    except Exception:
        return "__error__"


def main():
    e02 = json.loads((Path(RESULTS_DIR) / "e02_synthesis.json").read_text())
    e16 = json.loads((Path(RESULTS_DIR) / "e16_cross_model.json").read_text())
    programs_by_world = {}
    for run in e02["runs"] + e16["runs"]:
        if run.get("accepted") and run.get("code"):
            programs_by_world.setdefault(run["world"], []).append(
                {"model": run["model"], "code": run["code"]})

    flags_and_errors = []   # (flagged, is_error) per (program, probe)
    program_rows = []
    for spec_name, programs in programs_by_world.items():
        spec = WORLD_SPECS[spec_name]
        probes = random_probes(spec_name, random.Random(SEED))
        oracle_outs = [
            json.dumps(spec["oracle"](dict(state), action.to_dict()),
                       sort_keys=True, default=str)
            for state, action in probes
        ]
        predictions = [
            [predict(p["code"], dict(state), action) for state, action in probes]
            for p in programs
        ]
        # Ensemble mode per probe.
        modes = []
        for j in range(len(probes)):
            outs = [predictions[i][j] for i in range(len(programs))]
            modes.append(max(set(outs), key=outs.count))
        for i, program in enumerate(programs):
            agree = sum(predictions[i][j] == modes[j] for j in range(len(probes)))
            correct = sum(predictions[i][j] == oracle_outs[j] for j in range(len(probes)))
            for j in range(len(probes)):
                flagged = predictions[i][j] != modes[j]
                is_error = predictions[i][j] != oracle_outs[j]
                flags_and_errors.append((flagged, is_error))
            program_rows.append({
                "world": spec_name, "model": program["model"],
                "agreement_rate": agree / len(probes),
                "true_accuracy": correct / len(probes),
            })

    n_flagged = sum(f for f, _ in flags_and_errors)
    n_errors = sum(e for _, e in flags_and_errors)
    true_pos = sum(1 for f, e in flags_and_errors if f and e)
    precision = true_pos / n_flagged if n_flagged else None
    recall = true_pos / n_errors if n_errors else None
    rho = spearman([r["agreement_rate"] for r in program_rows],
                   [r["true_accuracy"] for r in program_rows])

    save_results("e23_self_check", {
        "n_probes_per_world": N_PROBES, "seed": SEED,
        "n_programs": len(program_rows),
        "n_program_probe_pairs": len(flags_and_errors),
        "n_flagged": n_flagged, "n_true_errors": n_errors,
        "flag_precision": precision,
        "flag_precision_ci": list(wilson_ci(true_pos, n_flagged)) if n_flagged else None,
        "flag_recall": recall,
        "flag_recall_ci": list(wilson_ci(true_pos, n_errors)) if n_errors else None,
        "spearman_agreement_vs_accuracy": rho,
        "program_rows": program_rows,
    })
    print(f"programs: {len(program_rows)}, pairs: {len(flags_and_errors)}")
    print(f"flag precision {precision}, recall {recall}")
    print(f"Spearman(agreement, true accuracy) = {rho:.3f}")


if __name__ == "__main__":
    main()
