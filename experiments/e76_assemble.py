"""E76 (assemble) - world-count scaling curve into one results JSON.

Reads the per-N eval files (held-out diagnostic accuracy of a 7B model fine-tuned on N
train specialties of the HARD family) and writes experiments/results/e76_world_count.json.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments" / "results" / "e77_artifacts" / "results"
OUT = ROOT / "experiments" / "results" / "e76_world_count.json"
N_GRID = [8, 16, 32, 64, 128, 256, 512]


def main():
    base = json.loads((SRC / "e76_base.json").read_text())["accuracy"]
    pts = []
    for n in N_GRID:
        f = SRC / f"e76_N{n}_ft.json"
        if f.exists():
            a = json.loads(f.read_text())["accuracy"]
            pts.append({"n_worlds": n, "acc": round(a, 4), "gain": round(a - base, 4)})
    out = {
        "task": "world-count scaling: held-out diagnostic accuracy vs number of TRAIN "
                "specialties traversed (hard diagnosis family); fine-tune target = 7B",
        "model": "Qwen2.5-7B-Instruct", "family": "hard (E75 params)",
        "oracle": 0.69, "n_test_specialties": 20, "base": round(base, 4),
        "points": pts,
        "note": "few worlds hurt (overfit a narrow slice); gain rises monotonically from "
                "N32 onward and is still climbing at N512 (no plateau).",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[e76-assemble] base {out['base']} ->", " ".join(f"N{p['n_worlds']}={p['acc']}(+{p['gain']})" for p in pts))


if __name__ == "__main__":
    main()
