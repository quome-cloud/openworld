"""Bootstrap 95% CI (10k resamples) for a results JSON's mean progression.

Usage: python3 bootstrap_ci.py results/nethack_results_baseline25.json [SOTA]
"""

import json
import random
import sys

SOTA = float(sys.argv[2]) if len(sys.argv) > 2 else 6.8


def main():
    doc = json.load(open(sys.argv[1]))
    xs = [e["progression"] * 100 for e in doc["episodes"]]
    n = len(xs)
    rng = random.Random(20260706)
    means = sorted(sum(rng.choices(xs, k=n)) / n for _ in range(10_000))
    mean = sum(xs) / n
    lo, hi = means[249], means[9749]
    print(f"n={n} mean={mean:.2f}  bootstrap 95% CI [{lo:.2f}, {hi:.2f}]")
    if lo > SOTA:
        verdict = f"CI excludes SOTA {SOTA} from above: decisively above SOTA"
    elif hi < SOTA:
        verdict = f"CI excludes SOTA {SOTA} from below: below SOTA"
    else:
        verdict = f"CI straddles SOTA {SOTA}: at SOTA level"
    print(verdict)
    return {"n": n, "mean": mean, "ci95": [lo, hi], "verdict": verdict}


if __name__ == "__main__":
    out = main()
    doc = json.load(open(sys.argv[1]))
    doc["bootstrap"] = out
    json.dump(doc, open(sys.argv[1], "w"), indent=1)
