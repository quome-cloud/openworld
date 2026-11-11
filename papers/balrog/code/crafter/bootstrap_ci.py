"""Bootstrap 95% CI (10k resamples, fixed rng seed) for a crafter summary
JSON (per_episode list of progression %). Usage:
  python3 bootstrap_ci.py results/summary_block25.json [SOTA=57.3]
Writes the bootstrap block back into the JSON (NetHack-arm CI format)."""
import json
import random
import sys

SOTA = float(sys.argv[2]) if len(sys.argv) > 2 else 57.3
doc = json.load(open(sys.argv[1]))
xs = doc['per_episode']
n = len(xs)
rng = random.Random(20260706)
means = sorted(sum(rng.choices(xs, k=n)) / n for _ in range(10_000))
mean = sum(xs) / n
lo, hi = means[249], means[9749]
if lo > SOTA:
    verdict = f"CI excludes SOTA {SOTA} from above: decisively above SOTA"
elif hi < SOTA:
    verdict = f"CI excludes SOTA {SOTA} from below: below SOTA"
else:
    verdict = f"CI straddles SOTA {SOTA}: at SOTA level"
doc['bootstrap'] = dict(n=n, mean=round(mean, 2),
                        ci95=[round(lo, 2), round(hi, 2)], verdict=verdict)
json.dump(doc, open(sys.argv[1], 'w'), indent=1)
print(f"n={n} mean={mean:.2f} CI95 [{lo:.2f}, {hi:.2f}] :: {verdict}")
