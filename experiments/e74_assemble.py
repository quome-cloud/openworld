"""E74 (assemble) - collect the LLM model-size sweep eval files into one results JSON the
paper pipeline reads. Picks up whatever sizes are present (14B/32B appear once the QLoRA
runs finish), so it can be re-run to extend the curve.

Reads experiments/results/e74_artifacts/eval_<tag>_{base,ft}.json (1.5B is the eval_dx_*
pair) and experiments/results/e74_diagnosis.json (offline floor/oracle); writes
experiments/results/e74_scaling.json.
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "experiments" / "results" / "e74_artifacts"
OUT = ROOT / "experiments" / "results" / "e74_scaling.json"

# (display name, params in B, eval-file tag, precision of the fine-tune)
SIZES = [("0.5B", 0.5, "0.5B", "bf16 LoRA"),
         ("1.5B", 1.5, "dx", "bf16 LoRA"),
         ("3B", 3.0, "3B", "bf16 LoRA"),
         ("7B", 7.0, "7B", "bf16 LoRA"),
         ("14B", 14.0, "14B", "4-bit QLoRA"),
         ("32B", 32.0, "32B", "4-bit QLoRA")]


def _read(tag, kind):
    f = ART / f"eval_{tag}_{kind}.json"
    return json.loads(f.read_text()) if f.exists() else None


def boot_ci(xs, n=5000, seed=0):
    """Bootstrap 95% CI of the mean of xs (resampling with replacement)."""
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(xs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return round(means[int(0.025 * n)], 4), round(means[int(0.975 * n)], 4)


def main():
    diag = json.loads((ROOT / "experiments" / "results" / "e74_diagnosis.json").read_text())
    sizes = []
    for name, pb, tag, prec in SIZES:
        bj, fj = _read(tag, "base"), _read(tag, "ft")
        if bj is None or fj is None:
            continue
        # bootstrap CIs over the held-out specialties (paired for the gain)
        keys = sorted(bj["per_specialty_accuracy"])
        bvals = [bj["per_specialty_accuracy"][k] for k in keys]
        fvals = [fj["per_specialty_accuracy"][k] for k in keys]
        gvals = [fv - bv for bv, fv in zip(bvals, fvals)]
        b_lo, b_hi = boot_ci(bvals)
        f_lo, f_hi = boot_ci(fvals)
        g_lo, g_hi = boot_ci(gvals)
        sizes.append({"name": name, "params_b": pb,
                      "base": bj["accuracy"], "base_ci": [b_lo, b_hi],
                      "ft": fj["accuracy"], "ft_ci": [f_lo, f_hi],
                      "gain": round(fj["accuracy"] - bj["accuracy"], 4), "gain_ci": [g_lo, g_hi],
                      "precision": prec})
    # n from any present eval file
    meta = {}
    for name, pb, tag, prec in SIZES:
        p = ART / f"eval_{tag}_base.json"
        if p.exists():
            d = json.loads(p.read_text())
            meta = {"n_test_cases": d["n_cases"], "n_test_specialties": d["n_held_out_specialties"]}
            break
    results = {
        "task": "world-test compute: fine-tuning an LLM on traversed diagnosis-specialty "
                "world models, then testing on held-out specialties, across model sizes",
        "model_family": "Qwen2.5-Instruct",
        "n_train_specialties": 60,
        **meta,
        "offline_floor": diag.get("prior_only_floor"),
        "offline_oracle": diag.get("oracle_ceiling"),
        "precision_note": "0.5B-7B bf16 LoRA; 14B/32B 4-bit QLoRA (a quantization step at the top end)",
        "sizes": sizes,
    }
    OUT.write_text(json.dumps(results, indent=2))
    print(f"[e74-assemble] {len(sizes)} sizes -> {OUT.name}")
    for s in sizes:
        print(f"  {s['name']:>5}: base {s['base']:.3f} -> ft {s['ft']:.3f}  "
              f"gain +{s['gain']:.3f} CI[{s['gain_ci'][0]:+.3f},{s['gain_ci'][1]:+.3f}]  [{s['precision']}]")


if __name__ == "__main__":
    main()
