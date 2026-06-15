"""E53 - Sheaf consistency: gluing local views into a global world (and catching
when you can't).

Semirings parse sub-worlds; sheaves check whether their local views glue into one
consistent global world. A field over L locations is observed by M overlapping
sensors (agents); each reports the locations in its footprint. The sheaf glues
consistent reports into the global field, and when a sensor is FAULTY it (1)
detects the inconsistency (the gluing obstruction is non-zero), (2) localizes the
fault (the agent implicated in the most overlap-disagreement), and (3) - because
overlaps are redundant - CORRECTS it by consensus across the agents that observe
each location, where naive averaging is silently corrupted by the fault.

Deterministic/offline/self-checking.
"""

import numpy as np

from openworld import glue, is_consistent, localize_fault, majority_glue
from openworld.sheaf import nerve_betti1, obstruction_norm

from common import save_results

L = 40                  # field locations
M = 30                  # sensors (agents)
WIN = 8                 # each sensor observes a window of WIN locations
SEED = 53


def build_cover(rng):
    cover = {}
    for i in range(M):
        start = rng.randint(0, L)
        cover[f"s{i}"] = [f"x{(start + k) % L}" for k in range(WIN)]
    return cover


def true_field():
    t = np.linspace(0, 1, L)
    f = np.sin(2 * np.pi * t) + 0.4 * np.sin(2 * np.pi * 3 * t)
    return {f"x{i}": float(f[i]) for i in range(L)}


def sections_with_faults(cover, field, faulty, rng):
    sec = {a: {v: field[v] for v in vs} for a, vs in cover.items()}
    for a in faulty:
        sec[a] = {v: field[v] + rng.uniform(-3, 3) for v in cover[a]}   # garbage
    return sec


def rmse(field, est):
    keys = [k for k in field if k in est]
    return float(np.sqrt(np.mean([(field[k] - est[k]) ** 2 for k in keys])))


def coverage(cover):
    cnt = {}
    for vs in cover.values():
        for v in vs:
            cnt[v] = cnt.get(v, 0) + 1
    return cnt


def main():
    rng = np.random.RandomState(SEED)
    cover = build_cover(rng)
    field = true_field()
    cov = coverage(cover)
    min_cov = min(cov.values())
    b1 = nerve_betti1(cover)

    rows = []
    field_snapshots = {}
    agents = list(cover)
    for nf in range(0, 9):
        frng = np.random.RandomState(100 + nf)
        faulty = set(frng.choice(agents, nf, replace=False)) if nf else set()
        sec = sections_with_faults(cover, field, faulty, frng)
        obstruction = obstruction_norm(cover, sec)
        detected = obstruction > 1e-9
        # localization: are the top-nf blamed agents the faulty ones?
        blame = {a: 0.0 for a in cover}
        from openworld.sheaf import disagreements
        for (a, b, _), d in disagreements(cover, sec).items():
            blame[a] += d
            blame[b] += d
        ranked = sorted(blame, key=blame.get, reverse=True)
        loc_acc = (len(set(ranked[:nf]) & faulty) / nf) if nf else 1.0
        # correction
        maj = majority_glue(cover, sec)
        avg = {v: float(np.mean([sec[a][v] for a in cover if v in sec[a]])) for v in field}
        rows.append({"n_faults": nf, "detected": detected,
                     "obstruction": round(obstruction, 3),
                     "localize_acc": round(loc_acc, 3),
                     "majority_rmse": round(rmse(field, maj), 4),
                     "average_rmse": round(rmse(field, avg), 4)})
        if nf in (0, 3):
            field_snapshots[nf] = {
                "truth": [field[f"x{i}"] for i in range(L)],
                "average": [avg[f"x{i}"] for i in range(L)],
                "majority": [maj[f"x{i}"] for i in range(L)]}

    # consistent case glues exactly
    sec0 = {a: {v: field[v] for v in vs} for a, vs in cover.items()}
    glued = glue(cover, sec0)
    glue_err = rmse(field, glued)

    results = {
        "locations": L, "sensors": M, "window": WIN,
        "min_coverage": min_cov, "nerve_betti1": b1,
        "glue_exact_error": round(glue_err, 6),
        "rows": rows, "snapshots": field_snapshots,
    }
    save_results("e53_sheaf", results)

    print(f"E53 - sheaf consistency ({M} sensors over {L} locations, each location "
          f"seen by >= {min_cov}, nerve b1={b1})\n")
    print(f"  consistent reports glue to the exact global field (error {glue_err:.1e})")
    print(f"  {'faults':>6}{'detected':>10}{'localize':>10}{'maj RMSE':>10}{'avg RMSE':>10}")
    for r in rows:
        print(f"  {r['n_faults']:>6}{str(r['detected']):>10}{r['localize_acc']:>10.2f}"
              f"{r['majority_rmse']:>10.3f}{r['average_rmse']:>10.3f}")

    # --- self-checks ---
    assert glue_err < 1e-9, "consistent local sections must glue to the exact global"
    assert is_consistent(cover, sec0), "no-fault case must be consistent"
    faulted = [r for r in rows if r["n_faults"] > 0]
    assert all(r["detected"] for r in faulted), "every fault must be detected"
    assert rows[1]["localize_acc"] == 1.0, "a single fault must be localized exactly"
    # correction: majority recovers the field; averaging is corrupted by faults
    mid = rows[3]                                   # 3 faults
    assert mid["majority_rmse"] < mid["average_rmse"], "majority should beat averaging"
    assert mid["majority_rmse"] < 0.05, "majority should essentially recover the field"
    print("\nall checks pass: glue, detect, localize, and correct via the sheaf.")


if __name__ == "__main__":
    main()
