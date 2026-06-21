"""E80 (diagnosis) - world-time compute on REAL clinical diagnosis (DDXPlus).

A WORLD is a real medical "specialty" -- a disjoint group of pathologies; an example is a real
patient whose true PATHOLOGY is in that specialty; the label is the real pathology. We give
each candidate pathology's typical-evidence PROFILE in the prompt (computed from data), so the
shared skill is "match the patient's findings to the best-fitting condition profile" -- exactly
the synthetic-diagnosis design, now on REAL patients. Pathologies are partitioned so each
appears in exactly one specialty: holding out whole specialty-worlds = holding out whole
conditions (strict, no leakage). Real labels -> rebuts the synthetic critique on the running
example.

Consumed by e80_common. Loads DDXPlus from HuggingFace (streamed; needs `datasets`).
"""

import ast
import random

CONFIG = {
    "ladder": [2, 4, 8],
    "abl_n": 6,
    "abl_noise": [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0],
    "cap": 80,
    "n_test": 4,
    "seeds": [0, 1],
    "base": "Qwen/Qwen2.5-0.5B-Instruct",
}

PER_PATH = 200          # patients collected per pathology
MAX_STREAM = 200000     # cap streamed rows
SPECIALTY_SIZE = 4      # pathologies per specialty-world
PROFILE_TOP = 8         # typical evidences shown per candidate pathology


def _base(ev):
    """'E_54_@_V_161' -> 'E_54'; keep base evidence code (presence), drop the value part."""
    return ev.split("_@_")[0]


def build_worlds():
    from datasets import load_dataset
    ds = load_dataset("aai530-group6/ddxplus", split="train", streaming=True)
    by_path = {}            # pathology -> list of evidence-code sets
    counts = {}
    for i, r in enumerate(ds):
        if i >= MAX_STREAM:
            break
        path = r.get("PATHOLOGY")
        evs = r.get("EVIDENCES")
        if isinstance(evs, str):                 # DDXPlus stores EVIDENCES as a stringified list
            try:
                evs = ast.literal_eval(evs)
            except (ValueError, SyntaxError):
                continue
        if not path or not evs:
            continue
        if counts.get(path, 0) >= PER_PATH:
            continue
        codes = sorted({_base(e) for e in evs})
        by_path.setdefault(path, []).append(codes)
        counts[path] = counts.get(path, 0) + 1

    paths = sorted(by_path)
    # per-pathology profile: the most DISTINCTIVE evidences (lift = P(code|pathology)/P(code)),
    # not the most common -- common findings (e.g. demographics) don't discriminate conditions.
    gfreq, total = {}, 0
    for p in paths:
        total += len(by_path[p])
        for codes in by_path[p]:
            for c in set(codes):
                gfreq[c] = gfreq.get(c, 0) + 1
    profile = {}
    for p in paths:
        n = len(by_path[p])
        freq = {}
        for codes in by_path[p]:
            for c in codes:
                freq[c] = freq.get(c, 0) + 1
        scored = [(c, (freq[c] / n) / (gfreq[c] / total)) for c in freq if freq[c] >= 0.2 * n]
        profile[p] = [c for c, _ in sorted(scored, key=lambda kv: -kv[1])[:PROFILE_TOP]]

    # disjoint partition of pathologies -> specialty-worlds (strict held-out by condition)
    rng = random.Random(80)
    rng.shuffle(paths)
    specialties = [paths[i:i + SPECIALTY_SIZE] for i in range(0, len(paths), SPECIALTY_SIZE)]
    specialties = [s for s in specialties if len(s) >= 2]

    worlds = {}
    for si, spec in enumerate(specialties):
        prof_block = "\n".join(f"  {p}: {', '.join(profile[p])}" for p in spec)
        rows = []
        for p in spec:
            for codes in by_path[p][:CONFIG["cap"] + 40]:
                present = ", ".join(codes) if codes else "(none)"
                prompt = ("You are a diagnostician. Candidate conditions and their typical "
                          f"findings (evidence codes):\n{prof_block}\n"
                          f"This patient presents with: {present}.\n"
                          "Which single condition best explains it? Reply with ONLY the "
                          "condition name exactly as listed.")
                rows.append({"prompt": prompt, "label": p})
        if len(rows) >= 60 and len({r["label"] for r in rows}) >= 2:
            worlds[f"specialty_{si:02d}"] = {"classes": spec, "rows": rows}
    return worlds


if __name__ == "__main__":
    w = build_worlds()
    print("diagnosis specialty-worlds:", len(w))
    for k, v in list(w.items())[:3]:
        print(" ", k, "classes:", v["classes"], "rows:", len(v["rows"]))
