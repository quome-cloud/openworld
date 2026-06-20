"""E80 (genomics) - world-time compute on REAL variant-effect data (ProteinGym DMS).

Each deep-mutational-scanning assay (one protein) is a WORLD; each single-residue mutation is
an example; the label is the REAL measured effect (ProteinGym's binary DMS_score_bin:
1 = functional/tolerated, 0 = deleterious). The shared skill is "predict a mutation's effect
from its biochemical context." We train on many assay-worlds and test on STRICTLY held-out
proteins (no protein appears in both) -- the mechanism test for world-time compute on real,
exactly-labeled genomics data.

A general LLM has no amino-acid-sequence prior, so mutations are represented by FEATURES
(wild-type/mutant residue, normalized position, and biochemical deltas), not raw sequence --
this asks whether traversing many real assays teaches transferable mutation-effect *rules*.
Bar: does held-out accuracy scale with the number of assay-worlds, and does it collapse when
the real labels are corrupted (the E78b exact-label ablation, on real fitness)?

Offline/deterministic given the downloaded ProteinGym CSVs (one per assay). Emits SFT/test
jsonl + manifest in the e73/e74 prompt-completion format.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Directory of ProteinGym substitution-assay CSVs (downloaded on the box; set OW_PG_DIR to
# wherever the zip unpacked). Each CSV has columns: mutant (e.g. "A45G"), DMS_score,
# DMS_score_bin (1 = functional/tolerated, 0 = deleterious).
PG_DIR = Path(os.environ.get("OW_PG_DIR",
              str(ROOT / "experiments" / "data" / "ProteinGym_substitutions")))

# world-count ladder + verified-vs-noisy ablation config (consumed by e80_common).
CONFIG = {
    "ladder": [2, 8, 32, 64, 128],
    "abl_n": 48,
    "abl_noise": [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0],
    "cap": 150,            # mutations per assay
    "n_test": 30,          # held-out proteins (strict, no leakage)
    "seeds": [0, 1],
    "base": "Qwen/Qwen2.5-0.5B-Instruct",
}


def build_worlds():
    """Each ProteinGym substitution assay (protein) -> a world of single-mutation examples
    labeled by the REAL measured effect. Keep assays with both classes and enough mutations."""
    worlds = {}
    for f in sorted(PG_DIR.glob("*.csv")):
        rows = load_assay(f)
        labs = {r["label"] for r in rows}
        if len(rows) >= 60 and len(labs) == 2:
            worlds[f.stem] = {"classes": ["tolerated", "deleterious"],
                              "rows": [{"prompt": r["prompt"], "label": r["label"]} for r in rows]}
    return worlds

# --- amino-acid biochemical property tables (Kyte-Doolittle hydropathy, side-chain volume A^3,
#     charge at pH7) -- the features a general model can use to learn mutation-effect rules. ---
HYDRO = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
         "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
         "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
VOLUME = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
          "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
          "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
          "Y": 193.6, "V": 140.0}
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 1}  # others 0
AAS = set(HYDRO)


def parse_mutant(m):
    """'A45G' -> ('A', 45, 'G'); returns None for multi-mutants or malformed."""
    m = m.strip()
    if ":" in m or ";" in m or len(m) < 3:
        return None
    wt, mut = m[0], m[-1]
    pos = m[1:-1]
    if wt not in AAS or mut not in AAS or not pos.isdigit():
        return None
    return wt, int(pos), mut


def featurize(wt, pos, mut, seqlen):
    """Biochemical context of a single substitution -> a compact, model-readable description."""
    dh = HYDRO[mut] - HYDRO[wt]
    dv = VOLUME[mut] - VOLUME[wt]
    dc = CHARGE.get(mut, 0) - CHARGE.get(wt, 0)
    relpos = pos / seqlen if seqlen else 0.0
    return {
        "wt": wt, "mut": mut, "relpos": round(relpos, 2),
        "d_hydropathy": round(dh, 1), "d_volume": round(dv, 0), "d_charge": dc,
        "proline_involved": int("P" in (wt, mut)),
        "glycine_involved": int("G" in (wt, mut)),
        "charge_flip": int(CHARGE.get(wt, 0) * CHARGE.get(mut, 0) < 0),
    }


def make_prompt(feat):
    loc = "N-term" if feat["relpos"] < 0.33 else ("C-term" if feat["relpos"] > 0.66 else "middle")
    return ("You are a protein variant-effect predictor. A single amino-acid substitution:\n"
            f"  wild-type residue: {feat['wt']}\n  mutant residue: {feat['mut']}\n"
            f"  position: {loc} (relative {feat['relpos']})\n"
            f"  change in hydropathy: {feat['d_hydropathy']}\n"
            f"  change in side-chain volume: {feat['d_volume']} A^3\n"
            f"  change in charge: {feat['d_charge']}\n"
            f"  proline involved: {'yes' if feat['proline_involved'] else 'no'}; "
            f"glycine involved: {'yes' if feat['glycine_involved'] else 'no'}; "
            f"charge reversal: {'yes' if feat['charge_flip'] else 'no'}\n"
            "Is this mutation TOLERATED (preserves function) or DELETERIOUS? "
            "Reply with ONLY 'tolerated' or 'deleterious'.")


LABELS = {1: "tolerated", 0: "deleterious"}


def load_assay(csv_path, max_per_assay=400):
    """Parse one ProteinGym assay CSV -> list of (prompt, label_str). Single substitutions only."""
    import csv as _csv
    rows = []
    with open(csv_path) as fh:
        reader = _csv.DictReader(fh)
        recs = [r for r in reader]
    # infer sequence length from the max parsed position (robust if mutated_sequence absent)
    parsed = []
    for r in recs:
        if "DMS_score_bin" not in r or r["DMS_score_bin"] == "":
            continue
        pm = parse_mutant(r.get("mutant", ""))
        if pm is None:
            continue
        parsed.append((pm, int(float(r["DMS_score_bin"]))))
    if not parsed:
        return []
    seqlen = max(p[0][1] for p in parsed)
    for (wt, pos, mut), lab in parsed:
        rows.append({"prompt": make_prompt(featurize(wt, pos, mut, seqlen)),
                     "completion": LABELS[lab], "answer": LABELS[lab]})
    return rows[:max_per_assay]


if __name__ == "__main__":
    # offline self-test of the model-readable core (no ProteinGym download needed)
    assert parse_mutant("A45G") == ("A", 45, "G")
    assert parse_mutant("M1V") == ("M", 1, "V")
    assert parse_mutant("A45G:L50P") is None and parse_mutant("x") is None
    f = featurize("A", 45, "G", 100)
    print("featurize A45G/len100:", f)
    print("prompt:\n" + make_prompt(f))
    print("ok: genomics mutation parser + featurizer + prompt")
