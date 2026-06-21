"""E80 genomics-ESM (Spearman + matched-data control): is world-time compute a real effect on
real variant-effect data, or just a data-scaling curve?

Perceptor: ESM-2 embeds each protein's wild-type sequence once -> per-residue embedding +
wt-marginal log-likelihood-ratio (wt->mut). A regression head over [embedding, LLR, biochem]
predicts the (per-assay z-scored) DMS_score. We report the FIELD-STANDARD metric: mean per-assay
Spearman on strictly held-out proteins, with the ESM zero-shot LLR Spearman as the baseline
(reproducing the ProteinGym leaderboard's zero-shot number) for comparison.

Three measurements:
  (a) ladder        -- Spearman vs number of train protein-worlds (examples/world fixed).
  (b) MATCHED-DATA  -- the decisive control: hold TOTAL training examples fixed and vary the
                       number of worlds they are spread across. If more worlds help at equal
                       data, world DIVERSITY is the lever (world-time compute); if flat, the
                       ladder was just a data-scaling curve.
  (c) ablation      -- corrupt a fraction of the (real) training labels; does Spearman collapse?

Needs torch + transformers + scipy. Runs on the GPU box.
  python3 e80_genomics_esm.py --bucket gs://openworld-bench/e80-genomics-esm
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PG_DIR = Path(os.environ.get("OW_PG_DIR",
              str(ROOT / "experiments" / "data" / "ProteinGym_substitutions")))
ESM_NAME = os.environ.get("OW_ESM", "facebook/esm2_t33_650M_UR50D")
OUT = ROOT / "experiments" / "results" / "e80_genomics_esm.json"
CACHE = ROOT / "experiments" / "results" / "e80_esm_cache.npz"
MAXLEN = 1022
CAP = 500                       # mutations kept per protein-world

LADDER = [2, 8, 32, 64, 128]
MATCHED_TOTAL = 3200           # fixed total training examples for the control
MATCHED_W = [2, 4, 8, 16, 32, 64]
ABL_W = 48
ABL_NOISE = [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0]
N_TEST = 40
SEEDS = [0, 1, 2]

HYDRO = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
         "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
         "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
VOLUME = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4,
          "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9,
          "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 1}
AAS = set(HYDRO)


def parse_mutant(m):
    m = (m or "").strip()
    if ":" in m or ";" in m or len(m) < 3:
        return None
    wt, mut, pos = m[0], m[-1], m[1:-1]
    if wt not in AAS or mut not in AAS or not pos.isdigit():
        return None
    return wt, int(pos), mut


def _read_assays():
    """protein -> (wt_seq, [(pos,wt,mut,dms_score)]). Continuous DMS_score for Spearman."""
    import csv as _csv
    out = {}
    for f in sorted(PG_DIR.glob("*.csv")):
        if f.name.startswith("._"):
            continue
        try:
            recs = list(_csv.DictReader(open(f)))
        except Exception:  # noqa: BLE001
            continue
        muts, wtseq = [], None
        for r in recs:
            s = r.get("DMS_score", "")
            pm = parse_mutant(r.get("mutant", ""))
            seq = r.get("mutated_sequence", "") or ""
            if s == "" or s is None or pm is None or not seq:
                continue
            wt, pos, mut = pm
            if pos > len(seq) or seq[pos - 1] != mut:
                continue
            try:
                sc = float(s)
            except ValueError:
                continue
            if wtseq is None:
                wtseq = seq[:pos - 1] + wt + seq[pos:]
            muts.append((pos, wt, mut, sc))
        if wtseq and len(wtseq) <= MAXLEN and len(muts) >= 80:
            out[f.stem] = (wtseq, muts[:CAP])
    return out


def embed():
    """ESM-2 perceptor -> per-world (X features, llr, y_z) cached. y_z = per-assay z-scored DMS."""
    if CACHE.exists():
        return np.load(CACHE, allow_pickle=True)["worlds"].item()
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, EsmForMaskedLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ESM_NAME)
    model = EsmForMaskedLM.from_pretrained(ESM_NAME, output_hidden_states=True).to(dev).eval()
    assays = _read_assays()
    print(f"[esm] embedding {len(assays)} proteins with {ESM_NAME}", flush=True)
    worlds = {}
    for k, (seq, muts) in assays.items():
        enc = tok(seq, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**enc)
        hid = o.hidden_states[-1][0].float().cpu().numpy()
        lp = F.log_softmax(o.logits[0].float(), dim=-1).cpu().numpy()
        X, llr, sc = [], [], []
        for pos, wt, mut, dms in muts:
            i = pos
            ll = lp[i, tok.convert_tokens_to_ids(mut)] - lp[i, tok.convert_tokens_to_ids(wt)]
            X.append(np.concatenate([hid[i], [ll, HYDRO[mut] - HYDRO[wt],
                     VOLUME[mut] - VOLUME[wt], CHARGE.get(mut, 0) - CHARGE.get(wt, 0)]]).astype(np.float32))
            llr.append(ll)
            sc.append(dms)
        sc = np.array(sc, dtype=np.float32)
        yz = (sc - sc.mean()) / (sc.std() + 1e-6)         # per-assay z-score (training target)
        worlds[k] = (np.stack(X), np.array(llr, np.float32), yz)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, worlds=np.array(worlds, dtype=object))
    return worlds


def _spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or np.std(a) == 0:
        return 0.0
    r = spearmanr(a, b).correlation
    return 0.0 if (r is None or np.isnan(r)) else float(r)


def train_head(Xtr, ytr, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(128, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    n = len(xt)
    for _ in range(40):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            lossf(net(xt[idx]).squeeze(1), yt[idx]).backward()
            opt.step()
    return net, mu, sd


def held_out_spearman(net, mu, sd, worlds, test_names):
    """Mean per-assay Spearman on held-out proteins (the ProteinGym protocol)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rs = []
    for nm in test_names:
        X, _, y = worlds[nm]
        with torch.no_grad():
            pred = net(torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)).squeeze(1).cpu().numpy()
        rs.append(_spearman(pred, y))
    return float(np.mean(rs))


def sample(worlds, names, per_world, rng):
    Xs, ys = [], []
    for nm in names:
        X, _, y = worlds[nm]
        idx = rng.permutation(len(X))[:per_world]
        Xs.append(X[idx])
        ys.append(y[idx])
    return np.concatenate(Xs), np.concatenate(ys)


def mean_ci(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None}
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
    t = {1: 12.71, 2: 4.30, 3: 3.18}.get(len(v) - 1, 1.96)
    return {"mean": round(m, 4), "ci": [round(m - t * sd / math.sqrt(len(v)), 4),
            round(m + t * sd / math.sqrt(len(v)), 4)], "seeds": [round(x, 4) for x in v]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="")
    args = ap.parse_args()
    worlds = embed()
    names = sorted(worlds)
    split = random.Random(80)
    order = names[:]
    split.shuffle(order)
    nte = min(N_TEST, len(order) // 3)
    test_names, pool = order[:nte], order[nte:]

    # ESM zero-shot LLR Spearman (the ProteinGym leaderboard's zero-shot number, our repro)
    zs = float(np.mean([_spearman(worlds[n][1], worlds[n][2]) for n in test_names]))

    res = {"domain": "genomics-esm", "esm": ESM_NAME, "n_worlds": len(names),
           "n_test_worlds": nte, "metric": "mean per-assay Spearman (held-out proteins)",
           "esm_llr_zeroshot_spearman": round(zs, 4),
           "config": {"ladder": LADDER, "matched_total": MATCHED_TOTAL, "matched_W": MATCHED_W,
                      "abl_W": ABL_W, "abl_noise": ABL_NOISE, "seeds": SEEDS},
           "ladder_raw": {}, "matched_raw": {}, "ablation_raw": {}}

    def upload():
        res["ladder"] = {str(n): mean_ci(list(v.values())) for n, v in res["ladder_raw"].items()}
        res["matched_data"] = {str(w): mean_ci(list(v.values())) for w, v in res["matched_raw"].items()}
        res["ablation"] = {t: mean_ci(list(v.values())) for t, v in res["ablation_raw"].items()}
        OUT.write_text(json.dumps(res, indent=2))
        if args.bucket:
            import subprocess
            subprocess.run(["gcloud", "storage", "cp", str(OUT),
                            f"{args.bucket}/e80_genomics_esm.json"], check=False)

    print(f"[esm] {len(names)} worlds | ESM zero-shot Spearman {zs:.3f}", flush=True)
    upload()

    # (a) ladder: vary #worlds, examples/world fixed (=80)
    for seed in SEEDS:
        p = pool[:]
        random.Random(seed).shuffle(p)
        rng = np.random.RandomState(100 + seed)
        for n in [x for x in LADDER if x <= len(p)]:
            net, mu, sd = train_head(*sample(worlds, p[:n], 80, rng), seed)
            r = held_out_spearman(net, mu, sd, worlds, test_names)
            res["ladder_raw"].setdefault(str(n), {})[str(seed)] = r
            print(f"[ladder s{seed} N={n}] spearman {r:.3f}", flush=True)
            upload()

    # (b) MATCHED-DATA control: total examples fixed = MATCHED_TOTAL, vary #worlds
    for seed in SEEDS:
        p = pool[:]
        random.Random(50 + seed).shuffle(p)
        rng = np.random.RandomState(200 + seed)
        for w in [x for x in MATCHED_W if x <= len(p)]:
            per = max(5, MATCHED_TOTAL // w)
            net, mu, sd = train_head(*sample(worlds, p[:w], per, rng), seed)
            r = held_out_spearman(net, mu, sd, worlds, test_names)
            res["matched_raw"].setdefault(str(w), {})[str(seed)] = r
            print(f"[matched s{seed} W={w} per={per} total~{w*per}] spearman {r:.3f}", flush=True)
            upload()

    # (c) ablation: corrupt training labels (shuffle a fraction within the training pool)
    for seed in SEEDS:
        p = pool[:]
        random.Random(700 + seed).shuffle(p)
        rng = np.random.RandomState(300 + seed)
        abln = p[:min(ABL_W, len(p))]
        for noise in ABL_NOISE:
            Xtr, ytr = sample(worlds, abln, 80, rng)
            if noise > 0:
                k = int(noise * len(ytr))
                fl = rng.permutation(len(ytr))[:k]
                ytr = ytr.copy()
                ytr[fl] = ytr[rng.permutation(len(ytr))[:k]]   # shuffle a fraction of labels
            net, mu, sd = train_head(Xtr, ytr, seed)
            r = held_out_spearman(net, mu, sd, worlds, test_names)
            res["ablation_raw"].setdefault(f"n{int(noise*100):02d}", {})[str(seed)] = r
            print(f"[abl s{seed} p={noise}] spearman {r:.3f}", flush=True)
            upload()

    upload()
    print("[esm] done", json.dumps({"zero_shot": zs, "ladder": res.get("ladder"),
          "matched_data": res.get("matched_data"), "ablation": res.get("ablation")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
