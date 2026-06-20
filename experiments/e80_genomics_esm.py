"""E80 genomics-ESM: world-time compute with a protein SEQUENCE-MODEL perceptor (ESM-2).

The text-LLM port showed no world-count scaling on variant effect -- a representation problem
(a general LLM has no sequence prior). Here the PERCEPTOR is ESM-2: each protein's wild-type
sequence is embedded once -> per-residue embeddings + masked(wt)-marginal log-likelihoods. Each
mutation's state = ESM embedding at the site + ESM log-likelihood-ratio (wt->mut) + biochemical
deltas. World-time compute then trains a small head across N protein-worlds and tests on STRICTLY
held-out proteins. Question: does a richer (ESM) representation make held-out accuracy SCALE with
the number of assay-worlds, where the text-LLM stayed flat?

Self-contained: ESM embedding (cached) + world-count ladder + verified-vs-noisy ablation + GCS
upload. Needs torch + transformers (ESM); runs on the GPU box.
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
ESM_NAME = os.environ.get("OW_ESM", "facebook/esm2_t30_150M_UR50D")
OUT = ROOT / "experiments" / "results" / "e80_genomics_esm.json"
CACHE = ROOT / "experiments" / "results" / "e80_esm_cache.npz"
MAXLEN = 1022
CAP = 200          # mutations per protein-world

CONFIG = {"ladder": [2, 8, 32, 64, 128], "abl_n": 48,
          "abl_noise": [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0],
          "n_test": 30, "seeds": [0, 1]}

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
    """protein -> (wt_seq, [(pos, wt, mut, label)])."""
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
            b = r.get("DMS_score_bin", "")
            pm = parse_mutant(r.get("mutant", ""))
            seq = r.get("mutated_sequence", "") or ""
            if b == "" or b is None or pm is None or not seq:
                continue
            wt, pos, mut = pm
            if pos > len(seq) or seq[pos - 1] != mut:
                continue
            if wtseq is None:
                wtseq = seq[:pos - 1] + wt + seq[pos:]
            muts.append((pos, wt, mut, int(float(b))))
        if wtseq and len(wtseq) <= MAXLEN and len(muts) >= 60 and len({m[3] for m in muts}) == 2:
            out[f.stem] = (wtseq, muts[:CAP * 2])
    return out


def embed():
    """ESM-2 perceptor -> per-mutation feature matrix per protein-world (cached)."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return z["worlds"].item()
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
        hid = o.hidden_states[-1][0].float().cpu().numpy()          # [L+2, D]
        lp = F.log_softmax(o.logits[0].float(), dim=-1).cpu().numpy()  # [L+2, vocab]
        X, y = [], []
        for pos, wt, mut, lab in muts:
            i = pos                                                  # CLS offset -> token idx
            llr = lp[i, tok.convert_tokens_to_ids(mut)] - lp[i, tok.convert_tokens_to_ids(wt)]
            feat = np.concatenate([hid[i], [llr, HYDRO[mut] - HYDRO[wt],
                                            VOLUME[mut] - VOLUME[wt],
                                            CHARGE.get(mut, 0) - CHARGE.get(wt, 0)]])
            X.append(feat.astype(np.float32))
            y.append(lab)
        worlds[k] = (np.stack(X), np.array(y, dtype=np.int64))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, worlds=np.array(worlds, dtype=object))
    return worlds


def train_eval(Xtr, ytr, Xte, yte, seed):
    """Small MLP head on ESM features -> held-out accuracy."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(128, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    n = len(xt)
    for ep in range(40):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            lossf(net(xt[idx]).squeeze(1), yt[idx]).backward()
            opt.step()
    with torch.no_grad():
        pred = (net(torch.tensor(Xte, dtype=torch.float32, device=dev)).squeeze(1) > 0).cpu().numpy()
    return float((pred == yte).mean())


def _cat(worlds, names, cap, rng, noise=0.0):
    Xs, ys = [], []
    for nm in names:
        X, y = worlds[nm]
        idx = rng.permutation(len(X))[:cap]
        yy = y[idx].copy()
        if noise > 0:
            flip = rng.random(len(idx)) < noise
            yy[flip] = 1 - yy[flip]
        Xs.append(X[idx])
        ys.append(yy)
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
    print(f"[esm] {len(names)} protein-worlds, feat dim {worlds[names[0]][0].shape[1]}", flush=True)

    split = random.Random(80)
    order = names[:]
    split.shuffle(order)
    nte = min(CONFIG["n_test"], len(order) // 3)
    test_names, pool = order[:nte], order[nte:]
    Xte, yte = _cat(worlds, test_names, CAP, np.random.RandomState(81))

    # ESM zero-shot LLR baseline (no training): LLR is feature index -4 (after the embedding)
    d = worlds[names[0]][0].shape[1]
    llr_te = Xte[:, d - 4]
    base = float(((llr_te > np.median(np.concatenate([worlds[n][0][:, d - 4] for n in pool]))).astype(int) == yte).mean())

    res = {"domain": "genomics-esm", "esm": ESM_NAME, "n_worlds": len(names),
           "n_test_worlds": nte, "feat_dim": d, "esm_llr_zeroshot_base": round(base, 4),
           "config": CONFIG, "ladder_raw": {}, "ablation_raw": {}}

    def upload():
        res["ladder"] = {str(n): mean_ci(list(v.values())) for n, v in res["ladder_raw"].items()}
        res["ablation"] = {t: mean_ci(list(v.values())) for t, v in res["ablation_raw"].items()}
        OUT.write_text(json.dumps(res, indent=2))
        if args.bucket:
            import subprocess
            subprocess.run(["gcloud", "storage", "cp", str(OUT),
                            f"{args.bucket}/e80_genomics_esm.json"], check=False)

    print(f"[esm] zero-shot LLR baseline acc {base:.3f}", flush=True)
    for seed in CONFIG["seeds"]:
        rng = np.random.RandomState(100 + seed)
        p = pool[:]
        random.Random(seed).shuffle(p)
        for n in [x for x in CONFIG["ladder"] if x <= len(p)]:
            Xtr, ytr = _cat(worlds, p[:n], CAP, rng)
            acc = train_eval(Xtr, ytr, Xte, yte, seed)
            res["ladder_raw"].setdefault(str(n), {})[str(seed)] = acc
            print(f"[ladder seed{seed} N={n}] {acc:.3f}", flush=True)
            upload()
    for seed in CONFIG["seeds"]:
        rng = np.random.RandomState(300 + seed)
        p = pool[:]
        random.Random(700 + seed).shuffle(p)
        abl = p[:min(CONFIG["abl_n"], len(p))]
        for noise in CONFIG["abl_noise"]:
            Xtr, ytr = _cat(worlds, abl, CAP, rng, noise=noise)
            acc = train_eval(Xtr, ytr, Xte, yte, seed)
            res["ablation_raw"].setdefault(f"n{int(noise * 100):02d}", {})[str(seed)] = acc
            print(f"[abl seed{seed} p={noise}] {acc:.3f}", flush=True)
            upload()
    upload()
    print("[esm] done", json.dumps({"base": base, "ladder": res.get("ladder"),
          "ablation": res.get("ablation")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
