"""E80 Bongard-RWR: world-time compute in the VISION modality on REAL images.

Each Bongard problem is a WORLD: a latent visual concept that separates 'left' images from
'right' images. We test per-world world-time compute the same way as text/grids -- but with a
frozen vision encoder + a small head (the ESM-genomics pattern), not an LLM. For each held-out
problem: DINOv2-encode its images (frozen), train a tiny head on the 12 support images
(left=0/right=1), predict the 2 held-out query images by exact side. Arms:
  - prototype  (no training: nearest class-mean) -- the no-world-time-compute baseline,
  - light TTT  (head trained on the 12 support features),
  - heavy TTT  (head trained on support + augmented views = MORE world-time compute),
  - corrupt    (heavy, but support labels shuffled) -- the verified-label discriminator.

Exact label (correct side); chance is 50%. Partial results upload after every problem.
  python3 e80_bongard.py --root /root/bongard-rwr-plus --bucket gs://openworld-bench/bongard
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

HERE = Path(__file__).resolve().parent
ENC = "facebook/dinov2-base"
LEVELS = [("light", 0, 120), ("heavy", 4, 300)]   # (name, n_aug_views, head_steps)
ABL = ("corrupt", 4, 300)


def load_problems(root):
    return json.load(open(Path(root) / "dataset.json"))["problems"]


def _aug_views(img, k, rng):
    """k extra rule-preserving views (flip + random resized crop) for world-time compute."""
    views = []
    w, h = img.size
    for _ in range(k):
        v = img
        if rng.random() < 0.5:
            v = v.transpose(Image.FLIP_LEFT_RIGHT)
        s = rng.uniform(0.7, 1.0)
        cw, ch = int(w * s), int(h * s)
        x0 = rng.randint(0, w - cw)
        y0 = rng.randint(0, h - ch)
        views.append(v.crop((x0, y0, x0 + cw, y0 + ch)))
    return views


@torch.no_grad()
def encode(paths, root, proc, model, device, aug=0, rng=None):
    """-> dict path -> list of feature vectors (1 + aug views each), L2-normalised."""
    feats = {}
    for p in paths:
        img = Image.open(Path(root) / p).convert("RGB")
        imgs = [img] + (_aug_views(img, aug, rng) if aug else [])
        px = proc(images=imgs, return_tensors="pt").to(device)
        out = model(**px).last_hidden_state[:, 0]          # CLS token
        out = torch.nn.functional.normalize(out, dim=-1)
        feats[p] = out.cpu().numpy()
    return feats


def _head_fit(X, y, steps, device, lr=0.05, wd=1e-3):
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    head = torch.nn.Linear(X.shape[1], 2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(head(Xt), yt).backward()
        opt.step()
    return head


def _predict_head(head, X, device):
    with torch.no_grad():
        return head(torch.tensor(X, dtype=torch.float32, device=device)).argmax(-1).cpu().numpy()


def _prototype(Xs, ys, Xq):
    mu = np.stack([Xs[ys == c].mean(0) for c in (0, 1)])
    return np.array([int(np.dot(mu[1], x) > np.dot(mu[0], x)) for x in Xq])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--bucket", default="")
    ap.add_argument("--n", type=int, default=300, help="held-out problems")
    args = ap.parse_args()
    dev = "cuda"

    probs = load_problems(args.root)
    rng0 = random.Random(80)
    rng0.shuffle(probs)
    # need >=6 per side + a query per side
    probs = [p for p in probs if len(p["left_images"]) >= 7 and len(p["right_images"]) >= 7][:args.n]
    print(f"[bongard] {len(probs)} held-out vision worlds", flush=True)

    proc = AutoImageProcessor.from_pretrained(ENC)
    model = AutoModel.from_pretrained(ENC, torch_dtype=torch.float32).to(dev).eval()

    res = {"experiment": "bongard-rwr", "encoder": ENC, "n_worlds": len(probs),
           "levels": [dict(name=n, aug=a, steps=s) for n, a, s in LEVELS + [ABL]],
           "arms": {}, "per_world": {}}
    for arm in ["prototype"] + [l[0] for l in LEVELS] + [ABL[0]]:
        res["per_world"][arm] = {}

    def upload():
        for arm, accs in res["per_world"].items():
            done = [a for a in accs.values() if a is not None]
            res["arms"][arm] = {"acc": round(sum(done) / len(done), 4) if done else None,
                                "n_done": len(done)}
        out = HERE / "results" / "e80_bongard.json"
        out.write_text(json.dumps(res, indent=2))
        if args.bucket:
            subprocess.run(["gcloud", "storage", "cp", str(out),
                            f"{args.bucket}/e80_bongard.json"], check=False)

    for pi, p in enumerate(probs):
        try:
            L, R = p["left_images"][:6], p["right_images"][:6]
            qL, qR = p["left_images"][6], p["right_images"][6]
            rng = random.Random(hash(p["id"]) % 2**32)
            # query features (1 view), support features per arm (aug views vary)
            qfeat = encode([qL, qR], args.root, proc, model, dev)
            Xq = np.stack([qfeat[qL][0], qfeat[qR][0]])
            yq = np.array([0, 1])

            # base (1-view) support features for prototype + light
            base = encode(L + R, args.root, proc, model, dev)
            Xs = np.stack([base[p0][0] for p0 in L + R])
            ys = np.array([0] * 6 + [1] * 6)

            res["per_world"]["prototype"][str(p["id"])] = float((_prototype(Xs, ys, Xq) == yq).mean())

            for name, aug, steps in LEVELS + [ABL]:
                if aug:
                    af = encode(L + R, args.root, proc, model, dev, aug=aug, rng=rng)
                    X = np.concatenate([af[p0] for p0 in L + R])
                    y = np.repeat([0] * 6 + [1] * 6, 1 + aug)
                else:
                    X, y = Xs.copy(), ys.copy()
                if name == ABL[0]:
                    y = y.copy()
                    rng.shuffle(y)
                head = _head_fit(X, y, steps, dev)
                acc = float((_predict_head(head, Xq, dev) == yq).mean())
                res["per_world"][name][str(p["id"])] = acc
        except Exception as e:  # noqa: BLE001
            print(f"[problem {p.get('id')}] FAILED {e}", flush=True)
            for arm in res["per_world"]:
                res["per_world"][arm].setdefault(str(p.get("id")), None)
        if pi % 10 == 0:
            upload()
            print(f"[{pi}/{len(probs)}] {({k: v['acc'] for k, v in res['arms'].items()})}", flush=True)
    upload()
    print("[bongard] done\n" + json.dumps(res["arms"], indent=2), flush=True)


if __name__ == "__main__":
    main()
