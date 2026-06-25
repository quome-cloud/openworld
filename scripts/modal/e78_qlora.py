"""Run E78 QLoRA (verified-planner distillation on Blocksworld) on Modal.

One GPU container: 4-bit QLoRA-SFT qwen2.5 on BFS-oracle labels (e78_finetune.py), then eval
base-vs-adapter on held-out instances scored by the verified validator (e78_eval.py). The HF
model and the LoRA adapter are staged on the shared volume so re-runs skip the download/retrain.

Data (experiments/results/e78_artifacts/{sft_train,test}.jsonl) is generated locally by
e78_data.py and mounted; the scripts only READ the mounted repo and write outputs to the
volume, so no writable copy is needed.

  modal run scripts/modal/e78_qlora.py::run --epochs 1 --eval-limit 24   # smoke
  modal run scripts/modal/e78_qlora.py::run                               # full (3 epochs, all test)
"""

import json
import os
import subprocess
import time
from pathlib import Path

import modal

PROJECT_SLUG = "openworld-e78"
GPU = "A10G"
RATE_PER_HR = 1.10
BASE_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"

vol = modal.Volume.from_name("research")
TENANT = f"/root/data/{PROJECT_SLUG}"
HF_CACHE = f"{TENANT}/hf-cache"
ADAPTER_DIR = f"{TENANT}/e78_adapter"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1", "transformers==4.45.2", "peft==0.13.2", "accelerate==1.0.1",
        "bitsandbytes==0.44.1", "datasets==3.0.1", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": HF_CACHE, "HF_HUB_CACHE": HF_CACHE, "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(
        ".", remote_path="/workspace/repo",
        ignore=[".git", "__pycache__", "*.pyc", ".claude", ".beads", ".runtime", "paper",
                "node_modules", "*.pdf", "*.log", "*.aux", "*.out", "*.pt",
                "experiments/results/e77_artifacts"],
    )
)

app = modal.App(f"{PROJECT_SLUG}-qlora")

REPO = "/workspace/repo"
EXP = f"{REPO}/experiments"
DATA = f"{REPO}/experiments/results/e78_artifacts"


def _run(cmd):
    """Run a script from the experiments dir with the repo importable; capture wall + output."""
    env = {**os.environ, "PYTHONPATH": REPO}
    t0 = time.time()
    p = subprocess.run(cmd, cwd=EXP, env=env, capture_output=True, text=True)
    return {"wall_s": round(time.time() - t0, 1), "rc": p.returncode,
            "out": p.stdout[-4000:], "err": p.stderr[-4000:]}


@app.function(image=image, gpu=GPU, timeout=7200, max_containers=2,
              volumes={"/root/data": vol},
              secrets=[modal.Secret.from_name("hf_token")])
def run(base: str = BASE_DEFAULT, epochs: float = 3.0, batch: int = 8,
        eval_limit: int = 0, do_train: bool = True):
    os.makedirs(TENANT, exist_ok=True)
    res = {"base": base, "epochs": epochs, "batch": batch, "gpu": GPU}

    if do_train:
        ft = _run(["python", "e78_finetune.py", "--base", base,
                   "--data", f"{DATA}/sft_train.jsonl", "--out", ADAPTER_DIR,
                   "--load_4bit", "--epochs", str(epochs), "--batch", str(batch)])
        vol.commit()  # persist the adapter
        res["train"] = ft
        if ft["rc"] != 0:
            Path(f"{TENANT}/qlora_result.json").write_text(json.dumps(res, indent=2))
            vol.commit()
            return res

    ev_cmd = ["python", "e78_eval.py", "--base", base, "--adapter", ADAPTER_DIR,
              "--tasks", f"{DATA}/test.jsonl", "--load_4bit",
              "--out", f"{TENANT}/e78_eval.json"]
    if eval_limit:
        ev_cmd += ["--limit", str(eval_limit)]
    ev = _run(ev_cmd)
    res["eval"] = ev

    ep = Path(f"{TENANT}/e78_eval.json")
    if ep.exists() and ev["rc"] == 0:
        full = json.loads(ep.read_text())
        res["summary"] = full.get("summary")
        res["mcnemar"] = full.get("mcnemar_ft_vs_base")
        res["n_eval"] = full.get("n")

    wall = (res.get("train", {}).get("wall_s", 0.0)) + ev["wall_s"]
    res["cost_usd_compute"] = round(wall / 3600 * RATE_PER_HR, 3)
    res["note_cost"] = "subprocess wall only; add ~one-time model download + cold start"
    Path(f"{TENANT}/qlora_result.json").write_text(json.dumps(res, indent=2))
    vol.commit()
    return res
