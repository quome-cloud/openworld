"""Run E78 (world-model-as-a-tool, Blocksworld/PlanBench) live on Modal.

The experiment needs a real LLM via Ollama (qwen2.5:7b). We run Ollama *inside* the
container (localhost:11434 — exactly what OllamaLLM defaults to), pull the model once onto
the shared volume so it's reused, copy the read-only mounted repo into a writable dir, and
run the experiment as a subprocess. Results + a cost meter land on the volume; the parsed
JSON is also returned so the caller can drop it into experiments/results/.

Two entrypoints:
  smoke  - tiny scope (E78_HORIZONS=4,8 x 3 each = 6 instances) to METER throughput/cost
           before committing the full run. Excludes the one-time model pull from its timer.
  full   - the real experiment (defaults: 6 horizons x 25 = 150 instances).

Launch (from repo root, venv active, env pinned):
  modal run scripts/modal/e78_world_model_tool.py::smoke
  modal run scripts/modal/e78_world_model_tool.py::full
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal

PROJECT_SLUG = "openworld-e78"
GPU = "A10G"
GPU_RATE_PER_HR = 1.10  # A10G list rate (verify-current; cost-and-budget.md)
OLLAMA_MODEL = "qwen2.5:7b"

vol = modal.Volume.from_name("research")
TENANT = f"/root/data/{PROJECT_SLUG}"
OLLAMA_MODELS = f"{TENANT}/ollama-models"  # persist the pulled model across containers

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "zstd", "ca-certificates")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .env({
        "OLLAMA_MODELS": OLLAMA_MODELS,
        "OLLAMA_KEEP_ALIVE": "30m",   # keep the 7B resident across the call storm
        "PYTHONUNBUFFERED": "1",
    })
    # build steps are all above; the local mount is last. exclude churn/bulk.
    .add_local_dir(
        ".", remote_path="/workspace/repo",
        ignore=[
            ".git", "__pycache__", "*.pyc", ".claude", ".beads", ".runtime",
            "paper", "node_modules", "*.pdf", "*.log", "*.aux", "*.out", "*.pt",
            "experiments/results/e77_artifacts",
        ],
    )
)

app = modal.App(f"{PROJECT_SLUG}-world-model-tool")


def _start_ollama():
    """Boot ollama serve, wait for the API, pull the model (idempotent; cached on volume)."""
    import urllib.request

    os.makedirs(OLLAMA_MODELS, exist_ok=True)
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ},
    )
    for _ in range(120):
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("ollama serve never came up")
    t0 = time.time()
    subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
    return proc, round(time.time() - t0, 1)


def _run_experiment(horizons, n_per_bucket):
    """Run the experiment from a WRITABLE copy of the mounted repo (the mount is read-only,
    so save_results would fail otherwise). Returns wall time + parsed result JSON."""
    work = "/root/work"
    if not os.path.exists(work):
        shutil.copytree("/workspace/repo", work)
    env = {**os.environ, "PYTHONPATH": work}
    if horizons:
        env["E78_HORIZONS"] = horizons
    if n_per_bucket:
        env["E78_N_PER_BUCKET"] = str(n_per_bucket)
    t0 = time.time()
    p = subprocess.run(
        ["python", "experiments/e78_world_model_tool.py", "--live"],
        cwd=work, env=env, capture_output=True, text=True,
    )
    wall = round(time.time() - t0, 1)
    rp = Path(work) / "experiments/results/e78_world_model_tool.json"
    data = json.loads(rp.read_text()) if rp.exists() else None
    return {
        "wall_s": wall,
        "returncode": p.returncode,
        "stdout_tail": p.stdout[-3000:],
        "stderr_tail": p.stderr[-3000:],
        "data": data,
    }


def _meter(wall_s, n_instances):
    cost = wall_s / 3600.0 * GPU_RATE_PER_HR
    return {
        "gpu": GPU, "rate_per_hr": GPU_RATE_PER_HR,
        "experiment_wall_s": wall_s,
        "experiment_cost_usd": round(cost, 4),
        "per_instance_s": round(wall_s / n_instances, 2) if n_instances else None,
    }


def _persist(tag, out, meta):
    os.makedirs(TENANT, exist_ok=True)
    rec = {**meta, **out}
    Path(f"{TENANT}/{tag}_result.json").write_text(json.dumps(rec, indent=2))
    vol.commit()


@app.function(image=image, gpu=GPU, timeout=1800, max_containers=2,
              volumes={"/root/data": vol})
def smoke():
    _, pull_s = _start_ollama()
    vol.commit()  # persist the pulled model for the full run
    out = _run_experiment("4,8", 3)
    n = (out["data"] or {}).get("n_instances", 6)
    meter = _meter(out["wall_s"], n)
    # extrapolate to the full 150-instance run (6 horizons x 25)
    full_n = 150
    proj_wall = out["wall_s"] / max(n, 1) * full_n
    projection = {
        "full_n": full_n,
        "projected_full_wall_s": round(proj_wall, 0),
        "projected_full_wall_min": round(proj_wall / 60, 1),
        "projected_full_cost_usd": round(proj_wall / 3600 * GPU_RATE_PER_HR, 2),
        "note": "linear extrapolation; long horizons cost more per instance, so full may run higher",
    }
    _persist("smoke", out, {"tag": "smoke", "model_pull_s": pull_s,
                            "meter": meter, "projection": projection})
    return {
        "returncode": out["returncode"], "model_pull_s": pull_s,
        "meter": meter, "projection": projection,
        "summary": (out["data"] or {}).get("summary"),
        "stdout_tail": out["stdout_tail"], "stderr_tail": out["stderr_tail"],
    }


@app.function(image=image, gpu=GPU, timeout=1800, max_containers=2,
              volumes={"/root/data": vol})
def probe():
    """Diagnose why live arms score 0: print the RAW model output, the parsed plan, and the
    verified per-round feedback for a couple of instances. Tells parsing-bug from genuine fail."""
    _start_ollama()
    work = "/root/work"
    if not os.path.exists(work):
        shutil.copytree("/workspace/repo", work)
    import importlib.util
    import sys
    sys.path.insert(0, work)
    sys.path.insert(0, f"{work}/experiments")
    import random as _random

    import blocksworld as bw
    # the experiment shares this entry file's module name; load it explicitly by path.
    _spec = importlib.util.spec_from_file_location(
        "e78_exp", f"{work}/experiments/e78_world_model_tool.py")
    e = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(e)

    propose, _sim = e.live_propose_and_sim(OLLAMA_MODEL)
    rng = _random.Random(78)
    insts = []
    for L in (2, 4):
        while len([i for i in insts if i["optimal_len"] == L]) < 2:
            pr = bw.gen_problem(4, L, rng)
            if pr:
                insts.append(pr)

    log = []
    for prob in insts:
        init, goal = prob["init"], prob["goal"]
        rec = {"optimal_len": prob["optimal_len"], "rounds": []}
        rec["optimal_plan"] = [f"{n}({p})" for n, p in bw.bfs_plan(init, goal)]
        history = []
        for r in range(e.MAX_ROUNDS):
            raw = propose(e.build_prompt(init, goal, history))
            plan = e.parse_plan(raw)
            rollout = bw.validate_plan(init, goal, plan)
            fb = e.feedback(rollout)
            rec["rounds"].append({
                "raw_model_output": raw[:600],
                "parsed_plan": [f"{n}({p})" for n, p in plan],
                "n_parsed": len(plan), "feedback": fb,
            })
            if rollout["first_illegal"] is None and rollout["reached"]:
                break
            history.append((plan, fb))
        log.append(rec)

    os.makedirs(TENANT, exist_ok=True)
    Path(f"{TENANT}/probe.json").write_text(json.dumps(log, indent=2))
    vol.commit()
    return log


@app.function(image=image, gpu=GPU, timeout=3600, max_containers=2,
              volumes={"/root/data": vol})
def batch(horizons: str = "2,4", n_per: int = 8):
    """Parametric subset run: does the verified tool give qwen2.5:7b ANY signal at chosen
    horizons? Decides whether the full sweep is worth it."""
    _, pull_s = _start_ollama()
    out = _run_experiment(horizons, n_per)
    n_inst = (out["data"] or {}).get("n_instances") or (len(horizons.split(",")) * n_per)
    meter = _meter(out["wall_s"], n_inst)
    tag = f"batch_{horizons.replace(',', '_')}_{n_per}"
    _persist(tag, out, {"tag": tag, "model_pull_s": pull_s, "meter": meter})
    return {
        "returncode": out["returncode"], "meter": meter,
        "summary": (out["data"] or {}).get("summary"),
        "by_horizon": (out["data"] or {}).get("by_horizon"),
        "mean_rounds": (out["data"] or {}).get("mean_rounds"),
    }


@app.function(image=image, gpu=GPU, timeout=14400, max_containers=2,
              volumes={"/root/data": vol})
def full():
    _, pull_s = _start_ollama()
    out = _run_experiment(None, None)  # defaults: 6 horizons x 25 = 150
    n = (out["data"] or {}).get("n_instances", 150)
    meter = _meter(out["wall_s"], n)
    _persist("full", out, {"tag": "full", "model_pull_s": pull_s, "meter": meter})
    return {
        "returncode": out["returncode"], "model_pull_s": pull_s, "meter": meter,
        "summary": (out["data"] or {}).get("summary"),
        "mcnemar": (out["data"] or {}).get("mcnemar"),
        "by_horizon": (out["data"] or {}).get("by_horizon"),
        "stdout_tail": out["stdout_tail"], "stderr_tail": out["stderr_tail"],
    }
