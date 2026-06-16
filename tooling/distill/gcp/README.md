# H100 spot harvest on GCP

Reproducible scripts to run the verified-trace teacher harvest
(`openworld.bench … --log-traces`) on a single ephemeral **H100 spot** VM, then
copy the traces back and delete the box. Harvest-only: training/eval stay on the
Mac (see `../README.md`).

## No secrets, by construction

- Auth is your **local** `gcloud auth login`. Credentials are **never** copied to
  the VM (the VM uses its own service-account `cloud-platform` scope only for its
  own lifecycle; nothing is exfiltrated).
- `config.env` (your project/zone) is **gitignored**. Only `config.env.example`
  is committed. Project IDs and zones aren't secrets, but we keep them out anyway.
- The repo is shipped to the VM as a tarball over `gcloud compute scp` (which
  tunnels over Google's control plane) — no git tokens on the box, and it carries
  the local uncommitted `num_ctx=8192` / `timeout=1800` run patch.

## Use

```bash
cp config.env.example config.env     # set PROJECT and a ZONE with H100 capacity
./run.sh                             # provision -> ship code -> launch harvest
# ... watch ~/harvest.log until "HARVEST COMPLETE" ...
./fetch_teardown.sh                  # pull traces to ./traces/harvest-70b, DELETE vm
```

| script | where | does |
|---|---|---|
| `provision.sh` | local | preflights H100 spot quota, creates the spot VM, waits for the driver |
| `run.sh` | local | provisions, tars+ships the repo, launches `remote_harvest.sh` |
| `remote_harvest.sh` | **VM** | installs Ollama, pulls the teacher, runs the harvest under `nohup` |
| `fetch_teardown.sh` | local | `scp`s traces back, then **deletes** the VM (`--keep` to skip delete) |

## Cost & spot behaviour

- `a3-highgpu-1g` (1× H100 80GB) spot ≈ **$3.5–4.5/hr**. Expect **3–8 hrs** +
  ~30 min setup ⇒ roughly **$15–40**.
- Spot can be **preempted**; the VM self-deletes on preemption. `--log-traces`
  **appends** and flushes per attempt, so re-running `./run.sh` continues the
  harvest on a fresh box with no lost completed trials.
- **Billing stops only when the VM is deleted.** Always finish with
  `fetch_teardown.sh` (or `gcloud compute instances delete`).

## Prerequisites

- `gcloud auth login` done locally, billing-enabled project.
- **H100 spot quota** in your region (metric `Preemptible NVIDIA H100 GPUs`).
  Most projects start at 0 — request an increase first; `provision.sh` preflights
  this and aborts early if it's zero.
