"""Backfill token/cost metadata in the arc3_traces dataset by RE-PARSING the on-disk transcripts with the
current summarize_transcript (which now reconstructs token usage from per-message usage when a run was cut
mid-stream and never emitted a result block). Updates ONLY the `transcript` block in each meta sidecar and
in runs.jsonl -- the verified `outcome` is left untouched (no env/roundtrip recompute). Pure parsing; plain
python3, safe to run while the sweep is going.

    python3 scripts/backfill_tokens.py
"""
import json, glob, sys
from pathlib import Path

ROOT = Path("/Users/jim/Desktop/openworld")
sys.path.insert(0, str(ROOT / "scripts"))
import capture_lib as c

TRANSCRIPT_FIELDS = ("session_id", "num_turns", "n_messages", "n_tool_calls", "tool_calls_by_name",
                     "n_text_blocks", "n_thinking_blocks", "n_user_msgs", "cost_usd", "cost_usd_estimated",
                     "cost_basis", "pricing_assumed", "tokens", "usage", "is_error", "api_error_status",
                     "duration_ms", "duration_api_ms", "ttft_ms")


def rebuilt_transcript_block(rec):
    tf = rec.get("transcript_file")
    if not tf:
        return None
    summ = c.summarize_transcript(c.TRACES / tf)
    return {k: summ.get(k) for k in TRANSCRIPT_FIELDS}


def main():
    # 1) update meta sidecars
    n_meta = 0
    for mp in glob.glob(str(c.META / "*.json")):
        try:
            rec = json.loads(open(mp).read())
        except Exception:
            continue
        blk = rebuilt_transcript_block(rec)
        if blk is None:
            continue
        rec["transcript"] = blk
        Path(mp).write_text(json.dumps(rec, indent=1))
        n_meta += 1

    # 2) update runs.jsonl in place (transcript block only; keep outcome + everything else)
    n_runs = 0; src_counts = {}
    if c.RUNS.exists():
        recs = [json.loads(l) for l in open(c.RUNS, errors="ignore") if l.strip()]
        for r in recs:
            blk = rebuilt_transcript_block(r)
            if blk is not None:
                r["transcript"] = blk
                src = (blk.get("tokens") or {}).get("source")
                src_counts[src] = src_counts.get(src, 0) + 1
                n_runs += 1
        recs.sort(key=lambda r: (r.get("started_at") or "", r.get("run_id")))
        with open(c.RUNS, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    print(f"[backfill] updated {n_meta} meta sidecars, {n_runs} runs.jsonl records")
    print(f"[backfill] token source distribution: {src_counts}")
    # summarize cost coverage now
    recs = [json.loads(l) for l in open(c.RUNS) if l.strip()] if c.RUNS.exists() else []
    agent = [r for r in recs if r["tier"] == "agent"]
    with_tokens = sum(1 for r in agent if (r.get("transcript", {}).get("tokens") or {}).get("total"))
    auth_cost = sum(1 for r in agent if r.get("transcript", {}).get("cost_usd") is not None)
    est_cost = sum(1 for r in agent if r.get("transcript", {}).get("cost_usd_estimated") is not None)
    tot_auth = sum(r["transcript"]["cost_usd"] for r in agent
                   if r.get("transcript", {}).get("cost_usd"))
    tot_est = sum(r["transcript"].get("cost_usd_estimated") or 0 for r in agent
                  if r.get("transcript", {}).get("cost_usd") is None)
    print(f"[backfill] agent runs: {len(agent)} | with token totals: {with_tokens} "
          f"| authoritative cost: {auth_cost} | estimated cost: {est_cost}")
    print(f"[backfill] cost: ${tot_auth:.2f} authoritative + ${tot_est:.2f} estimated "
          f"= ${tot_auth + tot_est:.2f} total")


if __name__ == "__main__":
    main()
