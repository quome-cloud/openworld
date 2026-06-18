"""E68 - Prototyping latency at scale: 100 code world models, built + timed by Claude Code.

Hand-authoring a simulator/world model is typically a multi-day effort. This benchmarks
the framework's intended path -- `openworld build "<description>"` drives Claude Code to
author a spec -- across 100 worlds spanning six sectors (healthcare, financial, legal,
cybersecurity, energy/climate, agentic-AI). For each we measure wall-clock from the
command to a spec that passes `validate_spec` (i.e. servable), and save the spec as a
reusable recipe under recipes/<sector>/<name>.json.

Agent-style experiment: wall-clock varies run-to-run (LLM latency); reproduces in
distribution (minutes-scale), not bit-for-bit. save_results runs BEFORE the asserts.
Run with `claude` on PATH; builds run concurrently (default 8).
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median

from common import save_results
from openworld.spec import validate_spec

PY = "/Users/jim/.pyenv/versions/3.9.18/bin/python"
ROOT = Path(__file__).resolve().parent.parent
RECIPES = ROOT / "recipes"
CONCURRENCY = int(os.environ.get("E68_CONCURRENCY", "8"))

WORLDS = {
 "healthcare": {
  "ed_triage": "an ED triage world: patients arrive with an acuity level and wait; actions triage, assign_bed, discharge; sicker patients are seen first",
  "icu_beds": "an ICU bed-allocation world across wards with bed counts; actions admit, transfer, discharge as beds free up",
  "sepsis_warning": "a sepsis early-warning world: vitals drift and a risk score rises; actions monitor, escalate, treat to lower risk",
  "med_reconciliation": "a medication reconciliation world: a patient's med list with interaction risk; actions add_med, remove_med, flag_interaction",
  "or_scheduling": "an operating-room scheduling world: cases with durations and rooms; actions book, delay, cancel without double-booking",
  "readmission_risk": "a discharge-readiness world with readmission risk; actions discharge, schedule_followup, readmit",
  "claims_adjudication": "a health-insurance claims adjudication world with claim states; actions submit, review, approve, deny",
  "vaccine_coldchain": "a vaccine cold-chain world: temperature and stock; actions ship, store, discard on temperature excursion",
  "ambulance_dispatch": "an ambulance dispatch world: incidents by priority and units; actions dispatch, arrive, return",
  "ward_staffing": "a nurse staffing world maintaining nurse-to-patient ratios across shifts; actions assign, call_float, overtime",
  "organ_matching": "an organ transplant waitlist world matching by compatibility and urgency; actions list, match, transplant",
  "glucose_mgmt": "a diabetes glucose-management world: blood glucose responds to dose, meals, exercise; actions dose_insulin, eat, exercise",
  "pharmacy_inventory": "a pharmacy inventory world with stock and expiry; actions dispense, reorder, expire",
  "infection_control": "a hospital infection-spread world (SIR-like) with isolation capacity; actions isolate, treat, screen",
  "appointment_sched": "a clinic appointment scheduling world with no-shows and overbooking; actions book, cancel, overbook",
  "dialysis_sched": "a dialysis chair scheduling world across sessions; actions schedule, swap, miss",
  "telehealth_queue": "a telehealth visit queue routed by specialty; actions enqueue, route, complete",
 },
 "financial": {
  "fraud_scoring": "a card-transaction fraud world: transactions get a risk score and holds; actions score, hold, release, block",
  "loan_underwriting": "a loan underwriting world with application states and risk; actions submit, verify, approve, decline",
  "aml_monitoring": "an AML transaction-monitoring world raising alerts on patterns; actions flag, investigate, file_sar, clear",
  "portfolio_rebalance": "a portfolio rebalancing world drifting from target weights; actions buy, sell, rebalance",
  "credit_limit": "a credit-limit management world with utilization and risk; actions increase, decrease, freeze",
  "trade_settlement": "a T+1 trade-settlement lifecycle world; actions match, affirm, settle, fail",
  "margin_call": "a margin-account world with equity vs maintenance margin; actions buy, sell, margin_call, liquidate",
  "kyc_onboarding": "a KYC onboarding world with verification tiers; actions submit_docs, verify, approve, reject",
  "payment_routing": "a payment routing world across rails by cost and speed; actions route, retry, reverse",
  "collections": "a debt-collections world with aging buckets; actions remind, negotiate, charge_off",
  "insurance_pricing": "an insurance premium-pricing world by risk pool; actions quote, bind, renew, lapse",
  "atm_cash": "an ATM cash-management world with balance and replenishment; actions withdraw, deposit, replenish",
  "fx_hedging": "an FX exposure-hedging world with open positions; actions open, close, hedge",
  "chargeback": "a card dispute/chargeback lifecycle world; actions open, represent, resolve",
  "liquidity_mgmt": "a bank liquidity/reserve management world; actions lend, borrow, hold to meet reserves",
  "ipo_allocation": "an IPO share-allocation world across bids with oversubscription; actions bid, allocate, scale_back",
  "budget_envelope": "a personal budgeting world with spending envelopes and rollovers; actions spend, transfer, rollover",
 },
 "legal": {
  "contract_review": "a contract-review workflow world with clause flags; actions draft, redline, approve, sign",
  "case_docketing": "a court case-docketing world with deadlines; actions file, schedule, continue, rule",
  "ediscovery": "an e-discovery review world labeling docs responsive/privileged; actions ingest, review, produce, withhold",
  "sol_tracking": "a statute-of-limitations tracking world per claim; actions accrue, toll, file, bar",
  "compliance_gating": "a regulatory compliance-gating world that gates actions; actions request, check, approve, block",
  "conflict_check": "a conflict-of-interest checking world for new matters; actions intake, screen, clear, decline",
  "ip_prosecution": "a patent prosecution lifecycle world; actions file, office_action, respond, grant",
  "billing_trust": "a legal billing and trust-accounting (IOLTA) world; actions bill, deposit, disburse, reconcile",
  "litigation_hold": "a litigation-hold preservation world; actions issue_hold, acknowledge, release",
  "nda_lifecycle": "an NDA lifecycle world with expiry and renewal; actions execute, amend, expire",
  "court_calendar": "a court calendar and judge-assignment world; actions assign, reschedule, hear",
  "immigration_case": "an immigration application status world; actions file, rfe, respond, decide",
  "discovery_deadlines": "a discovery request/response deadline world; actions serve, object, respond, compel",
  "settlement_negotiation": "a settlement negotiation world with offers; actions offer, counter, accept, reject",
  "regulatory_filing": "a regulatory filing-calendar world (e.g. SEC); actions prepare, file, amend, mark_late",
  "licensing_cle": "a professional licensing and CLE-credit tracking world; actions accrue_credits, renew, lapse",
  "foia_request": "a FOIA request-processing world with exemptions; actions submit, review, release, redact",
 },
 "cybersecurity": {
  "incident_response": "a security incident-response lifecycle world; actions detect, triage, contain, eradicate, recover",
  "access_control": "an RBAC access-request state machine world; actions request, approve, grant, revoke",
  "rate_limiter": "an API rate-limiter world (token bucket) with refill; actions request, refill, throttle",
  "patch_management": "a vulnerability patch-management world by severity; actions scan, patch, defer, verify",
  "intrusion_detection": "an intrusion-detection alert escalation world; actions alert, investigate, block, dismiss",
  "cert_rotation": "a TLS certificate rotation/expiry world; actions issue, rotate, expire, revoke",
  "ddos_mitigation": "a DDoS mitigation world with traffic scrubbing; actions detect, scrub, blackhole",
  "backup_restore": "a backup/restore lifecycle world with an RPO target; actions backup, verify, restore",
  "firewall_change": "a firewall rule-change management world; actions request, review, apply, rollback",
  "phishing_triage": "a phishing-report triage world; actions report, analyze, quarantine, educate",
  "secret_rotation": "a secrets/credential rotation world with leak risk; actions create, rotate, revoke",
  "autoscale": "a compute autoscaling world reacting to load; actions scale_up, scale_down, hold",
  "siem_correlation": "a SIEM event-correlation world raising incidents; actions ingest, correlate, escalate",
  "mfa_enrollment": "an MFA enrollment and recovery world; actions enroll, challenge, recover, lock",
  "change_mgmt": "an IT change-management (CAB) approval world; actions request, approve, deploy, rollback",
  "vuln_disclosure": "a coordinated vulnerability-disclosure timeline world; actions report, triage, fix, disclose",
  "honeypot": "a honeypot attacker-interaction world; actions probe, engage, alert",
 },
 "energy": {
  "grid_balancing": "a power-grid balancing world with supply, demand, and frequency; actions dispatch, curtail, shed_load",
  "ev_charging": "an EV charging-station world with a queue and load limit; actions plug, charge, throttle, unplug",
  "carbon_market": "a carbon cap-and-trade market world; actions emit, buy_credit, sell_credit, retire",
  "battery_storage": "a grid-battery arbitrage world with state of charge; actions charge, discharge, hold",
  "water_allocation": "a reservoir water-allocation world across users; actions release, ration, refill",
  "demand_response": "a demand-response event world; actions enroll, curtail, settle",
  "solar_farm": "a solar-farm output world with weather and curtailment; actions generate, curtail, store",
  "wind_dispatch": "a wind-generation dispatch world with forecasts; actions forecast, dispatch, curtail",
  "district_heating": "a district-heating network temperature-control world; actions heat, circulate, throttle",
  "microgrid": "an islanded-microgrid management world; actions island, reconnect, balance",
  "emissions_compliance": "a plant-emissions compliance world vs a cap; actions run, scrub, throttle",
  "fuel_inventory": "a power-plant fuel-inventory world with reorder; actions burn, reorder, ration",
  "heat_pump": "a building heat-pump and thermal-envelope world; actions heat, cool, setback",
  "hydrogen_storage": "a hydrogen production/storage cycle world; actions electrolyze, store, fuel",
  "peak_pricing": "a time-of-use peak-pricing demand-shifting world; actions consume, shift, store",
  "recycling_mrf": "a recycling material-recovery-facility flow world; actions sort, bale, reject_contam",
 },
 "agentic": {
  "tool_use_sandbox": "an agent world choosing tools to satisfy a request; actions call_tool, observe, finish",
  "react_loop": "a ReAct reasoning-loop world (thought, act, observe); actions think, act, observe, answer",
  "agent_negotiation": "a two-agent negotiation world over a deal; actions offer, counter, accept, walk_away",
  "task_decomposition": "an agent goal-decomposition world; actions plan, execute, replan, complete",
  "memory_retrieval": "an agent with episodic memory world; actions store, retrieve, act",
  "budget_agent": "an agent under a token/cost budget world; actions call, cache, stop",
  "self_correction": "a self-correcting agent world that verifies and fixes; actions attempt, verify, fix, accept",
  "tool_router": "a router world delegating to specialized agents; actions route, delegate, aggregate",
  "rate_limited_agent": "an agent respecting API rate limits world; actions request, backoff, retry",
  "rag_pipeline": "a retrieval-augmented generation pipeline world; actions query, retrieve, generate, cite",
  "agent_handoff": "an agent-to-human escalation world; actions handle, escalate, resolve",
  "permission_gate": "an agent permission-gated edit world; actions propose, approve, apply, deny",
  "approval_workflow": "an agent executing an approval-workflow world; actions submit, review, approve, execute",
  "explore_exploit": "an agent bandit explore/exploit world; actions explore, exploit, update",
  "actor_critic": "an actor-critic agent loop world with critique; actions act, critique, revise",
  "dialog_state": "a dialog agent managing conversation slots world; actions ask, fill_slot, confirm, complete",
 },
}


def build_one(sector, name, description):
    d = tempfile.mkdtemp(prefix=f"e68_{sector}_{name}_")
    t0 = time.time()
    status = "ok"
    try:
        subprocess.run([PY, "-m", "openworld.cli", "build", description, "--name", name],
                       cwd=d, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"sector": sector, "world": name, "status": "timeout", "seconds": 600, "validated": False}
    secs = round(time.time() - t0, 1)
    spec_path = Path(d) / "specs" / f"{name}.json"
    if not spec_path.exists():
        shutil.rmtree(d, ignore_errors=True)
        return {"sector": sector, "world": name, "status": "no_spec", "seconds": secs, "validated": False}
    spec = json.loads(spec_path.read_text())
    problems = validate_spec(spec)
    if not problems:                                    # save the recipe
        out = RECIPES / sector
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.json").write_text(json.dumps(spec, indent=2, default=str))
    else:
        status = "invalid"
    shutil.rmtree(d, ignore_errors=True)
    return {"sector": sector, "world": name, "status": status, "seconds": secs,
            "minutes": round(secs / 60, 2), "validated": not problems,
            "n_state": len(spec.get("state_schema", {})), "n_actions": len(spec.get("actions", [])),
            "has_perception": bool(spec.get("perception")), "has_emit": bool(spec.get("emit")),
            "has_objectives": bool(spec.get("objectives")), "has_composite": bool(spec.get("composite"))}


def pctl(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(p / 100 * len(s)))], 2)


def main():
    tasks = [(sec, n, d) for sec, ws in WORLDS.items() for n, d in ws.items()]
    print(f"building {len(tasks)} worlds, concurrency {CONCURRENCY} ...", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(build_one, *t): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result(); rows.append(r)
            print(f"  [{i}/{len(tasks)}] {r['sector']}/{r['world']}: "
                  f"{r.get('minutes','-')}min {r['status']}", flush=True)

    ok = [r for r in rows if r.get("validated")]
    mins = [r["minutes"] for r in ok]
    by_sector = {}
    for sec in WORLDS:
        sm = [r["minutes"] for r in ok if r["sector"] == sec]
        by_sector[sec] = {"n": sum(1 for r in rows if r["sector"] == sec),
                          "validated": len(sm),
                          "median_minutes": round(median(sm), 2) if sm else None}
    results = {
        "task": "natural-language description -> validated, servable code world model (openworld build)",
        "tool": "Claude Code via `openworld build` (headless claude -p, acceptEdits)",
        "n_worlds": len(rows), "n_validated": len(ok), "sectors": list(WORLDS),
        "median_minutes": round(median(mins), 2) if mins else None,
        "mean_minutes": round(mean(mins), 2) if mins else None,
        "p90_minutes": pctl(mins, 90), "max_minutes": round(max(mins), 2) if mins else None,
        "validation_rate": round(len(ok) / len(rows), 3) if rows else None,
        "by_sector": by_sector, "worlds": sorted(rows, key=lambda r: (r["sector"], r["world"])),
        "baseline_note": "hand-authoring a comparable simulator/world model is typically a "
                         "multi-day-to-multi-week effort; here each is minutes, end to end, "
                         "verified (validate_spec) and servable, saved as a reusable recipe.",
        "reproducibility": "wall-clock varies run-to-run (LLM latency); reproduces in "
                           "distribution (minutes-scale), not bit-for-bit.",
    }
    save_results("e68_prototyping_latency", results)    # BEFORE asserts

    assert len(ok) >= 0.8 * len(rows), f"only {len(ok)}/{len(rows)} validated"
    assert results["median_minutes"] is not None and results["median_minutes"] < 5, results["median_minutes"]
    print(f"\n[ok] {len(ok)}/{len(rows)} worlds built+validated; median {results['median_minutes']}min "
          f"mean {results['mean_minutes']} p90 {results['p90_minutes']} max {results['max_minutes']}")
    for sec, s in by_sector.items():
        print(f"  {sec:<14} {s['validated']}/{s['n']} validated, median {s['median_minutes']}min")


if __name__ == "__main__":
    main()
