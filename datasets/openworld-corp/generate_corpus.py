"""Generate the openworld-corp transcript dataset with an LLM.

Produces realistic, state-grounded internal text for the organization in
`org.py`, across multiple quarters with evolving metrics and scripted events:

  company-wide : all-hands, exec staff meeting, promotion committee, quarterly
                 planning, board update
  per division : division review, sprint retro, standup, a 1:1 for EVERY IC,
                 and (when an event calls for it) an incident postmortem
  Slack        : #general, #incidents, and a per-division channel

Every record carries the `ground_truth` metrics for its quarter/scope, so the
prose is state-recoverable (a `TextPerceptor` reads division health back out).

Pluggable backend (the "use APIs" path): `generate()` takes ANY `BaseLLM`.
  - Local default: OllamaLLM(model="qwen2.5:7b")
  - Hosted API: implement `class ApiLLM(BaseLLM): def chat(self, messages, **o): ...`
    and pass it in. No keys are stored in the repo.

To use on a REAL company: edit `org.py` (chart + metrics + events), or drop your
own transcripts into `corpus.json` following the schema in README.md.

Usage:  python datasets/openworld-corp/generate_corpus.py [--model qwen2.5:7b]
                 [--periods 3] [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root
from openworld import OllamaLLM                                  # noqa: E402

import org                                                       # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus.json"

SYSTEM = (
    "You write realistic, natural-sounding internal communications for a "
    "cloud-infrastructure company (meetings and chat). Spoken style with named "
    "speakers as 'Name: ...' lines; concrete and grounded in the numbers and "
    "events you are given; people cite their own metrics the way real operators "
    "do. No markdown headers, no narration. 130-240 words unless told otherwise."
)


def brief(name, s):
    d = s[name]
    ev = f" Event: {d['events']}" if d["events"] else ""
    return (f"{name} ({d['note']}): revenue ${d['revenue']}M, growth {d['growth']:.0%} "
            f"YoY, headcount {d['headcount']}, {d['open_roles']} open roles, "
            f"director {d['director']['name']}.{ev}")


# ---- company-wide prompts ---------------------------------------------------
def p_all_hands(period, s):
    body = "\n".join("  " + brief(n, s) for n in s)
    return (f"{org.COMPANY} company all-hands, {period}. CEO {org.CEO['name']} "
            f"reviews the quarter. Total revenue ${org.snapshot_total_cache}M.\n"
            f"Per-division data to walk through:\n{body}\n"
            "Call out the fastest and slowest growers, where headcount is going, "
            "and one strategic priority. A couple of directors chime in.")


def p_exec_staff(period, s):
    body = "\n".join("  " + brief(n, s) for n in s)
    return (f"{org.COMPANY} executive staff meeting, {period}: CEO {org.CEO['name']} "
            f"with the five division directors. Candid working session on the "
            f"portfolio: where to invest the next round of headcount and budget "
            f"given each division's growth and constraints.\n{body}\n"
            "Directors advocate for their divisions; the CEO pushes on ROI.")


def p_promo_committee(period, s):
    cands = []
    for n, d in s.items():
        if d["events"] and "promot" in d["events"].lower():
            cands.append(f"{n}: {d['events']}")
    extra = ("\nCandidate(s) on the table: " + "; ".join(cands)) if cands else ""
    return (f"{org.COMPANY} promotion committee, {period}. Directors and the CEO "
            f"calibrate promotions across divisions, balancing individual impact "
            f"against headcount and level budget. Fast-growing teams (database, "
            f"compute) have more impact to point to; mature teams less.{extra}\n"
            "Discuss who is ready and the trade-offs.")


def p_planning(period, s):
    body = "\n".join("  " + brief(n, s) for n in s)
    return (f"{org.COMPANY} quarterly planning, {period}. Set division-level goals "
            f"for next quarter. Database wants to push growth from "
            f"{s['database']['growth']:.0%} higher; storage is the cash cow funding "
            f"bets.\n{body}\nAgree on a headcount and investment split.")


# ---- per-division prompts ---------------------------------------------------
def p_review(name, period, s):
    d = s[name]
    ics = ", ".join(f"{i['name']} ({i['level']})" for i in d["ics"])
    return (f"{name.title()} division review at {org.COMPANY}, {period}. "
            f"{brief(name, s)} Team: {ics}. The director walks the team through "
            f"revenue and growth, hiring against the {d['open_roles']} open roles, "
            f"and the top priority. ICs ask questions.")


def p_retro(name, period, s):
    d = s[name]
    return (f"{name.title()} team sprint retro at {org.COMPANY}, {period}. "
            f"What went well, what didn't, action items. Context: {brief(name, s)} "
            "Keep it candid and specific to the work.")


def p_standup(name, period, s):
    ics = ", ".join(i["name"] for i in org.DIVISIONS[name]["ics"])
    return (f"Daily standup for the {name} team at {org.COMPANY} ({ics}), {period}. "
            f"Short status updates tied to priorities ({org.DIVISIONS[name]['note']}).")


def p_one_on_one(name, ic, period, s):
    d = s[name]
    return (f"A 1:1 between {d['director']['name']} (director) and {ic['name']} "
            f"({ic['level']} SWE) in {name}, {period}. {brief(name, s)} Discuss "
            f"{ic['name']}'s recent impact, growth, blockers, and path to promotion "
            f"— be specific about whether level and headcount support a promo now.")


def p_incident(name, period, s):
    return (f"Incident postmortem at {org.COMPANY}, {period}, {name} division. "
            f"{brief(name, s)} A production incident this quarter; walk the timeline, "
            "root cause, customer impact, and follow-ups. Blameless tone.")


def p_slack_div(name, period, s):
    return (f"A Slack thread in #{name} at {org.COMPANY} ({period}), 5-7 "
            f"'Name: message' lines, casual and concrete. Context: {brief(name, s)}")


def p_slack_general(period, s):
    return (f"A Slack thread in #general at {org.COMPANY} ({period}), 6-8 "
            f"'Name: message' lines: cross-team chatter, a launch shout-out, a "
            f"hiring ping. Total company revenue ${org.snapshot_total_cache}M.")


def p_slack_incidents(period, s):
    hot = next((n for n, d in s.items() if d["events"] and "incident" in d["events"].lower()),
               "networking")
    return (f"A Slack thread in #incidents at {org.COMPANY} ({period}), 5-7 "
            f"'Name: message' lines about a live issue in the {hot} division and "
            "its mitigation.")


def participants(scope, s):
    if scope == "company":
        return [{k: p.get(k) for k in ("name", "role", "level", "division")}
                for p in org.people()]
    d = org.DIVISIONS[scope]
    return ([{**d["director"], "division": scope}]
            + [{**i, "role": "SWE", "division": scope} for i in d["ics"]])


def gt(scope, s):
    if scope == "company":
        return {"total_revenue": sum(v["revenue"] for v in s.values()),
                "divisions": {n: {k: v[k] for k in ("revenue", "growth", "headcount",
                                                    "open_roles", "a")}
                              for n, v in s.items()}}
    v = s[scope]
    return {"division": scope, **{k: v[k] for k in ("revenue", "growth", "headcount",
                                                    "open_roles", "a")},
            "events": v["events"]}


def plan(period, s):
    """All (type, scope, prompt) items for one quarter."""
    items = [("all_hands", "company", p_all_hands(period, s)),
             ("exec_staff", "company", p_exec_staff(period, s)),
             ("promotion_committee", "company", p_promo_committee(period, s)),
             ("planning", "company", p_planning(period, s))]
    for n in org.DIVISIONS:
        items.append(("division_review", n, p_review(n, period, s)))
        items.append(("retro", n, p_retro(n, period, s)))
        items.append(("standup", n, p_standup(n, period, s)))
        for ic in org.DIVISIONS[n]["ics"]:
            items.append((f"one_on_one:{ic['name']}", n, p_one_on_one(n, ic, period, s)))
        if s[n]["events"] and "incident" in s[n]["events"].lower():
            items.append(("incident_postmortem", n, p_incident(n, period, s)))
    items.append(("slack", "general", p_slack_general(period, s)))
    items.append(("slack", "incidents", p_slack_incidents(period, s)))
    for n in org.DIVISIONS:
        items.append(("slack", n, p_slack_div(n, period, s)))
    return items


def generate(llm, periods=None, limit=None):
    periods = periods or len(org.PERIODS)
    records, idx = [], 0
    for qi in range(periods):
        period = org.PERIODS[qi]
        s = org.snapshot(qi)
        org.snapshot_total_cache = sum(v["revenue"] for v in s.values())
        for kind, scope, prompt in plan(period, s):
            if limit and len(records) >= limit:
                return _wrap(records)
            try:
                text = llm.ask(prompt, system=SYSTEM)
            except Exception as exc:                       # fail-soft per CLAUDE.md
                print(f"  [skip {kind}/{scope} {period}: {type(exc).__name__}]")
                continue
            base = kind.split(":")[0]
            rec = {"id": f"{kind.replace(':', '-')}-{scope}-{period}-{idx:03d}",
                   "type": base, "subject": kind.split(":")[1] if ":" in kind else None,
                   "period": period, "scope": scope,
                   "ground_truth": gt("company" if scope == "company" else
                                      (scope if scope in org.DIVISIONS else "company"), s)
                   if scope in org.DIVISIONS or scope == "company" else
                   gt("company", s)}
            if base == "slack":
                rec["channel"] = f"#{scope}"
                rec["division"] = scope if scope in org.DIVISIONS else None
                rec["messages"] = text
            else:
                rec["participants"] = participants(
                    "company" if scope == "company" else scope, s)
                rec["transcript"] = text
            records.append(rec)
            idx += 1
            print(f"  [{idx}] {kind}/{scope} {period} ({len(text)} chars)")
    return _wrap(records)


def _wrap(records):
    return {"company": org.COMPANY, "periods": org.PERIODS[:max(1, len({r['period'] for r in records}))],
            "n_records": len(records), "records": records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--periods", type=int, default=len(org.PERIODS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    llm = OllamaLLM(model=args.model, temperature=0.7, timeout=240,
                    options={"num_ctx": 8192})
    print(f"generating openworld-corp corpus with {args.model} "
          f"({args.periods} quarters) ...")
    corpus = generate(llm, periods=args.periods, limit=args.limit)
    OUT.write_text(json.dumps(corpus, indent=2))
    print(f"wrote {corpus['n_records']} records -> {OUT}")


if __name__ == "__main__":
    main()
