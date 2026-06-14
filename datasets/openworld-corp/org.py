"""The synthetic organization that grounds the openworld-corp transcript dataset.

A DigitalOcean-style PaaS company: a CEO over five product divisions, each with a
director and individual contributors, plus the per-division ground-truth metrics
that the generated transcripts are grounded in. This is the *seed* an LLM turns
into realistic meeting transcripts and Slack threads; swap it for your real org
chart + metrics (or point the loader at your real transcripts) to use the world
model on an actual company.

Every transcript in the corpus is generated FROM these numbers, so the dataset is
state-recoverable: a perceptor can read division health back out of the prose,
which is exactly what the E48 perception experiment measures.
"""

COMPANY = "Nimbus Cloud"               # a stand-in PaaS company
CEO = {"name": "Dana Okafor", "role": "CEO", "level": "E"}

# productivity `a` is the (latent) marginal-growth multiplier used by the world
# model; revenue/growth/headcount/open_roles are what meetings actually discuss.
DIVISIONS = {
    "database": {
        "a": 2.0, "revenue": 200, "growth": 0.42, "headcount": 10, "open_roles": 4,
        "director": {"name": "Priya Raman", "role": "Director", "level": "M2"},
        "ics": [{"name": "Sam Cole", "level": "L5"},
                {"name": "Wei Zhang", "level": "L4"},
                {"name": "Tomas Ruiz", "level": "L6"}],
        "note": "managed Postgres/Redis; fastest-growing, talent-constrained",
    },
    "serverless": {
        "a": 1.0, "revenue": 250, "growth": 0.28, "headcount": 8, "open_roles": 2,
        "director": {"name": "Lena Hart", "role": "Director", "level": "M2"},
        "ics": [{"name": "Omar Diallo", "level": "L5"},
                {"name": "Grace Kim", "level": "L4"},
                {"name": "Ivan Petrov", "level": "L4"}],
        "note": "functions platform; steady, competitive market",
    },
    "storage": {
        "a": 0.6, "revenue": 400, "growth": 0.11, "headcount": 12, "open_roles": 1,
        "director": {"name": "Carlos Mendez", "role": "Director", "level": "M3"},
        "ics": [{"name": "Aisha Bello", "level": "L6"},
                {"name": "Nora Fischer", "level": "L5"},
                {"name": "Hassan Ali", "level": "L4"}],
        "note": "legacy object store; largest revenue, slow growth, cash cow",
    },
    "compute": {
        "a": 1.5, "revenue": 300, "growth": 0.33, "headcount": 10, "open_roles": 3,
        "director": {"name": "Maya Singh", "role": "Director", "level": "M2"},
        "ics": [{"name": "Jonah Webb", "level": "L5"},
                {"name": "Eva Larsson", "level": "L5"},
                {"name": "Kofi Asante", "level": "L4"}],
        "note": "VMs/Kubernetes; strong, scaling fast",
    },
    "networking": {
        "a": 0.4, "revenue": 350, "growth": 0.08, "headcount": 9, "open_roles": 0,
        "director": {"name": "Ben Cohen", "role": "Director", "level": "M2"},
        "ics": [{"name": "Rosa Ortiz", "level": "L5"},
                {"name": "Liam Walsh", "level": "L4"},
                {"name": "Yuki Tanaka", "level": "L4"}],
        "note": "load balancers/VPC; mature, low growth, reliability-focused",
    },
}


# quarters covered by the corpus, with scripted org events per quarter so the
# transcripts reference real changes over time (promotions, attrition, launches).
PERIODS = ["2026-Q1", "2026-Q2", "2026-Q3"]
EVENTS = {
    1: {
        "database": "Sam Cole (L5) promoted to L6/Staff this quarter; Redis managed offering hit GA.",
        "compute": "Filled two of the open SWE reqs; Kofi Asante (L4) is up for L5.",
        "storage": "Carlos kicked off the legacy-cluster migration; revenue flat as expected.",
    },
    2: {
        "serverless": "Lost a senior engineer to a competitor; backfilling the L5 req.",
        "database": "Still talent-constrained; two reqs open for a full quarter now.",
        "compute": "Kubernetes autoscaler launch drove a strong quarter.",
        "networking": "Major load-balancer incident early in the quarter (see postmortem).",
    },
}


def total_revenue():
    return sum(d["revenue"] for d in DIVISIONS.values())


def snapshot(qi):
    """Per-division metrics for quarter index `qi` (0-based), evolved from the
    base: revenue compounds at the division's growth rate, fast-growing divisions
    hire against open roles, and scripted EVENTS attach per quarter. Deterministic.
    """
    snap = {}
    for name, d in DIVISIONS.items():
        fast = d["growth"] > 0.30
        hires = (1 if fast else 0) * qi
        snap[name] = {
            "a": d["a"],
            "revenue": round(d["revenue"] * (1 + d["growth"]) ** (qi / 4.0)),
            "growth": d["growth"],
            "headcount": d["headcount"] + hires,
            "open_roles": max(0, d["open_roles"] - hires) + (1 if (fast and qi == 2) else 0),
            "director": d["director"],
            "ics": d["ics"],
            "note": d["note"],
            "events": EVENTS.get(qi, {}).get(name, ""),
        }
    return snap


def snapshot_total(qi):
    return sum(s["revenue"] for s in snapshot(qi).values())


def people():
    """All named employees with role/level/division (for participant lists)."""
    out = [{**CEO, "division": None}]
    for name, d in DIVISIONS.items():
        out.append({**d["director"], "division": name})
        for ic in d["ics"]:
            out.append({**ic, "role": "SWE", "division": name})
    return out
