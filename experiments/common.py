"""Shared infrastructure for the openworld experiment campaign.

Ground-truth worlds, probe suites, Wilson confidence intervals, and a JSON
results writer. Every experiment script writes experiments/results/<name>.json
with enough metadata to regenerate paper tables and figures.
"""

from __future__ import annotations

import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path

from openworld import Action, OllamaConnectionError, OllamaLLM, World, WorldState
from openworld.transition import FunctionTransition

RESULTS_DIR = Path(__file__).parent / "results"
GENERATOR_MODEL = "qwen2.5:7b"
SMALL_MODEL = "qwen2.5:3b"


# ---------------------------------------------------------------------------
# Ground-truth worlds (deterministic oracles for every comparison)
# ---------------------------------------------------------------------------

def sprint_ground_truth(state, action):
    s = dict(state)
    name = action["name"]
    if name == "ship" and s["backlog"] > 0:
        s["backlog"] -= 1
        s["shipped"] += 1
        s["debt"] += 1
        s["bugs"] += s["debt"] // 4
    elif name == "fix":
        s["bugs"] = max(0, s["bugs"] - 2)
    elif name == "refactor":
        s["debt"] = max(0, s["debt"] - 2)
    return s


SPRINT_INITIAL = {"backlog": 12, "shipped": 0, "bugs": 0, "debt": 0}
SPRINT_ACTIONS = ["ship", "fix", "refactor"]
SPRINT_RULES = [
    "'ship' (when backlog > 0): backlog -1, shipped +1, debt +1, then bugs "
    "increase by debt // 4 (integer division, using the debt value after the +1).",
    "'fix': bugs decrease by 2, never below 0.",
    "'refactor': debt decreases by 2, never below 0.",
    "'noop' and unknown actions change nothing.",
]
SPRINT_DESCRIPTION = "An engineering team working a sprint backlog."


def orchard_ground_truth(state, action):
    s = dict(state)
    s["harvested"] = dict(s["harvested"])
    agent = action.get("agent")
    if action["name"] == "pick" and s["apples"] > 0 and agent:
        s["apples"] -= 1
        s["harvested"][agent] = s["harvested"].get(agent, 0) + 1
    return s


ORCHARD_INITIAL = {"apples": 10, "harvested": {"alice": 0}}
ORCHARD_ACTIONS = ["pick", "wait"]
ORCHARD_RULES = [
    "'pick' moves one apple from the orchard to the acting agent's count in 'harvested' "
    "(use action['agent'] as the key; create it at 0 if missing).",
    "Picking when no apples remain does nothing.",
    "'wait' and 'noop' leave the state unchanged.",
]
ORCHARD_DESCRIPTION = "Agents share an orchard with a limited pool of apples."


def triage_ground_truth(state, action):
    s = dict(state)
    name = action["name"]
    if name == "treat_critical" and s["critical_waiting"] > 0:
        s["critical_waiting"] -= 1
        s["treated"] += 1
        s["outcomes"] += 3
        s["spend"] += 3
    elif name == "treat_moderate" and s["moderate_waiting"] > 0:
        s["moderate_waiting"] -= 1
        s["treated"] += 1
        s["outcomes"] += 1
        s["spend"] += 1
    s["tick"] += 1
    if s["tick"] % 2 == 0 and s["critical_waiting"] > 0:
        s["critical_waiting"] -= 1
        s["deteriorated"] += 1
        s["outcomes"] -= 2
    return s


TRIAGE_INITIAL = {
    "tick": 0, "critical_waiting": 4, "moderate_waiting": 8,
    "treated": 0, "deteriorated": 0, "outcomes": 0, "spend": 0,
}
TRIAGE_ACTIONS = ["treat_critical", "treat_moderate", "wait"]
TRIAGE_RULES = [
    "'treat_critical' treats one waiting critical patient: critical_waiting -1, treated +1, outcomes +3, spend +3.",
    "'treat_moderate' treats one waiting moderate patient: moderate_waiting -1, treated +1, outcomes +1, spend +1.",
    "Treating when the matching queue is empty does nothing (besides the clock).",
    "After EVERY action (including 'wait' and 'noop'), tick increases by 1.",
    "Whenever the new tick is even and critical patients still wait, one of them "
    "deteriorates: critical_waiting -1, deteriorated +1, outcomes -2.",
]
TRIAGE_DESCRIPTION = (
    "An ICU triage queue. Critical and moderate patients wait for treatment; "
    "untreated critical patients deteriorate over time."
)

WORLD_SPECS = {
    "sprint": dict(
        description=SPRINT_DESCRIPTION, initial=SPRINT_INITIAL,
        actions=SPRINT_ACTIONS, rules=SPRINT_RULES, oracle=sprint_ground_truth,
    ),
    "orchard": dict(
        description=ORCHARD_DESCRIPTION, initial=ORCHARD_INITIAL,
        actions=ORCHARD_ACTIONS, rules=ORCHARD_RULES, oracle=orchard_ground_truth,
    ),
    "triage": dict(
        description=TRIAGE_DESCRIPTION, initial=TRIAGE_INITIAL,
        actions=TRIAGE_ACTIONS, rules=TRIAGE_RULES, oracle=triage_ground_truth,
    ),
}


def make_oracle_world(spec_name):
    spec = WORLD_SPECS[spec_name]
    return World(
        name=spec_name,
        description=spec["description"],
        initial_state=dict(spec["initial"]),
        actions=list(spec["actions"]),
        rules=list(spec["rules"]),
        transition=FunctionTransition(spec["oracle"]),
    )


# ---------------------------------------------------------------------------
# Probe suites: (state, action) pairs that exercise tricky branches
# ---------------------------------------------------------------------------

SPRINT_PROBES = [
    ({"backlog": 12, "shipped": 0, "bugs": 0, "debt": 0}, Action("ship")),
    ({"backlog": 5, "shipped": 7, "bugs": 0, "debt": 3}, Action("ship")),   # debt//4 fires
    ({"backlog": 5, "shipped": 7, "bugs": 2, "debt": 7}, Action("ship")),
    ({"backlog": 0, "shipped": 12, "bugs": 3, "debt": 5}, Action("ship")),  # empty backlog
    ({"backlog": 4, "shipped": 0, "bugs": 1, "debt": 0}, Action("fix")),    # clamp at 0
    ({"backlog": 4, "shipped": 0, "bugs": 5, "debt": 0}, Action("fix")),
    ({"backlog": 4, "shipped": 0, "bugs": 0, "debt": 1}, Action("refactor")),
    ({"backlog": 4, "shipped": 0, "bugs": 0, "debt": 6}, Action("refactor")),
    ({"backlog": 4, "shipped": 2, "bugs": 1, "debt": 2}, Action("noop")),
    ({"backlog": 1, "shipped": 11, "bugs": 9, "debt": 11}, Action("ship")),
]

SPRINT_PROBES_SCALED = [  # 10x out-of-distribution magnitudes
    ({"backlog": 120, "shipped": 0, "bugs": 0, "debt": 0}, Action("ship")),
    ({"backlog": 50, "shipped": 70, "bugs": 0, "debt": 39}, Action("ship")),
    ({"backlog": 50, "shipped": 70, "bugs": 20, "debt": 77}, Action("ship")),
    ({"backlog": 0, "shipped": 120, "bugs": 30, "debt": 50}, Action("ship")),
    ({"backlog": 40, "shipped": 0, "bugs": 1, "debt": 0}, Action("fix")),
    ({"backlog": 40, "shipped": 0, "bugs": 50, "debt": 0}, Action("fix")),
    ({"backlog": 40, "shipped": 0, "bugs": 0, "debt": 1}, Action("refactor")),
    ({"backlog": 40, "shipped": 0, "bugs": 0, "debt": 60}, Action("refactor")),
    ({"backlog": 40, "shipped": 20, "bugs": 10, "debt": 20}, Action("noop")),
    ({"backlog": 10, "shipped": 110, "bugs": 90, "debt": 110}, Action("ship")),
]


def probe_accuracy(transition, probes, oracle):
    """Fraction of probes where transition.step exactly matches the oracle."""
    matches = 0
    for state, action in probes:
        expected = oracle(dict(state), action.to_dict())
        try:
            actual = transition.step(WorldState(state), action)
        except Exception:
            continue
        if dict(actual) == expected:
            matches += 1
    return matches / len(probes)


# ---------------------------------------------------------------------------
# Stats and IO
# ---------------------------------------------------------------------------

def wilson_ci(successes, n, z=1.96):
    """95% Wilson score interval for a proportion. Returns (low, high)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def spearman(xs, ys):
    """Spearman rank correlation, no scipy dependency."""
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def mcnemar_p(b, c):
    """Exact two-sided McNemar test from discordant-pair counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def permutation_p_spearman(xs, ys, n_permutations=10000, seed=0):
    """Two-sided permutation p-value for the observed Spearman correlation."""
    import random as _random

    observed = abs(spearman(xs, ys))
    rng = _random.Random(seed)
    ys = list(ys)
    hits = 0
    for _ in range(n_permutations):
        rng.shuffle(ys)
        if abs(spearman(xs, ys)) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n_permutations + 1)


def _env_metadata():
    """Provenance stamped into every result: interpreter + analysis-stack versions,
    so a cached artifact records exactly what produced it (auditability)."""
    meta = {"python": platform.python_version()}
    for mod in ("numpy", "matplotlib"):
        try:
            meta[mod] = __import__(mod).__version__
        except Exception:
            pass
    return meta


def save_results(name, payload):
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = dict(payload)
    payload.setdefault("experiment", name)
    payload.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    payload.setdefault("platform", platform.platform())
    payload.setdefault("env", _env_metadata())
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {path}")
    return path


def require_ollama(model=GENERATOR_MODEL, **kwargs):
    llm = OllamaLLM(model=model, **kwargs)
    try:
        llm.ask("Reply with OK.")
    except OllamaConnectionError as exc:
        raise SystemExit(f"This experiment needs Ollama: {exc}")
    return llm


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
