"""Builder for OpenWorld-ContextBench (writes tasks.jsonl).

Each instance is a repair TASK plus a context_history of related, already-fixed
bugs on *different* modules that share the same fix pattern. The model must
transfer the pattern (not copy code). Run:

    python datasets/openworld-contextbench/build_tasks.py

then validate with `pytest tests/test_contextbench.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tasks.jsonl"


# --- context examples (the "already solved" related bugs) -------------------

RL_BUGGY = '''\
class RateLimiter:
    def __init__(self, capacity, refill):
        self.capacity = capacity
        self.refill = refill
        self.tokens = capacity

    def tick(self):
        self.tokens = self.tokens + self.refill
'''
RL_REF = '''\
class RateLimiter:
    def __init__(self, capacity, refill):
        self.capacity = capacity
        self.refill = refill
        self.tokens = capacity

    def tick(self):
        self.tokens = min(self.capacity, self.tokens + self.refill)
'''

BANK_BUGGY = '''\
class Account:
    def __init__(self):
        self.balance = 0

    def withdraw(self, amount):
        self.balance -= amount
        return self.balance
'''
BANK_REF = '''\
class Account:
    def __init__(self):
        self.balance = 0

    def withdraw(self, amount):
        if amount > self.balance:
            return self.balance
        self.balance -= amount
        return self.balance
'''

MERGE_BUGGY = '''\
def merge(intervals):
    out = []
    for start, end in intervals:
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out
'''
MERGE_REF = '''\
def merge(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out
'''


# --- the tasks to solve (different modules, same fix pattern) ---------------

SCORE_BUGGY = '''\
class ScoreBoard:
    def __init__(self, cap):
        self.cap = cap
        self.score = 0

    def add(self, points):
        self.score = self.score + points
        return self.score
'''
SCORE_REF = '''\
class ScoreBoard:
    def __init__(self, cap):
        self.cap = cap
        self.score = 0

    def add(self, points):
        self.score = min(self.cap, self.score + points)
        return self.score
'''
SCORE_PRE = '''\
def run(cap, adds):
    sb = ScoreBoard(cap)
    return [sb.add(p) for p in adds]
'''

WARE_BUGGY = '''\
class Warehouse:
    def __init__(self):
        self.count = 0

    def add(self, n):
        self.count += n
        return self.count

    def remove(self, n):
        self.count -= n
        return self.count

    def level(self):
        return self.count
'''
WARE_REF = '''\
class Warehouse:
    def __init__(self):
        self.count = 0

    def add(self, n):
        self.count += n
        return self.count

    def remove(self, n):
        if n > self.count:
            return self.count
        self.count -= n
        return self.count

    def level(self):
        return self.count
'''
WARE_PRE = '''\
def run(ops):
    w = Warehouse()
    out = []
    for op in ops:
        if op[0] == 'add':
            out.append(w.add(op[1]))
        elif op[0] == 'remove':
            out.append(w.remove(op[1]))
        else:
            out.append(w.level())
    return out
'''

STAT_BUGGY = '''\
def second_smallest(xs):
    return xs[1]
'''
STAT_REF = '''\
def second_smallest(xs):
    return sorted(xs)[1]
'''


def _world(slug, module_name, buggy):
    return {
        "name": f"contextbench:{slug}",
        "description": f"Program repair for the {module_name!r} module. Submit the "
                       "corrected source via submit_patch(params={'source': ...}).",
        "initial_state": {"source": buggy, "attempts": 0, "solved": False},
        "actions": ["submit_patch"],
        "rules": ["solved requires zero failures in both hidden suites"],
        "invariants": ["the public interface is unchanged"],
    }


def _task(slug, module_name, issue, buggy, ref, preamble, f2p, p2p):
    return {
        "instance_id": f"openworld-contextbench-{slug}",
        "module_name": module_name,
        "issue": issue,
        "buggy_source": buggy,
        "reference_source": ref,
        "test_preamble": preamble,
        "fail_to_pass": [list(p) for p in f2p],
        "pass_to_pass": [list(p) for p in p2p],
        "world": _world(slug, module_name, buggy),
    }


RAW = [
    dict(
        instance_id="openworld-contextbench-000-cap-at-max",
        pattern="cap a running value at a maximum with min()",
        task=_task(
            "000-cap-at-max", "scoreboard",
            "The scoreboard can exceed its cap. add() just keeps summing points with "
            "no ceiling, so the score climbs past the configured cap. It should never "
            "exceed the cap.",
            SCORE_BUGGY, SCORE_REF, SCORE_PRE,
            f2p=[("run(10, [6, 6])", "[6, 10]"), ("run(5, [10])", "[5]")],
            p2p=[("run(10, [3, 4])", "[3, 7]"), ("run(100, [20, 30])", "[20, 50]")],
        ),
        context_history=[dict(
            module_name="rate_limiter",
            issue="A token bucket overfilled because tick() kept adding refill tokens "
                  "with no ceiling. The fix capped the refill at the capacity.",
            buggy_source=RL_BUGGY, reference_source=RL_REF,
        )],
    ),
    dict(
        instance_id="openworld-contextbench-001-reject-underflow",
        pattern="reject an update that would push a value out of range",
        task=_task(
            "001-reject-underflow", "warehouse",
            "Warehouse stock goes negative. remove() subtracts the requested quantity "
            "even when there isn't enough on hand. A removal larger than the current "
            "level should be rejected and leave the level unchanged.",
            WARE_BUGGY, WARE_REF, WARE_PRE,
            f2p=[("run([('add',5),('remove',8),('level',)])", "[5, 5, 5]"),
                 ("run([('remove',1)])", "[0]")],
            p2p=[("run([('add',5),('remove',5),('level',)])", "[5, 0, 0]"),
                 ("run([('add',3),('remove',1),('level',)])", "[3, 2, 2]")],
        ),
        context_history=[dict(
            module_name="bank",
            issue="An account could go negative because withdraw() took more than the "
                  "balance. The fix rejected any withdrawal larger than the balance.",
            buggy_source=BANK_BUGGY, reference_source=BANK_REF,
        )],
    ),
    dict(
        instance_id="openworld-contextbench-002-sort-first",
        pattern="sort the input before indexing into it",
        task=_task(
            "002-sort-first", "stats",
            "second_smallest(xs) returns the wrong value when the list isn't already "
            "sorted — it just returns the element at index 1 instead of the second "
            "smallest value.",
            STAT_BUGGY, STAT_REF, "",
            f2p=[("second_smallest([5, 1, 3])", "3"), ("second_smallest([10, 4, 7])", "7")],
            p2p=[("second_smallest([1, 3, 5])", "3"), ("second_smallest([2, 4, 6, 8])", "4")],
        ),
        context_history=[dict(
            module_name="intervals",
            issue="merge() gave wrong results on unsorted input because it merged in "
                  "the given order. The fix sorted the intervals before merging.",
            buggy_source=MERGE_BUGGY, reference_source=MERGE_REF,
        )],
    ),
]


def main() -> None:
    with OUT.open("w", encoding="utf-8") as fh:
        for r in RAW:
            fh.write(json.dumps({
                "instance_id": r["instance_id"],
                "task": r["task"],
                "context_history": r["context_history"],
                "pattern": r["pattern"],
            }) + "\n")
    print(f"[wrote] {OUT} ({len(RAW)} instances)")


if __name__ == "__main__":
    main()
