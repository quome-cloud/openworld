"""Builder for OpenWorld-SWE-bench-STAGED (writes tasks.jsonl).

Companion to ``datasets/openworld-swebench`` (the atomic set). The v0 7b-ladder
run showed **no in-world lift** on the atomic instances: each bug is repaired by a
single edit fully described by the issue text, so single-shot and in-world
converge on attempt 1 and the failing-test feedback loop never does any work.

This dataset is the design response. Every instance is a **two-stage** repair:

  * **Stage 1** is the symptom the issue describes. The "obvious" patch a model
    writes from the issue alone fixes stage 1 and passes the first
    ``fail_to_pass`` test.
  * **Stage 2** is a *second*, related defect that the issue does NOT spell out
    and that stays latent until stage 1 is fixed. It surfaces only as a concrete
    failing-test error (input -> got/expected), which is exactly what the
    in-world loop feeds back and the single-shot condition never sees.

So the predicted result is **single-shot solves stage 1 but fails stage 2;
in-world reads the stage-2 error and finishes the repair** -> measurable lift.

Schema is byte-identical to the atomic set (same ``openworld.swebench`` loader).
``buggy_source`` still fails EVERY ``fail_to_pass`` and passes EVERY
``pass_to_pass`` (the dataset contract); the staging lives in the *model's*
intermediate patches, which the harness exercises at run time.

    python datasets/openworld-swebench-staged/build_tasks.py
    pytest tests/test_swebench_staged.py

The RAW list is the source of truth; tasks.jsonl is the generated artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tasks.jsonl"


# ---------------------------------------------------------------------------
# 1. Config parser — skip blanks (stage 1) AND split on the first '=' (stage 2)
# ---------------------------------------------------------------------------
CFG_BUGGY = '''\
def parse_config(lines):
    result = {}
    for line in lines:
        key, value = line.split('=')
        result[key] = value
    return result
'''

CFG_REF = '''\
def parse_config(lines):
    result = {}
    for line in lines:
        if not line.strip():
            continue
        key, value = line.split('=', 1)
        result[key] = value
    return result
'''


# ---------------------------------------------------------------------------
# 2. Discount — floor a negative pct (stage 1) AND cap it at 100 (stage 2)
# ---------------------------------------------------------------------------
DISC_BUGGY = '''\
def apply_discount(price, pct):
    return price - price * pct / 100
'''

DISC_REF = '''\
def apply_discount(price, pct):
    pct = max(0, min(100, pct))
    return price - price * pct / 100
'''


# ---------------------------------------------------------------------------
# 3. Histogram — increment instead of overwrite (stage 1) AND default unseen
#    keys to 0 in count() (stage 2). Stateful class.
# ---------------------------------------------------------------------------
HIST_BUGGY = '''\
class Histogram:
    def __init__(self):
        self.counts = {}

    def add(self, key):
        self.counts[key] = 1

    def count(self, key):
        return self.counts[key]
'''

HIST_REF = '''\
class Histogram:
    def __init__(self):
        self.counts = {}

    def add(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1

    def count(self, key):
        return self.counts.get(key, 0)
'''

HIST_PREAMBLE = '''\
def run(ops):
    h = Histogram()
    out = []
    for op in ops:
        if op[0] == 'add':
            h.add(op[1])
            out.append(None)
        else:
            out.append(h.count(op[1]))
    return out
'''


# ---------------------------------------------------------------------------
# 4. Median — average the two middle elements for even N (stage 1) AND guard
#    the empty list (stage 2)
# ---------------------------------------------------------------------------
MED_BUGGY = '''\
def median(nums):
    nums = sorted(nums)
    n = len(nums)
    return nums[n // 2]
'''

MED_REF = '''\
def median(nums):
    if not nums:
        return None
    nums = sorted(nums)
    n = len(nums)
    if n % 2 == 1:
        return nums[n // 2]
    return (nums[n // 2 - 1] + nums[n // 2]) / 2
'''


# ---------------------------------------------------------------------------
# 5. deep_get — None on a missing key (stage 1) AND None when an intermediate
#    value isn't a dict (stage 2: TypeError, not KeyError)
# ---------------------------------------------------------------------------
DG_BUGGY = '''\
def deep_get(d, keys):
    for k in keys:
        d = d[k]
    return d
'''

DG_REF = '''\
def deep_get(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d
'''


# ---------------------------------------------------------------------------
# 6. format_duration — zero-pad seconds (stage 1) AND break out hours past
#    3600s (stage 2)
# ---------------------------------------------------------------------------
DUR_BUGGY = '''\
def format_duration(seconds):
    minutes = seconds // 60
    secs = seconds % 60
    return str(minutes) + ":" + str(secs)
'''

DUR_REF = '''\
def format_duration(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours:
        return "{}:{:02d}:{:02d}".format(hours, minutes, secs)
    return "{}:{:02d}".format(minutes, secs)
'''


RAW = [
    dict(
        slug="config-parser-staged", module_name="config_parser",
        issue=(
            "parse_config() crashes with a ValueError on blank lines in the config "
            "file (it tries to unpack 'key=value' from an empty string). Blank or "
            "whitespace-only lines should just be skipped."
        ),
        buggy=CFG_BUGGY, ref=CFG_REF, preamble="",
        f2p=[
            ("parse_config(['a=1', '', 'b=2'])", "{'a': '1', 'b': '2'}"),
            ("parse_config(['url=http://x/?a=b'])", "{'url': 'http://x/?a=b'}"),
        ],
        p2p=[
            ("parse_config(['x=1'])", "{'x': '1'}"),
            ("parse_config(['a=1', 'b=2'])", "{'a': '1', 'b': '2'}"),
            ("parse_config([])", "{}"),
        ],
    ),
    dict(
        slug="discount-clamp-staged", module_name="discount",
        issue=(
            "apply_discount() lets a negative discount percentage INCREASE the "
            "price. A negative pct should be treated as no discount (0%)."
        ),
        buggy=DISC_BUGGY, ref=DISC_REF, preamble="",
        f2p=[
            ("apply_discount(100, -10)", "100.0"),
            ("apply_discount(50, 150)", "0.0"),
        ],
        p2p=[
            ("apply_discount(100, 20)", "80.0"),
            ("apply_discount(100, 0)", "100.0"),
        ],
    ),
    dict(
        slug="histogram-staged", module_name="histogram",
        issue=(
            "Histogram.add() overwrites the count instead of incrementing it: "
            "adding the same key twice still reports a count of 1. Repeated adds "
            "of a key should accumulate."
        ),
        buggy=HIST_BUGGY, ref=HIST_REF, preamble=HIST_PREAMBLE,
        f2p=[
            ("run([('add','x'),('add','x'),('count','x')])", "[None, None, 2]"),
            ("run([('count','z')])", "[0]"),
        ],
        p2p=[
            ("run([('add','a'),('count','a')])", "[None, 1]"),
            ("run([('add','a'),('add','b'),('count','b')])", "[None, None, 1]"),
        ],
    ),
    dict(
        slug="median-staged", module_name="median",
        issue=(
            "median() is wrong for even-length lists: it returns the upper of the "
            "two middle elements instead of their average. median([1,2,3,4]) should "
            "be 2.5, not 3."
        ),
        buggy=MED_BUGGY, ref=MED_REF, preamble="",
        f2p=[
            ("median([1, 2, 3, 4])", "2.5"),
            ("median([])", "None"),
        ],
        p2p=[
            ("median([3, 1, 2])", "2"),
            ("median([5])", "5"),
        ],
    ),
    dict(
        slug="deep-get-staged", module_name="deep_get",
        issue=(
            "deep_get(d, keys) walks a chain of keys into nested dicts but crashes "
            "with a KeyError when a key along the path is missing. A missing key "
            "should make it return None instead of raising."
        ),
        buggy=DG_BUGGY, ref=DG_REF, preamble="",
        f2p=[
            ("deep_get({'a': {'b': 2}}, ['a', 'x'])", "None"),
            ("deep_get({'a': 5}, ['a', 'b'])", "None"),
        ],
        p2p=[
            ("deep_get({'a': {'b': 2}}, ['a', 'b'])", "2"),
            ("deep_get({'x': 1}, ['x'])", "1"),
        ],
    ),
    dict(
        slug="format-duration-staged", module_name="format_duration",
        issue=(
            "format_duration(seconds) doesn't zero-pad the seconds field: 65 "
            "formats as '1:5' but should be '1:05'."
        ),
        buggy=DUR_BUGGY, ref=DUR_REF, preamble="",
        f2p=[
            ("format_duration(65)", "'1:05'"),
            ("format_duration(3665)", "'1:01:05'"),
        ],
        p2p=[
            ("format_duration(75)", "'1:15'"),
            ("format_duration(190)", "'3:10'"),
        ],
    ),
]


# Stage-1-only "obvious" patches: the fix a model writes from the issue text
# alone. NOT shipped in the dataset — used by tests/test_swebench_staged.py to
# assert the staging is real (each passes f2p[0], fails f2p[1]).
STAGE1_PATCHES = {
    "config-parser-staged": '''\
def parse_config(lines):
    result = {}
    for line in lines:
        if not line.strip():
            continue
        key, value = line.split('=')
        result[key] = value
    return result
''',
    "discount-clamp-staged": '''\
def apply_discount(price, pct):
    if pct < 0:
        pct = 0
    return price - price * pct / 100
''',
    "histogram-staged": '''\
class Histogram:
    def __init__(self):
        self.counts = {}

    def add(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1

    def count(self, key):
        return self.counts[key]
''',
    "median-staged": '''\
def median(nums):
    nums = sorted(nums)
    n = len(nums)
    if n % 2 == 1:
        return nums[n // 2]
    return (nums[n // 2 - 1] + nums[n // 2]) / 2
''',
    "deep-get-staged": '''\
def deep_get(d, keys):
    for k in keys:
        if k not in d:
            return None
        d = d[k]
    return d
''',
    "format-duration-staged": '''\
def format_duration(seconds):
    minutes = seconds // 60
    secs = seconds % 60
    return "{}:{:02d}".format(minutes, secs)
''',
}


def _world(slug: str, module_name: str, buggy: str) -> dict:
    return {
        "name": f"swebench-staged:{slug}",
        "description": (
            f"Program repair for the {module_name!r} module. Read the issue, then "
            "submit the complete corrected source via "
            "submit_patch(params={'source': ...})."
        ),
        "initial_state": {"source": buggy, "attempts": 0, "solved": False},
        "actions": ["submit_patch"],
        "rules": [
            "'submit_patch' replaces the module source and runs the hidden test "
            "suites; state records per-suite pass/fail counts and the first errors.",
            "The task is solved only when every fail_to_pass AND every pass_to_pass "
            "test passes (fix the bug without breaking regression tests).",
            "Once solved, further patches are ignored.",
        ],
        "invariants": [
            "a solved instance has zero failing tests in both suites",
            "the public interface of the module is unchanged",
        ],
    }


def build() -> list:
    instances = []
    for i, r in enumerate(RAW):
        instances.append({
            "instance_id": f"openworld-swebench-staged-{i:03d}-{r['slug']}",
            "module_name": r["module_name"],
            "issue": r["issue"],
            "buggy_source": r["buggy"],
            "reference_source": r["ref"],
            "test_preamble": r["preamble"],
            "fail_to_pass": [list(p) for p in r["f2p"]],
            "pass_to_pass": [list(p) for p in r["p2p"]],
            "world": _world(r["slug"], r["module_name"], r["buggy"]),
        })
    return instances


def main() -> None:
    instances = build()
    with OUT.open("w", encoding="utf-8") as fh:
        for inst in instances:
            fh.write(json.dumps(inst) + "\n")
    print(f"[wrote] {OUT} ({len(instances)} instances)")


if __name__ == "__main__":
    main()
