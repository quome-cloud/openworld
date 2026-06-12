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
    # -----------------------------------------------------------------------
    # 7. CSV row — strip fields (stage 1) AND drop a trailing empty field (s2)
    # -----------------------------------------------------------------------
    dict(
        slug="csv-row-staged", module_name="csv_row",
        issue=(
            "parse_row() doesn't strip surrounding whitespace from fields: "
            "'a, b , c' parses to ['a', ' b ', ' c']. Each field should have its "
            "leading/trailing spaces removed."
        ),
        buggy="def parse_row(line):\n    return line.split(',')\n",
        ref=(
            "def parse_row(line):\n"
            "    fields = [f.strip() for f in line.split(',')]\n"
            "    while fields and fields[-1] == '':\n"
            "        fields.pop()\n"
            "    return fields\n"
        ),
        preamble="",
        f2p=[
            ("parse_row('a, b , c')", "['a', 'b', 'c']"),
            ("parse_row('a,b,')", "['a', 'b']"),
        ],
        p2p=[
            ("parse_row('a,b,c')", "['a', 'b', 'c']"),
            ("parse_row('x')", "['x']"),
        ],
    ),
    # -----------------------------------------------------------------------
    # 8. Battery charge — cap at 100 (stage 1) AND floor at 0 (stage 2)
    # -----------------------------------------------------------------------
    dict(
        slug="charge-clamp-staged", module_name="charge",
        issue=(
            "charge() lets the level exceed 100%: charge(90, 20) returns 110. "
            "Charging should never push the level above 100."
        ),
        buggy="def charge(level, amount):\n    return level + amount\n",
        ref=(
            "def charge(level, amount):\n"
            "    return max(0, min(100, level + amount))\n"
        ),
        preamble="",
        f2p=[
            ("charge(90, 20)", "100"),
            ("charge(10, -50)", "0"),
        ],
        p2p=[
            ("charge(50, 10)", "60"),
            ("charge(0, 0)", "0"),
        ],
    ),
    # -----------------------------------------------------------------------
    # 9. Bounded stack — enforce capacity on push (stage 1) AND guard pop on
    #    empty (stage 2). Stateful class.
    # -----------------------------------------------------------------------
    dict(
        slug="bounded-stack-staged", module_name="bounded_stack",
        issue=(
            "BoundedStack ignores its capacity: pushing past `cap` keeps growing "
            "the stack. A push when the stack is already full should be rejected "
            "(don't append) and just return the current size."
        ),
        buggy=(
            "class BoundedStack:\n"
            "    def __init__(self, cap):\n"
            "        self.cap = cap\n"
            "        self.items = []\n\n"
            "    def push(self, x):\n"
            "        self.items.append(x)\n"
            "        return len(self.items)\n\n"
            "    def pop(self):\n"
            "        return self.items.pop()\n"
        ),
        ref=(
            "class BoundedStack:\n"
            "    def __init__(self, cap):\n"
            "        self.cap = cap\n"
            "        self.items = []\n\n"
            "    def push(self, x):\n"
            "        if len(self.items) >= self.cap:\n"
            "            return len(self.items)\n"
            "        self.items.append(x)\n"
            "        return len(self.items)\n\n"
            "    def pop(self):\n"
            "        if not self.items:\n"
            "            return None\n"
            "        return self.items.pop()\n"
        ),
        preamble=(
            "def run(cap, ops):\n"
            "    s = BoundedStack(cap)\n"
            "    out = []\n"
            "    for op in ops:\n"
            "        if op[0] == 'push':\n"
            "            out.append(s.push(op[1]))\n"
            "        else:\n"
            "            out.append(s.pop())\n"
            "    return out\n"
        ),
        f2p=[
            ("run(2, [('push',1),('push',2),('push',3)])", "[1, 2, 2]"),
            ("run(1, [('pop',)])", "[None]"),
        ],
        p2p=[
            ("run(3, [('push',1),('push',2),('pop',)])", "[1, 2, 2]"),
            ("run(2, [('push',5)])", "[1]"),
        ],
    ),
    # -----------------------------------------------------------------------
    # 10. Slugify — strip ends (stage 1) AND collapse internal whitespace (s2)
    # -----------------------------------------------------------------------
    dict(
        slug="slugify-staged", module_name="slugify",
        issue=(
            "slugify() turns leading/trailing spaces into dashes: ' Hello World ' "
            "becomes '-hello-world-'. Surrounding whitespace should be trimmed "
            "before slugifying."
        ),
        buggy="def slugify(text):\n    return text.lower().replace(' ', '-')\n",
        ref="def slugify(text):\n    return '-'.join(text.strip().lower().split())\n",
        preamble="",
        f2p=[
            ("slugify(' Hello World ')", "'hello-world'"),
            ("slugify('a  b')", "'a-b'"),
        ],
        p2p=[
            ("slugify('hello world')", "'hello-world'"),
            ("slugify('abc')", "'abc'"),
        ],
    ),
    # -----------------------------------------------------------------------
    # 11. Phone format — validate length (stage 1) AND strip non-digits (s2)
    # -----------------------------------------------------------------------
    dict(
        slug="phone-format-staged", module_name="phone_format",
        issue=(
            "format_phone() produces garbage for inputs that aren't exactly 10 "
            "digits (a 7-digit string yields an empty area code). If the input "
            "isn't 10 digits, return it unchanged."
        ),
        buggy=(
            "def format_phone(digits):\n"
            "    return '(' + digits[0:3] + ') ' + digits[3:6] + '-' + digits[6:10]\n"
        ),
        ref=(
            "def format_phone(digits):\n"
            "    cleaned = ''.join(c for c in digits if c.isdigit())\n"
            "    if len(cleaned) != 10:\n"
            "        return digits\n"
            "    return '(' + cleaned[0:3] + ') ' + cleaned[3:6] + '-' + cleaned[6:10]\n"
        ),
        preamble="",
        f2p=[
            ("format_phone('1234567')", "'1234567'"),
            ("format_phone('123-456-7890')", "'(123) 456-7890'"),
        ],
        p2p=[
            ("format_phone('1234567890')", "'(123) 456-7890'"),
            ("format_phone('5551234567')", "'(555) 123-4567'"),
        ],
    ),
    # -----------------------------------------------------------------------
    # 12. Running stats — guard max() on empty (stage 1) AND mean() on empty
    #     (stage 2). Stateful class.
    # -----------------------------------------------------------------------
    dict(
        slug="running-stats-staged", module_name="running_stats",
        issue=(
            "RunningStats.max() crashes with a ValueError when called before any "
            "value has been added. With no values, max() should return None."
        ),
        buggy=(
            "class RunningStats:\n"
            "    def __init__(self):\n"
            "        self.values = []\n\n"
            "    def add(self, x):\n"
            "        self.values.append(x)\n\n"
            "    def max(self):\n"
            "        return max(self.values)\n\n"
            "    def mean(self):\n"
            "        return sum(self.values) / len(self.values)\n"
        ),
        ref=(
            "class RunningStats:\n"
            "    def __init__(self):\n"
            "        self.values = []\n\n"
            "    def add(self, x):\n"
            "        self.values.append(x)\n\n"
            "    def max(self):\n"
            "        if not self.values:\n"
            "            return None\n"
            "        return max(self.values)\n\n"
            "    def mean(self):\n"
            "        if not self.values:\n"
            "            return 0.0\n"
            "        return sum(self.values) / len(self.values)\n"
        ),
        preamble=(
            "def run(ops):\n"
            "    rs = RunningStats()\n"
            "    out = []\n"
            "    for op in ops:\n"
            "        if op[0] == 'add':\n"
            "            rs.add(op[1])\n"
            "            out.append(None)\n"
            "        elif op[0] == 'max':\n"
            "            out.append(rs.max())\n"
            "        else:\n"
            "            out.append(rs.mean())\n"
            "    return out\n"
        ),
        f2p=[
            ("run([('max',)])", "[None]"),
            ("run([('mean',)])", "[0.0]"),
        ],
        p2p=[
            ("run([('add',3),('add',5),('max',)])", "[None, None, 5]"),
            ("run([('add',2),('add',4),('mean',)])", "[None, None, 3.0]"),
        ],
    ),
    # -----------------------------------------------------------------------
    # 13. Parse ints — empty string -> [] (stage 1) AND skip empty entries from
    #     double/trailing commas (stage 2)
    # -----------------------------------------------------------------------
    dict(
        slug="parse-ints-staged", module_name="parse_ints",
        issue=(
            "parse_ints() crashes with a ValueError on an empty string (it tries "
            "int('')). An empty string should return an empty list."
        ),
        buggy="def parse_ints(s):\n    return [int(x) for x in s.split(',')]\n",
        ref="def parse_ints(s):\n    return [int(x) for x in s.split(',') if x.strip()]\n",
        preamble="",
        f2p=[
            ("parse_ints('')", "[]"),
            ("parse_ints('1,,2')", "[1, 2]"),
        ],
        p2p=[
            ("parse_ints('1,2,3')", "[1, 2, 3]"),
            ("parse_ints('5')", "[5]"),
        ],
    ),
    # -----------------------------------------------------------------------
    # 14. Palindrome — ignore case (stage 1) AND ignore spaces (stage 2)
    # -----------------------------------------------------------------------
    dict(
        slug="palindrome-staged", module_name="palindrome",
        issue=(
            "is_palindrome() is case-sensitive, so 'Anna' is not recognized as a "
            "palindrome. The check should ignore letter case."
        ),
        buggy="def is_palindrome(s):\n    return s == s[::-1]\n",
        ref="def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]\n",
        preamble="",
        f2p=[
            ("is_palindrome('Anna')", "True"),
            ("is_palindrome('race car')", "True"),
        ],
        p2p=[
            ("is_palindrome('racecar')", "True"),
            ("is_palindrome('hello')", "False"),
        ],
    ),
    # -----------------------------------------------------------------------
    # 15. Truncate — keep total length <= n with ellipsis (stage 1) AND handle
    #     n too small for an ellipsis (stage 2)
    # -----------------------------------------------------------------------
    dict(
        slug="truncate-staged", module_name="truncate",
        issue=(
            "truncate(text, n) appends '...' but ignores it in the length budget, "
            "so the result can exceed n characters. The total length including the "
            "ellipsis must be at most n."
        ),
        buggy=(
            "def truncate(text, n):\n"
            "    if len(text) > n:\n"
            "        return text[:n] + '...'\n"
            "    return text\n"
        ),
        ref=(
            "def truncate(text, n):\n"
            "    if len(text) <= n:\n"
            "        return text\n"
            "    if n <= 3:\n"
            "        return text[:n]\n"
            "    return text[:n - 3] + '...'\n"
        ),
        preamble="",
        f2p=[
            ("truncate('hello world', 8)", "'hello...'"),
            ("truncate('hello', 2)", "'he'"),
        ],
        p2p=[
            ("truncate('hi', 5)", "'hi'"),
            ("truncate('abc', 3)", "'abc'"),
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
    "csv-row-staged": '''\
def parse_row(line):
    return [f.strip() for f in line.split(',')]
''',
    "charge-clamp-staged": '''\
def charge(level, amount):
    return min(100, level + amount)
''',
    "bounded-stack-staged": '''\
class BoundedStack:
    def __init__(self, cap):
        self.cap = cap
        self.items = []

    def push(self, x):
        if len(self.items) >= self.cap:
            return len(self.items)
        self.items.append(x)
        return len(self.items)

    def pop(self):
        return self.items.pop()
''',
    "slugify-staged": '''\
def slugify(text):
    return text.strip().lower().replace(' ', '-')
''',
    "phone-format-staged": '''\
def format_phone(digits):
    if len(digits) != 10:
        return digits
    return '(' + digits[0:3] + ') ' + digits[3:6] + '-' + digits[6:10]
''',
    "running-stats-staged": '''\
class RunningStats:
    def __init__(self):
        self.values = []

    def add(self, x):
        self.values.append(x)

    def max(self):
        if not self.values:
            return None
        return max(self.values)

    def mean(self):
        return sum(self.values) / len(self.values)
''',
    "parse-ints-staged": '''\
def parse_ints(s):
    if not s:
        return []
    return [int(x) for x in s.split(',')]
''',
    "palindrome-staged": '''\
def is_palindrome(s):
    s = s.lower()
    return s == s[::-1]
''',
    "truncate-staged": '''\
def truncate(text, n):
    if len(text) > n:
        return text[:n - 3] + '...'
    return text
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
