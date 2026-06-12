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


# --- new tasks (003-014) and their context examples ------------------------

# 003 floor-at-zero --------------------------------------------------------
BATTERY_BUGGY = '''\
class Battery:
    def __init__(self):
        self.charge = 100

    def drain(self, n):
        self.charge -= n
        return self.charge
'''
BATTERY_REF = '''\
class Battery:
    def __init__(self):
        self.charge = 100

    def drain(self, n):
        self.charge = max(0, self.charge - n)
        return self.charge
'''
BATTERY_PRE = '''\
def run(steps):
    b = Battery()
    return [b.drain(s) for s in steps]
'''
# context for 003: countdown timer that went negative
TIMER_BUGGY = '''\
class Countdown:
    def __init__(self, seconds):
        self.remaining = seconds

    def tick(self, n):
        self.remaining -= n
        return self.remaining
'''
TIMER_REF = '''\
class Countdown:
    def __init__(self, seconds):
        self.remaining = seconds

    def tick(self, n):
        self.remaining = max(0, self.remaining - n)
        return self.remaining
'''

# 004 guard-empty-before-divide (function) ---------------------------------
AVG_BUGGY = '''\
def average(xs):
    return sum(xs) / len(xs)
'''
AVG_REF = '''\
def average(xs):
    if not xs:
        return 0.0
    return sum(xs) / len(xs)
'''
# context for 004: density that crashed on no area
DENSITY_BUGGY = '''\
def density(mass, volume):
    return mass / volume
'''
DENSITY_REF = '''\
def density(mass, volume):
    if volume == 0:
        return 0.0
    return mass / volume
'''

# 005 default-for-missing-key (function) -----------------------------------
PRICE_BUGGY = '''\
def price_of(catalog, item):
    return catalog[item]
'''
PRICE_REF = '''\
def price_of(catalog, item):
    return catalog.get(item, 0)
'''
# context for 005: header lookup that raised on absent header
HEADER_BUGGY = '''\
def header_value(headers, name):
    return headers[name]
'''
HEADER_REF = '''\
def header_value(headers, name):
    return headers.get(name, '')
'''

# 006 include-final-element / off-by-one flush -----------------------------
CHUNK_BUGGY = '''\
def chunk_sums(xs, size):
    out = []
    cur = 0
    c = 0
    for x in xs:
        cur += x
        c += 1
        if c == size:
            out.append(cur)
            cur = 0
            c = 0
    return out
'''
CHUNK_REF = '''\
def chunk_sums(xs, size):
    out = []
    cur = 0
    c = 0
    for x in xs:
        cur += x
        c += 1
        if c == size:
            out.append(cur)
            cur = 0
            c = 0
    if c > 0:
        out.append(cur)
    return out
'''
# context for 006: line buffer that dropped the last unterminated line
SPLIT_BUGGY = '''\
def split_lines(text):
    out = []
    cur = ''
    for ch in text:
        if ch == '|':
            out.append(cur)
            cur = ''
        else:
            cur += ch
    return out
'''
SPLIT_REF = '''\
def split_lines(text):
    out = []
    cur = ''
    for ch in text:
        if ch == '|':
            out.append(cur)
            cur = ''
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out
'''

# 007 clamp-both-ends ------------------------------------------------------
VOLUME_BUGGY = '''\
class Volume:
    def __init__(self):
        self.level = 5

    def adjust(self, d):
        self.level = max(0, self.level + d)
        return self.level
'''
VOLUME_REF = '''\
class Volume:
    def __init__(self):
        self.level = 5

    def adjust(self, d):
        self.level = min(10, max(0, self.level + d))
        return self.level
'''
VOLUME_PRE = '''\
def run(vals):
    v = Volume()
    return [v.adjust(x) for x in vals]
'''
# context for 007: brightness slider that only clamped the low end
BRIGHT_BUGGY = '''\
class Brightness:
    def __init__(self):
        self.value = 50

    def step(self, d):
        self.value = max(0, self.value + d)
        return self.value
'''
BRIGHT_REF = '''\
class Brightness:
    def __init__(self):
        self.value = 50

    def step(self, d):
        self.value = min(100, max(0, self.value + d))
        return self.value
'''

# 008 normalize-before-compare ---------------------------------------------
TAGS_BUGGY = '''\
def count_unique_tags(tags):
    seen = set()
    for t in tags:
        seen.add(t)
    return len(seen)
'''
TAGS_REF = '''\
def count_unique_tags(tags):
    seen = set()
    for t in tags:
        seen.add(t.strip().lower())
    return len(seen)
'''
# context for 008: username dedupe that treated case/space as distinct
USER_BUGGY = '''\
def distinct_users(names):
    return len(set(names))
'''
USER_REF = '''\
def distinct_users(names):
    return len(set(n.strip().lower() for n in names))
'''

# 009 dedupe-preserving-order ----------------------------------------------
DEDUPE_BUGGY = '''\
def dedupe(xs):
    return list(xs)
'''
DEDUPE_REF = '''\
def dedupe(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
'''
# context for 009: recent-history list that kept duplicate entries
HIST_BUGGY = '''\
def recent(events):
    return [e for e in events]
'''
HIST_REF = '''\
def recent(events):
    seen = set()
    out = []
    for e in events:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out
'''

# 010 early-return-on-first-match ------------------------------------------
FIRSTNEG_BUGGY = '''\
def first_negative(xs):
    found = None
    for x in xs:
        if x < 0:
            found = x
    return found
'''
FIRSTNEG_REF = '''\
def first_negative(xs):
    for x in xs:
        if x < 0:
            return x
    return None
'''
# context for 010: search that returned the last hit instead of the first
FIND_BUGGY = '''\
def index_of(xs, target):
    result = -1
    for i, x in enumerate(xs):
        if x == target:
            result = i
    return result
'''
FIND_REF = '''\
def index_of(xs, target):
    for i, x in enumerate(xs):
        if x == target:
            return i
    return -1
'''

# 011 guard-empty-before-index ---------------------------------------------
HEADLINE_BUGGY = '''\
def headline(items):
    return items[0]
'''
HEADLINE_REF = '''\
def headline(items):
    if not items:
        return 'none'
    return items[0]
'''
# context for 011: peek() that crashed on an empty stack
PEEK_BUGGY = '''\
def peek(stack):
    return stack[-1]
'''
PEEK_REF = '''\
def peek(stack):
    if not stack:
        return None
    return stack[-1]
'''

# 012 reject-out-of-range (class) ------------------------------------------
THEATER_BUGGY = '''\
class Theater:
    def __init__(self, cap):
        self.cap = cap
        self.sold = 0

    def book(self, n):
        self.sold += n
        return self.sold
'''
THEATER_REF = '''\
class Theater:
    def __init__(self, cap):
        self.cap = cap
        self.sold = 0

    def book(self, n):
        if self.sold + n > self.cap:
            return self.sold
        self.sold += n
        return self.sold
'''
THEATER_PRE = '''\
def run(ops):
    t = Theater(10)
    return [t.book(n) for n in ops]
'''
# context for 012: parking lot that admitted cars past capacity
LOT_BUGGY = '''\
class ParkingLot:
    def __init__(self, spaces):
        self.spaces = spaces
        self.parked = 0

    def enter(self, n):
        self.parked += n
        return self.parked
'''
LOT_REF = '''\
class ParkingLot:
    def __init__(self, spaces):
        self.spaces = spaces
        self.parked = 0

    def enter(self, n):
        if self.parked + n > self.spaces:
            return self.parked
        self.parked += n
        return self.parked
'''

# 013 guard-empty-before-divide (function, percent) ------------------------
PCT_BUGGY = '''\
def percent_full(used, total):
    return round(100 * used / total, 1)
'''
PCT_REF = '''\
def percent_full(used, total):
    if total == 0:
        return 0.0
    return round(100 * used / total, 1)
'''
# context for 013: hit-rate that divided by zero on no requests
RATE_BUGGY = '''\
def hit_rate(hits, requests):
    return round(hits / requests, 2)
'''
RATE_REF = '''\
def hit_rate(hits, requests):
    if requests == 0:
        return 0.0
    return round(hits / requests, 2)
'''

# 014 default-for-missing-key (class) --------------------------------------
CONFIG_BUGGY = '''\
class Config:
    def __init__(self):
        self.data = {'theme': 'dark', 'lang': 'en'}

    def get_setting(self, key):
        return self.data[key]
'''
CONFIG_REF = '''\
class Config:
    def __init__(self):
        self.data = {'theme': 'dark', 'lang': 'en'}

    def get_setting(self, key):
        return self.data.get(key, 'default')
'''
CONFIG_PRE = '''\
def run(queries):
    c = Config()
    return [c.get_setting(q) for q in queries]
'''
# context for 014: feature-flag map that raised on an unknown flag
FLAG_BUGGY = '''\
class Flags:
    def __init__(self, enabled):
        self.enabled = enabled

    def is_on(self, name):
        return self.enabled[name]
'''
FLAG_REF = '''\
class Flags:
    def __init__(self, enabled):
        self.enabled = enabled

    def is_on(self, name):
        return self.enabled.get(name, False)
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
    dict(
        instance_id="openworld-contextbench-003-floor-at-zero",
        pattern="floor a decreasing value at zero with max(0, ...)",
        task=_task(
            "003-floor-at-zero", "battery",
            "Battery charge goes negative. drain() subtracts the requested amount with "
            "no lower bound, so draining more than the remaining charge produces a "
            "negative reading. Charge should never drop below zero.",
            BATTERY_BUGGY, BATTERY_REF, BATTERY_PRE,
            f2p=[("run([60, 60])", "[40, 0]"), ("run([150])", "[0]")],
            p2p=[("run([10, 20])", "[90, 70]"), ("run([30, 30])", "[70, 40]")],
        ),
        context_history=[dict(
            module_name="countdown",
            issue="A countdown timer ticked past zero into negative seconds because "
                  "tick() subtracted with no floor. The fix clamped remaining at zero "
                  "with max(0, ...).",
            buggy_source=TIMER_BUGGY, reference_source=TIMER_REF,
        )],
    ),
    dict(
        instance_id="openworld-contextbench-004-guard-empty-divide",
        pattern="guard against an empty collection before dividing, returning a default",
        task=_task(
            "004-guard-empty-divide", "stats_mean",
            "average(xs) crashes with ZeroDivisionError when given an empty list "
            "because it divides the sum by len(xs) unconditionally. An empty input "
            "should return 0.0 instead of raising.",
            AVG_BUGGY, AVG_REF, "",
            f2p=[("average([])", "0.0"), ("average(())", "0.0")],
            p2p=[("average([2, 4])", "3.0"), ("average([10])", "10.0")],
        ),
        context_history=[dict(
            module_name="physics",
            issue="density(mass, volume) raised ZeroDivisionError when volume was 0. "
                  "The fix returned 0.0 when the divisor was zero before dividing.",
            buggy_source=DENSITY_BUGGY, reference_source=DENSITY_REF,
        )],
    ),
    dict(
        instance_id="openworld-contextbench-005-default-missing-key",
        pattern="look up a key with dict.get(key, default) instead of indexing",
        task=_task(
            "005-default-missing-key", "catalog",
            "price_of(catalog, item) raises KeyError when the item isn't in the "
            "catalog because it indexes the dict directly. A missing item should "
            "yield a price of 0, not an exception.",
            PRICE_BUGGY, PRICE_REF, "",
            f2p=[("price_of({'a': 5}, 'b')", "0"), ("price_of({}, 'x')", "0")],
            p2p=[("price_of({'a': 5}, 'a')", "5"),
                 ("price_of({'a': 5, 'b': 7}, 'b')", "7")],
        ),
        context_history=[dict(
            module_name="http_headers",
            issue="header_value(headers, name) raised KeyError when a header was "
                  "absent. The fix used headers.get(name, '') to return an empty "
                  "string default instead of indexing.",
            buggy_source=HEADER_BUGGY, reference_source=HEADER_REF,
        )],
    ),
    dict(
        instance_id="openworld-contextbench-006-flush-final",
        pattern="flush the partially-accumulated final group after the loop ends",
        task=_task(
            "006-flush-final", "chunker",
            "chunk_sums(xs, size) drops the trailing partial chunk. It only appends a "
            "chunk total when exactly `size` items have been gathered, so leftover "
            "items at the end are silently lost. The final partial group must be "
            "flushed too.",
            CHUNK_BUGGY, CHUNK_REF, "",
            f2p=[("chunk_sums([1, 2, 3, 4, 5], 2)", "[3, 7, 5]"),
                 ("chunk_sums([1, 1, 1], 2)", "[2, 1]")],
            p2p=[("chunk_sums([1, 2, 3, 4], 2)", "[3, 7]"),
                 ("chunk_sums([5, 5], 1)", "[5, 5]")],
        ),
        context_history=[dict(
            module_name="tokenizer",
            issue="split_lines(text) dropped the last segment when the text didn't end "
                  "with a delimiter. The fix appended the leftover buffer after the "
                  "loop if it was non-empty.",
            buggy_source=SPLIT_BUGGY, reference_source=SPLIT_REF,
        )],
    ),
    dict(
        instance_id="openworld-contextbench-007-clamp-both-ends",
        pattern="clamp a value to both a lower and an upper bound",
        task=_task(
            "007-clamp-both-ends", "volume",
            "The volume control only clamps the low end. adjust() floors the level at "
            "0 but never caps it, so large positive adjustments push the level above "
            "the maximum of 10. The level must stay within 0..10 inclusive.",
            VOLUME_BUGGY, VOLUME_REF, VOLUME_PRE,
            f2p=[("run([20])", "[10]"), ("run([8, 5])", "[10, 10]")],
            p2p=[("run([2, -1])", "[7, 6]"), ("run([0])", "[5]")],
        ),
        context_history=[dict(
            module_name="brightness",
            issue="A brightness slider clamped at 0 but could exceed 100 on large "
                  "steps. The fix wrapped the value in min(100, max(0, ...)) to bound "
                  "both ends.",
            buggy_source=BRIGHT_BUGGY, reference_source=BRIGHT_REF,
        )],
    ),
    dict(
        instance_id="openworld-contextbench-008-normalize-before-compare",
        pattern="normalize (strip + lowercase) values before comparing/deduping",
        task=_task(
            "008-normalize-before-compare", "tagging",
            "count_unique_tags(tags) overcounts. It treats 'Red', ' red', and 'RED' as "
            "three different tags because it compares the raw strings. Tags should be "
            "stripped of whitespace and lowercased before counting distinct values.",
            TAGS_BUGGY, TAGS_REF, "",
            f2p=[("count_unique_tags(['Red', ' red', 'RED'])", "1"),
                 ("count_unique_tags([' a', 'a '])", "1")],
            p2p=[("count_unique_tags(['a', 'b'])", "2"),
                 ("count_unique_tags(['x'])", "1")],
        ),
        context_history=[dict(
            module_name="user_registry",
            issue="distinct_users(names) double-counted accounts that differed only by "
                  "case or surrounding spaces. The fix normalized each name with "
                  ".strip().lower() before building the set.",
            buggy_source=USER_BUGGY, reference_source=USER_REF,
        )],
    ),
    dict(
        instance_id="openworld-contextbench-009-dedupe-preserve-order",
        pattern="dedupe a sequence while preserving first-seen order",
        task=_task(
            "009-dedupe-preserve-order", "dedup",
            "dedupe(xs) doesn't actually remove duplicates — it returns the list "
            "unchanged. It should drop repeated elements while keeping the order of "
            "first appearance.",
            DEDUPE_BUGGY, DEDUPE_REF, "",
            f2p=[("dedupe([1, 1, 2, 1, 3])", "[1, 2, 3]"),
                 ("dedupe([5, 5, 5])", "[5]")],
            p2p=[("dedupe([1, 2, 3])", "[1, 2, 3]"), ("dedupe([])", "[]")],
        ),
        context_history=[dict(
            module_name="history",
            issue="recent(events) kept consecutive duplicate entries. The fix tracked "
                  "seen items in a set and appended each only the first time, "
                  "preserving order.",
            buggy_source=HIST_BUGGY, reference_source=HIST_REF,
        )],
    ),
    dict(
        instance_id="openworld-contextbench-010-early-return-first-match",
        pattern="return on the first match instead of overwriting until the last",
        task=_task(
            "010-early-return-first-match", "scan",
            "first_negative(xs) returns the LAST negative number, not the first. It "
            "keeps overwriting a variable for every negative it sees instead of "
            "returning as soon as it finds one. It should return the first negative "
            "(or None if there are none).",
            FIRSTNEG_BUGGY, FIRSTNEG_REF, "",
            f2p=[("first_negative([3, -2, -9])", "-2"),
                 ("first_negative([-1, -7])", "-1")],
            p2p=[("first_negative([1, 2, 3])", "None"),
                 ("first_negative([4, 5])", "None")],
        ),
        context_history=[dict(
            module_name="search",
            issue="index_of(xs, target) reported the last matching index because it "
                  "kept assigning on every hit. The fix returned immediately on the "
                  "first match.",
            buggy_source=FIND_BUGGY, reference_source=FIND_REF,
        )],
    ),
    dict(
        instance_id="openworld-contextbench-011-guard-empty-index",
        pattern="guard against an empty sequence before indexing, returning a default",
        task=_task(
            "011-guard-empty-index", "feed",
            "headline(items) raises IndexError on an empty input because it returns "
            "items[0] unconditionally. When there are no items it should return the "
            "string 'none' instead of crashing.",
            HEADLINE_BUGGY, HEADLINE_REF, "",
            f2p=[("headline([])", "'none'"), ("headline(())", "'none'")],
            p2p=[("headline(['a', 'b'])", "'a'"), ("headline(['x'])", "'x'")],
        ),
        context_history=[dict(
            module_name="stack",
            issue="peek(stack) raised IndexError when the stack was empty because it "
                  "indexed stack[-1] directly. The fix returned None when the stack "
                  "was empty before indexing.",
            buggy_source=PEEK_BUGGY, reference_source=PEEK_REF,
        )],
    ),
    dict(
        instance_id="openworld-contextbench-012-reject-over-capacity",
        pattern="reject an update that would exceed a maximum capacity",
        task=_task(
            "012-reject-over-capacity", "theater",
            "Ticket sales can exceed the theater's capacity. book() adds the requested "
            "seats with no capacity check, so a booking that would push sold seats past "
            "the cap is accepted. Such a booking should be rejected, leaving the sold "
            "count unchanged.",
            THEATER_BUGGY, THEATER_REF, THEATER_PRE,
            f2p=[("run([8, 5])", "[8, 8]"), ("run([12])", "[0]")],
            p2p=[("run([3, 4])", "[3, 7]"), ("run([10])", "[10]")],
        ),
        context_history=[dict(
            module_name="parking",
            issue="A parking lot let more cars in than it had spaces because enter() "
                  "never checked capacity. The fix rejected any entry that would "
                  "exceed the available spaces.",
            buggy_source=LOT_BUGGY, reference_source=LOT_REF,
        )],
    ),
    dict(
        instance_id="openworld-contextbench-013-guard-zero-percent",
        pattern="guard against a zero divisor before computing a ratio",
        task=_task(
            "013-guard-zero-percent", "usage",
            "percent_full(used, total) raises ZeroDivisionError when total is 0. It "
            "should treat an empty/zero-capacity total as 0.0% full rather than "
            "dividing by zero.",
            PCT_BUGGY, PCT_REF, "",
            f2p=[("percent_full(0, 0)", "0.0"), ("percent_full(5, 0)", "0.0")],
            p2p=[("percent_full(1, 2)", "50.0"), ("percent_full(3, 4)", "75.0")],
        ),
        context_history=[dict(
            module_name="cache_metrics",
            issue="hit_rate(hits, requests) divided by zero when there were no "
                  "requests yet. The fix returned 0.0 when requests was zero before "
                  "computing the ratio.",
            buggy_source=RATE_BUGGY, reference_source=RATE_REF,
        )],
    ),
    dict(
        instance_id="openworld-contextbench-014-default-missing-setting",
        pattern="look up a key with dict.get(key, default) instead of indexing (class)",
        task=_task(
            "014-default-missing-setting", "config",
            "Config.get_setting(key) raises KeyError for any key that isn't one of the "
            "known settings because it indexes the backing dict directly. An unknown "
            "key should return the string 'default' instead of raising.",
            CONFIG_BUGGY, CONFIG_REF, CONFIG_PRE,
            f2p=[("run(['missing'])", "['default']"),
                 ("run(['theme', 'x'])", "['dark', 'default']")],
            p2p=[("run(['theme'])", "['dark']"), ("run(['lang'])", "['en']")],
        ),
        context_history=[dict(
            module_name="feature_flags",
            issue="Flags.is_on(name) raised KeyError for an unknown flag because it "
                  "indexed the enabled map. The fix used enabled.get(name, False) so "
                  "unknown flags default to off.",
            buggy_source=FLAG_BUGGY, reference_source=FLAG_REF,
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
