"""Builder for the OpenWorld-SWE-bench dataset (writes tasks.jsonl).

Each instance is a buggy Python module + a natural-language issue + two hidden
test suites (fail_to_pass, pass_to_pass) + an explicit world spec. The `world`
block is generated generically from the instance. Run:

    python datasets/openworld-swebench/build_tasks.py

then validate with `pytest tests/test_swebench.py`. The instances here are the
source of truth; tasks.jsonl is the generated artifact the harness loads.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tasks.jsonl"


# ---------------------------------------------------------------------------
# 1. LRU cache — get() must update recency (cross-method state bug)
# ---------------------------------------------------------------------------
LRU_BUGGY = '''\
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = {}
        self.order = []

    def get(self, key):
        if key in self.data:
            return self.data[key]
        return -1

    def put(self, key, value):
        if key in self.data:
            self.order.remove(key)
        elif len(self.data) >= self.capacity:
            oldest = self.order.pop(0)
            del self.data[oldest]
        self.data[key] = value
        self.order.append(key)
'''

LRU_REF = '''\
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = {}
        self.order = []

    def get(self, key):
        if key in self.data:
            self.order.remove(key)
            self.order.append(key)
            return self.data[key]
        return -1

    def put(self, key, value):
        if key in self.data:
            self.order.remove(key)
        elif len(self.data) >= self.capacity:
            oldest = self.order.pop(0)
            del self.data[oldest]
        self.data[key] = value
        self.order.append(key)
'''

LRU_PREAMBLE = '''\
def run(ops):
    cache = LRUCache(2)
    out = []
    for op in ops:
        if op[0] == 'put':
            cache.put(op[1], op[2])
            out.append(None)
        else:
            out.append(cache.get(op[1]))
    return out
'''


# ---------------------------------------------------------------------------
# 2. Bank ledger — withdraw must reject overdraft
# ---------------------------------------------------------------------------
BANK_BUGGY = '''\
class Account:
    def __init__(self):
        self.balance = 0
        self.history = []

    def deposit(self, amount):
        self.balance += amount
        self.history.append(('deposit', amount))
        return self.balance

    def withdraw(self, amount):
        self.balance -= amount
        self.history.append(('withdraw', amount))
        return self.balance

    def entries(self):
        return len(self.history)
'''

BANK_REF = '''\
class Account:
    def __init__(self):
        self.balance = 0
        self.history = []

    def deposit(self, amount):
        self.balance += amount
        self.history.append(('deposit', amount))
        return self.balance

    def withdraw(self, amount):
        if amount > self.balance:
            return self.balance
        self.balance -= amount
        self.history.append(('withdraw', amount))
        return self.balance

    def entries(self):
        return len(self.history)
'''

BANK_PREAMBLE = '''\
def run(ops):
    acc = Account()
    out = []
    for op in ops:
        if op[0] == 'deposit':
            out.append(acc.deposit(op[1]))
        elif op[0] == 'withdraw':
            out.append(acc.withdraw(op[1]))
        else:
            out.append(acc.entries())
    return out
'''


# ---------------------------------------------------------------------------
# 3. Interval merge — must sort before merging
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 4. Token-bucket rate limiter — refill must cap at capacity
# ---------------------------------------------------------------------------
RL_BUGGY = '''\
class RateLimiter:
    def __init__(self, capacity, refill):
        self.capacity = capacity
        self.refill = refill
        self.tokens = capacity

    def tick(self):
        self.tokens = self.tokens + self.refill

    def allow(self):
        if self.tokens > 0:
            self.tokens -= 1
            return True
        return False
'''

RL_REF = '''\
class RateLimiter:
    def __init__(self, capacity, refill):
        self.capacity = capacity
        self.refill = refill
        self.tokens = capacity

    def tick(self):
        self.tokens = min(self.capacity, self.tokens + self.refill)

    def allow(self):
        if self.tokens > 0:
            self.tokens -= 1
            return True
        return False
'''

RL_PREAMBLE = '''\
def run(capacity, refill, ops):
    rl = RateLimiter(capacity, refill)
    out = []
    for op in ops:
        if op == 'tick':
            rl.tick()
            out.append(rl.tokens)
        else:
            out.append(rl.allow())
    return out
'''


# ---------------------------------------------------------------------------
# 5. Tokenizer — must flush the final token
# ---------------------------------------------------------------------------
TOK_BUGGY = '''\
def tokenize(s):
    tokens = []
    current = ''
    for ch in s:
        if ch in ' ,':
            if current:
                tokens.append(current)
                current = ''
        else:
            current += ch
    return tokens
'''

TOK_REF = '''\
def tokenize(s):
    tokens = []
    current = ''
    for ch in s:
        if ch in ' ,':
            if current:
                tokens.append(current)
                current = ''
        else:
            current += ch
    if current:
        tokens.append(current)
    return tokens
'''


# ---------------------------------------------------------------------------
# 6. Inventory — ship must reject overselling (but allow shipping exact stock)
# ---------------------------------------------------------------------------
INV_BUGGY = '''\
class Inventory:
    def __init__(self):
        self.stock = {}

    def receive(self, item, qty):
        self.stock[item] = self.stock.get(item, 0) + qty
        return self.stock[item]

    def ship(self, item, qty):
        self.stock[item] = self.stock.get(item, 0) - qty
        return self.stock[item]

    def level(self, item):
        return self.stock.get(item, 0)
'''

INV_REF = '''\
class Inventory:
    def __init__(self):
        self.stock = {}

    def receive(self, item, qty):
        self.stock[item] = self.stock.get(item, 0) + qty
        return self.stock[item]

    def ship(self, item, qty):
        available = self.stock.get(item, 0)
        if qty > available:
            return available
        self.stock[item] = available - qty
        return self.stock[item]

    def level(self, item):
        return self.stock.get(item, 0)
'''

INV_PREAMBLE = '''\
def run(ops):
    inv = Inventory()
    out = []
    for op in ops:
        if op[0] == 'receive':
            out.append(inv.receive(op[1], op[2]))
        elif op[0] == 'ship':
            out.append(inv.ship(op[1], op[2]))
        else:
            out.append(inv.level(op[1]))
    return out
'''


RAW = [
    dict(
        slug="lru-cache-recency", module_name="lru_cache",
        issue=(
            "Our LRU cache evicts the wrong entry. After I read a key with get(), "
            "a later put() that triggers eviction still throws out the key I just "
            "read instead of the genuinely least-recently-used one. Reads should "
            "count as recent use."
        ),
        buggy=LRU_BUGGY, ref=LRU_REF, preamble=LRU_PREAMBLE,
        f2p=[
            ("run([('put',1,1),('put',2,2),('get',1),('put',3,3),('get',2),('get',3),('get',1)])",
             "[None, None, 1, None, -1, 3, 1]"),
        ],
        p2p=[
            ("run([('put',1,1),('get',1)])", "[None, 1]"),
            ("run([('get',5)])", "[-1]"),
            ("run([('put',1,1),('put',1,9),('get',1)])", "[None, None, 9]"),
        ],
    ),
    dict(
        slug="bank-overdraft", module_name="bank",
        issue=(
            "Accounts can go negative. withdraw() happily takes more than the "
            "balance and records the withdrawal. A withdrawal larger than the "
            "current balance should be rejected: leave the balance unchanged and "
            "don't add it to the history."
        ),
        buggy=BANK_BUGGY, ref=BANK_REF, preamble=BANK_PREAMBLE,
        f2p=[
            ("run([('deposit',100),('withdraw',150)])", "[100, 100]"),
            ("run([('deposit',100),('withdraw',150),('withdraw',40),('entries',)])",
             "[100, 100, 60, 2]"),
        ],
        p2p=[
            ("run([('deposit',50),('withdraw',20)])", "[50, 30]"),
            ("run([('deposit',10),('deposit',5),('entries',)])", "[10, 15, 2]"),
        ],
    ),
    dict(
        slug="interval-merge-unsorted", module_name="intervals",
        issue=(
            "merge() returns wrong results when the input intervals aren't already "
            "sorted by start. For example merging [(1,3),(8,10),(2,6),(15,18)] "
            "leaves (1,3) and (2,6) unmerged. It should handle intervals in any "
            "order."
        ),
        buggy=MERGE_BUGGY, ref=MERGE_REF, preamble="",
        f2p=[
            ("merge([(1,3),(8,10),(2,6),(15,18)])", "[(1, 6), (8, 10), (15, 18)]"),
            ("merge([(5,6),(1,2)])", "[(1, 2), (5, 6)]"),
        ],
        p2p=[
            ("merge([(1,3),(2,6),(8,10)])", "[(1, 6), (8, 10)]"),
            ("merge([(1,4)])", "[(1, 4)]"),
            ("merge([(1,2),(3,4)])", "[(1, 2), (3, 4)]"),
        ],
    ),
    dict(
        slug="rate-limiter-overfill", module_name="rate_limiter",
        issue=(
            "The token bucket overfills. Calling tick() keeps adding refill tokens "
            "with no ceiling, so the limiter ends up with far more tokens than its "
            "capacity and stops limiting anything. Refilling must never exceed the "
            "configured capacity."
        ),
        buggy=RL_BUGGY, ref=RL_REF, preamble=RL_PREAMBLE,
        f2p=[
            ("run(2, 5, ['tick'])", "[2]"),
            ("run(3, 2, ['tick', 'tick'])", "[3, 3]"),
        ],
        p2p=[
            ("run(2, 1, ['allow', 'allow', 'allow'])", "[True, True, False]"),
            ("run(1, 1, ['allow', 'tick', 'allow'])", "[True, 1, True]"),
        ],
    ),
    dict(
        slug="tokenizer-drops-last", module_name="tokenizer",
        issue=(
            "tokenize() loses the final token when the string doesn't end in a "
            "separator. tokenize('a,b,c') returns ['a','b'] — the 'c' is dropped. "
            "Splitting on spaces and commas should keep the last token too."
        ),
        buggy=TOK_BUGGY, ref=TOK_REF, preamble="",
        f2p=[
            ("tokenize('a,b,c')", "['a', 'b', 'c']"),
            ("tokenize('hello world')", "['hello', 'world']"),
        ],
        p2p=[
            ("tokenize('a,b,')", "['a', 'b']"),
            ("tokenize('')", "[]"),
            ("tokenize('x ')", "['x']"),
        ],
    ),
    dict(
        slug="inventory-oversell", module_name="inventory",
        issue=(
            "Inventory can go negative. ship() subtracts the requested quantity "
            "even when we don't have enough on hand (and even for items we've "
            "never stocked). A shipment larger than the available level should be "
            "rejected and leave the level unchanged. Shipping exactly the "
            "available amount must still work."
        ),
        buggy=INV_BUGGY, ref=INV_REF, preamble=INV_PREAMBLE,
        f2p=[
            ("run([('receive','apple',5),('ship','apple',8),('level','apple')])", "[5, 5, 5]"),
            ("run([('ship','ghost',1)])", "[0]"),
        ],
        p2p=[
            ("run([('receive','apple',5),('ship','apple',5),('level','apple')])", "[5, 0, 0]"),
            ("run([('receive','pear',3),('ship','pear',1),('level','pear')])", "[3, 2, 2]"),
        ],
    ),
]


def _world(slug: str, module_name: str, buggy: str) -> dict:
    return {
        "name": f"swebench:{slug}",
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
            "instance_id": f"openworld-swebench-{i:03d}-{r['slug']}",
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
