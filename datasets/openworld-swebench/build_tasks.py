#!/usr/bin/env python3
"""Build tasks.jsonl for openworld-swebench.

Instances are defined here as plain dicts; this script computes each world's
initial state by actually running the hidden suites on the buggy source,
validates every instance (reference solves; buggy fails all fail_to_pass and
passes all pass_to_pass), and writes the JSONL artifact. Re-run after editing:

    python datasets/openworld-swebench/build_tasks.py
"""

import json
import sys
import textwrap
from pathlib import Path

from openworld.swebench import SWEBenchInstance, initial_world_state, run_instance_tests

OUT = Path(__file__).resolve().parent / "tasks.jsonl"

DEFAULT_RULES = [
    "submit_patch(params={'source': ...}) replaces the module and runs both hidden suites bit-exactly in a sandbox.",
    "The instance is solved when zero tests fail in both the fail_to_pass and pass_to_pass suites.",
    "Once solved, further actions are no-ops.",
    "Every submit_patch increments attempts by exactly one.",
]

DEFAULT_INVARIANTS = [
    "attempts never decreases",
    "solved implies zero failing tests in both suites",
    "state always carries the most recently submitted source",
]


def D(s):
    """Dedent an inline triple-quoted block and drop the leading newline."""
    return textwrap.dedent(s).lstrip("\n")


INSTANCES = [
    dict(
        instance_id="openworld-swebench-001-rate-limiter-window",
        module_name="rate_limiter",
        issue=(
            "Our RateLimiter(max_calls=3, window=10) lets a 4th call through.\n"
            "Repro: allow() at t=0, 1, 2 are accepted (good), but allow() at t=5\n"
            "is ALSO accepted even though 4 calls now sit inside one 10-second\n"
            "window. Denied calls also seem to mess up later decisions somehow."
        ),
        buggy_source=(
            "class RateLimiter:\n"
            "    def __init__(self, max_calls, window):\n"
            "        self.max_calls = max_calls\n"
            "        self.window = window\n"
            "        self.calls = []\n"
            "\n"
            "    def _evict(self, now):\n"
            "        self.calls = [t for t in self.calls if now - t < self.window]\n"
            "\n"
            "    def allow(self, now):\n"
            "        self._evict(now)\n"
            "        self.calls.append(now)\n"
            "        return len(self.calls) < self.max_calls\n"
        ),
        reference_source=(
            "class RateLimiter:\n"
            "    def __init__(self, max_calls, window):\n"
            "        self.max_calls = max_calls\n"
            "        self.window = window\n"
            "        self.calls = []\n"
            "\n"
            "    def _evict(self, now):\n"
            "        self.calls = [t for t in self.calls if now - t < self.window]\n"
            "\n"
            "    def allow(self, now):\n"
            "        self._evict(now)\n"
            "        if len(self.calls) >= self.max_calls:\n"
            "            return False\n"
            "        self.calls.append(now)\n"
            "        return True\n"
        ),
        test_preamble=(
            "def decisions(max_calls, window, times):\n"
            "    rl = RateLimiter(max_calls, window)\n"
            "    return [rl.allow(t) for t in times]\n"
        ),
        fail_to_pass=[
            ("decisions(3, 10, [0, 1, 2, 5])", "[True, True, True, False]"),
            ("decisions(1, 10, [0, 5, 11])", "[True, False, True]"),
            ("decisions(2, 10, [0, 1, 2, 12])", "[True, True, False, True]"),
        ],
        pass_to_pass=[
            ("decisions(3, 10, [0])", "[True]"),
            ("decisions(2, 5, [0, 10, 20])", "[True, True, True]"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-002-pagination-last-page",
        module_name="pagination",
        issue=(
            "Items vanish from the last page of results.\n"
            "With 7 items and per_page=3, page_count says 2 (should be 3) and\n"
            "the 7th item is never returned by get_page. Asking for the page\n"
            "after the last one raises IndexError instead of returning [].\n"
        ),
        buggy_source=(
            "def page_count(total, per_page):\n"
            "    return total // per_page\n"
            "\n"
            "def get_page(items, page, per_page):\n"
            "    if page < 1:\n"
            "        raise ValueError('pages are 1-indexed')\n"
            "    start = (page - 1) * per_page\n"
            "    return [items[i] for i in range(start, start + per_page)]\n"
            "\n"
            "def page_summary(items, per_page):\n"
            "    pages = page_count(len(items), per_page)\n"
            "    return [len(get_page(items, p, per_page)) for p in range(1, pages + 1)]\n"
        ),
        reference_source=(
            "def page_count(total, per_page):\n"
            "    return (total + per_page - 1) // per_page\n"
            "\n"
            "def get_page(items, page, per_page):\n"
            "    if page < 1:\n"
            "        raise ValueError('pages are 1-indexed')\n"
            "    start = (page - 1) * per_page\n"
            "    return items[start:start + per_page]\n"
            "\n"
            "def page_summary(items, per_page):\n"
            "    pages = page_count(len(items), per_page)\n"
            "    return [len(get_page(items, p, per_page)) for p in range(1, pages + 1)]\n"
        ),
        test_preamble="",
        fail_to_pass=[
            ("page_count(7, 3)", "3"),
            ("get_page([1, 2, 3, 4, 5, 6, 7], 3, 3)", "[7]"),
            ("get_page([1, 2], 2, 3)", "[]"),
            ("page_summary([1, 2, 3, 4, 5, 6, 7], 3)", "[3, 3, 1]"),
        ],
        pass_to_pass=[
            ("page_count(6, 3)", "2"),
            ("get_page([1, 2, 3, 4], 1, 2)", "[1, 2]"),
            ("page_summary([1, 2, 3, 4], 2)", "[2, 2]"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-003-lru-cache-eviction",
        module_name="lru_cache",
        issue=D('''
            Our LRUCache evicts the wrong entry under read traffic.
            Repro with capacity 2: put('a', 1), put('b', 2), get('a') returns 1
            (a hit), then put('c', 3). Now get('a') returns -1 -- the cache
            threw away the entry we read a moment ago and kept 'b' instead.
            Write-only workloads behave fine; it only goes wrong once reads
            are mixed in before an eviction.
        '''),
        buggy_source=D('''
            class LRUCache:
                """Fixed-capacity cache that evicts the least recently used key."""

                def __init__(self, capacity):
                    if capacity < 1:
                        raise ValueError('capacity must be at least 1')
                    self.capacity = capacity
                    self.data = {}
                    self.order = []  # least recently used first

                def __len__(self):
                    return len(self.data)

                def get(self, key):
                    """Return the cached value, or -1 on a miss."""
                    if key not in self.data:
                        return -1
                    return self.data[key]

                def put(self, key, value):
                    """Insert or update a key, evicting the LRU key if full."""
                    if key in self.data:
                        self.data[key] = value
                        self.order.remove(key)
                        self.order.append(key)
                        return
                    if len(self.data) >= self.capacity:
                        oldest = self.order.pop(0)
                        del self.data[oldest]
                    self.data[key] = value
                    self.order.append(key)
        '''),
        reference_source=D('''
            class LRUCache:
                """Fixed-capacity cache that evicts the least recently used key."""

                def __init__(self, capacity):
                    if capacity < 1:
                        raise ValueError('capacity must be at least 1')
                    self.capacity = capacity
                    self.data = {}
                    self.order = []  # least recently used first

                def __len__(self):
                    return len(self.data)

                def get(self, key):
                    """Return the cached value, or -1 on a miss."""
                    if key not in self.data:
                        return -1
                    self.order.remove(key)
                    self.order.append(key)
                    return self.data[key]

                def put(self, key, value):
                    """Insert or update a key, evicting the LRU key if full."""
                    if key in self.data:
                        self.data[key] = value
                        self.order.remove(key)
                        self.order.append(key)
                        return
                    if len(self.data) >= self.capacity:
                        oldest = self.order.pop(0)
                        del self.data[oldest]
                    self.data[key] = value
                    self.order.append(key)
        '''),
        test_preamble=D('''
            def run_ops(capacity, ops):
                cache = LRUCache(capacity)
                out = []
                for op in ops:
                    if op[0] == 'put':
                        cache.put(op[1], op[2])
                    else:
                        out.append(cache.get(op[1]))
                return out
        '''),
        fail_to_pass=[
            (
                "run_ops(2, [('put', 'a', 1), ('put', 'b', 2), ('get', 'a'), "
                "('put', 'c', 3), ('get', 'a'), ('get', 'b')])",
                "[1, 1, -1]",
            ),
            (
                "run_ops(2, [('put', 'a', 1), ('put', 'b', 2), ('get', 'a'), "
                "('get', 'a'), ('put', 'c', 3), ('get', 'b')])",
                "[1, 1, -1]",
            ),
            (
                "run_ops(3, [('put', 'x', 1), ('put', 'y', 2), ('put', 'z', 3), "
                "('get', 'x'), ('get', 'y'), ('put', 'w', 4), ('get', 'z'), ('get', 'x')])",
                "[1, 2, -1, 1]",
            ),
        ],
        pass_to_pass=[
            (
                "run_ops(1, [('put', 'a', 1), ('get', 'a'), ('put', 'b', 2), "
                "('get', 'a'), ('get', 'b')])",
                "[1, -1, 2]",
            ),
            (
                "run_ops(2, [('put', 'a', 1), ('put', 'b', 2), ('get', 'a'), ('get', 'b')])",
                "[1, 2]",
            ),
            ("run_ops(2, [('put', 'k', 1), ('put', 'k', 9), ('get', 'k')])", "[9]"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-004-text-wrap-long-word",
        module_name="textwrapper",
        issue=D('''
            wrap() silently loses text. wrap('a extraordinarily b', 6) comes
            back without 'extraordinarily' anywhere in the output -- the word
            just disappears. Worse, wrap('abcdefghij', 4) returns an empty
            list, so a whole paragraph can be erased. Any token longer than
            the width should be hard-broken into width-sized chunks, not
            dropped. Ordinary sentences wrap fine.
        '''),
        buggy_source=D('''
            def wrap(text, width):
                """Greedy word-wrap: returns a list of lines, each <= width chars."""
                if width < 1:
                    raise ValueError('width must be positive')
                lines = []
                line = ''
                for word in text.split():
                    if len(word) > width:
                        continue
                    if not line:
                        line = word
                    elif len(line) + 1 + len(word) <= width:
                        line = line + ' ' + word
                    else:
                        lines.append(line)
                        line = word
                if line:
                    lines.append(line)
                return lines

            def line_lengths(text, width):
                """Lengths of each wrapped line; handy for layout checks."""
                return [len(line) for line in wrap(text, width)]
        '''),
        reference_source=D('''
            def wrap(text, width):
                """Greedy word-wrap: returns a list of lines, each <= width chars."""
                if width < 1:
                    raise ValueError('width must be positive')
                pieces = []
                for word in text.split():
                    while len(word) > width:
                        pieces.append(word[:width])
                        word = word[width:]
                    if word:
                        pieces.append(word)
                lines = []
                line = ''
                for word in pieces:
                    if not line:
                        line = word
                    elif len(line) + 1 + len(word) <= width:
                        line = line + ' ' + word
                    else:
                        lines.append(line)
                        line = word
                if line:
                    lines.append(line)
                return lines

            def line_lengths(text, width):
                """Lengths of each wrapped line; handy for layout checks."""
                return [len(line) for line in wrap(text, width)]
        '''),
        test_preamble="",
        fail_to_pass=[
            ("wrap('a extraordinarily b', 6)", "['a', 'extrao', 'rdinar', 'ily b']"),
            ("wrap('hello worldwide', 5)", "['hello', 'world', 'wide']"),
            ("line_lengths('abcdefghij', 4)", "[4, 4, 2]"),
        ],
        pass_to_pass=[
            ("wrap('the quick brown fox', 10)", "['the quick', 'brown fox']"),
            ("wrap('', 5)", "[]"),
            ("line_lengths('a bb ccc', 3)", "[1, 2, 3]"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-005-interval-merge-touching",
        module_name="intervals",
        issue=D('''
            merge() leaves back-to-back intervals split. merge([[1, 2], [2, 3]])
            returns both intervals untouched instead of one block [[1, 3]].
            Our booking UI treats [1, 2] and [2, 3] as a single continuous
            reservation, so downstream consumers now render a phantom gap at
            the shared endpoint and the merged-block count is inflated.
            Genuinely overlapping and nested intervals merge correctly.
        '''),
        buggy_source=D('''
            def merge(intervals):
                """Merge a list of [start, end] intervals into sorted disjoint blocks."""
                if not intervals:
                    return []
                items = sorted([list(iv) for iv in intervals])
                out = [items[0]]
                for start, end in items[1:]:
                    last = out[-1]
                    if start < last[1]:
                        last[1] = max(last[1], end)
                    else:
                        out.append([start, end])
                return out

            def total_covered(intervals):
                """Total length of the line covered by the intervals."""
                return sum(end - start for start, end in merge(intervals))
        '''),
        reference_source=D('''
            def merge(intervals):
                """Merge a list of [start, end] intervals into sorted disjoint blocks."""
                if not intervals:
                    return []
                items = sorted([list(iv) for iv in intervals])
                out = [items[0]]
                for start, end in items[1:]:
                    last = out[-1]
                    if start <= last[1]:
                        last[1] = max(last[1], end)
                    else:
                        out.append([start, end])
                return out

            def total_covered(intervals):
                """Total length of the line covered by the intervals."""
                return sum(end - start for start, end in merge(intervals))
        '''),
        test_preamble="",
        fail_to_pass=[
            ("merge([[1, 2], [2, 3]])", "[[1, 3]]"),
            ("merge([[0, 1], [1, 2], [2, 3]])", "[[0, 3]]"),
            ("len(merge([[5, 8], [8, 10], [0, 2]]))", "2"),
        ],
        pass_to_pass=[
            ("merge([[1, 2], [4, 6]])", "[[1, 2], [4, 6]]"),
            ("merge([[1, 10], [2, 3]])", "[[1, 10]]"),
            ("total_covered([[0, 4], [2, 6]])", "6"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-006-bank-ledger-overdraft",
        module_name="ledger",
        issue=D('''
            Accounts are going negative. With the standard fee of 2 and a
            balance of 10, withdraw(9) is accepted and the balance lands at
            -1. Per the product spec a withdrawal must only be allowed when
            the amount PLUS the fee is covered; the only exception is the
            exact-balance close-out (withdrawing the full balance, fee
            waived), and that path still works. After an overdraft the
            account is wedged and later valid withdrawals get rejected.
        '''),
        buggy_source=D('''
            class Ledger:
                """Single-account ledger; withdrawals cost a flat fee."""

                FEE = 2

                def __init__(self, opening=0):
                    if opening < 0:
                        raise ValueError('opening balance cannot be negative')
                    self._balance = opening
                    self.rejected = 0

                def deposit(self, amount):
                    if amount <= 0:
                        raise ValueError('deposit must be positive')
                    self._balance += amount
                    return self._balance

                def withdraw(self, amount):
                    """Withdraw `amount`; returns True if accepted."""
                    if amount <= 0:
                        raise ValueError('withdrawal must be positive')
                    if amount == self._balance:
                        self._balance = 0
                        return True
                    if amount > self._balance:
                        self.rejected += 1
                        return False
                    self._balance -= amount + self.FEE
                    return True

                def balance(self):
                    return self._balance
        '''),
        reference_source=D('''
            class Ledger:
                """Single-account ledger; withdrawals cost a flat fee."""

                FEE = 2

                def __init__(self, opening=0):
                    if opening < 0:
                        raise ValueError('opening balance cannot be negative')
                    self._balance = opening
                    self.rejected = 0

                def deposit(self, amount):
                    if amount <= 0:
                        raise ValueError('deposit must be positive')
                    self._balance += amount
                    return self._balance

                def withdraw(self, amount):
                    """Withdraw `amount`; returns True if accepted."""
                    if amount <= 0:
                        raise ValueError('withdrawal must be positive')
                    if amount == self._balance:
                        self._balance = 0
                        return True
                    if amount + self.FEE > self._balance:
                        self.rejected += 1
                        return False
                    self._balance -= amount + self.FEE
                    return True

                def balance(self):
                    return self._balance
        '''),
        test_preamble=D('''
            def run_ledger(opening, withdrawals):
                led = Ledger(opening)
                results = [led.withdraw(a) for a in withdrawals]
                return (results, led.balance())
        '''),
        fail_to_pass=[
            ("run_ledger(10, [9])", "([False], 10)"),
            ("run_ledger(20, [19])", "([False], 20)"),
            ("run_ledger(5, [4, 1])", "([False, True], 2)"),
        ],
        pass_to_pass=[
            ("Ledger(0).deposit(50)", "50"),
            ("run_ledger(10, [10])", "([True], 0)"),
            ("run_ledger(10, [3])", "([True], 5)"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-007-tokenizer-quoted-strings",
        module_name="tokenizer",
        issue=D('''
            tokenize() breaks quoted phrases apart. tokenize('say "hello
            world" now') yields 4 tokens, two of which still contain stray
            double-quote characters, instead of 3 tokens with hello world
            kept together as a single token (quotes removed). count_tokens
            over-counts every input that uses quoting. Plain unquoted text,
            including runs of multiple spaces, tokenizes correctly.
        '''),
        buggy_source=D('''
            def tokenize(s):
                """Split a command line into tokens; double quotes group words."""
                return [t for t in s.split(' ') if t]

            def count_tokens(s):
                """Number of tokens in the input."""
                return len(tokenize(s))
        '''),
        reference_source=D('''
            def tokenize(s):
                """Split a command line into tokens; double quotes group words."""
                tokens = []
                current = ''
                in_quotes = False
                for ch in s:
                    if ch == '"':
                        in_quotes = not in_quotes
                    elif ch == ' ' and not in_quotes:
                        if current:
                            tokens.append(current)
                        current = ''
                    else:
                        current += ch
                if current:
                    tokens.append(current)
                return tokens

            def count_tokens(s):
                """Number of tokens in the input."""
                return len(tokenize(s))
        '''),
        test_preamble="",
        fail_to_pass=[
            ("tokenize('say \"hello world\" now')", "['say', 'hello world', 'now']"),
            ("count_tokens('a \"b c\" d')", "3"),
            ("tokenize('\"one two three\"')", "['one two three']"),
        ],
        pass_to_pass=[
            ("tokenize('alpha beta')", "['alpha', 'beta']"),
            ("tokenize('a  b   c')", "['a', 'b', 'c']"),
            ("count_tokens('')", "0"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-008-graph-bfs-distance",
        module_name="graph",
        issue=D('''
            shortest_len returns paths that are too long. On a graph where
            node 1 reaches node 5 both directly through 6 (two hops) and the
            long way around through 2-3-4 (four hops), shortest_len(edges, 1, 5)
            answers 4. It seems to find *a* path, not the shortest one --
            which path you get depends on the order the edges were listed.
            Trivial cases are fine: same start and goal gives 0, and
            disconnected nodes give -1. reachable() also looks correct.
        '''),
        buggy_source=D('''
            def _adjacency(edges):
                """Undirected adjacency map, neighbor lists in edge order."""
                adj = {}
                for a, b in edges:
                    adj.setdefault(a, []).append(b)
                    adj.setdefault(b, []).append(a)
                return adj

            def shortest_len(edges, a, b):
                """Number of edges on the shortest path a..b, or -1 if unreachable."""
                if a == b:
                    return 0
                adj = _adjacency(edges)
                frontier = [(a, 0)]
                seen = {a}
                while frontier:
                    node, dist = frontier.pop()
                    for nxt in adj.get(node, []):
                        if nxt == b:
                            return dist + 1
                        if nxt not in seen:
                            seen.add(nxt)
                            frontier.append((nxt, dist + 1))
                return -1

            def reachable(edges, a):
                """Sorted list of every node reachable from a (including a)."""
                adj = _adjacency(edges)
                seen = {a}
                stack = [a]
                while stack:
                    node = stack.pop()
                    for nxt in adj.get(node, []):
                        if nxt not in seen:
                            seen.add(nxt)
                            stack.append(nxt)
                return sorted(seen)
        '''),
        reference_source=D('''
            def _adjacency(edges):
                """Undirected adjacency map, neighbor lists in edge order."""
                adj = {}
                for a, b in edges:
                    adj.setdefault(a, []).append(b)
                    adj.setdefault(b, []).append(a)
                return adj

            def shortest_len(edges, a, b):
                """Number of edges on the shortest path a..b, or -1 if unreachable."""
                if a == b:
                    return 0
                adj = _adjacency(edges)
                frontier = [(a, 0)]
                seen = {a}
                while frontier:
                    node, dist = frontier.pop(0)
                    for nxt in adj.get(node, []):
                        if nxt == b:
                            return dist + 1
                        if nxt not in seen:
                            seen.add(nxt)
                            frontier.append((nxt, dist + 1))
                return -1

            def reachable(edges, a):
                """Sorted list of every node reachable from a (including a)."""
                adj = _adjacency(edges)
                seen = {a}
                stack = [a]
                while stack:
                    node = stack.pop()
                    for nxt in adj.get(node, []):
                        if nxt not in seen:
                            seen.add(nxt)
                            stack.append(nxt)
                return sorted(seen)
        '''),
        test_preamble="",
        fail_to_pass=[
            ("shortest_len([[1, 6], [6, 5], [1, 2], [2, 3], [3, 4], [4, 5]], 1, 5)", "2"),
            ("shortest_len([[0, 9], [9, 4], [0, 1], [1, 2], [2, 3], [3, 4]], 0, 4)", "2"),
            (
                "shortest_len([[0, 1], [1, 2], [2, 9], [0, 3], [3, 4], [4, 5], "
                "[5, 6], [6, 9]], 0, 9)",
                "3",
            ),
        ],
        pass_to_pass=[
            ("shortest_len([[1, 2]], 1, 1)", "0"),
            ("shortest_len([[1, 2], [3, 4]], 1, 4)", "-1"),
            ("reachable([[1, 2], [2, 3], [5, 6]], 1)", "[1, 2, 3]"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-009-priority-queue-stability",
        module_name="pqueue",
        issue=D('''
            PQueue is not FIFO for equal priorities. Push 'a', 'b', 'c' all
            at priority 1 and pop three times: you get c, b, a -- newest
            first. Our job runner depends on equal-priority jobs executing
            in submission order, so retries now jump ahead of jobs that have
            been waiting for minutes. Queues whose items all have distinct
            priorities drain in the right order.
        '''),
        buggy_source=D('''
            class PQueue:
                """Min-priority queue; ties should pop in insertion order."""

                def __init__(self):
                    self._items = []  # (priority, item)

                def __len__(self):
                    return len(self._items)

                def push(self, item, priority):
                    self._items.append((priority, item))

                def pop(self):
                    """Remove and return the item with the smallest priority."""
                    if not self._items:
                        raise IndexError('pop from empty queue')
                    best = 0
                    for i in range(1, len(self._items)):
                        if self._items[i][0] <= self._items[best][0]:
                            best = i
                    return self._items.pop(best)[1]
        '''),
        reference_source=D('''
            class PQueue:
                """Min-priority queue; ties should pop in insertion order."""

                def __init__(self):
                    self._items = []  # (priority, item)

                def __len__(self):
                    return len(self._items)

                def push(self, item, priority):
                    self._items.append((priority, item))

                def pop(self):
                    """Remove and return the item with the smallest priority."""
                    if not self._items:
                        raise IndexError('pop from empty queue')
                    best = 0
                    for i in range(1, len(self._items)):
                        if self._items[i][0] < self._items[best][0]:
                            best = i
                    return self._items.pop(best)[1]
        '''),
        test_preamble=D('''
            def drain(pushes):
                q = PQueue()
                for item, priority in pushes:
                    q.push(item, priority)
                out = []
                while len(q):
                    out.append(q.pop())
                return out
        '''),
        fail_to_pass=[
            ("drain([('a', 1), ('b', 1), ('c', 1)])", "['a', 'b', 'c']"),
            ("drain([('x', 2), ('y', 1), ('z', 1)])", "['y', 'z', 'x']"),
            ("drain([('a', 5), ('b', 5), ('c', 1)])", "['c', 'a', 'b']"),
        ],
        pass_to_pass=[
            ("drain([('a', 3), ('b', 1), ('c', 2)])", "['b', 'c', 'a']"),
            ("drain([('solo', 7)])", "['solo']"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-010-roman-numerals-subtractive",
        module_name="roman",
        issue=D('''
            from_roman misreads subtractive numerals. from_roman('IV')
            returns 6 instead of 4, and from_roman('MCMXCIV') gives 2216
            instead of 1994. Round-trips are broken too:
            from_roman(to_roman(49)) comes back as 71. to_roman output
            itself looks right -- purely additive numerals like 'XVI' and
            'MMXIII' convert back correctly.
        '''),
        buggy_source=D('''
            _VALUES = [
                (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
                (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
                (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'),
            ]

            _DIGIT = {'I': 1, 'V': 5, 'X': 10, 'L': 50,
                      'C': 100, 'D': 500, 'M': 1000}

            def to_roman(n):
                """Roman numeral for 1 <= n <= 3999."""
                if n < 1 or n > 3999:
                    raise ValueError('out of range')
                out = ''
                for value, symbol in _VALUES:
                    while n >= value:
                        out += symbol
                        n -= value
                return out

            def from_roman(s):
                """Integer value of a roman numeral string."""
                if not s:
                    raise ValueError('empty numeral')
                total = 0
                for ch in s:
                    if ch not in _DIGIT:
                        raise ValueError('bad symbol: ' + ch)
                    total += _DIGIT[ch]
                return total
        '''),
        reference_source=D('''
            _VALUES = [
                (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
                (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
                (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'),
            ]

            _DIGIT = {'I': 1, 'V': 5, 'X': 10, 'L': 50,
                      'C': 100, 'D': 500, 'M': 1000}

            def to_roman(n):
                """Roman numeral for 1 <= n <= 3999."""
                if n < 1 or n > 3999:
                    raise ValueError('out of range')
                out = ''
                for value, symbol in _VALUES:
                    while n >= value:
                        out += symbol
                        n -= value
                return out

            def from_roman(s):
                """Integer value of a roman numeral string."""
                if not s:
                    raise ValueError('empty numeral')
                total = 0
                for i, ch in enumerate(s):
                    if ch not in _DIGIT:
                        raise ValueError('bad symbol: ' + ch)
                    value = _DIGIT[ch]
                    if i + 1 < len(s) and _DIGIT[s[i + 1]] > value:
                        total -= value
                    else:
                        total += value
                return total
        '''),
        test_preamble="",
        fail_to_pass=[
            ("from_roman('IV')", "4"),
            ("from_roman('MCMXCIV')", "1994"),
            ("from_roman(to_roman(49))", "49"),
            ("from_roman(to_roman(944))", "944"),
        ],
        pass_to_pass=[
            ("to_roman(27)", "'XXVII'"),
            ("from_roman('XVI')", "16"),
            ("from_roman('MMXIII')", "2013"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-011-date-add-days-leap",
        module_name="datemath",
        issue=D('''
            add_days lands on February 29, 1900 -- a date that does not
            exist. is_leap(1900) returns True, and adding one day to
            (1900, 2, 28) yields (1900, 2, 29) instead of (1900, 3, 1).
            The year 2000 is handled correctly, so this only explodes on
            century years like 1900 and 2100; our billing horizon now
            crosses 2100 and schedules are off by a day after February.
        '''),
        buggy_source=D('''
            def is_leap(y):
                """True when year y is a leap year in the Gregorian calendar."""
                return y % 4 == 0

            def days_in_month(y, m):
                """Number of days in month m of year y."""
                if m < 1 or m > 12:
                    raise ValueError('month out of range')
                if m == 2:
                    return 29 if is_leap(y) else 28
                if m in (4, 6, 9, 11):
                    return 30
                return 31

            def add_days(y, m, d, n):
                """Date n days after (y, m, d), as a (year, month, day) tuple."""
                if n < 0:
                    raise ValueError('n must be non-negative')
                d += n
                while d > days_in_month(y, m):
                    d -= days_in_month(y, m)
                    m += 1
                    if m > 12:
                        m = 1
                        y += 1
                return (y, m, d)
        '''),
        reference_source=D('''
            def is_leap(y):
                """True when year y is a leap year in the Gregorian calendar."""
                return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)

            def days_in_month(y, m):
                """Number of days in month m of year y."""
                if m < 1 or m > 12:
                    raise ValueError('month out of range')
                if m == 2:
                    return 29 if is_leap(y) else 28
                if m in (4, 6, 9, 11):
                    return 30
                return 31

            def add_days(y, m, d, n):
                """Date n days after (y, m, d), as a (year, month, day) tuple."""
                if n < 0:
                    raise ValueError('n must be non-negative')
                d += n
                while d > days_in_month(y, m):
                    d -= days_in_month(y, m)
                    m += 1
                    if m > 12:
                        m = 1
                        y += 1
                return (y, m, d)
        '''),
        test_preamble="",
        fail_to_pass=[
            ("is_leap(1900)", "False"),
            ("days_in_month(2100, 2)", "28"),
            ("add_days(1900, 2, 28, 1)", "(1900, 3, 1)"),
            ("add_days(2100, 2, 27, 3)", "(2100, 3, 2)"),
        ],
        pass_to_pass=[
            ("is_leap(2000)", "True"),
            ("is_leap(2023)", "False"),
            ("add_days(2024, 2, 28, 1)", "(2024, 2, 29)"),
            ("add_days(2023, 12, 31, 1)", "(2024, 1, 1)"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-012-stats-variance-bias",
        module_name="stats",
        issue=D('''
            variance() is biased low compared with every stats package we
            checked. variance([1, 5]) returns 4.0 but pandas, R, and Excel
            all report 8.0 for the sample variance of that data; stdev() is
            off by the same factor. The discrepancy shrinks as samples get
            bigger, which is why our dashboards only looked wrong for small
            cohorts. mean() agrees with everything.
        '''),
        buggy_source=D('''
            def mean(xs):
                """Arithmetic mean."""
                if not xs:
                    raise ValueError('mean of empty data')
                return sum(xs) / len(xs)

            def variance(xs):
                """Sample variance (Bessel-corrected)."""
                if len(xs) < 2:
                    raise ValueError('variance needs at least two points')
                m = mean(xs)
                return sum((x - m) ** 2 for x in xs) / len(xs)

            def stdev(xs):
                """Sample standard deviation."""
                return math.sqrt(variance(xs))
        '''),
        reference_source=D('''
            def mean(xs):
                """Arithmetic mean."""
                if not xs:
                    raise ValueError('mean of empty data')
                return sum(xs) / len(xs)

            def variance(xs):
                """Sample variance (Bessel-corrected)."""
                if len(xs) < 2:
                    raise ValueError('variance needs at least two points')
                m = mean(xs)
                return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

            def stdev(xs):
                """Sample standard deviation."""
                return math.sqrt(variance(xs))
        '''),
        test_preamble="",
        fail_to_pass=[
            ("variance([1, 5])", "8.0"),
            ("variance([2, 4, 6])", "4.0"),
            ("stdev([3, 7, 11])", "4.0"),
        ],
        pass_to_pass=[
            ("mean([1, 2, 3, 4])", "2.5"),
            ("mean([5, 5, 5])", "5.0"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-013-rpn-operand-order",
        module_name="rpn",
        issue=D('''
            Subtraction and division come out backwards in eval_rpn.
            eval_rpn(['7', '2', '-']) returns -5 instead of 5, and
            eval_rpn(['8', '4', '/']) gives 0.5 instead of 2.0. Addition
            and multiplication are fine, which is presumably why this
            slipped through. Nested expressions compound the damage:
            ['5', '1', '2', '-', '*'] should be -5 and is not.
            safe_div still correctly returns None for a zero divisor.
        '''),
        buggy_source=D('''
            def safe_div(a, b):
                """a / b, or None when b is zero."""
                if b == 0:
                    return None
                return a / b

            def eval_rpn(tokens):
                """Evaluate a reverse-polish expression of ints and + - * /."""
                stack = []
                for tok in tokens:
                    if tok in ('+', '-', '*', '/'):
                        if len(stack) < 2:
                            raise ValueError('not enough operands')
                        right = stack.pop()
                        left = stack.pop()
                        if tok == '+':
                            stack.append(left + right)
                        elif tok == '*':
                            stack.append(left * right)
                        elif tok == '-':
                            stack.append(right - left)
                        else:
                            stack.append(right / left)
                    else:
                        stack.append(int(tok))
                if len(stack) != 1:
                    raise ValueError('malformed expression')
                return stack[0]
        '''),
        reference_source=D('''
            def safe_div(a, b):
                """a / b, or None when b is zero."""
                if b == 0:
                    return None
                return a / b

            def eval_rpn(tokens):
                """Evaluate a reverse-polish expression of ints and + - * /."""
                stack = []
                for tok in tokens:
                    if tok in ('+', '-', '*', '/'):
                        if len(stack) < 2:
                            raise ValueError('not enough operands')
                        right = stack.pop()
                        left = stack.pop()
                        if tok == '+':
                            stack.append(left + right)
                        elif tok == '*':
                            stack.append(left * right)
                        elif tok == '-':
                            stack.append(left - right)
                        else:
                            stack.append(left / right)
                    else:
                        stack.append(int(tok))
                if len(stack) != 1:
                    raise ValueError('malformed expression')
                return stack[0]
        '''),
        test_preamble="",
        fail_to_pass=[
            ("eval_rpn(['7', '2', '-'])", "5"),
            ("eval_rpn(['8', '4', '/'])", "2.0"),
            ("eval_rpn(['5', '1', '2', '-', '*'])", "-5"),
            ("eval_rpn(['10', '2', '3', '-', '/'])", "-10.0"),
        ],
        pass_to_pass=[
            ("eval_rpn(['2', '3', '+'])", "5"),
            ("eval_rpn(['4', '5', '*'])", "20"),
            ("safe_div(9, 0)", "None"),
            ("safe_div(9, 3)", "3.0"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-014-cart-coupon-stacking",
        module_name="cart",
        issue=D('''
            Discounts stack when a coupon is applied twice. Customers who
            tap "apply" twice on a 10% coupon are getting roughly 19% off:
            with a 200 subtotal we charged 162 instead of 180. Applying a
            coupon is supposed to REPLACE whatever discount is active, not
            combine with it. Carts with no coupon or a single application
            total correctly.
        '''),
        buggy_source=D('''
            class Cart:
                """Shopping cart with line items and a percent-off coupon."""

                def __init__(self):
                    self._items = []
                    self._coupons = []

                def add(self, name, price, qty=1):
                    """Add a line item; price is in whole cents."""
                    if price < 0 or qty < 1:
                        raise ValueError('bad line item')
                    self._items.append((name, price, qty))

                def apply_coupon(self, pct):
                    """Activate a pct-percent discount on the whole cart."""
                    if pct < 0 or pct > 100:
                        raise ValueError('pct must be between 0 and 100')
                    self._coupons.append(pct)

                def subtotal(self):
                    return sum(price * qty for _, price, qty in self._items)

                def total(self):
                    """Cart total after the active discount, floored to cents."""
                    total = self.subtotal()
                    for pct in self._coupons:
                        total = total * (100 - pct) // 100
                    return total
        '''),
        reference_source=D('''
            class Cart:
                """Shopping cart with line items and a percent-off coupon."""

                def __init__(self):
                    self._items = []
                    self._coupons = []

                def add(self, name, price, qty=1):
                    """Add a line item; price is in whole cents."""
                    if price < 0 or qty < 1:
                        raise ValueError('bad line item')
                    self._items.append((name, price, qty))

                def apply_coupon(self, pct):
                    """Activate a pct-percent discount on the whole cart."""
                    if pct < 0 or pct > 100:
                        raise ValueError('pct must be between 0 and 100')
                    self._coupons = [pct]

                def subtotal(self):
                    return sum(price * qty for _, price, qty in self._items)

                def total(self):
                    """Cart total after the active discount, floored to cents."""
                    total = self.subtotal()
                    for pct in self._coupons:
                        total = total * (100 - pct) // 100
                    return total
        '''),
        test_preamble=D('''
            def checkout(lines, coupons):
                cart = Cart()
                for name, price, qty in lines:
                    cart.add(name, price, qty)
                for pct in coupons:
                    cart.apply_coupon(pct)
                return cart.total()
        '''),
        fail_to_pass=[
            ("checkout([('book', 100, 2)], [10, 10])", "180"),
            ("checkout([('pen', 50, 2)], [20, 50])", "50"),
            ("checkout([('mug', 80, 1)], [25, 25])", "60"),
        ],
        pass_to_pass=[
            ("checkout([('a', 40, 3)], [])", "120"),
            ("checkout([('a', 100, 1)], [15])", "85"),
            ("checkout([('a', 10, 2), ('b', 5, 4)], [50])", "20"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-015-matrix-multiply-index",
        module_name="matrix",
        issue=D('''
            matmul gives wrong numbers -- or crashes -- on non-square
            inputs. matmul([[1, 2]], [[3], [4]]) raises IndexError instead
            of returning [[11]], and multiplying by a non-symmetric square
            matrix returns values that look transposed. Maddeningly, our
            smoke test that multiplies by a diagonal matrix passes, which
            is why this shipped. shape and transpose behave.
        '''),
        buggy_source=D('''
            def shape(m):
                """(rows, cols) of a matrix stored as a list of rows."""
                return (len(m), len(m[0]) if m else 0)

            def transpose(m):
                """Transpose of m."""
                if not m:
                    return []
                return [[m[r][c] for r in range(len(m))] for c in range(len(m[0]))]

            def matmul(a, b):
                """Matrix product a @ b."""
                rows_a, cols_a = shape(a)
                rows_b, cols_b = shape(b)
                if cols_a != rows_b:
                    raise ValueError('incompatible shapes')
                return [
                    [sum(a[i][k] * b[j][k] for k in range(cols_a)) for j in range(cols_b)]
                    for i in range(rows_a)
                ]
        '''),
        reference_source=D('''
            def shape(m):
                """(rows, cols) of a matrix stored as a list of rows."""
                return (len(m), len(m[0]) if m else 0)

            def transpose(m):
                """Transpose of m."""
                if not m:
                    return []
                return [[m[r][c] for r in range(len(m))] for c in range(len(m[0]))]

            def matmul(a, b):
                """Matrix product a @ b."""
                rows_a, cols_a = shape(a)
                rows_b, cols_b = shape(b)
                if cols_a != rows_b:
                    raise ValueError('incompatible shapes')
                return [
                    [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
                    for i in range(rows_a)
                ]
        '''),
        test_preamble="",
        fail_to_pass=[
            ("matmul([[1, 2]], [[3], [4]])", "[[11]]"),
            ("matmul([[1, 0], [0, 1]], [[1, 2], [3, 4]])", "[[1, 2], [3, 4]]"),
            ("matmul([[1, 2], [3, 4]], [[0, 1], [2, 3]])", "[[4, 7], [8, 15]]"),
        ],
        pass_to_pass=[
            ("matmul([[1, 2], [3, 4]], [[2, 0], [0, 2]])", "[[2, 4], [6, 8]]"),
            ("matmul([[5]], [[7]])", "[[35]]"),
            ("shape([[1, 2, 3]])", "(1, 3)"),
            ("transpose([[1, 2], [3, 4]])", "[[1, 3], [2, 4]]"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-016-histogram-bin-edge",
        module_name="histogram",
        issue=D('''
            histogram crashes whenever a sample equals the upper bound.
            histogram([0, 5, 10], 0, 10, 2) raises IndexError. Digging in,
            bin_index(10, 0, 10, 5) returns 5, but with 5 bins the valid
            indices are 0..4. A value exactly at the top of the range is
            supposed to be counted in the last bin (the convention every
            plotting library uses). Interior values bin correctly.
        '''),
        buggy_source=D('''
            def bin_index(x, lo, hi, bins):
                """Index of the equal-width bin that x falls into."""
                if bins < 1:
                    raise ValueError('need at least one bin')
                if x < lo or x > hi:
                    raise ValueError('value out of range')
                width = (hi - lo) / bins
                return int((x - lo) / width)

            def histogram(xs, lo, hi, bins):
                """Counts per bin for the samples in xs."""
                counts = [0] * bins
                for x in xs:
                    counts[bin_index(x, lo, hi, bins)] += 1
                return counts
        '''),
        reference_source=D('''
            def bin_index(x, lo, hi, bins):
                """Index of the equal-width bin that x falls into."""
                if bins < 1:
                    raise ValueError('need at least one bin')
                if x < lo or x > hi:
                    raise ValueError('value out of range')
                if x == hi:
                    return bins - 1
                width = (hi - lo) / bins
                return int((x - lo) / width)

            def histogram(xs, lo, hi, bins):
                """Counts per bin for the samples in xs."""
                counts = [0] * bins
                for x in xs:
                    counts[bin_index(x, lo, hi, bins)] += 1
                return counts
        '''),
        test_preamble="",
        fail_to_pass=[
            ("bin_index(10, 0, 10, 5)", "4"),
            ("histogram([0, 5, 10], 0, 10, 2)", "[1, 2]"),
            ("histogram([10, 10], 0, 10, 4)", "[0, 0, 0, 2]"),
        ],
        pass_to_pass=[
            ("bin_index(0, 0, 10, 5)", "0"),
            ("bin_index(7, 0, 10, 5)", "3"),
            ("histogram([1, 2, 3], 0, 10, 2)", "[3, 0]"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-017-query-string-repeats",
        module_name="querystring",
        issue=D('''
            parse_qs drops data in two ways. Repeated keys lose all but one
            value: 'a=1&a=2' parses to just one entry for 'a' (the spec says
            values for a repeated key accumulate in order). And any value
            containing '=' gets truncated: 'sig=ab=cd' comes back as 'ab'
            instead of 'ab=cd' -- only the FIRST '=' separates key from
            value. get_first consequently returns the wrong value for
            repeated keys. Simple one-shot pairs parse fine.
        '''),
        buggy_source=D('''
            def parse_qs(s):
                """Parse 'k=v&k2=v2' into a dict of key -> list of values."""
                result = {}
                if not s:
                    return result
                for part in s.split('&'):
                    if not part:
                        continue
                    pieces = part.split('=')
                    key = pieces[0]
                    value = pieces[1] if len(pieces) > 1 else ''
                    result[key] = [value]
                return result

            def get_first(s, key):
                """First value submitted for `key`, or None if absent."""
                values = parse_qs(s).get(key)
                if not values:
                    return None
                return values[0]
        '''),
        reference_source=D('''
            def parse_qs(s):
                """Parse 'k=v&k2=v2' into a dict of key -> list of values."""
                result = {}
                if not s:
                    return result
                for part in s.split('&'):
                    if not part:
                        continue
                    key, _, value = part.partition('=')
                    result.setdefault(key, []).append(value)
                return result

            def get_first(s, key):
                """First value submitted for `key`, or None if absent."""
                values = parse_qs(s).get(key)
                if not values:
                    return None
                return values[0]
        '''),
        test_preamble="",
        fail_to_pass=[
            ("parse_qs('a=1&a=2')", "{'a': ['1', '2']}"),
            ("parse_qs('sig=ab=cd')", "{'sig': ['ab=cd']}"),
            ("get_first('tag=x&tag=y', 'tag')", "'x'"),
        ],
        pass_to_pass=[
            ("parse_qs('a=1&b=2')", "{'a': ['1'], 'b': ['2']}"),
            ("parse_qs('')", "{}"),
            ("get_first('x=5', 'x')", "'5'"),
            ("get_first('x=5', 'y')", "None"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-018-scheduler-containment",
        module_name="scheduler",
        issue=D('''
            We are double-booking rooms. book(0, 10) followed by book(3, 4)
            both return True, even though 3-4 sits entirely inside the
            existing 0-10 booking. Partial overlaps are rejected correctly
            and back-to-back meetings (one ending exactly when the next
            starts) are still allowed, but any request fully contained in
            an existing slot slips through and ends up in bookings().
        '''),
        buggy_source=D('''
            class Scheduler:
                """Books half-open [start, end) slots on a single resource."""

                def __init__(self):
                    self._slots = []

                def book(self, start, end):
                    """Reserve [start, end); returns False on any conflict."""
                    if start >= end:
                        raise ValueError('start must be before end')
                    for s, e in self._slots:
                        if start <= s < end or start < e <= end:
                            return False
                    self._slots.append((start, end))
                    return True

                def bookings(self):
                    """All accepted slots, sorted by start time."""
                    return sorted(self._slots)
        '''),
        reference_source=D('''
            class Scheduler:
                """Books half-open [start, end) slots on a single resource."""

                def __init__(self):
                    self._slots = []

                def book(self, start, end):
                    """Reserve [start, end); returns False on any conflict."""
                    if start >= end:
                        raise ValueError('start must be before end')
                    for s, e in self._slots:
                        if start < e and s < end:
                            return False
                    self._slots.append((start, end))
                    return True

                def bookings(self):
                    """All accepted slots, sorted by start time."""
                    return sorted(self._slots)
        '''),
        test_preamble=D('''
            def run_bookings(requests):
                sch = Scheduler()
                return [sch.book(s, e) for s, e in requests]

            def final_bookings(requests):
                sch = Scheduler()
                for s, e in requests:
                    sch.book(s, e)
                return sch.bookings()
        '''),
        fail_to_pass=[
            ("run_bookings([(0, 10), (3, 4)])", "[True, False]"),
            ("run_bookings([(0, 10), (2, 5), (2, 5)])", "[True, False, False]"),
            ("final_bookings([(1, 9), (4, 5)])", "[(1, 9)]"),
        ],
        pass_to_pass=[
            ("run_bookings([(0, 4), (4, 8)])", "[True, True]"),
            ("run_bookings([(2, 6), (4, 8)])", "[True, False]"),
            ("run_bookings([(5, 6), (1, 3)])", "[True, True]"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-019-runlength-segments",
        module_name="segments",
        issue=D('''
            compress loses the tail of every sequence. compress([1, 1, 2, 2, 2])
            returns [[1, 2]] -- the run of three 2s is just gone -- and
            compress([5]) returns []. As a result expand(compress(xs)) never
            equals xs for non-empty input, which corrupts every payload that
            goes through our encode/decode path. expand() on its own behaves
            correctly, and compress([]) is fine.
        '''),
        buggy_source=D('''
            def compress(xs):
                """Run-length encode xs into [value, count] pairs."""
                pairs = []
                if not xs:
                    return pairs
                current = xs[0]
                count = 1
                for x in xs[1:]:
                    if x == current:
                        count += 1
                    else:
                        pairs.append([current, count])
                        current = x
                        count = 1
                return pairs

            def expand(pairs):
                """Inverse of compress: rebuild the original list."""
                out = []
                for value, count in pairs:
                    if count < 0:
                        raise ValueError('negative count')
                    out.extend([value] * count)
                return out
        '''),
        reference_source=D('''
            def compress(xs):
                """Run-length encode xs into [value, count] pairs."""
                pairs = []
                if not xs:
                    return pairs
                current = xs[0]
                count = 1
                for x in xs[1:]:
                    if x == current:
                        count += 1
                    else:
                        pairs.append([current, count])
                        current = x
                        count = 1
                pairs.append([current, count])
                return pairs

            def expand(pairs):
                """Inverse of compress: rebuild the original list."""
                out = []
                for value, count in pairs:
                    if count < 0:
                        raise ValueError('negative count')
                    out.extend([value] * count)
                return out
        '''),
        test_preamble="",
        fail_to_pass=[
            ("compress([1, 1, 2, 2, 2])", "[[1, 2], [2, 3]]"),
            ("compress([5])", "[[5, 1]]"),
            ("expand(compress([3, 3, 3]))", "[3, 3, 3]"),
            ("expand(compress([1, 2, 2]))", "[1, 2, 2]"),
        ],
        pass_to_pass=[
            ("compress([])", "[]"),
            ("expand([[7, 2], [8, 1]])", "[7, 7, 8]"),
            ("expand([])", "[]"),
        ],
    ),
    dict(
        instance_id="openworld-swebench-020-thermostat-hysteresis",
        module_name="thermostat",
        issue=D('''
            The furnace is short-cycling. With Thermostat(target=20, band=2)
            and the room holding steady at 20 degrees, step(20) alternates
            heat, off, heat, off on every single tick. Hysteresis is supposed
            to hold the current mode until the temperature actually exits the
            band (below target-band to start heating, above target+band to
            stop); instead the relay chatters constantly anywhere near the
            target. Far below target it heats steadily and far above it
            stays off, so the extremes look fine.
        '''),
        buggy_source=D('''
            class Thermostat:
                """Bang-bang heater control with a hysteresis band."""

                def __init__(self, target, band):
                    if band <= 0:
                        raise ValueError('band must be positive')
                    self.target = target
                    self.band = band
                    self.mode = 'off'

                def step(self, temp):
                    """Advance one tick with the measured temp; returns the mode."""
                    if self.mode == 'off':
                        if temp < self.target + self.band:
                            self.mode = 'heat'
                    else:
                        if temp > self.target - self.band:
                            self.mode = 'off'
                    return self.mode
        '''),
        reference_source=D('''
            class Thermostat:
                """Bang-bang heater control with a hysteresis band."""

                def __init__(self, target, band):
                    if band <= 0:
                        raise ValueError('band must be positive')
                    self.target = target
                    self.band = band
                    self.mode = 'off'

                def step(self, temp):
                    """Advance one tick with the measured temp; returns the mode."""
                    if self.mode == 'off':
                        if temp < self.target - self.band:
                            self.mode = 'heat'
                    else:
                        if temp > self.target + self.band:
                            self.mode = 'off'
                    return self.mode
        '''),
        test_preamble=D('''
            def run_modes(target, band, temps):
                t = Thermostat(target, band)
                return [t.step(x) for x in temps]
        '''),
        fail_to_pass=[
            (
                "run_modes(20, 2, [20, 20, 20, 20])",
                "['off', 'off', 'off', 'off']",
            ),
            (
                "run_modes(20, 2, [15, 19, 19, 23, 21])",
                "['heat', 'heat', 'heat', 'off', 'off']",
            ),
            (
                "run_modes(70, 3, [69, 69, 69])",
                "['off', 'off', 'off']",
            ),
        ],
        pass_to_pass=[
            ("run_modes(20, 2, [10, 11, 12])", "['heat', 'heat', 'heat']"),
            ("run_modes(20, 2, [30, 29, 28])", "['off', 'off', 'off']"),
        ],
    ),
]


def build():
    records = []
    for spec in INSTANCES:
        instance = SWEBenchInstance(
            instance_id=spec["instance_id"],
            module_name=spec["module_name"],
            issue=spec["issue"],
            buggy_source=spec["buggy_source"],
            reference_source=spec["reference_source"],
            test_preamble=spec.get("test_preamble", ""),
            fail_to_pass=[tuple(t) for t in spec["fail_to_pass"]],
            pass_to_pass=[tuple(t) for t in spec["pass_to_pass"]],
            world={},
        )
        ref = run_instance_tests(instance.reference_source, instance)
        if not ref["solved"]:
            sys.exit(
                f"{instance.instance_id}: reference does not solve: "
                f"{ref['fail_to_pass']['errors'] + ref['pass_to_pass']['errors']}"
            )
        buggy = run_instance_tests(instance.buggy_source, instance)
        if buggy["fail_to_pass"]["passed"] != 0:
            sys.exit(f"{instance.instance_id}: a fail_to_pass test passes on the buggy source")
        if buggy["pass_to_pass"]["failed"] != 0:
            sys.exit(
                f"{instance.instance_id}: buggy source breaks pass_to_pass: "
                f"{buggy['pass_to_pass']['errors']}"
            )
        instance.world = {
            "name": f"swebench:{instance.instance_id}",
            "description": (
                f"Program repair as a world model for module '{instance.module_name}'. "
                "Submit a corrected module via submit_patch(params={'source': ...})."
            ),
            "initial_state": initial_world_state(instance),
            "actions": ["submit_patch"],
            "rules": list(spec.get("rules", DEFAULT_RULES)),
            "invariants": list(spec.get("invariants", DEFAULT_INVARIANTS)),
        }
        record = dict(
            instance_id=instance.instance_id,
            module_name=instance.module_name,
            issue=instance.issue,
            buggy_source=instance.buggy_source,
            reference_source=instance.reference_source,
            test_preamble=instance.test_preamble,
            fail_to_pass=[list(t) for t in instance.fail_to_pass],
            pass_to_pass=[list(t) for t in instance.pass_to_pass],
            world=instance.world,
        )
        records.append(record)
        print(f"  ok {instance.instance_id} "
              f"(f2p {len(instance.fail_to_pass)}, p2p {len(instance.pass_to_pass)})")
    OUT.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
        encoding="utf-8",
    )
    print(f"[saved] {OUT} ({len(records)} instances)")


if __name__ == "__main__":
    build()
