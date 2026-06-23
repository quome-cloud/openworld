"""The coding world: program repair as a world model.

A CodingTask is a buggy Python function plus a hidden unit-test suite. The
world's symbolic state tracks the current source and its test results; the
single meaningful action is submit_patch(source=...), whose transition runs
the tests bit-exactly in a restricted sandbox. Episodes end when all tests
pass or the attempt budget is exhausted.

This is a Code World Model turned on its natural domain: the environment
dynamics ARE code execution, so the simulator is exact by construction and
every reward signal (tests passed) is verifiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .state import Action, WorldState
from .transition import Transition
from .world import World

# Tests run with the same restricted builtins philosophy as transition code,
# but a slightly wider set since real functions need more of the language.
_TEST_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs", "all", "any", "bool", "chr", "dict", "divmod", "enumerate",
        "filter", "float", "frozenset", "int", "isinstance", "issubclass",
        "iter", "len", "list", "map", "max", "min", "next", "ord", "pow",
        "range", "repr", "reversed", "round", "set", "sorted", "str", "sum",
        "tuple", "zip", "ValueError", "KeyError", "TypeError", "IndexError",
        "ZeroDivisionError", "StopIteration", "Exception", "AssertionError",
    )
}


@dataclass
class CodingTask:
    """One program-repair problem.

    tests are (call_expression, expected_repr) pairs evaluated against the
    submitted source, e.g. ("median([3, 1, 2])", "2").
    """

    name: str
    description: str
    buggy_source: str
    tests: List[Tuple[str, str]]
    function_name: str = ""
    reference_source: str = ""  # known-good solution, for oracle baselines


def run_tests(
    source: str,
    tests: List[Tuple[str, str]],
    timeout_seconds: float = 5.0,
    extra_builtins: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute `source`, run each test expression, compare repr() to expected.

    Returns {"passed": n, "failed": n, "errors": [readable failure strings]}.
    A source that fails to exec marks every test as failed.

    On POSIX the suite runs in a forked child that the parent SIGKILLs at the
    deadline. In-process alarms are insufficient: a generated patch containing
    a bare `except:` inside its loop swallows any exception an alarm handler
    raises (observed in the wild) - only a hard kill is loop-proof.
    """
    import os

    if timeout_seconds and hasattr(os, "fork"):
        return _run_tests_forked(source, tests, timeout_seconds, extra_builtins)
    return _run_tests_inline(source, tests, extra_builtins)


def _run_tests_inline(
    source: str,
    tests: List[Tuple[str, str]],
    extra_builtins: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import math

    namespace: Dict[str, Any] = {
        "__builtins__": {**_TEST_BUILTINS, **(extra_builtins or {})},
        "math": math,
        "__name__": "<submission>",
    }
    errors: List[str] = []
    try:
        exec(compile(source, "<submission>", "exec"), namespace)
    except Exception as exc:
        return {
            "passed": 0,
            "failed": len(tests),
            "errors": [f"source failed to execute: {exc!r}"],
        }
    passed = 0
    for expression, expected in tests:
        try:
            result = eval(expression, namespace)  # noqa: S307 - sandboxed namespace
            if repr(result) == expected:
                passed += 1
            else:
                errors.append(f"{expression} -> {result!r}, expected {expected}")
        except Exception as exc:
            errors.append(f"{expression} raised {exc!r}")
    return {"passed": passed, "failed": len(tests) - passed, "errors": errors}


def _run_tests_forked(
    source: str,
    tests: List[Tuple[str, str]],
    timeout_seconds: float,
    extra_builtins: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import json
    import os
    import signal
    import time

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: run the suite, report through the pipe, hard-exit
        try:
            os.close(read_fd)
            payload = json.dumps(_run_tests_inline(source, tests, extra_builtins)).encode("utf-8")
            os.write(write_fd, payload)
            os.close(write_fd)
        finally:
            os._exit(0)

    os.close(write_fd)
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    while True:
        done_pid, _status = os.waitpid(pid, os.WNOHANG)
        if done_pid:
            break
        if time.monotonic() > deadline:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            timed_out = True
            break
        time.sleep(0.02)

    chunks = []
    while True:
        chunk = os.read(read_fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_fd)

    if timed_out:
        return {"passed": 0, "failed": len(tests),
                "errors": ["source timed out (killed; possible infinite loop)"]}
    data = b"".join(chunks)
    if not data:
        return {"passed": 0, "failed": len(tests),
                "errors": ["test process died without reporting"]}
    return json.loads(data.decode("utf-8"))


class CodeFixTransition(Transition):
    """Dynamics of the coding world: submitting a patch runs the test suite."""

    def __init__(self, task: CodingTask):
        self.task = task

    def step(self, state: WorldState, action: Action) -> WorldState:
        s = state.copy()
        if s.get("solved"):
            return s
        if action.name == "submit_patch":
            source = str(action.params.get("source", "")) or s["source"]
            result = run_tests(source, self.task.tests)
            s["source"] = source
            s["tests_passed"] = result["passed"]
            s["tests_failed"] = result["failed"]
            s["last_errors"] = result["errors"][:3]
            s["attempts"] += 1
            s["solved"] = result["failed"] == 0
        return s


def build_codefix_world(task: CodingTask) -> World:
    """A World wrapping one CodingTask. State carries visible test feedback."""
    initial = run_tests(task.buggy_source, task.tests)
    return World(
        name=f"codefix:{task.name}",
        description=(
            f"Program repair. Task: {task.description} "
            "Submit corrected source via submit_patch(params={'source': ...})."
        ),
        initial_state={
            "task": task.name,
            "source": task.buggy_source,
            "tests_passed": initial["passed"],
            "tests_failed": initial["failed"],
            "last_errors": initial["errors"][:3],
            "attempts": 0,
            "solved": initial["failed"] == 0,
        },
        actions=["submit_patch"],
        transition=CodeFixTransition(task),
    )


# ---------------------------------------------------------------------------
# Benchmark suite: ten classic bug archetypes, self-contained and offline.
# ---------------------------------------------------------------------------

BENCHMARK: List[CodingTask] = [
    CodingTask(
        name="sum_range_off_by_one",
        description="sum_to(n) should return the sum of integers 1..n inclusive.",
        buggy_source="def sum_to(n):\n    total = 0\n    for i in range(1, n):\n        total += i\n    return total\n",
        reference_source="def sum_to(n):\n    total = 0\n    for i in range(1, n + 1):\n        total += i\n    return total\n",
        tests=[("sum_to(1)", "1"), ("sum_to(5)", "15"), ("sum_to(10)", "55")],
        function_name="sum_to",
    ),
    CodingTask(
        name="max_wrong_comparison",
        description="largest(xs) should return the largest element of a non-empty list.",
        buggy_source="def largest(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
        reference_source="def largest(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
        tests=[("largest([3, 1, 2])", "3"), ("largest([-5, -2, -9])", "-2"), ("largest([7])", "7")],
        function_name="largest",
    ),
    CodingTask(
        name="palindrome_case",
        description="is_palindrome(s) should ignore letter case.",
        buggy_source="def is_palindrome(s):\n    return s == s[::-1]\n",
        reference_source="def is_palindrome(s):\n    t = s.lower()\n    return t == t[::-1]\n",
        tests=[("is_palindrome('Level')", "True"), ("is_palindrome('abc')", "False"), ("is_palindrome('Noon')", "True")],
        function_name="is_palindrome",
    ),
    CodingTask(
        name="count_vowels_missing",
        description="count_vowels(s) should count a, e, i, o, u case-insensitively.",
        buggy_source="def count_vowels(s):\n    count = 0\n    for ch in s:\n        if ch in 'aeiu':\n            count += 1\n    return count\n",
        reference_source="def count_vowels(s):\n    count = 0\n    for ch in s.lower():\n        if ch in 'aeiou':\n            count += 1\n    return count\n",
        tests=[("count_vowels('book')", "2"), ("count_vowels('AEIOU')", "5"), ("count_vowels('xyz')", "0")],
        function_name="count_vowels",
    ),
    CodingTask(
        name="fib_wrong_base",
        description="fib(n) should return the nth Fibonacci number with fib(0)=0, fib(1)=1.",
        buggy_source="def fib(n):\n    if n <= 1:\n        return 1\n    return fib(n - 1) + fib(n - 2)\n",
        reference_source="def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
        tests=[("fib(0)", "0"), ("fib(1)", "1"), ("fib(7)", "13")],
        function_name="fib",
    ),
    CodingTask(
        name="median_unsorted",
        description="median(xs) should return the median of an odd-length list.",
        buggy_source="def median(xs):\n    return xs[len(xs) // 2]\n",
        reference_source="def median(xs):\n    ys = sorted(xs)\n    return ys[len(ys) // 2]\n",
        tests=[("median([3, 1, 2])", "2"), ("median([9, 1, 5, 3, 7])", "5"), ("median([4])", "4")],
        function_name="median",
    ),
    CodingTask(
        name="dedupe_order",
        description="dedupe(xs) should remove duplicates while keeping first-seen order.",
        buggy_source="def dedupe(xs):\n    return list(set(xs))\n",
        reference_source="def dedupe(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n",
        tests=[("dedupe([3, 1, 3, 2, 1])", "[3, 1, 2]"), ("dedupe([5, 5, 5])", "[5]"), ("dedupe([])", "[]")],
        function_name="dedupe",
    ),
    CodingTask(
        name="balanced_parens_early",
        description="balanced(s) should check whether parentheses are balanced.",
        buggy_source="def balanced(s):\n    depth = 0\n    for ch in s:\n        if ch == '(':\n            depth += 1\n        elif ch == ')':\n            depth -= 1\n    return depth == 0\n",
        reference_source="def balanced(s):\n    depth = 0\n    for ch in s:\n        if ch == '(':\n            depth += 1\n        elif ch == ')':\n            depth -= 1\n            if depth < 0:\n                return False\n    return depth == 0\n",
        tests=[("balanced('(())')", "True"), ("balanced(')(')", "False"), ("balanced('(()')", "False")],
        function_name="balanced",
    ),
    CodingTask(
        name="word_count_split",
        description="word_count(s) should count whitespace-separated words, handling repeats of spaces.",
        buggy_source="def word_count(s):\n    return len(s.split(' '))\n",
        reference_source="def word_count(s):\n    return len(s.split())\n",
        tests=[("word_count('a b  c')", "3"), ("word_count('hello')", "1"), ("word_count('')", "0")],
        function_name="word_count",
    ),
    CodingTask(
        name="running_max_reset",
        description="running_max(xs) should return the running maximum list of xs.",
        buggy_source="def running_max(xs):\n    out = []\n    for x in xs:\n        best = x\n        out.append(best)\n    return out\n",
        reference_source="def running_max(xs):\n    out = []\n    best = None\n    for x in xs:\n        best = x if best is None or x > best else best\n        out.append(best)\n    return out\n",
        tests=[("running_max([1, 3, 2, 5, 4])", "[1, 3, 3, 5, 5]"), ("running_max([2])", "[2]"), ("running_max([5, 1])", "[5, 5]")],
        function_name="running_max",
    ),
    CodingTask(
        name="binary_search_last",
        description="binary_search(xs, target) should return the index of target in sorted list xs, or -1 if absent.",
        buggy_source="def binary_search(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if xs[mid] == target:\n            return mid\n        elif xs[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
        reference_source="def binary_search(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if xs[mid] == target:\n            return mid\n        elif xs[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
        tests=[("binary_search([1, 3, 5], 5)", "2"), ("binary_search([1, 3, 5], 1)", "0"), ("binary_search([2, 4], 3)", "-1")],
        function_name="binary_search",
    ),
    CodingTask(
        name="anagram_normalize",
        description="is_anagram(a, b) should ignore letter case and spaces.",
        buggy_source="def is_anagram(a, b):\n    return sorted(a) == sorted(b)\n",
        reference_source="def is_anagram(a, b):\n    na = sorted(a.lower().replace(' ', ''))\n    nb = sorted(b.lower().replace(' ', ''))\n    return na == nb\n",
        tests=[("is_anagram('Listen', 'Silent')", "True"), ("is_anagram('hello', 'world')", "False"), ("is_anagram('a gentleman', 'elegant man')", "True")],
        function_name="is_anagram",
    ),
    CodingTask(
        name="flatten_no_expand",
        description="flatten(xs) should flatten one level: a list of lists becomes a single list.",
        buggy_source="def flatten(xs):\n    out = []\n    for x in xs:\n        out.append(x)\n    return out\n",
        reference_source="def flatten(xs):\n    out = []\n    for x in xs:\n        out.extend(x)\n    return out\n",
        tests=[("flatten([[1, 2], [3], [4, 5]])", "[1, 2, 3, 4, 5]"), ("flatten([[7]])", "[7]"), ("flatten([])", "[]")],
        function_name="flatten",
    ),
    CodingTask(
        name="clamp_swapped",
        description="clamp(x, lo, hi) should clamp x into the inclusive range [lo, hi].",
        buggy_source="def clamp(x, lo, hi):\n    return min(lo, max(hi, x))\n",
        reference_source="def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
        tests=[("clamp(5, 0, 10)", "5"), ("clamp(-3, 0, 10)", "0"), ("clamp(15, 0, 10)", "10")],
        function_name="clamp",
    ),
    CodingTask(
        name="average_int_division",
        description="average(xs) should return the arithmetic mean as a float.",
        buggy_source="def average(xs):\n    return sum(xs) // len(xs)\n",
        reference_source="def average(xs):\n    return sum(xs) / len(xs)\n",
        tests=[("average([1, 2])", "1.5"), ("average([3])", "3.0"), ("average([1, 2, 3, 5])", "2.75")],
        function_name="average",
    ),
    CodingTask(
        name="count_overlapping",
        description="count_overlapping(s, sub) should count occurrences of sub in s, including overlapping ones.",
        buggy_source="def count_overlapping(s, sub):\n    return s.count(sub)\n",
        reference_source="def count_overlapping(s, sub):\n    count = 0\n    for i in range(len(s) - len(sub) + 1):\n        if s[i:i + len(sub)] == sub:\n            count += 1\n    return count\n",
        tests=[("count_overlapping('aaa', 'aa')", "2"), ("count_overlapping('abcabc', 'abc')", "2"), ("count_overlapping('xyz', 'q')", "0")],
        function_name="count_overlapping",
    ),
    CodingTask(
        name="reverse_words_chars",
        description="reverse_words(s) should reverse the order of the words, not the characters.",
        buggy_source="def reverse_words(s):\n    return s[::-1]\n",
        reference_source="def reverse_words(s):\n    return ' '.join(s.split()[::-1])\n",
        tests=[("reverse_words('hello world')", "'world hello'"), ("reverse_words('a b c')", "'c b a'"), ("reverse_words('solo')", "'solo'")],
        function_name="reverse_words",
    ),
    CodingTask(
        name="second_largest_dup",
        description="second_largest(xs) should return the second-largest DISTINCT value in xs.",
        buggy_source="def second_largest(xs):\n    return sorted(xs)[-2]\n",
        reference_source="def second_largest(xs):\n    return sorted(set(xs))[-2]\n",
        tests=[("second_largest([5, 5, 3])", "3"), ("second_largest([1, 2, 3])", "2"), ("second_largest([9, 9, 9, 4])", "4")],
        function_name="second_largest",
    ),
    CodingTask(
        name="rle_last_run",
        description="rle(s) should run-length encode s as letter+count pairs, e.g. 'aaabb' -> 'a3b2'.",
        buggy_source="def rle(s):\n    if not s:\n        return ''\n    out = ''\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i - 1]:\n            count += 1\n        else:\n            out += s[i - 1] + str(count)\n            count = 1\n    return out\n",
        reference_source="def rle(s):\n    if not s:\n        return ''\n    out = ''\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i - 1]:\n            count += 1\n        else:\n            out += s[i - 1] + str(count)\n            count = 1\n    out += s[-1] + str(count)\n    return out\n",
        tests=[("rle('aaabb')", "'a3b2'"), ("rle('x')", "'x1'"), ("rle('')", "''")],
        function_name="rle",
    ),
    CodingTask(
        name="gcd_returns_zero",
        description="gcd(a, b) should return the greatest common divisor (gcd(x, 0) == x).",
        buggy_source="def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return b\n",
        reference_source="def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
        tests=[("gcd(12, 18)", "6"), ("gcd(7, 3)", "1"), ("gcd(5, 0)", "5")],
        function_name="gcd",
    ),
]
