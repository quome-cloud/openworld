"""Tests for the E126 local-coder pipeline robustness fixes (the ornith:35b empty-response bug)."""
from experiments.e126_lib import history_budget, build_history, extract_code, CHARS_PER_TOKEN


def test_history_budget_fits_within_context_window():
    # The whole point: task + history + a response must fit in num_ctx. The OLD code used a flat
    # 24000-char budget regardless of num_ctx -> overflow at 8192. The new budget must be SMALLER than
    # the overflow value, and leave real response headroom.
    num_ctx = 8192
    task_chars = 1500
    b = history_budget(num_ctx, task_chars=task_chars)
    ctx_chars = num_ctx * CHARS_PER_TOKEN
    # task + history must leave >= ~1000 tokens of response room
    assert task_chars + b <= ctx_chars - 1000 * CHARS_PER_TOKEN
    assert b < 24000          # strictly tighter than the buggy flat budget
    assert b > 0


def test_history_budget_scales_with_num_ctx_and_caps():
    assert history_budget(4096) < history_budget(16384)      # bigger window -> bigger budget
    assert history_budget(256000) <= 60000                   # capped, never builds a 700KB prompt
    assert history_budget(2048) >= 0                          # tiny window never goes negative


def test_build_history_respects_budget():
    log = [{"round": i, "script": "X" * 1000, "stdout": "Y" * 1000, "best_after": 0} for i in range(20)]
    h = build_history(log, budget=4000)
    assert len(h) <= 4000 + 2200          # at most a one-record overshoot past budget
    assert "round 19" in h                # newest round always kept
    assert "round 0" not in h             # oldest trimmed away


def test_build_history_empty():
    assert build_history([], budget=4000).startswith("(none yet")


def test_extract_code_prefers_fenced_block():
    assert extract_code("blah\n```python\nimport numpy\n```\nthanks") == "import numpy"


def test_extract_code_returns_empty_for_prose_not_garbage():
    # the bug: prose was fed to the interpreter as 'code'. Now prose -> "" so the round is retried.
    assert extract_code("I think the answer is to move left then click.") == ""
    assert extract_code("") == ""
    assert extract_code("   ") == ""


def test_extract_code_accepts_bare_code_without_fence():
    assert extract_code("g.reset()\nfor a in [1,2,3]: g.step(a)") != ""
