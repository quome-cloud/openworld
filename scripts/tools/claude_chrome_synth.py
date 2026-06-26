"""claude_chrome_synth.py — Playwright driver for claude.ai synthesis (T369).

Drives a logged-in chrome.ai session via Playwright to post synthesis prompts
and scrape code-block responses.  Requires a persistent Chrome profile with
claude.ai already logged in (operator performs one-time login via VNC + HA068).

Usage:
  # Post a prompt from a file, print JSON result to stdout
  python3 claude_chrome_synth.py --prompt-file prompt.txt

  # Post a prompt string directly
  python3 claude_chrome_synth.py --prompt "Write a Python function..."

  # Smoke-test: verify browser opens claude.ai correctly
  python3 claude_chrome_synth.py --smoke-test

Output (stdout, JSON):
  {"status": "ok", "code_blocks": ["def predict(...)..."], "raw_response": "..."}
  {"status": "error", "error": "...", "code_blocks": []}

Design notes:
- Uses launch_persistent_context so the Chrome profile (with cookies) survives
  across runs.  No login handling — if the session expires, fails loud.
- One new claude.ai conversation per call.  Tabs accumulate; caller can pass
  --reuse-tab to keep a single tab for throughput.
- Response completion is detected by polling for the "Copy" button that appears
  under the final assistant message only after generation stops.
- Extraction: grabs ```python ... ``` blocks; falls back to all ``` blocks.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

CHROME_PROFILE = Path.home() / ".config" / "google-chrome"
CLAUDE_URL = "https://claude.ai/new"
RESPONSE_TIMEOUT = 300   # seconds max to wait for generation to finish
POLL_INTERVAL = 1.5      # seconds between completion checks
MAX_RETRIES = 2          # retries on transient DOM errors


def extract_code_blocks(text: str) -> list[str]:
    """Extract ```python ... ``` blocks; fall back to all ``` blocks."""
    blocks = re.findall(r"```python\s*(.*?)```", text, re.S)
    if not blocks:
        blocks = re.findall(r"```\s*(.*?)```", text, re.S)
    return [b.strip() for b in blocks if b.strip()]


def run_synth(prompt: str, reuse_tab: bool = False, headless: bool = False) -> dict:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(CHROME_PROFILE),
            channel="chrome",
            headless=headless,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        try:
            # Use existing tab or open new one
            page = ctx.pages[0] if (reuse_tab and ctx.pages) else ctx.new_page()
            page.goto(CLAUDE_URL, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_timeout(2000)

            # Detect login gate — if redirected to /login, session expired
            if "/login" in page.url or "/auth" in page.url:
                return {"status": "error",
                        "error": "session_expired: claude.ai redirected to login",
                        "code_blocks": []}

            # Find the prompt input (contenteditable div or textarea)
            input_sel = (
                "div[contenteditable='true'][data-placeholder],"
                "div[contenteditable='true'].ProseMirror,"
                "textarea[placeholder*='message'],"
                "div[contenteditable='true']"
            )
            try:
                page.wait_for_selector(input_sel, timeout=15_000)
            except PWTimeout:
                return {"status": "error",
                        "error": "input_not_found: could not locate prompt input",
                        "code_blocks": []}

            inp = page.locator(input_sel).first

            # Type the prompt (use fill for plain text, then Enter to submit)
            inp.click()
            page.keyboard.insert_text(prompt)
            page.wait_for_timeout(500)

            # Submit (Enter key; shift-enter for newlines inside the prompt)
            page.keyboard.press("Enter")

            # Wait for response to complete.
            # Strategy: poll for a "Copy" button inside an assistant message
            # that was NOT present before we submitted.
            started = time.time()
            raw_text = ""
            while time.time() - started < RESPONSE_TIMEOUT:
                page.wait_for_timeout(int(POLL_INTERVAL * 1000))

                # Check for error state (network error, rate limit UI, etc.)
                err_sel = "div[data-testid='error-message'], div.claude-error"
                if page.locator(err_sel).count() > 0:
                    err_txt = page.locator(err_sel).first.inner_text()
                    return {"status": "error",
                            "error": f"claude_error: {err_txt[:200]}",
                            "code_blocks": []}

                # Look for the stop-generating button (generation in progress)
                stop_sel = "button[aria-label*='Stop'], button[data-testid*='stop']"
                generating = page.locator(stop_sel).count() > 0
                if generating:
                    continue  # still generating

                # Generation stopped — try to read the assistant response
                # Assistant messages typically in: div[data-message-author-role='assistant']
                # or .claude-message, or .font-claude-message
                msg_sel = (
                    "div[data-message-author-role='assistant'],"
                    "div.font-claude-message,"
                    ".claude-message"
                )
                msgs = page.locator(msg_sel)
                if msgs.count() == 0:
                    # No assistant message yet — wait a bit more
                    time.sleep(0.5)
                    continue

                # Take the last assistant message (most recent)
                last_msg = msgs.last
                raw_text = last_msg.inner_text()
                if raw_text.strip():
                    break
            else:
                return {"status": "error",
                        "error": f"timeout: no response in {RESPONSE_TIMEOUT}s",
                        "code_blocks": []}

            code_blocks = extract_code_blocks(raw_text)
            return {"status": "ok", "code_blocks": code_blocks, "raw_response": raw_text}

        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc), "code_blocks": []}
        finally:
            ctx.close()


def smoke_test() -> bool:
    """Verify browser opens claude.ai without login redirect."""
    result = run_synth("Say exactly: SMOKE_OK", headless=False)
    if result["status"] == "error":
        print(f"SMOKE FAIL: {result['error']}", file=sys.stderr)
        return False
    if "SMOKE_OK" in result.get("raw_response", ""):
        print("SMOKE PASS: claude.ai responded correctly")
        return True
    print(f"SMOKE WARN: responded but no SMOKE_OK marker — inspect output")
    print(json.dumps(result, indent=2))
    return True


def main():
    ap = argparse.ArgumentParser(description="Drive claude.ai via Playwright")
    ap.add_argument("--prompt", default="", help="Prompt string to post")
    ap.add_argument("--prompt-file", default="", dest="prompt_file",
                    help="File containing the prompt")
    ap.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                    help="Open claude.ai, check login, print status and exit")
    ap.add_argument("--headless", action="store_true",
                    help="Run browser headless (default: visible for debugging)")
    ap.add_argument("--reuse-tab", action="store_true", dest="reuse_tab",
                    help="Reuse existing tab instead of opening a new one")
    ap.add_argument("--out", default="", help="Write JSON result to this file")
    args = ap.parse_args()

    if args.smoke_test:
        ok = smoke_test()
        sys.exit(0 if ok else 1)

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = sys.stdin.read()

    if not prompt.strip():
        print(json.dumps({"status": "error", "error": "empty prompt", "code_blocks": []}))
        sys.exit(1)

    result = run_synth(prompt, reuse_tab=args.reuse_tab, headless=args.headless)
    out_json = json.dumps(result, ensure_ascii=False, indent=2)
    print(out_json)
    if args.out:
        Path(args.out).write_text(out_json)
    sys.exit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
