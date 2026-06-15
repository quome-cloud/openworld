"""Drive a Claude Code session inside tmux for the build/optimize phases.

The build/optimize phases are frontier-LLM-driven: we open an interactive
``claude`` session in a tmux pane, send it an opening instruction, and watch the
filesystem for the artifact it produces (a world spec). This keeps a human able
to peek at / steer the pane while the CLI tracks progress. If tmux or claude are
unavailable, callers degrade to printing the scaffold + prompt for the user to
run manually.

Pure stdlib (subprocess/shutil/time); only used by the optional CLI.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def tmux_available() -> bool:
    return have("tmux")


def claude_available() -> bool:
    return have("claude")


def _tmux(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=True, text=True, check=check)


def start_session(session: str, cwd: Path, command: str = "claude") -> None:
    _tmux("new-session", "-d", "-s", session, "-c", str(cwd), command, check=True)


def send(session: str, text: str) -> None:
    _tmux("send-keys", "-t", session, text, "Enter")


def capture(session: str) -> str:
    return _tmux("capture-pane", "-t", session, "-p").stdout


def kill(session: str) -> None:
    _tmux("kill-session", "-t", session)


def session_exists(session: str) -> bool:
    return _tmux("has-session", "-t", session).returncode == 0


def drive(session: str, cwd: Path, message: str, wait_for: Path,
          timeout: float = 900.0, poll: float = 2.0,
          on_tail: Optional[Callable[[str], None]] = None,
          keep: bool = True) -> bool:
    """Open a claude session, send `message`, and wait until `wait_for` exists.

    Returns True if the file appeared before `timeout`. Leaves the session alive
    (keep=True) so the user can keep collaborating with Claude in the pane.
    """
    start_session(session, cwd)
    time.sleep(2.5)                                  # let claude boot
    send(session, message)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if wait_for.exists():
            return True
        if on_tail:
            on_tail(capture(session))
        if not session_exists(session):
            return wait_for.exists()
        time.sleep(poll)
    if not keep:
        kill(session)
    return wait_for.exists()
