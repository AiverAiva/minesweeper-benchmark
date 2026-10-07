"""Timestamped progress lines on stderr.

A run can sit silent for minutes on one streaming completion, so each step
narrates what it is doing. Stdout stays reserved for final summaries, so a
run can be piped without the narration mixed in.
"""

from __future__ import annotations

import sys
import time


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def describe_action(action: dict | None) -> str:
    """Render a move as `open R4C4`, or `<no move>` when the reply had none."""
    if action is None:
        return "<no move>"
    at = action.get("at") or {}
    return f"{action.get('kind')} R{at.get('row')}C{at.get('col')}"


def duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s"
