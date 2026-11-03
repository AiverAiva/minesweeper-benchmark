"""Relabel an existing JSONL log with the exact mine-count rules.

No model call. Each multi-step episode is replayed from its fixture. Each
one-step record is restored from the gold prefix of the same length. After
every replayed step the board is compared with the log. An unrecognized
label is applied when the record shows that a move landed; otherwise rescore
stops instead of letting the replay drift.
"""

from __future__ import annotations

from pathlib import Path

from msbench.engine import EngineError, EngineSession
from msbench.run import load_position, resolve_fixture
from msbench.score import LABELS, LEGAL, classify
from msbench.summary import active_steps, load_jsonl

_COUNTED = ("supported", "lucky", "mine", "misflag", "illegal")


def rescore_logs(paths: list[Path], hints: list[Path], binary: str | Path | None) -> str:
    blocks = []
    tallies: dict[tuple[str, str], dict] = {}
    with EngineSession(binary) as session:
        for path in paths:
            text, rows = rescore_log(session, path, hints)
            blocks.append(text)
            for row in rows:
                bucket = tallies.setdefault(row["condition"], _empty_tally())
                _add_tally(bucket, row)
    blocks.append(_render_conditions(tallies))
    return "\n".join(blocks).rstrip() + "\n"


def rescore_log(session: EngineSession, path: Path, hints: list[Path]) -> tuple[str, list[dict]]:
    records = load_jsonl(path)
    lines = [f"rescore {path}"]
    rows: list[dict] = []
    if not records:
        lines.append("  empty log")
        return "\n".join(lines), rows
    for key, steps in _episodes(records):
        lines.append(f"episode {key}")
        fixture_path, fixture = resolve_fixture(steps[0], hints)
        lines.append(f"  fixture {fixture_path}")
        if steps[0].get("mode") == "one_step":
            comparisons = [_rescore_one_step(session, path, fixture, record) for record in steps]
        else:
            comparisons = _rescore_multi(session, path, fixture, active_steps(steps))
        for row in comparisons:
            if row.get("skip"):
                lines.append(f"  {row['text']}")
                continue
            rows.append(row)
            lines.append(f"  {row['text']}  tier_needed={row['tier'] or '-'}")
    changed = sum(1 for row in rows if row["old"] != row["new"])
    lines.append(
        f"changed {changed} of {len(rows)}  "
        f"reasoning {_fmt_rate([row['old'] for row in rows])} -> {_fmt_rate([row['new'] for row in rows])}"
    )
    return "\n".join(lines), rows


def _rescore_one_step(session, path: Path, fixture: dict, record: dict) -> dict:
    index = record.get("step_index")
    prefix = fixture.get("gold") or []
    if isinstance(index, int):
        prefix = prefix[:index]
    text, _status = load_position(session, fixture, prefix)
    return _replay_step(session, path, record, text)


def _rescore_multi(session, path: Path, fixture: dict, steps: list[dict]) -> list[dict]:
    text, status = load_position(session, fixture, [])
    rows = []
    for record in steps:
        if status != "playing":
            rows.append(_after_terminal(path, record, text))
            continue
        row = _replay_step(session, path, record, text)
        rows.append(row)
        if row.get("status_after"):
            status = row["status_after"]
        if row.get("text"):
            # The replayed board, once checked, is the next step's start.
            text = row.get("board") or text
    return rows


def _replay_step(session, path: Path, record: dict, before: str) -> dict:
    index = record.get("step_index")
    _check_before(path, record, before)
    label = record.get("label")
    if label is None:
        return _unlabeled(path, record, before)
    if label not in LABELS:
        return _unknown(session, path, record, before)
    outcome = classify(session, record.get("action"))
    after = outcome.get("text_after") or before
    _check_after(path, record, before, after)
    return _row(record, outcome, after)


def _unlabeled(path: Path, record: dict, before: str) -> dict:
    index = record.get("step_index")
    if _move_landed(record):
        raise EngineError(
            "replay",
            f"{path} step {index}: unlabeled record changed the board",
        )
    board = record.get("board_text")
    if not _has_board(board):
        raise EngineError(
            "replay",
            f"{path} step {index}: unlabeled record has no board_text to compare",
        )
    if before != board:
        raise EngineError(
            "replay",
            f"{path} step {index}: board before the move does not match board_text",
        )
    return {
        "skip": True,
        "text": f"{_step_label(record)}  unlabeled, board unchanged",
        "board": before,
    }


def _unknown(session, path: Path, record: dict, before: str) -> dict:
    index = record.get("step_index")
    label = record.get("label")
    if not _move_landed(record):
        raise EngineError(
            "replay",
            f"{path} step {index}: unrecognized label {label!r} and no applied move to replay",
        )
    action = record.get("action")
    if not isinstance(action, dict):
        raise EngineError(
            "replay",
            f"{path} step {index}: unrecognized label {label!r} has no action",
        )
    outcome = classify(session, action)
    after = outcome.get("text_after") or before
    _check_after(path, record, before, after)
    return _row(record, outcome, after)


def _after_terminal(path: Path, record: dict, board: str) -> dict:
    index = record.get("step_index")
    logged = record.get("board_text")
    if _has_board(logged) and logged != board:
        raise EngineError(
            "replay",
            f"{path} step {index}: board does not match the log after the game ended",
        )
    if _move_landed(record) or record.get("label") not in (None, *LABELS):
        raise EngineError(
            "replay",
            f"{path} step {index}: record after the game ended cannot be replayed",
        )
    return {
        "skip": True,
        "text": f"{_step_label(record)}  skipped (game already over)",
        "board": board,
    }


def _row(record: dict, outcome: dict, board: str) -> dict:
    old = str(record.get("label"))
    new = str(outcome.get("label"))
    text = f"{_step_label(record)}  {_describe(record.get('action'))}  {old} -> {new}"
    return {
        "old": old,
        "new": new,
        "tier": outcome.get("tier_needed"),
        "text": text,
        "status_after": outcome.get("status_after"),
        "board": board,
        "condition": _condition(record),
    }


def _check_before(path: Path, record: dict, before: str) -> None:
    board = record.get("board_text")
    if _has_board(board) and before != board:
        raise EngineError(
            "replay",
            f"{path} step {record.get('step_index')}: board before the move does not match board_text",
        )


def _check_after(path: Path, record: dict, before: str, after: str) -> None:
    expected = _expected_after(record, before)
    if expected is None:
        raise EngineError(
            "replay",
            f"{path} step {record.get('step_index')}: log has no board text to compare",
        )
    if after != expected:
        raise EngineError(
            "replay",
            f"{path} step {record.get('step_index')}: replayed board does not match the log",
        )


def _expected_after(record: dict, before: str) -> str | None:
    del before
    text_after = record.get("text_after")
    if _has_board(text_after):
        return text_after
    board = record.get("board_text")
    if _has_board(board) and not _move_landed(record):
        return board
    return None


def _move_landed(record: dict) -> bool:
    action = record.get("action")
    if not isinstance(action, dict) or not action.get("kind"):
        return False
    text_after = record.get("text_after")
    board = record.get("board_text")
    if _has_board(text_after) and text_after != board:
        return True
    if record.get("status_after") in ("won", "lost"):
        return True
    return False


def _has_board(text: object) -> bool:
    return isinstance(text, str) and "board:" in text


def _condition(record: dict) -> tuple[str, str]:
    version = record.get("prompt_version")
    board = record.get("board_format")
    return (str(version or "none"), str(board or "none"))


def _empty_tally() -> dict:
    return {"counts": {label: 0 for label in _COUNTED}, "lucky_to_supported": 0, "supported_to_lucky": 0, "legal": []}


def _add_tally(bucket: dict, row: dict) -> None:
    new = row["new"]
    old = row["old"]
    if new in bucket["counts"]:
        bucket["counts"][new] += 1
    if new in LEGAL:
        bucket["legal"].append(new)
    if old == "lucky" and new == "supported":
        bucket["lucky_to_supported"] += 1
    if old == "supported" and new == "lucky":
        bucket["supported_to_lucky"] += 1


def _render_conditions(tallies: dict[tuple[str, str], dict]) -> str:
    lines = ["conditions"]
    if not tallies:
        lines.append("  none")
        return "\n".join(lines)
    for version, board in sorted(tallies):
        bucket = tallies[(version, board)]
        counts = bucket["counts"]
        parts = "  ".join(f"{label} {counts[label]}" for label in _COUNTED)
        lines.append(f"  prompt_version={version} board_format={board}")
        lines.append(f"    {parts}")
        lines.append(f"    reasoning {_fmt_rate(bucket['legal'])}")
        lines.append(
            f"    lucky->supported {bucket['lucky_to_supported']}  "
            f"supported->lucky {bucket['supported_to_lucky']}"
        )
    return "\n".join(lines)


def _episodes(records: list[dict]) -> list[tuple[str, list[dict]]]:
    grouped: dict[str, list[dict]] = {}
    order: list[str] = []
    for record in records:
        key = f"{record.get('mode') or 'log'}:{record.get('episode_id')}"
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(record)
    return [(key, grouped[key]) for key in order]


def _step_label(record: dict) -> str:
    index = record.get("step_index")
    return f"step {index}" if isinstance(index, int) else "step ?"


def _describe(action: dict | None) -> str:
    if not action:
        return "no-action"
    at = action.get("at") or {}
    return f"{action.get('kind')} row={at.get('row')} col={at.get('col')}"


def _fmt_rate(labels: list[str]) -> str:
    legal = [label for label in labels if label in LEGAL]
    if not legal:
        return "n/a"
    supported = sum(label == "supported" for label in legal)
    return f"{supported / len(legal):.3f} ({supported}/{len(legal)})"
