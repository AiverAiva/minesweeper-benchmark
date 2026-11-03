"""Relabel an existing JSONL log with the exact mine-count rules.

No model call. Each multi-step episode is replayed from its fixture. Each
one-step record is restored from the gold prefix of the same length.
"""

from __future__ import annotations

from pathlib import Path

from msbench.engine import EngineSession
from msbench.run import load_position, resolve_fixture
from msbench.score import LABELS, LEGAL, classify
from msbench.summary import active_steps, load_jsonl


def rescore_logs(paths: list[Path], hints: list[Path], binary: str | Path | None) -> str:
    blocks = []
    with EngineSession(binary) as session:
        for path in paths:
            blocks.append(rescore_log(session, path, hints))
    return "\n".join(blocks).rstrip() + "\n"


def rescore_log(session: EngineSession, path: Path, hints: list[Path]) -> str:
    records = load_jsonl(path)
    lines = [f"rescore {path}"]
    if not records:
        lines.append("  empty log")
        return "\n".join(lines)
    old_labels: list[str] = []
    new_labels: list[str] = []
    changed = 0
    scored = 0
    for key, steps in _episodes(records):
        lines.append(f"episode {key}")
        fixture_path, fixture = resolve_fixture(steps[0], hints)
        lines.append(f"  fixture {fixture_path}")
        if steps[0].get("mode") == "one_step":
            comparisons = [_rescore_one_step(session, fixture, record) for record in steps]
        else:
            comparisons = _rescore_multi(session, fixture, active_steps(steps))
        for old, new, tier, text in comparisons:
            if old is None:
                lines.append(f"  {text}")
                continue
            scored += 1
            old_labels.append(old)
            new_labels.append(new)
            if old != new:
                changed += 1
            lines.append(f"  {text}  tier_needed={tier or '-'}")
    lines.append(
        f"changed {changed} of {scored}  "
        f"reasoning {_fmt_rate(old_labels)} -> {_fmt_rate(new_labels)}"
    )
    return "\n".join(lines)


def _rescore_one_step(session, fixture: dict, record: dict) -> tuple:
    if record.get("label") not in LABELS:
        return None, None, None, f"{_step_label(record)}  skipped (unlabeled)"
    index = record.get("step_index")
    prefix = fixture.get("gold") or []
    if isinstance(index, int):
        prefix = prefix[:index]
    load_position(session, fixture, prefix)
    old, new, tier, text, _after = _apply_label(session, record)
    return old, new, tier, text


def _rescore_multi(session, fixture: dict, steps: list[dict]) -> list[tuple]:
    load_position(session, fixture, [])
    status = "playing"
    rows = []
    for record in steps:
        if record.get("label") not in LABELS:
            rows.append((None, None, None, f"{_step_label(record)}  skipped (unlabeled)"))
            continue
        if status != "playing":
            rows.append((None, None, None, f"{_step_label(record)}  skipped (game already over)"))
            continue
        old, new, tier, text, after = _apply_label(session, record)
        if after:
            status = after
        rows.append((old, new, tier, text))
    return rows


def _apply_label(session, record: dict) -> tuple:
    outcome = classify(session, record.get("action"))
    old = str(record.get("label"))
    new = str(outcome.get("label"))
    text = f"{_step_label(record)}  {_describe(record.get('action'))}  {old} -> {new}"
    return old, new, outcome.get("tier_needed"), text, outcome.get("status_after")


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
