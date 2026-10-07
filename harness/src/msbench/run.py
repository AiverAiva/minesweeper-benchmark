"""One-step snapshots along the gold path, and multi-step play until the game ends."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from msbench.client import Completion, clip, complete_chat
from msbench.config import ConfigError, Settings, api_key, require_api
from msbench.engine import EngineError, EngineSession, repo_root
from msbench.progress import describe_action, duration, log
from msbench.prompt import PROMPT_SHA256, render_prompt
from msbench.score import GoldError, classify, hidden_count, replay_gold


@dataclass
class RunMeta:
    run_id: str
    model: str
    protocol: str
    track: str
    samples: int = 1
    temperature: float = 0


def new_run_id() -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def meta_from_settings(settings: Settings, run_id: str | None = None) -> RunMeta:
    return RunMeta(
        run_id=run_id or new_run_id(),
        model=settings.model,
        protocol=settings.protocol,
        track=settings.track,
        samples=settings.samples,
        temperature=settings.temperature,
    )


def load_fixture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def default_log_path(mode: str, run_id: str) -> Path:
    return repo_root() / "runs" / f"{run_id}-{mode}.jsonl"


# Consecutive steps without board progress before a multi-step episode stops.
# Guesses and rejected moves are recorded and play continues; only a stuck
# board (temp 0 repeats the same reply forever) needs this backstop.
STALL_LIMIT = 3


def check_gold(paths: list[Path], binary: str | None = None) -> None:
    with EngineSession(binary) as session:
        for path in paths:
            fixture = load_fixture(path)
            replay_gold(session, fixture)
            print(f"ok {path}  gold_steps={len(fixture['gold'])}  max_tier={fixture['max_tier']}")


def run_paths(
    paths: list[Path],
    *,
    mode: str,
    settings: Settings,
    out: Path | None = None,
    complete=None,
    step_cap: int | None = None,
) -> tuple[list[dict], Path]:
    if mode == "multi_step" and settings.samples != 1:
        raise ConfigError("multi_step plays each position once; keep MS_SAMPLES at 1")
    key = api_key()
    if complete is None:
        require_api(settings, key)
        complete = _http_complete(settings, key)
    meta = meta_from_settings(settings)
    destination = out or default_log_path(mode, meta.run_id)
    records: list[dict] = []
    handle = None
    failed = False
    started = time.monotonic()
    log(
        f"run {meta.run_id} start: mode={mode} model={meta.model} protocol={meta.protocol} "
        f"track={meta.track} temperature={meta.temperature} samples={meta.samples} "
        f"fixtures={len(paths)}"
    )
    log(f"streaming records to {destination}")

    def emit(record: dict) -> None:
        nonlocal handle
        if handle is None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            handle = destination.open("w", encoding="utf-8", newline="\n")
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()

    try:
        with EngineSession(settings.binary) as session:
            log(f"engine session started: {session.binary}")
            for index, path in enumerate(paths, start=1):
                fixture = load_fixture(path)
                header = (
                    f"fixture {index}/{len(paths)} {path.stem}: seed={fixture.get('seed')} "
                    f"{fixture.get('rows')}x{fixture.get('cols')} {fixture.get('mines_total')} mines "
                    f"bbbv={fixture.get('bbbv')}"
                )
                if mode == "one_step":
                    header += f" gold_steps={len(fixture.get('gold') or [])}"
                log(header)
                records.extend(
                    run_fixture(
                        session,
                        fixture,
                        path.stem,
                        mode,
                        meta,
                        complete,
                        step_cap=step_cap,
                        on_record=emit,
                    )
                )
    except Exception:
        failed = True
        raise
    finally:
        if handle is not None:
            handle.close()
        spent = duration(time.monotonic() - started)
        if failed:
            log(f"run aborted after {spent}: {len(records)} records in {destination}")
        else:
            log(f"run done in {spent}: {len(records)} records -> {destination}")
    return records, destination


def run_fixture(
    session: EngineSession,
    fixture: dict,
    fixture_name: str,
    mode: str,
    meta: RunMeta,
    complete,
    *,
    step_cap: int | None = None,
    on_record: Callable[[dict], None] | None = None,
) -> list[dict]:
    if mode == "one_step":
        return run_one_step(session, fixture, fixture_name, meta, complete, on_record=on_record)
    if mode == "multi_step":
        if meta.samples != 1:
            raise ConfigError("multi_step plays each position once; keep MS_SAMPLES at 1")
        return run_multi_step(
            session, fixture, fixture_name, meta, complete, step_cap=step_cap, on_record=on_record
        )
    raise ConfigError(f"unknown mode {mode}")


def run_one_step(
    session, fixture, fixture_name, meta: RunMeta, complete, *, on_record: Callable[[dict], None] | None = None
) -> list[dict]:
    gold = fixture.get("gold") or []
    if not gold:
        raise GoldError("fixture has no gold steps; one_step needs them for snapshots")
    episode_id = episode_identity(fixture_name, fixture)
    total = len(gold)
    records = []
    for index, step in enumerate(gold):
        tier = step.get("tier")
        for sample_index in range(meta.samples):
            board_text, status = load_position(session, fixture, gold[:index])
            if status != "playing":
                raise EngineError(
                    "terminal",
                    f"snapshot {index + 1}/{total} of {fixture_name} is {status}, expected playing",
                )
            head = f"{fixture_name} step {index + 1}/{total}"
            if meta.samples > 1:
                head += f" sample {sample_index + 1}/{meta.samples}"
            log(f"{head}: asking model (hidden={hidden_count(board_text)}, pos_tier={tier})")
            records.append(
                ask(
                    session,
                    complete,
                    meta,
                    fixture,
                    board_text,
                    mode="one_step",
                    fixture_name=fixture_name,
                    episode_id=episode_id,
                    item_id=f"{episode_id}:step{index}",
                    step_index=index,
                    sample_index=sample_index,
                    position_tier=tier,
                )
            )
            log(f"{head}: {step_summary(records[-1], board_text)}")
            if on_record is not None:
                on_record(records[-1])
    return records


def run_multi_step(
    session,
    fixture,
    fixture_name,
    meta: RunMeta,
    complete,
    *,
    step_cap: int | None = None,
    on_record: Callable[[dict], None] | None = None,
) -> list[dict]:
    board_text, status = load_position(session, fixture, [])
    if status != "playing":
        raise EngineError("terminal", f"episode start of {fixture_name} is {status}, expected playing")
    cap = fixture["rows"] * fixture["cols"] if step_cap is None else step_cap
    episode_id = episode_identity(fixture_name, fixture)
    log(
        f"{fixture_name} episode start: {fixture.get('rows')}x{fixture.get('cols')} "
        f"{fixture.get('mines_total')} mines bbbv={fixture.get('bbbv')}, step cap {cap}"
    )
    records = []
    stalls = 0
    while status == "playing" and len(records) < cap:
        head = f"{fixture_name} step {len(records) + 1}"
        log(f"{head}: asking model (hidden={hidden_count(board_text)})")
        record = ask(
            session,
            complete,
            meta,
            fixture,
            board_text,
            mode="multi_step",
            fixture_name=fixture_name,
            episode_id=episode_id,
            item_id=f"{episode_id}:step{len(records)}",
            step_index=len(records),
            sample_index=0,
            position_tier=None,
        )
        records.append(record)
        log(f"{head}: {step_summary(record, board_text)}")
        if record.get("transport_error"):
            record["stop_reason"] = "transport"
        elif record.get("status_after") == "lost":
            record["stop_reason"] = "mine"
        elif record.get("status_after") == "won":
            record["stop_reason"] = "cleared"
        elif record.get("text_after") and record.get("text_after") != board_text:
            status = record.get("status_after") or status
            board_text = record.get("text_after") or board_text
            stalls = 0
            if len(records) >= cap:
                record["stop_reason"] = "step_cap"
            else:
                if on_record is not None:
                    on_record(record)
                continue
        else:
            stalls += 1
            if len(records) >= cap:
                record["stop_reason"] = "step_cap"
            elif stalls >= STALL_LIMIT:
                record["stop_reason"] = record.get("label")
            else:
                log(f"{head}: no board change (stall {stalls}/{STALL_LIMIT})")
                if on_record is not None:
                    on_record(record)
                continue
        if on_record is not None:
            on_record(record)
        _episode_end(fixture_name, records)
        return records
    _episode_end(fixture_name, records)
    return records


def step_summary(record: dict, board_before: str) -> str:
    """One line on what the model played and what the move changed."""
    if record.get("transport_error"):
        return f"transport error: {record['transport_error']}"
    played = f"{describe_action(record.get('action'))} -> {record.get('label')}"
    detail = record.get("detail")
    if detail:
        played += f" ({clip(str(detail), 120)})"
    parts = [played]
    if record.get("status_after"):
        parts.append(f"status={record['status_after']}")
    hidden = str(record.get("hidden_count"))
    after = record.get("text_after")
    if after:
        hidden += f"->{hidden_count(after)}"
    parts.append(f"hidden {hidden}")
    latency, tokens = _step_timing(record)
    if latency is not None:
        parts.append(f"{latency / 1000:.1f}s")
    if tokens is not None:
        parts.append(f"{tokens} tok")
    return ", ".join(parts)


def _step_timing(record: dict) -> tuple[float | None, int | None]:
    attempts = record.get("attempts") or []
    latency = next(
        (item.get("latency_ms") for item in reversed(attempts) if item.get("latency_ms") is not None),
        None,
    )
    tokens = next(
        (item.get("total_tokens") for item in reversed(attempts) if item.get("total_tokens") is not None),
        None,
    )
    return latency, tokens


def _episode_end(fixture_name: str, records: list[dict]) -> None:
    reason = records[-1].get("stop_reason") if records else "nothing played"
    log(f"{fixture_name} episode end: stop={reason}, steps={len(records)}")


def ask(
    session,
    complete,
    meta: RunMeta,
    fixture: dict,
    board_text: str,
    *,
    mode: str,
    fixture_name: str,
    episode_id: str,
    item_id: str,
    step_index: int,
    sample_index: int,
    position_tier: str | None,
) -> dict:
    completion: Completion = complete(render_prompt(board_text))
    judged = session.forced()
    record = {
        "schema": 1,
        "run_id": meta.run_id,
        "model": meta.model,
        "protocol": meta.protocol,
        "track": meta.track,
        "temperature": meta.temperature,
        "item_id": item_id,
        "episode_id": episode_id,
        "fixture": fixture_name,
        "seed": fixture.get("seed"),
        "rows": fixture.get("rows"),
        "cols": fixture.get("cols"),
        "mines_total": fixture.get("mines_total"),
        "bbbv": fixture.get("bbbv"),
        "position_tier": position_tier,
        "step_index": step_index,
        "sample_index": sample_index,
        "mode": mode,
        "prompt_sha256": PROMPT_SHA256,
        "board_text": board_text,
        "response_text": completion.text,
        "reasoning_text": completion.reasoning,
        "label": None,
        "action": None,
        "forced_tier": judged.get("tier"),
        "forced_actions": judged.get("actions"),
        "forced_count": len(judged.get("actions") or []),
        "forced_open_count": sum(
            1 for action in judged.get("actions") or [] if action.get("kind") == "open"
        ),
        "hidden_count": hidden_count(board_text),
        "status_after": None,
        "text_after": None,
        "stop_reason": None,
        "detail": None,
        "transport_error": completion.transport_error,
        "attempts": completion.attempts,
    }
    if position_tier is None:
        record["position_tier"] = record["forced_tier"]
    if completion.transport_error:
        return record
    record.update(classify(session, completion.action, judged))
    return record


def load_position(session: EngineSession, fixture: dict, prefix: list[dict]) -> tuple[str, str]:
    opened = session.new_fixture(fixture)
    text = opened.get("text") or ""
    status = opened.get("status")
    for step in prefix:
        applied = session.apply(step["action"])
        if not applied.get("ok"):
            raise EngineError(
                str(applied.get("error") or "apply"),
                f"gold replay failed: {applied.get('detail')}",
            )
        text = applied.get("text") or ""
        status = applied.get("status")
    return text, status


def episode_identity(fixture_name: str, fixture: dict) -> str:
    return (
        f"{fixture_name}:{fixture.get('seed')}:"
        f"{fixture.get('rows')}x{fixture.get('cols')}m{fixture.get('mines_total')}"
    )


def _http_complete(settings: Settings, key: str):
    def complete(prompt: str) -> Completion:
        return complete_chat(
            prompt=prompt,
            base=settings.base,
            key=key,
            model=settings.model,
            temperature=settings.temperature,
            max_retries=settings.max_retries,
            timeout=settings.timeout,
            report=log,
        )

    return complete
