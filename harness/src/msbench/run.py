"""One-step snapshots along the gold path, and multi-step play until the first miss."""

from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from msbench.client import Completion, complete_chat
from msbench.config import ConfigError, Settings, api_key, require_api
from msbench.engine import EngineError, EngineSession, repo_root
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


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def default_log_path(mode: str, run_id: str) -> Path:
    return repo_root() / "runs" / f"{run_id}-{mode}.jsonl"


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
        raise ConfigError("多步目前每個局面只走一次。MS_SAMPLES 請維持 1。")
    key = api_key()
    if complete is None:
        require_api(settings, key)
        complete = _http_complete(settings, key)
    meta = meta_from_settings(settings)
    destination = out or default_log_path(mode, meta.run_id)
    records: list[dict] = []
    failed = False
    try:
        with EngineSession(settings.binary) as session:
            for path in paths:
                fixture = load_fixture(path)
                records.extend(
                    run_fixture(
                        session,
                        fixture,
                        path.stem,
                        mode,
                        meta,
                        complete,
                        step_cap=step_cap,
                    )
                )
    except Exception:
        failed = True
        raise
    finally:
        if records:
            write_jsonl(destination, records)
            if failed:
                print(f"已寫入未完成的紀錄 {destination}", file=sys.stderr)
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
) -> list[dict]:
    if mode == "one_step":
        return run_one_step(session, fixture, fixture_name, meta, complete)
    if mode == "multi_step":
        if meta.samples != 1:
            raise ConfigError("多步目前每個局面只走一次。MS_SAMPLES 請維持 1。")
        return run_multi_step(
            session, fixture, fixture_name, meta, complete, step_cap=step_cap
        )
    raise ConfigError(f"未知模式 {mode}")


def run_one_step(session, fixture, fixture_name, meta: RunMeta, complete) -> list[dict]:
    gold = fixture.get("gold") or []
    if not gold:
        raise GoldError("fixture 沒有 gold 步，無法切一步快照")
    episode_id = episode_identity(fixture_name, fixture)
    records = []
    for index, step in enumerate(gold):
        for sample_index in range(meta.samples):
            board_text, status = load_position(session, fixture, gold[:index])
            if status != "playing":
                raise EngineError("terminal", f"第 {index} 步快照的狀態是 {status}")
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
                    position_tier=step.get("tier"),
                )
            )
    return records


def run_multi_step(
    session,
    fixture,
    fixture_name,
    meta: RunMeta,
    complete,
    *,
    step_cap: int | None = None,
) -> list[dict]:
    board_text, status = load_position(session, fixture, [])
    if status != "playing":
        raise EngineError("terminal", f"開局狀態是 {status}")
    cap = fixture["rows"] * fixture["cols"] if step_cap is None else step_cap
    episode_id = episode_identity(fixture_name, fixture)
    records = []
    while status == "playing" and len(records) < cap:
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
        if record.get("transport_error"):
            record["stop_reason"] = "transport"
            return records
        if record.get("label") != "supported":
            record["stop_reason"] = record.get("label")
            return records
        status = record.get("status_after")
        board_text = record.get("text_after") or ""
        if status == "won":
            record["stop_reason"] = "cleared"
            return records
        if status != "playing":
            record["stop_reason"] = record.get("label")
            return records
    if records and records[-1].get("stop_reason") is None:
        records[-1]["stop_reason"] = "step_cap"
    return records


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
    record.update(classify(session, completion.text, judged))
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
                f"重放 gold 失敗: {applied.get('detail')}",
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
        )

    return complete
