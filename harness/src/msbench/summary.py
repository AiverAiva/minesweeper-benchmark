"""Aggregate a JSONL run. Scores stay in rates; tokens and latency stay beside them."""

from __future__ import annotations

import json
import random
from pathlib import Path

from msbench.score import LABELS, LEGAL

TIERS = ("single", "subset", "enumerate")
TIER_NAMES = {"single": "單格", "subset": "子集", "enumerate": "枚舉"}
CLASS_NAMES = {
    "supported": "必走",
    "lucky": "安全但推不出",
    "mine": "踩雷",
    "illegal": "非法",
    "format": "格式",
}
BAND_NAMES = {"beginner": "初級", "intermediate": "中級", "other": "其他"}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def summarize(records: list[dict], *, draws: int = 1000, seed: int = 0) -> str:
    return render(analyze(records, draws=draws, seed=seed))


def analyze(records: list[dict], *, draws: int = 1000, seed: int = 0) -> dict:
    return {
        "models": _column(records, "model"),
        "protocols": _column(records, "protocol"),
        "tracks": _column(records, "track"),
        "prompts": _column(records, "prompt_sha256"),
        "one_step": _one_step(records, draws=draws, seed=seed),
        "multi_step": _multi_step(records, draws=draws, seed=seed + 1),
        "requests": _requests(records),
    }


def paired_bootstrap(
    left: list[dict], right: list[dict], *, draws: int = 1000, seed: int = 0
) -> dict:
    diffs = _paired_diffs(left, right)
    stats = bootstrap_mean(diffs, draws=draws, seed=seed)
    return {"n": len(diffs), **stats} if stats is not None else {"n": 0, "mean": None, "low": None, "high": None}


def bootstrap_mean(values: list[float], *, draws: int = 1000, seed: int = 0) -> dict | None:
    if not values:
        return None
    point = sum(values) / len(values)
    rng = random.Random(seed)
    count = len(values)
    means = []
    for _ in range(draws):
        total = 0.0
        for _ in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)
    return {"mean": point, "low": percentile(means, 0.025), "high": percentile(means, 0.975)}


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def size_band(record: dict) -> str:
    if record.get("rows") == 9 and record.get("cols") == 9 and record.get("mines_total") == 10:
        return "beginner"
    if record.get("rows") == 16 and record.get("cols") == 16 and record.get("mines_total") == 40:
        return "intermediate"
    return "other"


def _one_step(records: list[dict], *, draws: int, seed: int) -> dict:
    scheduled = [record for record in records if record.get("mode") == "one_step"]
    body = [record for record in scheduled if record.get("label") in LABELS]
    counts = _counts(body)
    scores = _legal_scores(body)
    return {
        "scheduled": len(scheduled),
        "n": len(body),
        "counts": counts,
        "rates": _rates(counts, len(body)),
        "n_legal": counts["supported"] + counts["lucky"] + counts["mine"],
        "reasoning": _mean(scores),
        "reasoning_pooled": _ratio(counts["supported"], counts["supported"] + counts["lucky"] + counts["mine"]),
        "reasoning_ci": bootstrap_mean(scores, draws=draws, seed=seed),
        "by_tier": {tier: _tier_slice([record for record in body if record.get("position_tier") == tier]) for tier in TIERS},
        "chance": _chance(body),
        "transport_rate": _ratio(len(scheduled) - len(body), len(scheduled)),
    }


def _tier_slice(records: list[dict]) -> dict:
    counts = _counts(records)
    legal = counts["supported"] + counts["lucky"] + counts["mine"]
    return {
        "n": len(records),
        "n_legal": legal,
        "reasoning": _mean(_legal_scores(records)),
    }


def _multi_step(records: list[dict], *, draws: int, seed: int) -> dict:
    episodes = [_score_episode(steps) for _key, steps in _episodes(records)]
    scored = [episode for episode in episodes if not episode["transport"]]
    clears = [1.0 if episode["cleared"] else 0.0 for episode in scored]
    prefixes = [episode["prefix"] for episode in scored]
    bands: dict[str, list[float]] = {}
    for episode in scored:
        bands.setdefault(episode["band"], []).append(episode["prefix"])
    return {
        "episodes": len(episodes),
        "scored": len(scored),
        "transport": len(episodes) - len(scored),
        "transport_rate": _ratio(len(episodes) - len(scored), len(episodes)),
        "clear_rate": _mean(clears),
        "clear_ci": bootstrap_mean(clears, draws=draws, seed=seed),
        "prefix_mean": _mean(prefixes),
        "prefix_by_band": {band: _mean(values) for band, values in bands.items()},
    }


def _requests(records: list[dict]) -> dict:
    attempts = [attempt for record in records for attempt in (record.get("attempts") or [])]
    retries = sum(max(0, len(record.get("attempts") or []) - 1) for record in records)
    body = [record for record in records if record.get("label") in LABELS]
    inputs = _numbers(body, "input_tokens")
    outputs = _numbers(body, "output_tokens")
    reasoning = _numbers(body, "reasoning_tokens")
    latencies = [
        attempt["latency_ms"]
        for attempt in attempts
        if isinstance(attempt.get("latency_ms"), (int, float))
    ]
    length_records = [
        record for record in body if _last(record).get("finish_reason") == "length"
    ]
    format_records = [record for record in body if record.get("label") == "format"]
    both = [record for record in length_records if record.get("label") == "format"]
    return {
        "requests": len(attempts),
        "retries": retries,
        "retry_rate": _ratio(retries, len(attempts)),
        "input": _spread(inputs),
        "output": _spread(outputs),
        "reasoning": _spread(reasoning),
        "latency": {
            "n": len(latencies),
            "median": percentile(latencies, 0.5) if latencies else None,
            "p90": percentile(latencies, 0.9) if latencies else None,
        },
        "episode_cost": _episode_costs(records),
        "length": len(length_records),
        "format": len(format_records),
        "length_and_format": len(both),
        "body": len(body),
    }


def _episode_costs(records: list[dict]) -> dict:
    totals = []
    for _key, steps in _episodes(records):
        total = 0
        known = False
        for step in steps:
            attempt = _last(step)
            for key in ("input_tokens", "output_tokens"):
                value = attempt.get(key)
                if isinstance(value, (int, float)):
                    total += value
                    known = True
        if known:
            totals.append(float(total))
    return {
        "n": len(totals),
        "sum": sum(totals) if totals else None,
        "median": percentile(totals, 0.5) if totals else None,
    }


def _score_episode(steps: list[dict]) -> dict:
    transport = any(step.get("label") not in LABELS for step in steps)
    prefix = 0
    for step in steps:
        if step.get("label") == "supported":
            prefix += 1
        else:
            break
    return {
        "transport": transport,
        "cleared": bool(steps) and steps[-1].get("stop_reason") == "cleared" and not transport,
        "prefix": prefix,
        "band": size_band(steps[0]) if steps else "other",
    }


def _episodes(records: list[dict]) -> list[tuple[str, list[dict]]]:
    grouped: dict[str, list[dict]] = {}
    order: list[str] = []
    for record in records:
        if record.get("mode") != "multi_step":
            continue
        key = str(record.get("episode_id"))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(record)
    return [
        (key, sorted(grouped[key], key=lambda record: record.get("step_index") or 0))
        for key in order
    ]


def _legal_scores(records: list[dict]) -> list[float]:
    scores = []
    for samples in _item_groups(records).values():
        legal = [sample for sample in samples if sample.get("label") in LEGAL]
        if not legal:
            continue
        scores.append(sum(sample.get("label") == "supported" for sample in legal) / len(legal))
    return scores


def _chance(records: list[dict]) -> float | None:
    ratios = []
    for samples in _item_groups(records).values():
        sample_ratios = []
        for record in samples:
            hidden = record.get("hidden_count") or 0
            if hidden:
                sample_ratios.append((record.get("forced_open_count") or 0) / hidden)
        if sample_ratios:
            ratios.append(sum(sample_ratios) / len(sample_ratios))
    return _mean(ratios)


def _paired_diffs(left: list[dict], right: list[dict]) -> list[float]:
    left_scores = _score_map(left)
    right_scores = _score_map(right)
    keys = sorted(set(left_scores) & set(right_scores))
    return [right_scores[key] - left_scores[key] for key in keys]


def _score_map(records: list[dict]) -> dict[str, float]:
    body = [record for record in records if record.get("mode") == "one_step" and record.get("label") in LABELS]
    scores = {}
    for item_id, samples in _item_groups(body).items():
        legal = [sample for sample in samples if sample.get("label") in LEGAL]
        if not legal:
            continue
        scores[item_id] = sum(sample.get("label") == "supported" for sample in legal) / len(legal)
    return scores


def _item_groups(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record.get("item_id")), []).append(record)
    return groups


def _counts(records: list[dict]) -> dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for record in records:
        label = record.get("label")
        if label in counts:
            counts[label] += 1
    return counts


def _rates(counts: dict[str, int], total: int) -> dict[str, float | None]:
    if total == 0:
        return {label: None for label in LABELS}
    return {label: counts[label] / total for label in LABELS}


def _numbers(records: list[dict], key: str) -> list[float]:
    values = []
    for record in records:
        value = _last(record).get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _spread(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "sum": None, "median": None, "iqr": None}
    low = percentile(values, 0.25)
    high = percentile(values, 0.75)
    return {
        "n": len(values),
        "sum": sum(values),
        "median": percentile(values, 0.5),
        "iqr": high - low,
    }


def _last(record: dict) -> dict:
    attempts = record.get("attempts") or []
    if not attempts:
        return {}
    last = attempts[-1]
    return last if isinstance(last, dict) else {}


def _column(records: list[dict], key: str) -> list:
    found = []
    for record in records:
        value = record.get(key)
        if value not in (None, "") and value not in found:
            found.append(value)
    return found


def _ratio(part: int, whole: int) -> float | None:
    if whole == 0:
        return None
    return part / whole


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def render(report: dict) -> str:
    lines = ["msbench 摘要"]
    lines.append(f"模型: {_join(report['models'])}")
    lines.append(f"協定: {_join(report['protocols'])}")
    lines.append(f"軌道: {_join(report['tracks'])}")
    lines.append(f"提示前綴 sha256: {_join(report['prompts'])}")
    lines.append("")
    lines.extend(_render_one_step(report["one_step"]))
    lines.append("")
    lines.extend(_render_multi_step(report["multi_step"]))
    lines.append("")
    lines.extend(_render_requests(report["requests"]))
    return "\n".join(lines) + "\n"


def render_paired(stats: dict) -> str:
    if stats["mean"] is None:
        return f"配對題數: {stats['n']}\n沒有兩邊都算得進推理率的題。\n"
    return (
        f"配對題數: {stats['n']}\n"
        f"推理率差（第二份減第一份）: {_num(stats['mean'])}  "
        f"95% 區間 [{_num(stats['low'])}, {_num(stats['high'])}]\n"
    )


def _render_one_step(block: dict) -> list[str]:
    lines = [f"一步  有正文 {block['n']}  排程 {block['scheduled']}"]
    if block["n"] == 0:
        lines.append("沒有一步正文。")
        return lines
    for label in LABELS:
        lines.append(
            f"  {CLASS_NAMES[label]} {label:<10} {block['counts'][label]}/{block['n']}  {_num(block['rates'][label])}"
        )
    lines.append(f"  五類合計 {_num(sum(rate for rate in block['rates'].values() if rate is not None))}")
    lines.append(
        f"  推理率 {_num(block['reasoning'])}  {_ci(block['reasoning_ci'])}"
        f"  n={block['n']}  n_legal={block['n_legal']}"
    )
    for tier in TIERS:
        sliver = block["by_tier"][tier]
        lines.append(
            f"  層級 {TIER_NAMES[tier]} {tier:<10} 推理率 {_num(sliver['reasoning'])}"
            f"  n={sliver['n']}  n_legal={sliver['n_legal']}"
        )
    lines.append(f"  機會基線（必走 open / 未開格，逐題平均） {_num(block['chance'])}")
    return lines


def _render_multi_step(block: dict) -> list[str]:
    lines = [
        f"多步  計分局 {block['scored']}  傳輸失敗局 {block['transport']}  全部 {block['episodes']}"
    ]
    if block["episodes"] == 0:
        lines.append("沒有多步紀錄。")
        return lines
    lines.append(f"  通關率 {_num(block['clear_rate'])}  {_ci(block['clear_ci'])}")
    lines.append(f"  平均前綴 {_num(block['prefix_mean'])}")
    for band in ("beginner", "intermediate", "other"):
        if band not in block["prefix_by_band"]:
            continue
        lines.append(f"  {BAND_NAMES[band]}平均前綴 {_num(block['prefix_by_band'][band])}")
    return lines


def _render_requests(block: dict) -> list[str]:
    lines = ["請求"]
    lines.append(
        f"  次數 {block['requests']}  重試 {block['retries']}  重試率 {_num(block['retry_rate'])}"
    )
    lines.append(f"  input tokens {_spread_text(block['input'])}")
    lines.append(f"  output tokens {_spread_text(block['output'])}")
    lines.append(f"  reasoning tokens {_spread_text(block['reasoning'])}")
    latency = block["latency"]
    lines.append(
        f"  延遲毫秒 中位數 {_num(latency['median'])}  P90 {_num(latency['p90'])}  n={latency['n']}"
    )
    cost = block["episode_cost"]
    lines.append(
        f"  多步每局 token 總和 中位數 {_num(cost['median'])}  總和 {_num(cost['sum'], 0)}  局數 {cost['n']}"
    )
    lines.append(
        f"  finish_reason=length {block['length']}/{block['body']}  "
        f"其中格式失敗 {block['length_and_format']}  "
        f"格式失敗共 {block['format']}"
    )
    return lines


def _spread_text(spread: dict) -> str:
    if spread["n"] == 0:
        return "空"
    return f"總和 {_num(spread['sum'], 0)}  中位數 {_num(spread['median'])}  IQR {_num(spread['iqr'])}  n={spread['n']}"


def _ci(stats: dict | None) -> str:
    if stats is None:
        return "95% 區間 空"
    return f"95% 區間 [{_num(stats['low'])}, {_num(stats['high'])}]"


def _join(values: list) -> str:
    if not values:
        return "空"
    return ", ".join(str(value) for value in values)


def _num(value, digits: int = 3) -> str:
    if value is None:
        return "空"
    return f"{value:.{digits}f}"
