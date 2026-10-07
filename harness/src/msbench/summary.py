"""Aggregate a JSONL run. Scores stay in rates; tokens and latency stay beside them."""

from __future__ import annotations

import json
import random
from pathlib import Path

from msbench.score import LABELS, LEGAL

TIERS = ("single", "subset", "enumerate")


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
    labeled = [
        record
        for _key, steps in _episodes(records)
        for record in steps
        if record.get("label") in LABELS
    ]
    return {
        "episodes": len(episodes),
        "scored": len(scored),
        "transport": len(episodes) - len(scored),
        "transport_rate": _ratio(len(episodes) - len(scored), len(episodes)),
        "counts": _counts(labeled),
        "clear_rate": _mean(clears),
        "clear_ci": bootstrap_mean(clears, draws=draws, seed=seed),
        "prefix_mean": _mean(prefixes),
    }


def _requests(records: list[dict]) -> dict:
    attempts = [attempt for record in records for attempt in (record.get("attempts") or [])]
    retries = sum(max(0, len(record.get("attempts") or []) - 1) for record in records)
    body = [record for record in records if record.get("label") in LABELS]
    inputs = _numbers(body, "input_tokens")
    outputs = _numbers(body, "output_tokens")
    reasoning = _numbers(body, "reasoning_tokens")
    totals = _totals(body)
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
        "total": _spread(totals),
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
        total = 0.0
        known = False
        for step in steps:
            value = _billed(_last(step))
            if value is None:
                continue
            total += value
            known = True
        if known:
            totals.append(total)
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


def _billed(attempt: dict) -> float | None:
    """Tokens the provider billed for one call.

    reasoning_tokens is already inside completion_tokens, so it is not added again.
    """
    value = attempt.get("total_tokens")
    if isinstance(value, (int, float)):
        return float(value)
    prompt = attempt.get("input_tokens")
    completion = attempt.get("output_tokens")
    if isinstance(prompt, (int, float)) and isinstance(completion, (int, float)):
        return float(prompt) + float(completion)
    return None


def _totals(records: list[dict]) -> list[float]:
    values = []
    for record in records:
        value = _billed(_last(record))
        if value is not None:
            values.append(value)
    return values


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
    lines = ["msbench summary"]
    lines.append(f"models: {_join(report['models'])}")
    lines.append(f"protocol: {_join(report['protocols'])}")
    lines.append(f"track: {_join(report['tracks'])}")
    lines.append(f"prompt sha256: {_join(report['prompts'])}")
    lines.append("")
    lines.extend(_render_one_step(report["one_step"]))
    lines.append("")
    lines.extend(_render_multi_step(report["multi_step"]))
    lines.append("")
    lines.extend(_render_requests(report["requests"]))
    return "\n".join(lines) + "\n"


def render_paired(stats: dict) -> str:
    if stats["mean"] is None:
        return f"paired items: {stats['n']}\nno items scored on both sides.\n"
    return (
        f"paired items: {stats['n']}\n"
        f"reasoning rate diff (second minus first): {_num(stats['mean'])}  "
        f"95% CI [{_num(stats['low'])}, {_num(stats['high'])}]\n"
    )


def _render_one_step(block: dict) -> list[str]:
    lines = [f"one-step  scored {block['n']}  scheduled {block['scheduled']}"]
    if block["n"] == 0:
        lines.append("no scored one-step records.")
        return lines
    for label in LABELS:
        lines.append(
            f"  {label:<10} {block['counts'][label]}/{block['n']}  {_num(block['rates'][label])}"
        )
    lines.append(
        f"  reasoning rate {_num(block['reasoning'])}  {_ci(block['reasoning_ci'])}"
        f"  n={block['n']}  n_legal={block['n_legal']}"
    )
    for tier in TIERS:
        sliver = block["by_tier"][tier]
        lines.append(
            f"  tier {tier:<10} reasoning {_num(sliver['reasoning'])}"
            f"  n={sliver['n']}  n_legal={sliver['n_legal']}"
        )
    lines.append(f"  chance baseline (forced opens / hidden cells, mean per item) {_num(block['chance'])}")
    return lines


def _render_multi_step(block: dict) -> list[str]:
    lines = [
        f"multi-step  scored {block['scored']}  transport {block['transport']}  total {block['episodes']}"
    ]
    if block["episodes"] == 0:
        lines.append("no multi-step records.")
        return lines
    lines.append(f"  clear rate {_num(block['clear_rate'])}  {_ci(block['clear_ci'])}")
    lines.append(
        "  " + "  ".join(f"{label} {block['counts'][label]}" for label in LABELS)
    )
    lines.append(f"  mean prefix {_num(block['prefix_mean'])}")
    return lines


def _render_requests(block: dict) -> list[str]:
    lines = ["requests"]
    lines.append(
        f"  calls {block['requests']}  retries {block['retries']}  retry rate {_num(block['retry_rate'])}"
    )
    lines.append(
        "  tokens per request  "
        f"output median {_num(block['output']['median'])}  "
        f"reasoning median {_num(block['reasoning']['median'])}  "
        f"total median {_num(block['total']['median'])}"
    )
    cost = block["episode_cost"]
    lines.append(
        f"  multi-step episode tokens  median {_num(cost['median'])}  sum {_num(cost['sum'], 0)}  episodes {cost['n']}"
    )
    lines.append(
        f"  finish_reason=length {block['length']}/{block['body']}  "
        f"format failures among them {block['length_and_format']}  "
        f"format failures total {block['format']}"
    )
    return lines


def _ci(stats: dict | None) -> str:
    if stats is None:
        return "95% CI none"
    return f"95% CI [{_num(stats['low'])}, {_num(stats['high'])}]"


def _join(values: list) -> str:
    if not values:
        return "none"
    return ", ".join(str(value) for value in values)


def _num(value, digits: int = 3) -> str:
    if value is None:
        return "none"
    return f"{value:.{digits}f}"
