import pytest

from msbench.summary import analyze, bootstrap_mean, paired_bootstrap, summarize


def attempt(latency, inp, out, reasoning=None, finish="stop", http=200):
    return {
        "attempt": 0,
        "http_status": http,
        "latency_ms": latency,
        "finish_reason": finish,
        "input_tokens": inp,
        "output_tokens": out,
        "reasoning_tokens": reasoning,
    }


def one_step(item, label, tier, forced_open, hidden, latency, inp, out, reasoning=None, finish="stop"):
    return {
        "mode": "one_step",
        "item_id": item,
        "episode_id": "board",
        "label": label,
        "position_tier": tier,
        "sample_index": 0,
        "step_index": 0,
        "forced_open_count": forced_open,
        "hidden_count": hidden,
        "rows": 9,
        "cols": 9,
        "mines_total": 10,
        "attempts": [attempt(latency, inp, out, reasoning, finish)],
    }


def multi(episode, step, label, stop, rows, cols, mines, inp=None, out=None):
    return {
        "mode": "multi_step",
        "episode_id": episode,
        "item_id": f"{episode}:step{step}",
        "step_index": step,
        "label": label,
        "stop_reason": stop,
        "rows": rows,
        "cols": cols,
        "mines_total": mines,
        "attempts": [attempt(None, inp, out)],
    }


def sample_log():
    return [
        one_step("a", "supported", "single", 1, 4, 10, 10, 5),
        one_step("b", "lucky", "subset", 2, 4, 30, 20, 6),
        one_step("c", "mine", "enumerate", 1, 2, 50, 30, 7, reasoning=4),
        one_step("d", "illegal", "single", 1, 4, 70, 40, 8),
        one_step("e", "format", "subset", 0, 5, 90, 50, 9, finish="length"),
        {
            "mode": "one_step",
            "item_id": "f",
            "label": None,
            "transport_error": "timed out",
            "attempts": [attempt(100, None, None)],
        },
        multi("clear", 0, "supported", None, 9, 9, 10, 10, 1),
        multi("clear", 1, "supported", "cleared", 9, 9, 10, 10, 1),
        multi("miss", 0, "lucky", "lucky", 16, 16, 40, 8, 2),
        multi("drop", 0, None, "transport", 3, 3, 2),
    ]


def test_five_rates_sum_to_one_and_reasoning_excludes_format_and_illegal():
    report = analyze(sample_log(), draws=50)
    one = report["one_step"]
    assert one["n"] == 5
    assert one["scheduled"] == 6
    assert one["transport_rate"] == pytest.approx(1 / 6)
    assert sum(one["rates"].values()) == pytest.approx(1)
    assert one["rates"]["supported"] == pytest.approx(0.2)
    assert one["n_legal"] == 3
    assert one["reasoning"] == pytest.approx(1 / 3)
    assert one["reasoning_ci"]["low"] <= one["reasoning"] <= one["reasoning_ci"]["high"]
    assert one["by_tier"]["single"] == {"n": 2, "n_legal": 1, "reasoning": pytest.approx(1)}
    assert one["by_tier"]["subset"]["n"] == 2
    assert one["by_tier"]["subset"]["n_legal"] == 1
    assert one["by_tier"]["subset"]["reasoning"] == pytest.approx(0)
    assert one["by_tier"]["enumerate"]["reasoning"] == pytest.approx(0)
    assert one["chance"] == pytest.approx(0.3)


def test_multi_step_clear_rate_and_prefix_skip_transport():
    multi_step = analyze(sample_log(), draws=50)["multi_step"]
    assert multi_step["episodes"] == 3
    assert multi_step["scored"] == 2
    assert multi_step["transport"] == 1
    assert multi_step["clear_rate"] == pytest.approx(0.5)
    assert multi_step["prefix_mean"] == pytest.approx(1)
    assert multi_step["counts"] == {
        "supported": 2,
        "lucky": 1,
        "mine": 0,
        "illegal": 0,
        "format": 0,
    }


def test_tokens_use_accepted_attempts_and_latency_uses_every_request():
    requests = analyze(sample_log(), draws=20)["requests"]
    assert requests["requests"] == 10
    assert requests["retries"] == 0
    assert requests["input"]["sum"] == pytest.approx(178)
    assert requests["input"]["median"] == pytest.approx(15)
    assert requests["output"]["sum"] == pytest.approx(39)
    assert requests["reasoning"]["n"] == 1
    assert requests["reasoning"]["sum"] == pytest.approx(4)
    assert requests["total"]["n"] == 8
    assert requests["total"]["sum"] == pytest.approx(217)
    assert requests["total"]["median"] == pytest.approx(20.5)
    assert requests["latency"]["n"] == 6
    assert requests["latency"]["median"] == pytest.approx(60)
    assert requests["latency"]["p90"] == pytest.approx(95)
    assert requests["episode_cost"]["median"] == pytest.approx(16)
    assert requests["episode_cost"]["sum"] == pytest.approx(32)
    assert requests["length"] == 1
    assert requests["format"] == 1
    assert requests["length_and_format"] == 1
    assert requests["body"] == 8


def test_retry_rate_counts_failed_attempts_and_tokens_come_from_the_last_one():
    record = one_step("a", "supported", "single", 1, 4, 8, 3, 1)
    record["attempts"] = [
        attempt(5, None, None, http=500, finish=None),
        attempt(8, 3, 1),
    ]
    record["attempts"][1]["attempt"] = 1
    requests = analyze([record])["requests"]
    assert requests["requests"] == 2
    assert requests["retries"] == 1
    assert requests["retry_rate"] == pytest.approx(0.5)
    assert requests["input"]["sum"] == pytest.approx(3)
    assert requests["latency"]["n"] == 2


def test_reasoning_rate_is_empty_when_every_body_is_format_or_illegal():
    records = [
        one_step("a", "format", "single", 1, 4, 1, 1, 1),
        one_step("b", "illegal", "subset", 1, 4, 1, 1, 1),
    ]
    one = analyze(records)["one_step"]
    assert one["reasoning"] is None
    assert one["n_legal"] == 0
    assert sum(one["rates"].values()) == pytest.approx(1)


def test_repeated_samples_are_averaged_within_the_item_first():
    records = [
        one_step("a", "supported", "single", 1, 4, 1, 1, 1),
        one_step("b", "lucky", "single", 1, 4, 1, 1, 1),
        one_step("b", "lucky", "single", 1, 4, 1, 1, 1),
        one_step("b", "supported", "single", 1, 4, 1, 1, 1),
    ]
    records[1]["sample_index"] = 0
    records[2]["sample_index"] = 1
    records[3]["sample_index"] = 2
    one = analyze(records)["one_step"]
    assert one["reasoning"] == pytest.approx((1 + 1 / 3) / 2)


def test_bootstrap_is_stable_for_a_seed():
    assert bootstrap_mean([1, 0, 0, 1], draws=30, seed=1) == bootstrap_mean([1, 0, 0, 1], draws=30, seed=1)


def test_paired_bootstrap_uses_items_present_on_both_sides():
    left = [
        one_step("a", "supported", "single", 1, 4, 1, 1, 1),
        one_step("b", "lucky", "single", 1, 4, 1, 1, 1),
        one_step("c", "format", "single", 1, 4, 1, 1, 1),
    ]
    right = [
        one_step("a", "supported", "single", 1, 4, 1, 1, 1),
        one_step("b", "supported", "single", 1, 4, 1, 1, 1),
        one_step("c", "supported", "single", 1, 4, 1, 1, 1),
    ]
    stats = paired_bootstrap(left, right, draws=40, seed=0)
    assert stats["n"] == 2
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["low"] <= stats["mean"] <= stats["high"]


def test_summary_text_names_the_classes():
    text = summarize(sample_log(), draws=20)
    assert "supported" in text
    assert "1/5  0.200" in text
    assert "reasoning rate 0.333" in text
    assert "clear rate 0.500" in text
    assert "mean prefix 1.000" in text
    assert "chance baseline" in text
    assert "tokens per request  output median 5.500  reasoning median 4.000  total median 20.500" in text
    assert "multi-step episode tokens  median 16.000  sum 32  episodes 2" in text
