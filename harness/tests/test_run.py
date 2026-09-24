from msbench.client import Completion
from msbench.prompt import PREFIX
from msbench.run import RunMeta, run_multi_step, run_one_step
from msbench.score import LABELS
from textbook import TEXTBOOK

META = RunMeta(run_id="test", model="scripted", protocol="instruct", track="adhoc")


def scripted(action: dict | None, transport: str | None = None) -> Completion:
    return Completion(
        text="" if transport else "noted",
        action=None if transport else action,
        transport_error=transport,
        attempts=[
            {
                "attempt": 0,
                "http_status": None if transport else 200,
                "latency_ms": 1.0,
                "finish_reason": None if transport else "stop",
                "input_tokens": None if transport else 8,
                "output_tokens": None if transport else 3,
                "reasoning_tokens": None,
                "error": transport,
            }
        ],
    )


def test_one_step_reloads_each_gold_position(session, five_by_five):
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        action = five_by_five["gold"][len(prompts) - 1]["action"]
        return scripted(action)

    records = run_one_step(session, five_by_five, "five", META, complete)
    assert len(records) == len(five_by_five["gold"])
    assert prompts[0][len(PREFIX) :] == five_by_five["initial_text"]
    assert prompts[1][len(PREFIX) :] == five_by_five["gold"][0]["text_after"]
    assert [record["label"] for record in records] == ["supported"] * len(records)
    assert records[0]["position_tier"] == five_by_five["gold"][0]["tier"]
    assert records[0]["board_text"] == five_by_five["initial_text"]
    assert records[1]["board_text"] == five_by_five["gold"][0]["text_after"]


def test_repeated_samples_reload_the_same_position(session):
    meta = RunMeta("test", "scripted", "instruct", "adhoc", samples=2)
    records = run_one_step(
        session,
        TEXTBOOK,
        "textbook",
        meta,
        lambda _prompt: scripted({"kind": "open", "at": {"row": 3, "col": 1}}),
    )
    assert [record["sample_index"] for record in records] == [0, 1]
    assert {record["item_id"] for record in records} == {records[0]["item_id"]}
    assert all(record["label"] == "supported" and record["status_after"] == "won" for record in records)


def test_multi_step_gold_follower_clears(session, five_by_five):
    cursor = {"n": 0}

    def complete(_prompt):
        action = five_by_five["gold"][cursor["n"]]["action"]
        cursor["n"] += 1
        return scripted(action)

    records = run_multi_step(session, five_by_five, "five", META, complete)
    assert len(records) == len(five_by_five["gold"])
    assert [record["label"] for record in records] == ["supported"] * len(records)
    assert records[-1]["stop_reason"] == "cleared"
    assert records[-1]["status_after"] == "won"
    assert records[0]["stop_reason"] is None


def test_lucky_click_continues_and_can_still_clear(session):
    calls = {"n": 0}

    def complete(_prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return scripted({"kind": "open", "at": {"row": 2, "col": 1}})
        return scripted({"kind": "open", "at": {"row": 3, "col": 1}})

    records = run_multi_step(session, TEXTBOOK, "textbook", META, complete)
    assert [record["label"] for record in records] == ["lucky", "supported"]
    assert records[0]["stop_reason"] is None
    assert records[0]["status_after"] == "playing"
    assert records[-1]["stop_reason"] == "cleared"
    assert records[-1]["status_after"] == "won"


def test_illegal_move_continues_and_can_recover(session):
    calls = {"n": 0}

    def complete(_prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return scripted({"kind": "open", "at": {"row": 1, "col": 2}})
        return scripted({"kind": "open", "at": {"row": 3, "col": 1}})

    records = run_multi_step(session, TEXTBOOK, "textbook", META, complete)
    assert [record["label"] for record in records] == ["illegal", "supported"]
    assert records[0]["stop_reason"] is None
    assert records[-1]["stop_reason"] == "cleared"


def test_mine_stops_the_episode(session):
    records = run_multi_step(
        session,
        TEXTBOOK,
        "textbook",
        META,
        lambda _prompt: scripted({"kind": "open", "at": {"row": 1, "col": 1}}),
    )
    assert len(records) == 1
    assert records[0]["label"] == "mine"
    assert records[0]["stop_reason"] == "mine"
    assert records[0]["status_after"] == "lost"


def test_format_stalls_three_times_then_stops(session):
    calls = {"n": 0}

    def complete(_prompt):
        calls["n"] += 1
        return scripted(None)

    records = run_multi_step(session, TEXTBOOK, "textbook", META, complete)
    assert calls["n"] == 3
    assert len(records) == 3
    assert [record["label"] for record in records] == ["format"] * 3
    assert [record["stop_reason"] for record in records] == [None, None, "format"]
    assert all(record["label"] in LABELS for record in records)


def test_step_cap_stops_a_supported_game_that_is_still_playing(session, five_by_five):
    assert len(five_by_five["gold"]) >= 2
    action = five_by_five["gold"][0]["action"]
    records = run_multi_step(
        session,
        five_by_five,
        "five",
        META,
        lambda _prompt: scripted(action),
        step_cap=1,
    )
    assert len(records) == 1
    assert records[0]["label"] == "supported"
    assert records[0]["status_after"] == "playing"
    assert records[0]["stop_reason"] == "step_cap"


def test_transport_mid_episode_stops_and_is_unlabeled(session, five_by_five):
    calls = {"n": 0}

    def complete(_prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return scripted(five_by_five["gold"][0]["action"])
        return scripted("", transport="timed out")

    records = run_multi_step(session, five_by_five, "five", META, complete)
    assert records[0]["label"] == "supported"
    assert records[0]["stop_reason"] is None
    assert records[1]["label"] is None
    assert records[1]["transport_error"] == "timed out"
    assert records[1]["stop_reason"] == "transport"
    assert records[1]["forced_count"] is not None
