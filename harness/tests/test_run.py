import json

from msbench.client import Completion
from msbench.config import Settings
from msbench.engine import EngineSession
from msbench.prompt import PREFIX
from msbench.run import RunMeta, continue_log, episode_is_resumable, playable_prefix, run_multi_step, run_one_step
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


def test_multi_step_narrates_every_step(session, five_by_five, capsys):
    cursor = {"n": 0}

    def complete(_prompt):
        action = five_by_five["gold"][cursor["n"]]["action"]
        cursor["n"] += 1
        return scripted(action)

    run_multi_step(session, five_by_five, "five", META, complete)
    err = capsys.readouterr().err
    assert "episode start" in err
    assert "asking model" in err
    assert "-> supported" in err
    assert "episode end: stop=cleared" in err


def test_transport_resume_replays_prefix_and_retries_the_failed_step(session, five_by_five):
    calls = {"n": 0}

    def interrupt(_prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return scripted(five_by_five["gold"][0]["action"])
        return scripted("", transport="timed out")

    prior = run_multi_step(session, five_by_five, "five", META, interrupt)
    assert prior[-1]["stop_reason"] == "transport"
    assert episode_is_resumable(prior)
    assert [record["label"] for record in playable_prefix(prior)] == ["supported"]

    gold_index = {"n": 1}
    prompts = []

    def resume(prompt):
        prompts.append(prompt)
        action = five_by_five["gold"][gold_index["n"]]["action"]
        gold_index["n"] += 1
        return scripted(action)

    new = run_multi_step(session, five_by_five, "five", META, resume, prior_records=prior)
    assert new[0]["step_index"] == prior[-1]["step_index"] == 1
    assert prompts[0][len(PREFIX) :] == five_by_five["gold"][0]["text_after"]
    assert new[0]["board_text"] == five_by_five["gold"][0]["text_after"]
    assert new[-1]["stop_reason"] == "cleared"
    assert new[-1]["status_after"] == "won"


def test_crash_resume_starts_at_the_next_step(session, five_by_five):
    prior = run_multi_step(
        session,
        five_by_five,
        "five",
        META,
        lambda _prompt: scripted(five_by_five["gold"][0]["action"]),
        step_cap=1,
    )
    crashed = dict(prior[0])
    crashed["stop_reason"] = None
    assert episode_is_resumable([crashed])

    gold_index = {"n": 1}
    prompts = []

    def resume(prompt):
        prompts.append(prompt)
        action = five_by_five["gold"][gold_index["n"]]["action"]
        gold_index["n"] += 1
        return scripted(action)

    new = run_multi_step(session, five_by_five, "five", META, resume, prior_records=[crashed])
    assert new[0]["step_index"] == 1
    assert prompts[0][len(PREFIX) :] == five_by_five["gold"][0]["text_after"]
    assert new[-1]["stop_reason"] == "cleared"


def test_finished_episodes_are_not_resumable(session, five_by_five):
    cursor = {"n": 0}

    def follow_gold(_prompt):
        action = five_by_five["gold"][cursor["n"]]["action"]
        cursor["n"] += 1
        return scripted(action)

    cleared = run_multi_step(session, five_by_five, "five", META, follow_gold)
    assert cleared[-1]["stop_reason"] == "cleared"
    assert not episode_is_resumable(cleared)

    mine = run_multi_step(
        session,
        TEXTBOOK,
        "textbook",
        META,
        lambda _prompt: scripted({"kind": "open", "at": {"row": 1, "col": 1}}),
    )
    assert mine[-1]["stop_reason"] == "mine"
    assert not episode_is_resumable(mine)

    capped = run_multi_step(
        session,
        five_by_five,
        "five",
        META,
        lambda _prompt: scripted(five_by_five["gold"][0]["action"]),
        step_cap=1,
    )
    assert capped[-1]["stop_reason"] == "step_cap"
    assert not episode_is_resumable(capped)


def test_stall_episode_is_resumable_with_a_fresh_limit(session):
    stalled = run_multi_step(session, TEXTBOOK, "textbook", META, lambda _prompt: scripted(None))
    assert stalled[-1]["stop_reason"] == "format"
    assert episode_is_resumable(stalled)
    assert [record["label"] for record in stalled] == ["format", "format", "format"]

    calls = {"n": 0}

    def resume(_prompt):
        calls["n"] += 1
        if calls["n"] <= 2:
            return scripted(None)
        return scripted({"kind": "open", "at": {"row": 3, "col": 1}})

    new = run_multi_step(session, TEXTBOOK, "textbook", META, resume, prior_records=stalled)
    assert new[0]["step_index"] == 3
    assert [record["label"] for record in new] == ["format", "format", "supported"]
    assert new[-1]["stop_reason"] == "cleared"


def test_stall_count_survives_a_transport_gap(session):
    calls = {"n": 0}

    def interrupt(_prompt):
        calls["n"] += 1
        if calls["n"] <= 2:
            return scripted(None)
        return scripted("", transport="HTTP 502")

    prior = run_multi_step(session, TEXTBOOK, "textbook", META, interrupt)
    assert [record.get("label") for record in prior] == ["format", "format", None]
    assert episode_is_resumable(prior)

    new = run_multi_step(
        session,
        TEXTBOOK,
        "textbook",
        META,
        lambda _prompt: scripted(None),
        prior_records=prior,
    )
    assert len(new) == 1
    assert new[0]["label"] == "format"
    assert new[0]["stop_reason"] == "format"
    assert new[0]["step_index"] == 2


def test_continue_log_appends_to_the_jsonl(tmp_path, engine_binary, five_by_five):
    board = tmp_path / "five.json"
    board.write_text(json.dumps(five_by_five), encoding="utf-8")
    calls = {"n": 0}

    def interrupt(_prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return scripted(five_by_five["gold"][0]["action"])
        return scripted("", transport="timed out")

    with EngineSession(engine_binary) as session:
        prior = run_multi_step(session, five_by_five, "five", META, interrupt)
    log_path = tmp_path / "run.jsonl"
    log_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in prior),
        encoding="utf-8",
    )

    gold_index = {"n": 1}

    def resume(_prompt):
        action = five_by_five["gold"][gold_index["n"]]["action"]
        gold_index["n"] += 1
        return scripted(action)

    settings = Settings(
        base="",
        model="scripted",
        protocol="instruct",
        temperature=0,
        max_retries=2,
        samples=1,
        timeout=30,
        track="adhoc",
        binary=str(engine_binary),
    )
    records, destination = continue_log(
        log_path,
        settings=settings,
        fixture_paths=[board],
        complete=resume,
    )
    assert destination == log_path
    saved = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert saved[1]["stop_reason"] == "transport"
    assert saved[-1]["stop_reason"] == "cleared"
    assert records[-1]["stop_reason"] == "cleared"
    assert any(record.get("step_index") == 1 and record.get("label") == "supported" for record in saved)


def test_continue_log_out_drops_the_transport_tail(tmp_path, engine_binary, five_by_five):
    board = tmp_path / "five.json"
    board.write_text(json.dumps(five_by_five), encoding="utf-8")
    calls = {"n": 0}

    def interrupt(_prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return scripted(five_by_five["gold"][0]["action"])
        return scripted("", transport="timed out")

    with EngineSession(engine_binary) as session:
        prior = run_multi_step(session, five_by_five, "five", META, interrupt)
    log_path = tmp_path / "run.jsonl"
    out = tmp_path / "continued.jsonl"
    log_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in prior),
        encoding="utf-8",
    )
    gold_index = {"n": 1}

    def resume(_prompt):
        action = five_by_five["gold"][gold_index["n"]]["action"]
        gold_index["n"] += 1
        return scripted(action)

    settings = Settings(
        base="",
        model="scripted",
        protocol="instruct",
        temperature=0,
        max_retries=2,
        samples=1,
        timeout=30,
        track="adhoc",
        binary=str(engine_binary),
    )
    continue_log(
        log_path,
        settings=settings,
        fixture_paths=[board],
        out=out,
        complete=resume,
    )
    saved = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert all(record.get("stop_reason") != "transport" for record in saved)
    assert saved[-1]["stop_reason"] == "cleared"


def test_one_step_narrates_the_position_tier(session, five_by_five, capsys):
    calls = {"n": 0}

    def complete(_prompt):
        action = five_by_five["gold"][calls["n"]]["action"]
        calls["n"] += 1
        return scripted(action)

    run_one_step(session, five_by_five, "five", META, complete)
    err = capsys.readouterr().err
    assert "step 1/" in err
    assert "pos_tier=" in err
    assert "-> supported" in err
