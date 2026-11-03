import json
import subprocess

from msbench.score import classify, hidden_count, replay_gold
from textbook import OPENING, TEXTBOOK

FORCED_OPEN = {"kind": "open", "at": {"row": 3, "col": 1}}


def test_gold_path_clears(session):
    replay_gold(session, TEXTBOOK)


def test_generated_gold_clears(session, five_by_five):
    assert len(five_by_five["gold"]) >= 2
    replay_gold(session, five_by_five)


def test_opening_hides_five_cells():
    assert hidden_count(OPENING) == 5


def test_forced_call_wins(session):
    session.new_fixture(TEXTBOOK)
    result = classify(session, {"kind": "open", "at": {"row": 3, "col": 1}})
    assert result["label"] == "supported"
    assert result["status_after"] == "won"
    assert result["forced_tier"] == "enumerate"
    assert result["tier_needed"] == "enumerate"


def test_unforced_safe_open_is_lucky(session):
    session.new_fixture(TEXTBOOK)
    result = classify(session, {"kind": "open", "at": {"row": 2, "col": 1}})
    assert result["label"] == "lucky"
    assert result["status_after"] == "playing"
    assert result["action"] == {"kind": "open", "at": {"row": 2, "col": 1}}


def _action_key(action: dict) -> tuple:
    at = action.get("at") or {}
    return (action.get("kind"), at.get("row"), at.get("col"))


def test_mine_count_open_is_supported_when_single_already_fired(session, engine_binary, tmp_path):
    """Single already forced some other cell, so forced() never counted this one."""
    found = None
    for seed in range(1, 21):
        path = tmp_path / f"beginner-{seed}.json"
        proc = subprocess.run(
            [
                str(engine_binary),
                "generate",
                "--seed",
                str(seed),
                "--level",
                "beginner",
                "--attempts",
                "40",
                "--out",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            continue
        fixture = json.loads(path.read_text(encoding="utf-8"))
        session.new_fixture(fixture)
        for step in fixture["gold"]:
            forced = session.forced()
            proven = session.exact()
            forced_keys = {_action_key(item) for item in forced["actions"]}
            candidate = next(
                (
                    action
                    for action in proven["actions"]
                    if action.get("kind") == "open"
                    and action.get("tier") == "global"
                    and _action_key(action) not in forced_keys
                ),
                None,
            )
            if forced.get("tier") == "single" and candidate is not None:
                result = classify(
                    session,
                    {"kind": "open", "at": candidate["at"]},
                    forced,
                    proven,
                )
                assert result["label"] == "supported"
                assert result["tier_needed"] == "global"
                assert result["status_after"] != "lost"
                found = (seed, candidate["at"])
                break
            applied = session.apply(step["action"])
            if not applied.get("ok") or applied.get("status") != "playing":
                break
        if found:
            break
    assert found, "no beginner position had a mine-count-only safe open beside a single-cell force"


def test_flagging_a_safe_cell_is_a_misflag(session):
    session.new_fixture(TEXTBOOK)
    result = classify(session, {"kind": "flag", "at": {"row": 3, "col": 1}})
    assert result["label"] == "misflag"
    assert result["status_after"] == "playing"


def test_opening_a_mine_is_a_mine(session):
    session.new_fixture(TEXTBOOK)
    result = classify(session, {"kind": "open", "at": {"row": 1, "col": 1}})
    assert result["label"] == "mine"
    assert result["status_after"] == "lost"


def test_already_open_is_illegal_and_the_board_stays(session):
    session.new_fixture(TEXTBOOK)
    before = session.forced()["actions"]
    result = classify(session, {"kind": "open", "at": {"row": 1, "col": 2}})
    assert result["label"] == "illegal"
    assert session.forced()["actions"] == before
    assert FORCED_OPEN in before


def test_missing_call_is_format_and_does_not_move(session):
    session.new_fixture(TEXTBOOK)
    before = session.forced()["actions"]
    result = classify(session, None)
    assert result["label"] == "format"
    assert result["action"] is None
    assert session.forced()["actions"] == before


def test_missing_call_does_not_move(session):
    session.new_fixture(TEXTBOOK)
    result = classify(session, None)
    assert result["label"] == "format"
    assert FORCED_OPEN in session.forced()["actions"]
    follow = classify(session, {"kind": "open", "at": {"row": 3, "col": 1}})
    assert follow["label"] == "supported"
    assert follow["status_after"] == "won"
