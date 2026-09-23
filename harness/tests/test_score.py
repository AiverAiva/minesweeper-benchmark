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


def test_prose_around_the_forced_open_wins(session):
    session.new_fixture(TEXTBOOK)
    result = classify(session, "ok\nACTION: open row=3 col=1\nthanks")
    assert result["label"] == "supported"
    assert result["status_after"] == "won"
    assert result["forced_tier"] == "enumerate"


def test_unforced_safe_open_is_lucky(session):
    session.new_fixture(TEXTBOOK)
    result = classify(session, "I think\nACTION: open row=2 col=1\n")
    assert result["label"] == "lucky"
    assert result["status_after"] == "playing"
    assert result["action"] == {"kind": "open", "at": {"row": 2, "col": 1}}


def test_opening_a_mine_is_a_mine(session):
    session.new_fixture(TEXTBOOK)
    result = classify(session, "ACTION: open row=1 col=1")
    assert result["label"] == "mine"
    assert result["status_after"] == "lost"


def test_already_open_is_illegal_and_the_board_stays(session):
    session.new_fixture(TEXTBOOK)
    before = session.forced()["actions"]
    result = classify(session, "ACTION: open row=1 col=2")
    assert result["label"] == "illegal"
    assert session.forced()["actions"] == before
    assert FORCED_OPEN in before


def test_prose_without_an_action_is_format_and_does_not_move(session):
    session.new_fixture(TEXTBOOK)
    before = session.forced()["actions"]
    result = classify(session, "I cannot decide")
    assert result["label"] == "format"
    assert result["action"] is None
    assert session.forced()["actions"] == before


def test_two_action_lines_do_not_move(session):
    session.new_fixture(TEXTBOOK)
    result = classify(session, "ACTION: open row=3 col=1\nACTION: open row=2 col=1")
    assert result["label"] == "format"
    assert FORCED_OPEN in session.forced()["actions"]
    follow = classify(session, "ACTION: open row=3 col=1")
    assert follow["label"] == "supported"
    assert follow["status_after"] == "won"
