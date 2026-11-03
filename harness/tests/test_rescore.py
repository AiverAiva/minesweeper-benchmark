import json

from msbench.__main__ import main
from msbench.engine import EngineSession
from textbook import OPENING, TEXTBOOK, WON


def _record(episode, action, label, board, after, **extra):
    return {
        "mode": "multi_step",
        "fixture": "textbook",
        "seed": 0,
        "rows": 3,
        "cols": 3,
        "mines_total": 2,
        "step_index": 0,
        "episode_id": episode,
        "label": label,
        "action": action,
        "board_text": board,
        "text_after": after,
        "prompt_version": "v0",
        "board_format": "grid",
        **extra,
    }


def test_rescore_relabels_a_log_without_a_model(tmp_path, engine_binary, capsys):
    with EngineSession(engine_binary) as session:
        session.new_fixture(TEXTBOOK)
        flagged = session.apply({"kind": "flag", "at": {"row": 3, "col": 1}})
    board = tmp_path / "textbook.json"
    board.write_text(json.dumps(TEXTBOOK), encoding="utf-8")
    log = tmp_path / "run.jsonl"
    rows = [
        _record("open", {"kind": "open", "at": {"row": 3, "col": 1}}, "lucky", OPENING, WON),
        _record("flag", {"kind": "flag", "at": {"row": 3, "col": 1}}, "lucky", OPENING, flagged["text"]),
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    code = main(["rescore", str(log), "--fixture", str(board), "--binary", str(engine_binary)])
    assert code == 0
    out = capsys.readouterr().out
    assert "open row=3 col=1  lucky -> supported" in out
    assert "flag row=3 col=1  lucky -> misflag" in out
    assert "tier_needed=enumerate" in out
    assert "changed 2 of 2" in out
    assert "reasoning 0.000 (0/2) -> 0.500 (1/2)" in out
    assert "prompt_version=v0 board_format=grid" in out
    assert "lucky->supported 1" in out
    assert "supported->lucky 0" in out
    assert "sk-" not in out


def test_rescore_stops_when_the_replayed_board_differs(tmp_path, engine_binary, capsys):
    board = tmp_path / "textbook.json"
    board.write_text(json.dumps(TEXTBOOK), encoding="utf-8")
    log = tmp_path / "run.jsonl"
    bad = WON.replace("121", "111")
    row = _record("open", {"kind": "open", "at": {"row": 3, "col": 1}}, "lucky", OPENING, bad)
    log.write_text(json.dumps(row) + "\n", encoding="utf-8")
    code = main(["rescore", str(log), "--fixture", str(board), "--binary", str(engine_binary)])
    assert code == 1
    err = capsys.readouterr().err
    assert "step 0" in err
    assert str(log) in err
    assert "does not match" in err


def test_rescore_applies_an_unknown_label_when_the_move_landed(tmp_path, engine_binary, capsys):
    board = tmp_path / "textbook.json"
    board.write_text(json.dumps(TEXTBOOK), encoding="utf-8")
    log = tmp_path / "run.jsonl"
    row = _record("open", {"kind": "open", "at": {"row": 3, "col": 1}}, "rescued", OPENING, WON)
    log.write_text(json.dumps(row) + "\n", encoding="utf-8")
    code = main(["rescore", str(log), "--fixture", str(board), "--binary", str(engine_binary)])
    assert code == 0
    out = capsys.readouterr().out
    assert "rescued -> supported" in out


def test_rescore_rejects_an_unknown_label_with_no_move(tmp_path, engine_binary, capsys):
    board = tmp_path / "textbook.json"
    board.write_text(json.dumps(TEXTBOOK), encoding="utf-8")
    log = tmp_path / "run.jsonl"
    row = _record("open", None, "rescued", OPENING, None)
    log.write_text(json.dumps(row) + "\n", encoding="utf-8")
    code = main(["rescore", str(log), "--fixture", str(board), "--binary", str(engine_binary)])
    assert code == 1
    err = capsys.readouterr().err
    assert "step 0" in err
    assert "unrecognized label" in err
