import json

from msbench.__main__ import main
from textbook import TEXTBOOK


def test_rescore_relabels_a_log_without_a_model(tmp_path, engine_binary, capsys):
    board = tmp_path / "textbook.json"
    board.write_text(json.dumps(TEXTBOOK), encoding="utf-8")
    log = tmp_path / "run.jsonl"
    common = {
        "mode": "multi_step",
        "fixture": "textbook",
        "seed": 0,
        "rows": 3,
        "cols": 3,
        "mines_total": 2,
        "step_index": 0,
        "label": "lucky",
    }
    rows = [
        {
            **common,
            "episode_id": "open",
            "action": {"kind": "open", "at": {"row": 3, "col": 1}},
        },
        {
            **common,
            "episode_id": "flag",
            "action": {"kind": "flag", "at": {"row": 3, "col": 1}},
        },
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
    assert "sk-" not in out
