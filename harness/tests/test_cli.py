import json

from msbench.__main__ import main
from msbench.config import Settings
from msbench.run import run_paths
from test_run import scripted
from textbook import TEXTBOOK


def test_gold_command_replays_the_textbook(tmp_path, engine_binary, capsys):
    path = tmp_path / "textbook.json"
    path.write_text(json.dumps(TEXTBOOK), encoding="utf-8")
    assert main(["gold", str(path), "--binary", str(engine_binary)]) == 0
    assert "gold_steps=1" in capsys.readouterr().out


def test_one_step_refuses_to_start_without_an_env_key(monkeypatch, tmp_path, engine_binary, capsys):
    monkeypatch.delenv("MS_API_KEY", raising=False)
    path = tmp_path / "textbook.json"
    path.write_text(json.dumps(TEXTBOOK), encoding="utf-8")
    code = main(
        [
            "one-step",
            str(path),
            "--base",
            "https://example.com/v1",
            "--model",
            "m",
            "--temperature",
            "0",
            "--samples",
            "1",
            "--binary",
            str(engine_binary),
        ]
    )
    assert code == 1
    assert "MS_API_KEY" in capsys.readouterr().err


def test_run_writes_a_jsonl_log_without_a_key(tmp_path, engine_binary):
    path = tmp_path / "textbook.json"
    path.write_text(json.dumps(TEXTBOOK), encoding="utf-8")
    out = tmp_path / "out.jsonl"
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
    records, destination = run_paths(
        [path],
        mode="multi_step",
        settings=settings,
        out=out,
        complete=lambda _prompt: scripted("ACTION: open row=3 col=1"),
    )
    assert destination == out
    assert records[-1]["stop_reason"] == "cleared"
    saved = out.read_text(encoding="utf-8")
    assert json.loads(saved)["label"] == "supported"
    assert "sk-" not in saved


def test_summary_command_prints_chinese(tmp_path, capsys):
    path = tmp_path / "run.jsonl"
    path.write_text(
        json.dumps({"mode": "one_step", "item_id": "a", "label": "format", "attempts": []}) + "\n",
        encoding="utf-8",
    )
    assert main(["summary", str(path)]) == 0
    assert "格式" in capsys.readouterr().out
