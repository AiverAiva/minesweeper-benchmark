import json
import os
import subprocess

import pytest

from msbench.engine import EngineSession, repo_root


@pytest.fixture(autouse=True)
def isolated_dotenv(monkeypatch):
    monkeypatch.setenv("MS_ENV_FILE", "")


@pytest.fixture(scope="session")
def engine_binary():
    name = "msengine.exe" if os.name == "nt" else "msengine"
    binary = repo_root() / "target" / "debug" / name
    if not binary.is_file():
        pytest.fail(f"找不到 {binary}。先在倉庫根目錄執行 cargo build -p minesweeper-engine。")
    return binary


@pytest.fixture
def session(engine_binary):
    with EngineSession(engine_binary) as engine:
        yield engine


@pytest.fixture(scope="session")
def five_by_five(engine_binary, tmp_path_factory):
    path = tmp_path_factory.mktemp("boards") / "five.json"
    subprocess.run(
        [
            str(engine_binary),
            "generate",
            "--seed",
            "7",
            "--rows",
            "5",
            "--cols",
            "5",
            "--mines",
            "2",
            "--attempts",
            "20",
            "--out",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(path.read_text(encoding="utf-8"))
