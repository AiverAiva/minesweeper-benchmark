"""Long-lived `msengine session` process.

Python asks for the player view and for judgments. The process applies the rules.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path


class EngineError(RuntimeError):
    def __init__(self, error: str, detail: str = ""):
        self.error = error
        self.detail = detail
        message = error if not detail else f"{error}: {detail}"
        super().__init__(message)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "Cargo.toml").is_file() and (parent / "engine" / "Cargo.toml").is_file():
            return parent
    raise EngineError("missing_workspace", "cannot locate the minesweeper Cargo workspace")


def default_binary() -> Path:
    override = os.environ.get("MSENGINE")
    if override:
        return Path(override)
    root = repo_root()
    name = "msengine.exe" if os.name == "nt" else "msengine"
    debug = root / "target" / "debug" / name
    release = root / "target" / "release" / name
    if debug.is_file():
        return debug
    if release.is_file():
        return release
    return debug


class EngineSession:
    def __init__(self, binary: str | Path | None = None):
        self.binary = Path(binary) if binary is not None else default_binary()
        self.proc: subprocess.Popen[bytes] | None = None
        self.stderr: list[bytes] = []
        self._thread: threading.Thread | None = None

    def __enter__(self) -> EngineSession:
        if not self.binary.is_file():
            raise EngineError(
                "missing_binary",
                f"engine binary not found: {self.binary}. "
                "Build it with `cargo build -p minesweeper-engine` at the repo root, or set MSENGINE.",
            )
        self.proc = subprocess.Popen(
            [str(self.binary), "session"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._thread.start()
        pong = self._raw({"op": "ping"})
        if pong.get("op") != "pong":
            raise EngineError("protocol", "msengine session did not reply to ping with pong")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        proc = self.proc
        if proc is None:
            return
        if proc.stdin is not None:
            proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.proc = None

    def new_fixture(self, fixture: dict) -> dict:
        return self._checked({"op": "new_fixture", "fixture": fixture})

    def forced(self) -> dict:
        return self._checked({"op": "forced"})

    def apply(self, action: dict) -> dict:
        return self._checked({"op": "apply", "action": action})

    def _checked(self, payload: dict) -> dict:
        response = self._raw(payload)
        if response.get("ok") or response.get("error") == "illegal":
            return response
        raise EngineError(str(response.get("error") or "parse"), str(response.get("detail") or ""))

    def _raw(self, payload: dict) -> dict:
        proc = self.proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise EngineError("eof", "msengine session is not running")
        if proc.poll() is not None:
            raise EngineError("eof", self._stderr_text() or f"msengine exited with code {proc.returncode}")
        proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        proc.stdin.flush()
        raw = proc.stdout.readline()
        if not raw:
            detail = self._stderr_text() or "msengine closed stdout"
            raise EngineError("eof", detail)
        text = raw.decode("utf-8").strip()
        try:
            response = json.loads(text)
        except json.JSONDecodeError as error:
            raise EngineError("parse", text[:500]) from error
        if not isinstance(response, dict):
            raise EngineError("parse", text[:500])
        return response

    def _drain_stderr(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        kept = 0
        while True:
            chunk = proc.stderr.read(4096)
            if not chunk:
                break
            if kept < 65536:
                self.stderr.append(chunk)
                kept += len(chunk)

    def _stderr_text(self) -> str:
        return b"".join(self.stderr).decode("utf-8", "replace")[:500]
