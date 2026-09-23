"""Local read-only page for runs/*.jsonl.

Listens on 127.0.0.1 and serves only .jsonl files whose names match, inside runs/.
From the repository root:

    uv run --project harness python viewer/serve.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "runs"
VIEWER = HERE / "index.html"
NAME = re.compile(r"^[A-Za-z0-9._-]+\.jsonl$")


def log_file(name: str) -> Path | None:
    if not NAME.fullmatch(name):
        return None
    root = RUNS.resolve()
    file = (root / name).resolve()
    if file.parent != root or not file.is_file():
        return None
    return file


def line_count(path: Path) -> int:
    count = 0
    last = b"\n"
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            count += chunk.count(b"\n")
            last = chunk[-1:]
    if last not in (b"\n", b""):
        count += 1
    return count


def list_runs() -> list[dict]:
    if not RUNS.is_dir():
        return []
    items = []
    root = RUNS.resolve()
    for file in RUNS.glob("*.jsonl"):
        resolved = file.resolve()
        if resolved.parent != root or not resolved.is_file() or not NAME.fullmatch(file.name):
            continue
        items.append(
            {
                "name": file.name,
                "bytes": resolved.stat().st_size,
                "records": line_count(resolved),
            }
        )
    items.sort(key=lambda item: item["name"], reverse=True)
    return items


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            if not VIEWER.is_file():
                self._send(500, "text/plain; charset=utf-8", b"viewer/index.html is missing")
                return
            self._send(200, "text/html; charset=utf-8", VIEWER.read_bytes())
            return
        if path == "/api/runs":
            body = json.dumps(list_runs(), ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
            return
        prefix = "/api/runs/"
        if path.startswith(prefix):
            file = log_file(path[len(prefix):])
            if file is None:
                self._send(404, "text/plain; charset=utf-8", b"No such log")
                return
            self._send(200, "text/plain; charset=utf-8", file.read_bytes())
            return
        self._send(404, "text/plain; charset=utf-8", b"Not found")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="Read runs/*.jsonl in a browser.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true", help="open the page instead of only printing the URL")
    args = parser.parse_args(argv)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        print(f"Could not bind 127.0.0.1:{args.port} ({error})", file=sys.stderr)
        return 1
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    print(f"Open {url}", flush=True)
    print(f"Logs {RUNS}", flush=True)
    print("Stop with Ctrl+C", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
