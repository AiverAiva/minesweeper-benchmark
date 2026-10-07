"""CLI for the text harness.

Settings come from harness/.env. A shell variable with the same name wins.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from msbench.client import FatalApiError
from msbench.config import ConfigError, load_settings
from msbench.engine import EngineError
from msbench.run import check_gold, run_paths
from msbench.score import GoldError
from msbench.summary import load_jsonl, render_paired, paired_bootstrap, summarize


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(
        prog="msbench",
        description="Text harness for the minesweeper reasoning benchmark. Rules live in msengine; this CLI handles prompts, the API, and scoring.",
        epilog=(
            "Settings live in harness/.env; see harness/.env.example.\n"
            "MS_API_BASE is an API root ending in /v1. MS_MODEL is the model id. MS_API_KEY is the key.\n"
            "A variable already set in the environment wins over the file.\n"
            "MS_PROTOCOL is only recorded (instruct or thinking), default instruct.\n"
            "Requests stream with no token cap; the model decides when to stop, and usage is taken from the response total.\n"
            "Board paths resolve against the current directory. Run from the repo root with uv run --project harness msbench <command>."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gold = sub.add_parser("gold", help="replay a frozen board along its gold path to confirm the engine and file agree")
    gold.add_argument("fixtures", nargs="+", metavar="board.json", help="board JSON written by msengine generate")
    gold.add_argument("--binary", default=None)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "fixtures",
        nargs="+",
        metavar="board.json",
        help="board JSON written by msengine generate",
    )
    shared.add_argument("--out", default=None)
    shared.add_argument("--base", default=None)
    shared.add_argument("--model", default=None)
    shared.add_argument("--protocol", default=None)
    shared.add_argument("--temperature", type=float, default=None)
    shared.add_argument("--samples", type=int, default=None)
    shared.add_argument("--max-retries", dest="max_retries", type=int, default=None)
    shared.add_argument("--timeout", type=float, default=None)
    shared.add_argument("--track", default=None)
    shared.add_argument("--binary", default=None)

    sub.add_parser("one-step", parents=[shared], help="ask once at every position along the gold path")
    sub.add_parser("multi-step", parents=[shared], help="play from the opening until the first non-forced move")

    summary = sub.add_parser("summary", help="summarize JSONL records")
    summary.add_argument("logs", nargs="+")

    args = parser.parse_args(argv)
    try:
        if args.command == "gold":
            check_gold([Path(path) for path in args.fixtures], args.binary)
            return 0
        if args.command == "summary":
            return _summary([Path(path) for path in args.logs])
        settings = load_settings(args)
        records, destination = run_paths(
            [Path(path) for path in args.fixtures],
            mode=args.command.replace("-", "_"),
            settings=settings,
            out=Path(args.out) if args.out else None,
        )
    except (ConfigError, EngineError, GoldError, FatalApiError, OSError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"wrote {destination}  records={len(records)}")
    print(summarize(records), end="")
    return 0


def _summary(paths: list[Path]) -> int:
    bundles = [(path, load_jsonl(path)) for path in paths]
    for path, records in bundles:
        print(f"# {path}")
        print(summarize(records), end="")
        print()
    if len(bundles) == 2:
        print(render_paired(paired_bootstrap(bundles[0][1], bundles[1][1])), end="")
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
