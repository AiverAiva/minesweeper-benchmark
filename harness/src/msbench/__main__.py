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
        description="踩地雷推理的文字測驗。規則在 msengine，這裡負責提示、API 與計分。",
        epilog=(
            "設定寫在 harness/.env，範例是 harness/.env.example。\n"
            "MS_API_BASE 是以 /v1 結尾的 API 根。MS_MODEL 是模型 id。MS_API_KEY 是金鑰。\n"
            "已經在環境裡的同名變數會蓋過這個檔案。\n"
            "MS_PROTOCOL 只寫進紀錄（instruct 或 thinking），預設 instruct。\n"
            "請求不送 token 上限。模型自己停，用量照回應裡的 total 記入。\n"
            "盤面路徑照當前目錄解析，在倉庫根目錄用 uv run --project harness msbench <命令> 執行。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gold = sub.add_parser("gold", help="沿 gold 走完一張凍結盤面，確認引擎與檔案一致")
    gold.add_argument("fixtures", nargs="+", metavar="board.json", help="msengine generate 寫出的盤面 JSON")
    gold.add_argument("--binary", default=None)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "fixtures",
        nargs="+",
        metavar="board.json",
        help="msengine generate 寫出的盤面 JSON",
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

    sub.add_parser("one-step", parents=[shared], help="沿 gold 路徑的每個位置問一次")
    sub.add_parser("multi-step", parents=[shared], help="從開局下到第一次非必走")

    summary = sub.add_parser("summary", help="彙總 JSONL 紀錄")
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
