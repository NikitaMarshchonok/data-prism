"""Command-line analytical regression gate for Data Prism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.quality_evaluation import run_quality_evaluation


DEFAULT_CONFIG = Path(__file__).resolve().parent / "evaluation" / "quality_gate.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic analytical quality benchmarks."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Quality-gate JSON configuration (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON report. The report is always printed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with args.config.expanduser().open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        report = run_quality_evaluation(config)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            output = args.output.expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, sort_keys=True))
        return 1

    print(rendered, end="")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
