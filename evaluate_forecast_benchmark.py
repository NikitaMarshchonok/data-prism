"""Run the frozen synthetic rolling-origin protocol, preserving adverse results."""
import argparse
import json
from pathlib import Path

from src.forecast_benchmark import run_forecast_benchmark


DEFAULT_CONFIG = Path(__file__).resolve().parent / 'evaluation' / 'forecast_benchmark.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        report = run_forecast_benchmark(config)
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding='utf-8')
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({'status': 'error', 'error': str(error)}, sort_keys=True))
        return 1
    print(rendered, end='')
    # Predictive losses are valid results, not execution failures. A green exit
    # means all scheduled runs/candidates completed the protocol, not useful ML.
    return 0 if report['protocol_integrity'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
