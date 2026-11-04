"""Local frozen-model inference. Never fit, choose candidates or send a file."""
import argparse
import csv
import json
from pathlib import Path

from evaluate_forecast import read_forecast_csv
from src.forecast_artifact import load_artifact, predict_artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description='Predict using a local JSON daily model without retraining.')
    parser.add_argument('--model', type=Path, required=True, help='Data-bearing model snapshot from --save-model; not a pickle or joblib file.')
    parser.add_argument('--history-csv', type=Path,
                        help='Complete original training history plus new released daily actuals. Must preserve the original prefix.')
    parser.add_argument('--forecast-origin', help='Required with new history; explicit ISO end-of-day issue date.')
    args = parser.parse_args(argv)
    try:
        if args.history_csv is not None and args.forecast_origin is None:
            raise ValueError('New history requires an explicit issue date.')
        artifact = load_artifact(args.model)
        history = read_forecast_csv(args.history_csv) if args.history_csv is not None else None
        report = predict_artifact(artifact, history=history, forecast_origin=args.forecast_origin)
    except (OSError, ValueError, TypeError, UnicodeError, csv.Error, FloatingPointError, OverflowError):
        report = {'status': 'error', 'message': 'Unable to validate the local model or its available history; no prediction was accepted.'}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
