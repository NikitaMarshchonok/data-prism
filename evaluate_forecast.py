"""Local daily evaluation with optional explicit, data-bearing model persistence."""
import argparse
import csv
import io
import json
from pathlib import Path

import pandas as pd

from src.forecasting import MAX_ROWS, ForecastTask, forecast_series, forecast_task_settings
from src.forecast_result import validate_forecast_result
from src.forecast_backtest import historical_forecast_backtest
from src.forecast_artifact import create_artifact, save_artifact


MAX_CSV_BYTES = 16 * 1024 * 1024


def read_forecast_csv(path):
    """Shared bounded local reader; no upload, mutations or silent truncation."""
    with Path(path).expanduser().open('rb') as source:
        raw = source.read(MAX_CSV_BYTES + 1)
    if len(raw) > MAX_CSV_BYTES:
        raise ValueError('Forecast CSV exceeds the 16 MiB local evaluation limit.')
    decoded = raw.decode('utf-8-sig')
    header = next(csv.reader(io.StringIO(decoded)), [])
    if not header or len(header) > 64 or len(header) != len(set(header)):
        raise ValueError('CSV requires 1–64 uniquely named columns.')
    return pd.read_csv(io.StringIO(decoded), nrows=MAX_ROWS + 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a single daily numeric series with temporal backtests.")
    parser.add_argument("--csv", type=Path, required=True, help="Local UTF-8 comma-delimited CSV; never uploaded.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--time", required=True, help="ISO YYYY-MM-DD daily date column.")
    parser.add_argument("--horizon", type=int, required=True, help="Number of future days, 1–90.")
    parser.add_argument("--seasonal-period", type=int, default=7, help="Explicit seasonal baseline period in days.")
    parser.add_argument("--reporting-delay-days", type=int, default=0,
                        help="Assert a fixed target release delay, 0–30 whole calendar days; no data vintages are verified.")
    parser.add_argument('--forecast-origin', help='Explicit ISO YYYY-MM-DD end-of-day origin; input must end exactly at origin minus delay. Omit for retrospective mode.')
    parser.add_argument('--save-model', type=Path,
                        help='Explicitly create a private local JSON model file. Contains target lags and fitted parameters; never overwrites. No server retention or cloud backup.')
    args = parser.parse_args(argv)
    try:
        task = ForecastTask(args.target, args.time, args.horizon, args.seasonal_period, args.reporting_delay_days, args.forecast_origin)
        data = read_forecast_csv(args.csv)
        report = forecast_series(data, task)
        try:
            if not isinstance(report, dict):
                raise ValueError('A structured forecast result is required.')
            if report.get('status') == 'ok':
                report['result_integrity'] = validate_forecast_result(report, data, forecast_task_settings(task))
                report['historical_backtest'] = historical_forecast_backtest(data, task, candidate_suite=report['candidate_suite'])
        except ValueError:
            report = {'status': 'error', 'message': 'The forecast result failed its evidence-consistency check; no result was accepted.',
                      'error_type': 'ValueError'}
    except (OSError, ValueError, UnicodeError, csv.Error) as error:
        # Do not echo paths, row values or arbitrary parser diagnostics.
        report = {"status": "error", "message": "Unable to read a valid bounded UTF-8 CSV or forecast task.",
                  "error_type": type(error).__name__}
    if args.save_model is not None and report.get('status') == 'ok':
        try:
            artifact = create_artifact(data, task, report)
            save_artifact(artifact, args.save_model)
            report['model_artifact'] = {'status': 'saved', 'contract': artifact['contract'],
                                        'model_id': artifact['model_id'], 'sha256': artifact['sha256'],
                                        'data_bearing': True, 'storage': 'explicit_local_file'}
        except (OSError, ValueError, FloatingPointError, OverflowError):
            report['model_artifact'] = {'status': 'error', 'message': 'Unable to create the requested local model. Existing paths are never replaced.'}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report.get('status') == 'ok' and report.get('model_artifact', {}).get('status') != 'error' else 1


if __name__ == "__main__":
    raise SystemExit(main())
