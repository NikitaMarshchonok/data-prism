"""Local daily-forecast evaluation. No upload, serving or model persistence."""
import argparse
import csv
import io
import json
from pathlib import Path

import pandas as pd

from src.forecasting import MAX_ROWS, ForecastTask, forecast_series


MAX_CSV_BYTES = 16 * 1024 * 1024


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a single daily numeric series with temporal backtests.")
    parser.add_argument("--csv", type=Path, required=True, help="Local UTF-8 comma-delimited CSV; never uploaded.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--time", required=True, help="ISO YYYY-MM-DD daily date column.")
    parser.add_argument("--horizon", type=int, required=True, help="Number of future days, 1–90.")
    parser.add_argument("--seasonal-period", type=int, default=7, help="Explicit seasonal baseline period in days.")
    parser.add_argument("--reporting-delay-days", type=int, default=0,
                        help="Assert a fixed target release delay, 0–30 whole calendar days; no data vintages are verified.")
    args = parser.parse_args(argv)
    try:
        task = ForecastTask(args.target, args.time, args.horizon, args.seasonal_period, args.reporting_delay_days)
        with args.csv.expanduser().open("rb") as source:
            raw = source.read(MAX_CSV_BYTES + 1)
        if len(raw) > MAX_CSV_BYTES:
            raise ValueError("Forecast CSV exceeds the 16 MiB local evaluation limit.")
        decoded = raw.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(decoded)), [])
        if not header or len(header) > 64 or len(header) != len(set(header)):
            raise ValueError("CSV requires 1–64 uniquely named columns.")
        data = pd.read_csv(io.StringIO(decoded), nrows=MAX_ROWS + 1)
        report = forecast_series(data, task)
    except (OSError, ValueError, UnicodeError, csv.Error) as error:
        # Do not echo paths, row values or arbitrary parser diagnostics.
        report = {"status": "error", "message": "Unable to read a valid bounded UTF-8 CSV or forecast task.",
                  "error_type": type(error).__name__}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
