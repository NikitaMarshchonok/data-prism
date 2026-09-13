"""Print a local aggregate pilot report without starting the web application."""

import argparse
import json
import sqlite3

from vibedash.pilot_metrics import build_pilot_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True, help='Path to analysis_jobs.sqlite3 on the host being measured')
    parser.add_argument('--days', type=int, default=30, help='Analysis creation cohort, 1–30 days')
    args = parser.parse_args()
    try:
        report = build_pilot_report(args.database, args.days)
    except (OSError, ValueError, sqlite3.Error):
        parser.exit(2, 'Could not read pilot metrics. Check the database path, schema, and days (1–30).\n')
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
