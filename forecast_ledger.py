"""Explicit local forecast journal operations; no network or automatic training."""
import argparse
import json
from pathlib import Path
import sqlite3

from src.forecast_ledger import ForecastLedger


def read_actuals(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate actual fields.')
            result[key] = value
        return result

    def reject(value):
        raise ValueError('Actuals must use finite JSON numbers.')

    with Path(path).expanduser().open('rb') as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError('Actuals input exceeds 64 KiB.')
    try:
        payload = json.loads(raw.decode('utf-8'), object_pairs_hook=unique, parse_constant=reject)
    except (RecursionError, OverflowError) as error:
        raise ValueError('Invalid actuals input.') from error
    if type(payload) is not dict or set(payload) != {'as_of_date', 'actuals'}:
        raise ValueError('Provide exactly as_of_date and actuals.')
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description='Manage an opt-in private local forecast/outcome journal.')
    parser.add_argument('--ledger', type=Path, required=True, help='Private journal directory; initialize explicitly before use.')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('init', help='Create a new private directory/database; never adopt or overwrite an existing path.')
    for name in ('report', 'actuals', 'revisions', 'delete'):
        command = commands.add_parser(name)
        command.add_argument('--prediction-id', required=True)
        if name == 'actuals':
            command.add_argument('--input', type=Path, required=True, help='UTF-8 JSON: as_of_date and 1–90 date/actual rows.')
            command.add_argument('--correction-reason', help='Required to revise an existing value; the old revision remains.')
        if name == 'delete':
            command.add_argument('--confirm-delete', action='store_true', help='Explicit logical deletion, not secure erasure or backup deletion.')
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            ForecastLedger.create(args.ledger)
            report = {'status': 'initialized', 'storage': 'private_local_only'}
        else:
            ledger = ForecastLedger(args.ledger)
            if args.command == 'actuals':
                payload = read_actuals(args.input)
                report = ledger.accept_actuals(args.prediction_id, payload['actuals'],
                                               as_of_date=payload['as_of_date'],
                                               correction_reason=args.correction_reason)
            elif args.command == 'report':
                report = ledger.evaluate(args.prediction_id)
            elif args.command == 'revisions':
                report = {'status': 'ok', 'revisions': ledger.revisions(args.prediction_id)}
            else:
                if not args.confirm_delete:
                    raise ValueError('Explicit deletion confirmation required.')
                report = ledger.delete(args.prediction_id)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, sqlite3.Error,
            FloatingPointError, OverflowError, RecursionError):
        print(json.dumps({'status': 'error', 'message': 'Local journal operation failed; no successful result is asserted.'}))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
