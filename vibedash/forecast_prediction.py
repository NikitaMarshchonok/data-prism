"""Bounded, transient CSV input for web inference from an owned frozen model."""
import csv
import io

import pandas as pd

from src.forecast_artifact import ForecastHistoryError
from src.forecasting import MAX_ROWS


MAX_HISTORY_BYTES = 2 * 1024 * 1024
MAX_HISTORY_REQUEST_BYTES = MAX_HISTORY_BYTES + 64 * 1024
MAX_HISTORY_COLUMNS = 64


class ForecastHistoryTooLarge(ForecastHistoryError):
    """Input exceeds the declared web size budget."""


def load_prediction_history(upload):
    """Validate record widths before pandas can infer an index or mangle a header."""
    if upload is None or not (upload.filename or '').lower().endswith('.csv'):
        raise ForecastHistoryError('Choose one UTF-8 CSV file with the full updated history.')
    raw = upload.stream.read(MAX_HISTORY_BYTES + 1)
    if len(raw) > MAX_HISTORY_BYTES:
        raise ForecastHistoryTooLarge('The history CSV must be no larger than 2 MiB.')
    if b'\x00' in raw:
        raise ForecastHistoryError('Upload a valid UTF-8 CSV without binary content.')
    try:
        text = raw.decode('utf-8-sig')
        records = csv.reader(io.StringIO(text, newline=''), strict=True)
        header = next(records, [])
        names = [name.strip() for name in header]
        if not 2 <= len(names) <= MAX_HISTORY_COLUMNS or not all(names) or len(set(names)) != len(names):
            raise ForecastHistoryError('Use 2–64 nonempty, uniquely named CSV columns.')
        rows = 0
        for rows, record in enumerate(records, 1):
            if rows > MAX_ROWS:
                raise ForecastHistoryError(f'Use at most {MAX_ROWS:,} daily history rows.')
            if len(record) != len(header):
                raise ForecastHistoryError('Every CSV record must have the same number of fields as the header.')
        if rows == 0:
            raise ForecastHistoryError('The CSV has no daily observations.')
        frame = pd.read_csv(io.StringIO(text), skip_blank_lines=False)
        if list(frame.columns) != header or len(frame) != rows or not isinstance(frame.index, pd.RangeIndex):
            raise ForecastHistoryError('Use a comma-delimited CSV with one header and consistent data records.')
        return frame
    except (UnicodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError):
        raise ForecastHistoryError('The file could not be read as a valid comma-delimited UTF-8 CSV.') from None
